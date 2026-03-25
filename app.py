"""
LLM 기반 주식 투자 신호 시스템 — Streamlit 앱

실행: streamlit run app.py
"""

import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from utils import TICKERS, EXPERIMENT_DIR

# ── 페이지 설정 ────────────────────────────────────────────
st.set_page_config(
    page_title="LLM 주식 신호 시스템",
    page_icon="📈",
    layout="wide",
)


# ── DART 캐시 점검 (앱 시작 시 1회) ───────────────────────
@st.cache_resource
def _check_dart_cache() -> str:
    """DART corp_codes pkl 캐시 유효성 점검.

    오늘 날짜 캐시가 없거나 읽기 실패 시 구 캐시를 삭제하고
    OpenDartReader 초기화로 재생성.

    Returns:
        "" : 정상
        str: 오류 메시지 (재생성 실패 시)
    """
    docs_cache = os.path.join(os.path.dirname(__file__), "docs_cache")
    today_fn = os.path.join(
        docs_cache,
        f"opendartreader_corp_codes_{datetime.today().strftime('%Y%m%d')}.pkl",
    )

    # 오늘 날짜 캐시가 있으면 읽기 테스트
    if os.path.exists(today_fn):
        try:
            pd.read_pickle(today_fn)
            return ""  # 정상
        except Exception:
            pass  # 호환 불가 → 아래에서 삭제 후 재생성

    # 구 캐시(오늘 것 포함) 전체 삭제
    for old in glob.glob(os.path.join(docs_cache, "opendartreader_corp_codes_*.pkl")):
        try:
            os.remove(old)
        except OSError:
            pass

    # OpenDartReader 재초기화 → 캐시 자동 재생성
    try:
        import OpenDartReader as _odr
        dart_key = os.environ.get("DARTS_API_KEY", "")
        if not dart_key:
            return "DARTS_API_KEY 환경변수가 설정되지 않았습니다."
        _odr(api_key=dart_key)
        return ""
    except Exception as e:
        return f"DART 캐시 재생성 실패: {e}"


_dart_warn = _check_dart_cache()
if _dart_warn:
    st.warning(f"DART 초기화 경고: {_dart_warn}")


# ── 상수 ──────────────────────────────────────────────────
COND_LABELS = {
    "cond1": "cond1 — 종목명만",
    "cond2": "cond2 — + 재무지표",
    "cond3": "cond3 — + 애널리스트 리포트",
    "cond4": "cond4 — + DART 실적 (권장)",
}

SIGNAL_STYLE = {
    "Buy":     ("초록", "#d4edda", "#155724"),
    "Sell":    ("빨강", "#f8d7da", "#721c24"),
    "Neutral": ("회색", "#e2e3e5", "#383d41"),
}


# ── 헬퍼 ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def load_backtest_results(cond: str) -> pd.DataFrame | None:
    """results/experiment/{cond}/latest/{cond}_results.csv 로드."""
    path = os.path.join(EXPERIMENT_DIR, cond, "latest", f"{cond}_results.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype={"ticker": str})
    except Exception:
        return None


def get_ticker_backtest(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """ticker 컬럼 형식(int 또는 zero-padded str) 무관하게 필터링."""
    # 저장 형식이 int일 수 있어 양쪽 비교
    ticker_int = str(int(ticker)) if ticker.isdigit() else ticker
    mask = df["ticker"].astype(str).str.lstrip("0") == ticker_int.lstrip("0")
    return df[mask]


def fmt_val(val, suffix="", decimals=1, na_str="N/A") -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return na_str
    return f"{val:.{decimals}f}{suffix}"


def signal_badge(signal: str) -> str:
    _, bg, fg = SIGNAL_STYLE.get(signal, ("", "#e2e3e5", "#383d41"))
    label = {"Buy": "매수 (Buy)", "Sell": "매도 (Sell)", "Neutral": "중립 (Neutral)"}.get(signal, signal)
    return (
        f'<div style="background:{bg};color:{fg};padding:20px 30px;'
        f'border-radius:12px;text-align:center;font-size:2rem;font-weight:bold;'
        f'margin:10px 0;">{label}</div>'
    )


# ── 사이드바 ───────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ 분석 설정")
    st.divider()

    ticker_options = [f"{name} ({ticker})" for name, ticker in TICKERS.items()]
    selected_label = st.selectbox("종목 선택", ticker_options, index=0)
    selected_name, selected_ticker = selected_label.rsplit(" (", 1)
    selected_ticker = selected_ticker.rstrip(")")

    selected_cond = st.selectbox(
        "분석 조건",
        options=list(COND_LABELS.keys()),
        format_func=lambda k: COND_LABELS[k],
        index=3,  # cond4 기본
    )

    st.divider()
    analyze_btn = st.button("🔍 분석하기", use_container_width=True, type="primary")

    st.divider()
    st.caption("분석 시 최신 데이터 자동 갱신 후 Gemini API를 호출합니다.")


# ── 메인 화면 ──────────────────────────────────────────────
st.title("📈 LLM 기반 주식 투자 신호 시스템")
st.caption("Google Gemini 기반 | 향후 20거래일 방향성 예측")

if not analyze_btn:
    st.info("사이드바에서 종목과 분석 조건을 선택한 뒤 **🔍 분석하기** 버튼을 눌러주세요.")
    st.stop()


# ── 분석 실행 ──────────────────────────────────────────────
import concurrent.futures
from forward_test import run_forward

result = None
with st.status("분석 중...", expanded=True) as _status:
    try:
        _status.write(f"**{selected_name}** 실시간 데이터 수집 중 (FDR / DART)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
            _future = _ex.submit(run_forward, selected_ticker, selected_cond)
            try:
                result = _future.result(timeout=180)   # 최대 3분
            except concurrent.futures.TimeoutError:
                _status.update(label="시간 초과", state="error", expanded=True)
                st.error(
                    "분석 시간이 초과되었습니다 (3분). "
                    "FDR 또는 DART API가 응답하지 않을 수 있습니다. "
                    "잠시 후 다시 시도해주세요."
                )
                st.stop()

        _status.write("LLM 신호 분석 완료!")
        _status.update(label="분석 완료", state="complete", expanded=False)

    except Exception as _e:
        _status.update(label="오류 발생", state="error", expanded=True)
        st.error(f"**{type(_e).__name__}**: {_e}")
        st.stop()

if result is None:
    st.stop()


# ── 1. 상단: 종목 정보 ─────────────────────────────────────
st.subheader(f"{result['name']}  ({result['ticker']})")
col_p, col_d, col_c = st.columns(3)
col_p.metric("현재가", f"{int(result['price']):,}원")
col_d.metric("분석 날짜", result["date"])
col_c.metric("데이터 기준일", result.get("context_date", "-"))

st.divider()


# ── 2. 신호 박스 ───────────────────────────────────────────
signal     = result["signal"]
confidence = result["confidence"]

st.markdown(signal_badge(signal), unsafe_allow_html=True)

st.markdown(f"**신뢰도**: {confidence}%")
st.progress(confidence / 100)

st.divider()


# ── 3. 투자 근거 ───────────────────────────────────────────
st.subheader("📋 투자 근거")
reasons = result.get("reasons", [])
if reasons:
    for r in reasons:
        st.markdown(f"- {r}")
else:
    st.caption("근거 없음")

st.divider()


# ── 4. 재무지표 (cond2 이상) ───────────────────────────────
ctx = result.get("context_used", {})
if selected_cond in ("cond2", "cond3", "cond4"):
    st.subheader("📊 재무지표")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("PER",    fmt_val(ctx.get("per")))
    col2.metric("PBR",    fmt_val(ctx.get("pbr")))
    col3.metric("ROE",    fmt_val(ctx.get("roe"), suffix="%"))
    col4.metric("시가총액", fmt_val(ctx.get("market_cap"), suffix="조원", decimals=1))

    col5, col6, col7 = st.columns(3)
    col5.metric("52주 위치",      fmt_val(ctx.get("price_position_52w"), suffix="%"))
    col6.metric("1개월 수익률",   fmt_val(ctx.get("momentum_1m"), suffix="%"))
    col7.metric("거래량 변화율",  fmt_val(ctx.get("volume_change"), suffix="%"))

    st.divider()


# ── 5. 최근 리포트 (cond3 이상) ────────────────────────────
if selected_cond in ("cond3", "cond4"):
    st.subheader("📄 최근 애널리스트 리포트")
    reports = ctx.get("recent_reports", [])
    if reports:
        rows = []
        for r in reports:
            tp = r.get("target_price")
            tp_str = f"{tp:,}원" if tp else "-"
            rows.append({"제목": r["title"], "목표주가": tp_str})
        st.table(pd.DataFrame(rows))
    else:
        st.caption("최근 30일 이내 리포트 없음")

    st.divider()


# ── 6. 분기 실적 (cond4만) ─────────────────────────────────
if selected_cond == "cond4":
    st.subheader("🏭 분기 실적 (DART 기준)")

    col_a, col_b, col_c2 = st.columns(3)
    rev_growth = ctx.get("revenue_growth")
    op_margin  = ctx.get("operating_margin")
    debt       = ctx.get("debt_ratio")
    div_yield  = ctx.get("dividend_yield")

    col_a.metric(
        "매출 성장률 (YoY)",
        fmt_val(rev_growth, suffix="%"),
        delta=f"{rev_growth:+.1f}%" if rev_growth is not None else None,
    )
    col_b.metric("영업이익률",  fmt_val(op_margin, suffix="%"))
    col_c2.metric("부채비율",   fmt_val(debt, suffix="%"))

    st.metric("배당수익률", fmt_val(div_yield, suffix="%"))

    st.divider()


# ── 7. 백테스팅 성과 ───────────────────────────────────────
st.subheader("📉 백테스팅 성과")

bt_df = load_backtest_results(selected_cond)

if bt_df is None:
    st.info(f"{selected_cond} 실험 결과가 아직 없습니다. `python src/llm_experiment.py --cond {selected_cond}` 실행 후 확인하세요.")
else:
    ticker_df = get_ticker_backtest(bt_df, selected_ticker)

    if ticker_df.empty:
        st.caption(f"{selected_name}의 {selected_cond} 이력 없음")
    else:
        total   = len(ticker_df)
        buy_df  = ticker_df[ticker_df["signal"] == "Buy"]
        sell_df = ticker_df[ticker_df["signal"] == "Sell"]

        # 신호별 히트율 계산
        def hit_rate(df: pd.DataFrame, direction: str) -> float | None:
            if df.empty:
                return None
            if direction == "Buy":
                return (df["return_20d"] > 0).mean() * 100
            elif direction == "Sell":
                return (df["return_20d"] < 0).mean() * 100
            return None

        buy_hr   = hit_rate(buy_df, "Buy")
        sell_hr  = hit_rate(sell_df, "Sell")
        avg_ret  = ticker_df["return_20d"].mean()

        st.caption(f"조건: **{COND_LABELS[selected_cond]}** | 총 **{total}**개월 백테스트 이력")

        col_bt1, col_bt2, col_bt3 = st.columns(3)
        col_bt1.metric(
            "Buy 히트율",
            f"{buy_hr:.1f}%" if buy_hr is not None else "N/A",
            help="Buy 신호 후 20거래일 수익률 > 0 비율",
        )
        col_bt2.metric(
            "Sell 히트율",
            f"{sell_hr:.1f}%" if sell_hr is not None else "N/A",
            help="Sell 신호 후 20거래일 수익률 < 0 비율",
        )
        col_bt3.metric(
            "평균 20일 수익률",
            f"{avg_ret:+.2f}%",
        )

        # 신호별 상세 테이블
        with st.expander("신호별 상세 통계"):
            stats = (
                ticker_df.groupby("signal")["return_20d"]
                .agg(
                    건수="count",
                    평균수익률="mean",
                    히트율=lambda x: (x > 0).mean() * 100,
                )
                .round(2)
            )
            st.dataframe(stats)

        # 시계열 차트
        with st.expander("수익률 추이"):
            chart_df = (
                ticker_df[["signal_date", "return_20d", "signal"]]
                .sort_values("signal_date")
                .set_index("signal_date")
            )
            st.line_chart(chart_df[["return_20d"]])

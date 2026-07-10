"""
LLM 기반 주식 투자 신호 시스템 — Streamlit 앱

실행: streamlit run app.py

탭 구성:
  ① 개별 종목 분석 — 종목·조건·모델 선택 후 실시간 신호 생성 (당일 캐시)
  ② 전체 종목 한눈에 — forward 캐시(20종목×5조건) 신호 매트릭스 (API 호출 없음)
  ③ 백테스트 성과·모델 비교 — results/analysis 기반 모델별 성과·유의성 (API 호출 없음)
"""

import glob
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# .env를 앱 시작 시점에 로드 — forward_test는 버튼 클릭 시 lazy import되므로
# _check_dart_cache()가 실행되는 시점엔 아직 키가 환경에 없어 경고가 뜬다
load_dotenv()

_src = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, os.path.join(_src, "collect"))
sys.path.insert(0, os.path.join(_src, "experiment"))
sys.path.insert(0, _src)

from utils import TICKERS, EXPERIMENT_DIR, FORWARD_DIR, ANALYSIS_DIR
from compare import COND_LABELS, COND_ORDER, DEFAULT_MODEL

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
        from opendartreader import OpenDartReader as _odr
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
# 연구용 조건(cond4_no_reports, cond4_blind)은 개별 분석 UI 미노출 — 사용자 편의 조건만 표시
UI_CONDS = ["cond1", "cond2", "cond3", "cond4"]

# 개별 분석용 모델 목록 — 앵커(DEFAULT_MODEL)/gemma가 우선(앞 배치), gpt/claude는 별도 키·호출 비용 필요
UI_MODELS = [DEFAULT_MODEL, "gemma-4-31b-it", "gpt-5.4-mini", "claude-haiku-4-5"]

SIGNAL_STYLE = {
    "Buy":     ("초록", "#d4edda", "#155724"),
    "Sell":    ("빨강", "#f8d7da", "#721c24"),
    "Neutral": ("회색", "#e2e3e5", "#383d41"),
}


# ── 헬퍼 ──────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)  # 백테스팅 결과 5분 캐시 — 빈번한 파일 재로드 방지
def load_backtest_results(cond: str, model: str) -> pd.DataFrame | None:
    """results/experiment/{cond}/{model}/latest/{cond}_results.csv 로드."""
    path = os.path.join(EXPERIMENT_DIR, cond, model, "latest", f"{cond}_results.csv")
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


def signal_cell_style(v) -> str:
    """신호 매트릭스 셀 색상 — 'Buy 85%' 형태 문자열의 접두어로 판별."""
    if isinstance(v, str):
        for sig, (_, bg, fg) in SIGNAL_STYLE.items():
            if v.startswith(sig):
                return f"background-color:{bg};color:{fg}"
    return ""


def list_forward_dates() -> list[str]:
    """results/forward/ 하위의 날짜(YYYY-MM-DD) 폴더 목록 (최신순)."""
    if not os.path.isdir(FORWARD_DIR):
        return []
    dates = [
        d for d in os.listdir(FORWARD_DIR)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)
        and os.path.isdir(os.path.join(FORWARD_DIR, d))
    ]
    return sorted(dates, reverse=True)


def list_forward_models(date: str) -> list[str]:
    """해당 날짜 폴더의 모델 하위폴더 목록."""
    base = os.path.join(FORWARD_DIR, date)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


@st.cache_data(show_spinner=False)
def load_forward_signals(date: str, model: str, file_names: tuple[str, ...]) -> pd.DataFrame:
    """forward 캐시 JSON들 → long DataFrame(ticker/name/cond/signal/confidence).

    cond에 언더스코어 포함(cond4_no_reports) → 파일명 파싱 대신 JSON 본문에서 읽음.
    손상 파일은 개별 skip. file_names가 캐시 키에 포함 — 탭1에서 새 신호가
    생성되면(파일 추가) 즉시 재로드되고, 불변 폴더는 재로드 없음.
    """
    rows = []
    for fname in file_names:
        try:
            with open(os.path.join(FORWARD_DIR, date, model, fname), encoding="utf-8") as f:
                d = json.load(f)
            rows.append({
                "ticker":     str(d["ticker"]).zfill(6),
                "name":       d["name"],
                "cond":       d["cond"],
                "signal":     d["signal"],
                "confidence": d["confidence"],
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def list_analysis_models() -> list[str]:
    """results/analysis/ 하위에서 latest/all_comparison.csv가 있는 모델 폴더만.

    레거시 최상위 날짜 폴더(20260411 등)는 자연히 걸러짐.
    """
    if not os.path.isdir(ANALYSIS_DIR):
        return []
    return sorted(
        m for m in os.listdir(ANALYSIS_DIR)
        if os.path.exists(os.path.join(ANALYSIS_DIR, m, "latest", "all_comparison.csv"))
    )


@st.cache_data(ttl=300, show_spinner=False)
def load_analysis_csv(model: str, filename: str) -> pd.DataFrame | None:
    """results/analysis/{model}/latest/{filename} 로드. 없으면 None."""
    path = os.path.join(ANALYSIS_DIR, model, "latest", filename)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


# ── 메인 화면 ──────────────────────────────────────────────
st.title("📈 LLM 기반 주식 투자 신호 시스템")
st.caption("멀티모델 LLM 기반 | 향후 20거래일 방향성 예측")

tab1, tab2, tab3 = st.tabs(["🔍 개별 종목 분석", "📋 전체 종목 한눈에", "📊 백테스트 성과·모델 비교"])


# ═══════════════════════════════════════════════════════════
# 탭 1 — 개별 종목 분석
# ═══════════════════════════════════════════════════════════
with tab1:
    col_t, col_c, col_m, col_b = st.columns([3, 3, 3, 2])

    ticker_options = [f"{name} ({ticker})" for name, ticker in TICKERS.items()]
    selected_label = col_t.selectbox("종목 선택", ticker_options, index=0)
    selected_name, selected_ticker = selected_label.rsplit(" (", 1)
    selected_ticker = selected_ticker.rstrip(")")

    selected_cond = col_c.selectbox(
        "분석 조건",
        options=UI_CONDS,
        format_func=lambda k: COND_LABELS[k] + (" — 권장" if k == "cond4" else ""),
        index=3,  # cond4 기본
    )

    selected_model = col_m.selectbox(
        "모델",
        options=UI_MODELS,
        format_func=lambda m: f"{m} (기본)" if m == UI_MODELS[0] else m,
        index=0,
    )

    col_b.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)  # 버튼 세로 정렬
    analyze_btn = col_b.button("🔍 분석하기", use_container_width=True, type="primary")

    # provider 판별은 llm_experiment._provider와 동일하게 접두어 기준 — 모델 개명/추가에 안전
    if selected_model.startswith("gemma"):
        st.caption("gemma는 rate-limit으로 응답이 지연될 수 있습니다 (재시도 대기 포함 최대 3분).")
    elif selected_model.startswith(("gpt", "claude")):
        st.caption("이 모델은 해당 provider의 API 키(.env)와 소액의 호출 비용이 필요합니다.")
    else:
        st.caption("분석 시 최신 데이터 자동 갱신 후 선택한 모델의 API를 호출합니다.")

    # ── 분석 실행 ──────────────────────────────────────────
    if analyze_btn:
        import concurrent.futures
        from forward_test import run_forward

        with st.status("분석 중...", expanded=True) as _status:
            try:
                _status.write(f"**{selected_name}** 실시간 데이터 수집 중 (FDR / DART)...")

                # with 블록 대신 shutdown(wait=False) — with exit은 워커 완료까지 블로킹해
                # Future.result(timeout=)의 타임아웃이 무의미해짐
                _ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                _future = _ex.submit(run_forward, selected_ticker, selected_cond, selected_model)
                try:
                    result = _future.result(timeout=180)   # 최대 3분
                    st.session_state["fw_result"] = result
                    _status.write("LLM 신호 분석 완료!")
                    _status.update(label="분석 완료", state="complete", expanded=False)
                except concurrent.futures.TimeoutError:
                    _status.update(label="시간 초과", state="error", expanded=True)
                    st.error(
                        "분석 시간이 초과되었습니다 (3분). "
                        "FDR/DART API 미응답 또는 모델 rate-limit 재시도 대기일 수 있습니다. "
                        "백그라운드 호출이 완료되면 당일 캐시에 저장되므로, 잠시 후 다시 시도하면 즉시 표시될 수 있습니다."
                    )
                finally:
                    _ex.shutdown(wait=False)  # 진행 중 워커를 기다리지 않고 즉시 반환
            except Exception as _e:
                _status.update(label="오류 발생", state="error", expanded=True)
                st.error(f"**{type(_e).__name__}**: {_e}")

    fw = st.session_state.get("fw_result")
    if fw is None:
        st.info("종목·분석 조건·모델을 선택한 뒤 **🔍 분석하기** 버튼을 눌러주세요.")
    else:
        # 렌더링 분기는 셀렉트박스가 아닌 마지막 분석 결과(fw) 기준 — 위젯 변경 후 불일치 방지
        fw_cond  = fw["cond"]
        fw_model = fw.get("model", "-")
        ctx = fw.get("context_used", {})

        # ── 1. 상단: 종목 정보 ─────────────────────────────
        # 조건·모델을 헤더에 명시 — 셀렉트박스를 바꾸고 재분석 안 한 상태에서
        # 아래 패널이 어떤 설정의 결과인지 오인하지 않도록
        st.subheader(f"{fw['name']}  ({fw['ticker']})")
        col_p, col_d, col_cd, col_mo = st.columns(4)
        col_p.metric("현재가", f"{int(fw['price']):,}원")
        col_d.metric("분석 날짜", fw["date"])
        col_cd.metric("분석 조건", fw_cond)
        col_mo.metric("사용 모델", fw_model)

        st.divider()

        # ── 2. 신호 박스 ───────────────────────────────────
        st.markdown(signal_badge(fw["signal"]), unsafe_allow_html=True)
        st.markdown(f"**신뢰도**: {fw['confidence']}%")
        st.progress(fw["confidence"] / 100)

        st.divider()

        # ── 3. 투자 근거 ───────────────────────────────────
        st.subheader("📋 투자 근거")
        reasons = fw.get("reasons", [])
        if reasons:
            for r in reasons:
                st.markdown(f"- {r}")
        else:
            st.caption("근거 없음")

        st.divider()

        # ── 4. 재무지표 (cond2 이상) ───────────────────────
        if fw_cond in ("cond2", "cond3", "cond4"):
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

        # ── 5. 최근 리포트 (cond3 이상) ────────────────────
        if fw_cond in ("cond3", "cond4"):
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

        # ── 6. DART 실적 (cond4만) ─────────────────────────
        if fw_cond == "cond4":
            fp = ctx.get("fiscal_period")
            rn = ctx.get("report_name")
            if fp and rn:
                st.subheader(f"🏭 {fp} 실적 (DART {rn})")
            else:
                st.subheader("🏭 최근 실적 (DART 정기보고서)")  # 구 캐시 JSON — 기간 필드 없음

            col_a, col_b2, col_c2 = st.columns(3)
            rev_growth = ctx.get("revenue_growth")
            op_margin  = ctx.get("operating_margin")
            debt       = ctx.get("debt_ratio")
            div_yield  = ctx.get("dividend_yield")

            col_a.metric(
                "매출 성장률 (YoY)",
                fmt_val(rev_growth, suffix="%"),
                delta=f"{rev_growth:+.1f}%" if (rev_growth is not None and not pd.isna(rev_growth)) else None,
            )
            col_b2.metric("영업이익률",  fmt_val(op_margin, suffix="%"))
            col_c2.metric("부채비율",   fmt_val(debt, suffix="%"))

            st.metric("배당수익률", fmt_val(div_yield, suffix="%"))

            st.divider()

        # ── 7. 백테스팅 성과 ───────────────────────────────
        st.subheader("📉 백테스팅 성과")

        bt_df = load_backtest_results(fw_cond, fw_model)

        if bt_df is None:
            st.info(
                f"{fw_cond} × {fw_model} 실험 결과가 아직 없습니다. "
                f"`python src/experiment/llm_experiment.py --cond {fw_cond} --model {fw_model}` 실행 후 확인하세요."
            )
        else:
            ticker_df = get_ticker_backtest(bt_df, fw["ticker"])

            if ticker_df.empty:
                st.caption(f"{fw['name']}의 {fw_cond} 이력 없음")
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

                st.caption(f"조건: **{COND_LABELS[fw_cond]}** | 모델: **{fw_model}** | 총 **{total}**개월 백테스트 이력")

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
                        .agg(건수="count", 평균수익률="mean")
                        .round(2)
                    )
                    for sig, g in ticker_df.groupby("signal")["return_20d"]:
                        hr = (g < 0).mean() if sig == "Sell" else (g > 0).mean()
                        stats.loc[sig, "히트율"] = round(hr * 100, 2)
                    st.dataframe(stats)

                # 시계열 차트
                with st.expander("수익률 추이"):
                    chart_df = (
                        ticker_df[["signal_date", "return_20d", "signal"]]
                        .sort_values("signal_date")
                        .set_index("signal_date")
                    )
                    st.line_chart(chart_df[["return_20d"]])


# ═══════════════════════════════════════════════════════════
# 탭 2 — 전체 종목 한눈에 (forward 캐시 읽기 전용, API 호출 없음)
# ═══════════════════════════════════════════════════════════
with tab2:
    fw_dates = list_forward_dates()
    if not fw_dates:
        st.info(
            "forward 신호 캐시가 없습니다. "
            "`python src/experiment/forward_run_all.py` 실행 후 확인하세요."
        )
    else:
        col_fd, col_fm = st.columns(2)
        sel_date = col_fd.selectbox("신호 생성 날짜", fw_dates, index=0)
        fw_models = list_forward_models(sel_date)

        if not fw_models:
            st.info(f"{sel_date} 폴더에 모델 하위폴더가 없습니다.")
            sig_df = pd.DataFrame()
        else:
            # 앵커 모델이 있으면 기본 선택 (폴더명 알파벳순이라 claude가 먼저 옴)
            _default_idx = fw_models.index(UI_MODELS[0]) if UI_MODELS[0] in fw_models else 0
            sel_fw_model = col_fm.selectbox("모델", fw_models, index=_default_idx)

            _fw_folder = os.path.join(FORWARD_DIR, sel_date, sel_fw_model)
            _file_names = tuple(sorted(f for f in os.listdir(_fw_folder) if f.endswith(".json")))
            sig_df = load_forward_signals(sel_date, sel_fw_model, _file_names)

            if sig_df.empty:
                st.info(f"{sel_date} 캐시에 신호 데이터가 없습니다.")

        if not sig_df.empty:
            st.caption(f"**{sel_date}** 생성 캐시 기준 · {sig_df['ticker'].nunique()}종목 × {sig_df['cond'].nunique()}조건 (API 호출 없음)")

            # 신호 분포 요약
            dist = pd.crosstab(sig_df["cond"], sig_df["signal"])
            dist = dist.reindex(index=[c for c in COND_ORDER if c in dist.index],
                                columns=[s for s in ("Buy", "Neutral", "Sell") if s in dist.columns])
            st.markdown("**신호 분포 (조건 × 신호)**")
            st.dataframe(dist, use_container_width=True)

            # 종목 × 조건 신호 매트릭스
            st.markdown("**종목별 신호 매트릭스**")
            mat_df = sig_df.copy()
            mat_df["종목"] = mat_df["name"] + " (" + mat_df["ticker"] + ")"
            mat_df["cell"] = mat_df["signal"] + " " + mat_df["confidence"].astype(str) + "%"
            # 잔류 중복 파일(수동 백업본 등) 방어 — pivot은 중복 인덱스에서 ValueError
            mat_df = mat_df.drop_duplicates(subset=["ticker", "cond"], keep="last")
            pivot = mat_df.pivot(index="종목", columns="cond", values="cell")
            pivot = pivot[[c for c in COND_ORDER if c in pivot.columns]]
            # TICKERS 정의 순서로 행 정렬
            row_order = [f"{name} ({ticker})" for name, ticker in TICKERS.items() if f"{name} ({ticker})" in pivot.index]
            pivot = pivot.reindex(row_order).fillna("-")
            st.dataframe(pivot.style.map(signal_cell_style), use_container_width=True, height=740)


# ═══════════════════════════════════════════════════════════
# 탭 3 — 백테스트 성과·모델 비교 (analysis 캐시 읽기 전용)
# ═══════════════════════════════════════════════════════════
with tab3:
    an_models = list_analysis_models()
    if not an_models:
        st.info(
            "분석 결과가 없습니다. "
            "`python src/experiment/compare.py --all --model <모델명>` 실행 후 확인하세요."
        )
    else:
        # ── A. Buy 신호 성과 — 모델 비교 ────────────────────
        st.subheader("🎯 Buy 신호 성과 — 모델 비교 (20거래일)")
        buy_cols = st.columns(len(an_models))
        BUY_RENAME = {
            "label": "조건", "n": "신호수", "mean": "평균수익률(%)", "hit_rate": "Hit(%)",
            "sharpe": "Sharpe", "mean_excess": "초과수익(%)", "hit_rate_excess": "초과Hit(%)",
            "conf_mean": "평균신뢰도",
        }
        for col, m in zip(buy_cols, an_models):
            with col:
                st.markdown(f"**{m}**")
                comp = load_analysis_csv(m, "all_comparison.csv")
                if comp is None:
                    st.caption("해당 모델 결과 없음")
                    continue
                buy = comp[(comp["signal"] == "Buy") & (comp["label"].isin(COND_ORDER))].copy()
                buy["label"] = buy["label"].map(lambda k: COND_LABELS.get(k, k))
                # 구 스키마 CSV 방어 — 존재하는 컬럼만 선택 (섹션 B와 동일 패턴)
                buy = buy[[c for c in BUY_RENAME if c in buy.columns]].rename(columns=BUY_RENAME)
                st.dataframe(buy.set_index("조건").round(2), use_container_width=True)

        st.divider()

        # ── B. 베이스라인 대비 전체 성과 ────────────────────
        st.subheader("⚖️ 베이스라인 대비 전체 성과 (전 신호, 20거래일)")
        st.caption("컨센서스(애널리스트 투자의견)·골든크로스(기술분석)가 베이스라인")
        FULL_RENAME = {
            "strategy": "전략", "n": "신호수", "mean_ret": "평균수익률(%)", "hit_rate": "Hit(%)",
            "sharpe": "Sharpe", "excess_mean": "초과수익(%)", "excess_hit_rate": "초과Hit(%)",
            "excess_sharpe": "초과Sharpe",
        }
        full_cols = st.columns(len(an_models))
        for col, m in zip(full_cols, an_models):
            with col:
                st.markdown(f"**{m}**")
                full = load_analysis_csv(m, "full_comparison.csv")
                if full is None:
                    st.caption("해당 모델 결과 없음")
                    continue
                full = full[[c for c in FULL_RENAME if c in full.columns]].rename(columns=FULL_RENAME)
                st.dataframe(full.set_index("전략").round(2), use_container_width=True)

        st.divider()

        # ── C. 통계적 유의성 ────────────────────────────────
        st.subheader("📐 통계적 유의성 검정 (Buy 신호)")
        sel_an_model = st.selectbox("검정 결과 모델", an_models, index=0)
        sig = load_analysis_csv(sel_an_model, "all_significance.csv")
        if sig is None:
            st.caption("해당 모델 검정 결과 없음")
        else:
            # 헤드라인: cond4 vs 컨센서스 (Mann-Whitney, 절대수익)
            head = sig[
                (sig["group_a"] == "cond4") & (sig["group_b"] == "Consensus")
                & (sig["test"] == "mann_whitney") & (sig["metric"] == "return_20d")
            ]
            if not head.empty:
                r = head.iloc[0]
                # two-sided 검정이라 유의성만으로는 방향을 모름 — mean_diff 부호로 우위/열위 명시
                _direction = "우위" if r["mean_diff"] > 0 else "열위"
                st.metric(
                    "cond4 vs 컨센서스 (Mann-Whitney, 20d 절대수익)",
                    f"p = {r['p_value']:.4f} {r['significance']}",
                    delta=f"{r['mean_diff']:+.2f}%p (cond4 {_direction})",
                    help="two-sided 검정 — p<0.05는 '차이가 유의함'을 의미하며, 방향(우위/열위)은 평균 차이 부호 기준",
                )

            core = sig[sig["category"] == "core"].copy()
            styled = core.style.map(
                lambda v: "background-color:#d4edda" if isinstance(v, (int, float)) and v < 0.05 else "",
                subset=["p_value"],
            )
            st.dataframe(styled, use_container_width=True, hide_index=True)
            st.caption("유의 수준: *** p<0.001 · ** p<0.01 · * p<0.05 · . p<0.10 · ns 비유의")

            with st.expander("전체 검정 결과 (보조 비교 포함)"):
                st.dataframe(sig, use_container_width=True, hide_index=True)

        # ── D. 연도별·시장국면별 분석 ───────────────────────
        with st.expander("📅 연도별·시장국면별 분석 (breakdown)"):
            yearly = load_analysis_csv(sel_an_model, "breakdown_yearly.csv")
            regime = load_analysis_csv(sel_an_model, "breakdown_regime.csv")
            if yearly is not None:
                st.markdown("**연도별**")
                st.dataframe(yearly, use_container_width=True, hide_index=True)
            if regime is not None:
                st.markdown("**시장 국면별 (상승/하락)**")
                st.dataframe(regime, use_container_width=True, hide_index=True)
            if yearly is None and regime is None:
                st.caption("해당 모델 breakdown 결과 없음")

"""탭 2 — 개별 종목 분석 (오늘 기준 실시간 신호 생성, LLM API 호출).

다섯 탭 중 유일하게 외부 API를 쓴다. 생성된 신호는 정식 평가 표본에 섞이지 않도록
results/forward_demo/로 격리한다(demote_to_demo 참고).

대상 종목은 백테스트 20종목이 아니라 KRX 상장 보통주 전 종목이다. 백테스트·forward는 방법을
검증하는 통제 실험이고, 실제 분석은 아무 종목에나 적용되어야 하기 때문. 대신 20종목
밖에서는 ① 과거 성과를 붙일 수 없고 ② 리포트 커버리지가 떨어지므로, 둘 다 화면에서
명시한다(숨기면 부실을 감추는 것이 된다).
"""

import concurrent.futures
import glob
import json
import os
import threading
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app_ui import ROOT_DIR
from app_ui.shared import (
    BACKTEST_TICKERS, COND_LABELS, FORWARD_DEMO_DIR, FORWARD_DIR, REPORTS_DIR,
    SIGNAL_STYLE, TICKERS, UI_CONDS, UI_MODELS, load_backtest_results,
)

# 리포트를 입력에 포함하는 조건 — 리포트 0건이면 실질 입력이 줄어든다는 안내가 필요하다
REPORT_CONDS_UI = ("cond3", "cond4")


# ── DART 캐시 점검 (분석 버튼 클릭 시 1회) ────────────────
@st.cache_resource
def _check_dart_cache() -> str:
    """DART corp_codes pkl 캐시 유효성 점검.

    오늘 날짜 캐시가 없거나 읽기 실패 시 구 캐시를 삭제하고
    OpenDartReader 초기화로 재생성(법인코드 ~11MB 다운로드).

    호출 시점 주의: 앱 시작 시가 아니라 이 탭의 분석 버튼에서 호출한다.
    날짜가 바뀐 첫 실행이면 재생성에 수십 초가 걸리는데, DART가 필요 없는
    나머지 네 탭(캐시 읽기 전용)까지 그 대기에 묶이기 때문.
    @st.cache_resource라 프로세스당 1회만 수행된다.

    Returns:
        "" : 정상
        str: 오류 메시지 (재생성 실패 시)
    """
    docs_cache = os.path.join(ROOT_DIR, "docs_cache")
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


# ── 종목 목록 ─────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)  # 상장 목록은 하루 단위로만 바뀐다
def load_krx_stocks() -> list[tuple[str, str]]:
    """KRX 상장 종목 (표시 라벨, 티커) 목록. 시가총액 내림차순.

    시총 순으로 정렬하는 이유는 셀렉트박스에서 검색 없이 훑을 때 아는 이름이
    먼저 나와야 하기 때문. 라벨에 티커를 붙여 동명 종목이 겹치지 않게 한다.

    우선주는 제외한다. 우선주는 별도 종목코드를 갖지만 DART 재무제표는 보통주 기준
    하나뿐이라 EPS가 매칭되지 않아 PER이 비고(삼성전자우처럼 시총 100조가 넘어도
    마찬가지다), 발행주식수도 보통주 기준이라 시가총액이 어긋난다. 분석이 성립하지
    않는 종목을 목록에 두면 시연에서 빈 화면을 고르게 된다.

    판별은 KRX 종목코드 규약을 쓴다 — 보통주는 끝자리가 0이고 우선주는 5/7/9/K/L 등이다.
    이름 규칙(`...우`로 끝남)은 성우·이오플로우·에코글로우 같은 보통주를 잘못 걸러낸다.
    실측에서 코드 규칙으로 걸린 113개는 전부 이름에도 '우'가 들어가 오탐이 없었다.

    조회 실패 시 백테스트 20종목으로 폴백한다 — 네트워크가 없어도 시연은 되어야 한다.
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX").dropna(subset=["Code", "Name"])
        df = df[df["Code"].str[-1] == "0"]
        if "Marcap" in df.columns:
            df = df.sort_values("Marcap", ascending=False, na_position="last")
        out = [(f"{n} ({c})", c) for c, n in zip(df["Code"], df["Name"])]
        if out:
            return out
    except Exception:
        pass
    return [(f"{n} ({t})", t) for n, t in TICKERS.items()]


def ensure_reports(ticker: str) -> None:
    """20종목 밖 종목의 리포트 캐시를 당일 기준으로 확보한다.

    get_today_context가 data/reports/{ticker}.csv를 직접 읽으므로, 파일이 없으면
    cond3·cond4가 리포트 없이 돌아간다. 30일치만 받는 이유는 forward가 그 창만 쓰기
    때문이다(백테스트용 전체 이력은 crawl.py 담당).

    **파일이 있어도 오늘 받은 것이 아니면 다시 받는다.** 예전에 받아둔 파일은 그때의
    30일 창이라, 시간이 지나면 get_today_context가 보는 창(오늘 기준 30일)과 겹치지
    않아 리포트가 있는 종목이 "리포트 없음"으로 나온다. 제출·시연이 수집일보다 몇 달
    뒤라 실제로 발생하는 경로다. 판정은 파일 수정시각으로 하며(마지막 리포트 날짜로
    하면 원래 리포트가 뜸한 종목을 매번 다시 받게 된다) 하루 1회로 제한된다.

    백테스트 20종목은 건드리지 않는다. 그 CSV는 실험 입력이고 crawl.py가 전체 이력을
    관리하는 파일이라, 앱이 30일치로 덮어쓰면 실험 데이터를 훼손한다.

    실패해도 예외를 올리지 않는다 — 리포트는 없으면 없는 대로 분석이 성립하고,
    화면에서 "리포트 없음"으로 안내된다.
    """
    # TICKERS가 아니라 BACKTEST_TICKERS로 판정한다. 분석 시 종목명을 TICKERS에
    # 주입하므로, TICKERS로 보면 한 번 분석한 종목이 20종목으로 취급돼 이후 영영
    # 리포트를 받지 않는다
    if ticker in BACKTEST_TICKERS:
        return

    path = os.path.join(REPORTS_DIR, f"{ticker}.csv")
    today = datetime.today().date()
    if os.path.exists(path) and datetime.fromtimestamp(os.path.getmtime(path)).date() == today:
        return

    try:
        from crawl import fetch_reports
        since = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        recs = fetch_reports(ticker, since_date=since, max_pages=3)
        if recs:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            pd.DataFrame(recs).to_csv(path, index=False, encoding="utf-8-sig")
        elif os.path.exists(path):
            # 30일 내 리포트가 없는데 옛 파일이 남아 있으면 그 옛 행이 계속 읽힌다.
            # 빈 CSV를 쓰면 get_today_context의 read_csv가 터지므로 파일을 지운다
            os.remove(path)
    except Exception:
        pass


# ── 헬퍼 ──────────────────────────────────────────────────
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


def fmt_market_cap(mc_jo) -> str:
    """context_used의 시가총액(조원 단위) → 화면 표기. 프롬프트 쪽과 같은 기준.

    1조 미만을 조원 소수 1자리로 쓰면 1000억 미만이 전부 "0.0조원"이 된다.
    0은 구 형식(소수 1자리 저장) 캐시에서 뭉개진 값이므로 숫자로 내지 않는다.
    """
    if mc_jo is None or pd.isna(mc_jo) or mc_jo == 0:
        return "N/A"
    if mc_jo >= 1:
        return f"{mc_jo:.1f}조원"
    return f"{mc_jo * 1e4:,.0f}억원"


def signal_badge(signal: str) -> str:
    bg, fg = SIGNAL_STYLE.get(signal, ("#e2e3e5", "#383d41"))
    label = {"Buy": "매수 (Buy)", "Sell": "매도 (Sell)", "Neutral": "중립 (Neutral)"}.get(signal, signal)
    return (
        f'<div style="background:{bg};color:{fg};padding:20px 30px;'
        f'border-radius:12px;text-align:center;font-size:2rem;font-weight:bold;'
        f'margin:10px 0;">{label}</div>'
    )


# ── 수집 데이터 내보내기 ──────────────────────────────────
# 화면에 이미 있는 값(fw)만 쓴다. API·DART 재호출이 없어 버튼이 즉시 반응한다.
#
# 한계를 분명히 해둔다. fw["context_used"]는 LLM 호출 이후 만들어지는 **화면 표시 전용**
# 축약본이라, 프롬프트에 실제로 들어간 52주 최고/최저가와 매출·영업이익·순이익·
# 영업현금흐름의 절대액이 들어 있지 않다. 그래서 이 파일은 "LLM에 투입된 컨텍스트"가
# 아니라 "화면에 표시된 수집 지표"다. 절대액까지 내보내려면 get_today_context를 다시
# 호출해야 하는데(약 6초 + DART 호출) 다운로드 버튼 뒤에 두기엔 무겁다.
EXPORT_FIELDS = [
    ("per",                "PER"),
    ("pbr",                "PBR"),
    ("roe",                "ROE(%)"),
    ("market_cap",         "시가총액(조원)"),
    ("price_position_52w", "52주내위치(%)"),
    ("momentum_1m",        "1개월수익률(%)"),
    ("volume_change",      "거래량변화율(%)"),
    ("revenue_growth",     "매출성장률YoY(%)"),
    ("operating_margin",   "영업이익률(%)"),
    ("debt_ratio",         "부채비율(%)"),
]


def build_export_frame(fw: dict) -> pd.DataFrame:
    """지표를 한 행짜리 wide 표로.

    여러 종목을 받아 그대로 이어붙일 수 있는 형태를 택했다. 항목/값 long 형식은
    사람이 읽기엔 낫지만 종목 간 비교로 쌓을 수 없다.

    결측은 빈 칸으로 둔다. 20종목 밖에서는 PER 결측(적자)·리포트 0건이 정상이라
    임의의 기본값으로 채우면 없는 값을 있는 것처럼 만든다.
    """
    ctx = fw.get("context_used", {})
    row = {
        "종목코드":   fw["ticker"],
        "종목명":     fw["name"],
        "기준일":     fw["date"],
        "현재가":     fw["price"],
        "신호":       fw["signal"],
        "신뢰도(%)":  fw["confidence"],
        "분석조건":   fw["cond"],
        "모델":       fw.get("model", ""),
    }
    row.update({label: ctx.get(key) for key, label in EXPORT_FIELDS})
    return pd.DataFrame([row])


def build_export_markdown(fw: dict) -> str:
    """사람이 읽는 브리핑. 리포트 목록과 판단 근거까지 담는다(CSV는 중첩을 못 담는다)."""
    ctx = fw.get("context_used", {})
    lines = [
        f"# {fw['name']} ({fw['ticker']})",
        "",
        f"- 기준일: {fw['date']}",
        f"- 현재가: {int(fw['price']):,}원",
        f"- 신호: **{fw['signal']}** (신뢰도 {fw['confidence']}%)",
        f"- 분석 조건: {COND_LABELS.get(fw['cond'], fw['cond'])}",
        f"- 모델: {fw.get('model', '-')}",
        "",
        "## 판단 근거",
    ]
    lines += [f"- {r}" for r in fw.get("reasons", [])] or ["- (없음)"]

    lines += ["", "## 수집 지표", "", "| 항목 | 값 |", "|---|---|"]
    for key, label in EXPORT_FIELDS:
        v = ctx.get(key)
        lines.append(f"| {label} | {'' if v is None or pd.isna(v) else v} |")

    reports = ctx.get("recent_reports", [])
    lines += ["", f"## 애널리스트 리포트 ({len(reports)}건)"]
    if reports:
        lines += ["", "| 날짜 | 제목 | 목표주가 |", "|---|---|---|"]
        for r in reports:
            tp = r.get("target_price")
            lines.append(f"| {r.get('date', '')} | {r['title']} | {f'{tp:,}원' if tp else '-'} |")
    else:
        lines.append("")
        lines.append("최근 30일 이내 리포트가 없습니다. 커버리지가 낮은 종목에서 나타납니다.")

    lines += [
        "",
        "---",
        "",
        "화면에 표시된 수집 지표입니다. LLM 프롬프트에 실제로 투입된 원문이 아니며, "
        "매출·영업이익 등의 절대액과 52주 최고/최저가는 포함되지 않습니다.",
    ]
    return "\n".join(lines)


def render_export_section(fw: dict) -> None:
    """CSV / JSON / MD 내려받기 버튼 3개."""
    st.subheader("⬇️ 수집 데이터 내려받기")
    st.caption(
        "이 종목에 대해 시스템이 수집·산출한 값입니다. 화면에 있는 값을 그대로 내보내므로 "
        "추가 API 호출이 없습니다. 백테스트 20종목이 아니라 **KRX 상장 보통주 전 종목**이 대상이며, "
        "커버리지가 낮은 종목에서는 일부 지표가 빈 칸일 수 있습니다."
    )

    stem = f"{fw['ticker']}_{fw['name']}_{fw['date']}"
    col_csv, col_json, col_md = st.columns(3)

    # utf-8-sig — 엑셀에서 한글 헤더가 깨지지 않게 (data/reports CSV와 같은 기준)
    col_csv.download_button(
        "CSV",
        data=build_export_frame(fw).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}.csv",
        mime="text/csv",
        width="stretch",
        help="지표 한 행. 여러 종목을 받아 이어붙이기 좋은 형식입니다.",
    )
    col_json.download_button(
        "JSON",
        data=json.dumps(fw, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name=f"{stem}.json",
        mime="application/json",
        width="stretch",
        help="분석 결과 원본. 판단 근거와 리포트 목록까지 중첩 구조 그대로 담깁니다.",
    )
    col_md.download_button(
        "Markdown",
        data=build_export_markdown(fw).encode("utf-8"),
        file_name=f"{stem}.md",
        mime="text/markdown",
        width="stretch",
        help="사람이 읽는 브리핑 형식.",
    )


def demote_to_demo(batch_path: str) -> bool:
    """정식 forward 경로에 생긴 앱 시연 신호를 forward_demo/로 이동.

    run_forward의 저장 경로는 고정이라(코드 동결) 생성 후 옮기는 방식을 쓴다.
    주간 배치로 이미 존재하던 파일은 호출자가 걸러내므로 여기선 무조건 이동한다.

    Returns:
        이동 성공 여부 (파일이 아직 없으면 False)
    """
    if not os.path.exists(batch_path):
        return False
    demo_path = os.path.join(FORWARD_DEMO_DIR, os.path.relpath(batch_path, FORWARD_DIR))
    os.makedirs(os.path.dirname(demo_path), exist_ok=True)
    os.replace(batch_path, demo_path)
    # 이동으로 빈 껍데기만 남은 {날짜}/{model} 폴더 정리.
    # 주간 배치 폴더는 다른 파일이 남아 있어 rmdir이 실패하므로 안전하다
    for d in (os.path.dirname(batch_path), os.path.dirname(os.path.dirname(batch_path))):
        try:
            os.rmdir(d)
        except OSError:
            break
    return True


@st.cache_resource
def forward_job_runtime():
    """세션 간 공유하는 forward 실행기와 진행 중 작업 레지스트리."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=2), {}, threading.Lock()


def run_forward_and_demote(run_forward, ticker: str, cond: str, model: str, batch_path: str) -> dict:
    """forward 실행 후 결과를 즉시 시연 폴더로 격리.

    화면이 3분 후 타임아웃되거나 브라우저 세션이 끊겨도 이 워커는
    run_forward가 끝난 직후 demote까지 수행하므로 정식 평가 경로에 남지 않는다.
    """
    result = run_forward(ticker, cond, model)
    if not demote_to_demo(batch_path):
        raise RuntimeError("시연 결과를 forward_demo로 격리하지 못했습니다.")
    return result


def render() -> None:
    # vertical_alignment="bottom" — 라벨 없는 버튼을 옆 셀렉트박스 baseline에 맞춘다.
    # 예전에는 빈 div로 라벨 높이만큼 밀어냈는데, 위젯 높이가 바뀌면 같이 틀어지는 값이었다.
    col_t, col_c, col_m, col_b = st.columns([3, 3, 3, 2], vertical_alignment="bottom")

    stocks = load_krx_stocks()
    ticker_options = [lb for lb, _ in stocks]
    _label_to_ticker = dict(stocks)
    # 기본값은 삼성전자 — 시총 1위라 목록 선두이기도 하고 백테스트 이력이 있어 화면이 다 찬다
    _t_idx = next((i for i, (_, t) in enumerate(stocks) if t == "005930"), 0)
    selected_label  = col_t.selectbox("종목 선택", ticker_options, index=_t_idx)
    selected_ticker = _label_to_ticker[selected_label]
    selected_name   = selected_label.rsplit(" (", 1)[0]

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

    analyze_btn = col_b.button("🔍 분석하기", width="stretch", type="primary")

    # provider 판별은 llm_experiment._provider와 동일하게 접두어 기준 — 모델 개명/추가에 안전
    if selected_model.startswith("gemma"):
        st.caption("gemma는 rate-limit으로 응답이 지연될 수 있습니다 (재시도 대기 포함 최대 3분).")
    elif selected_model.startswith(("gpt", "claude")):
        st.caption("이 모델은 해당 provider의 API 키(.env)와 소액의 호출 비용이 필요합니다.")
    else:
        st.caption("분석 시 최신 데이터 자동 갱신 후 선택한 모델의 API를 호출합니다.")

    # 20종목은 백테스트로 검증된 구간, 그 밖은 같은 파이프라인을 처음 적용하는 종목이다.
    # 고르기 전에 알려야 결과를 같은 무게로 읽지 않는다
    if selected_ticker not in BACKTEST_TICKERS:
        # 종목명 뒤에 조사를 붙이지 않는다 — 2,700종목이면 받침 유무가 제각각이라
        # "클래시스은"처럼 틀린 문장이 화면에 그대로 나온다
        st.caption(
            f"ℹ️ **{selected_name}** — 백테스트 검증 대상 20종목 밖입니다. 신호 생성은 동일한 "
            "파이프라인으로 이뤄지지만 과거 성과 이력은 표시되지 않으며, 애널리스트 리포트가 "
            "적거나 없을 수 있습니다."
        )

    # ── 분석 실행 ──────────────────────────────────────────
    if analyze_btn:
        from forward_test import run_forward
        from forward_verify import verify_ticker

        # 직전 결과를 먼저 비움 — 타임아웃·오류로 갱신에 실패하면 이전 종목의
        # 결과가 그대로 남아 새로 선택한 종목의 결과처럼 보인다
        st.session_state.pop("fw_result", None)
        st.session_state.pop("fw_meta", None)

        # 캐시 판별 — run_forward는 캐시 반환 여부를 알려주지 않으므로 호출 전에
        # 저장 경로 존재를 직접 확인한다 (forward_test.py의 경로 규약과 동일).
        # 정식 배치 캐시(주간 실행 산출물)와 시연 캐시를 구분해서 본다
        _today_str  = datetime.today().strftime("%Y-%m-%d")
        _fname      = f"{selected_ticker}_{selected_cond}.json"
        _batch_path = os.path.join(FORWARD_DIR, _today_str, selected_model, _fname)
        _demo_path  = os.path.join(FORWARD_DEMO_DIR, _today_str, selected_model, _fname)

        _was_cached  = os.path.exists(_batch_path)
        _demo_cached = (not _was_cached) and os.path.exists(_demo_path)

        with st.status("분석 중...", expanded=True) as _status:
            try:
                result = None

                if _demo_cached:
                    # 시연 캐시는 run_forward가 못 찾는 경로에 있어 직접 읽는다(재호출 방지)
                    _status.write(f"**{selected_name}** 시연 캐시 사용 — LLM 재호출 없이 반환합니다.")
                    with open(_demo_path, encoding="utf-8") as _f:
                        result = json.load(_f)
                elif _was_cached:
                    # 정식 배치 캐시는 직접 읽는다. run_forward를 거치지 않아 워커 생성도 없다.
                    _status.write(f"**{selected_name}** 당일 정식 배치 캐시 확인 — LLM 재호출 없이 반환합니다.")
                    with open(_batch_path, encoding="utf-8") as _f:
                        result = json.load(_f)
                else:
                    # get_today_context는 TICKERS에서 종목명을 찾고, 못 찾으면 티커 코드를
                    # 그대로 이름으로 쓴다. 그 이름이 LLM 프롬프트에 들어가므로(cond1은
                    # 종목명이 입력의 전부다) 20종목 밖은 "005490"을 회사명으로 받게 된다.
                    # src/는 코드 동결이라 레지스트리에 이름을 주입해 해결한다.
                    # 20종목은 이미 있는 항목이라 주입해도 값이 바뀌지 않는다.
                    TICKERS.setdefault(selected_name, selected_ticker)

                    # 리포트 CSV가 없으면 cond3·cond4가 리포트 없이 돌아간다. 먼저 채운다
                    if selected_cond in REPORT_CONDS_UI:
                        _status.write(f"**{selected_name}** 애널리스트 리포트 확인 중...")
                        ensure_reports(selected_ticker)

                    # DART 법인코드 점검은 실제로 수집이 필요한 경우에만
                    # (캐시 반환 경로는 DART를 쓰지 않는다 — 함수 docstring 참고)
                    _dart_warn = _check_dart_cache()
                    if _dart_warn:
                        st.warning(f"DART 초기화 경고: {_dart_warn}")

                    _executor, _jobs, _jobs_lock = forward_job_runtime()
                    _job_key = (_today_str, selected_ticker, selected_cond, selected_model)
                    with _jobs_lock:
                        # 완료된 실패 작업은 제거해 다음 클릭에서 재시도할 수 있게 한다.
                        _done_keys = [k for k, f in _jobs.items() if f.done()]
                        for _key in _done_keys:
                            _jobs.pop(_key, None)

                        _future = _jobs.get(_job_key)
                        if _future is None:
                            _future = _executor.submit(
                                run_forward_and_demote,
                                run_forward,
                                selected_ticker,
                                selected_cond,
                                selected_model,
                                _batch_path,
                            )
                            _jobs[_job_key] = _future
                            _already_running = False
                        else:
                            _already_running = True

                    if _already_running:
                        _status.write(f"**{selected_name}** 동일 분석이 이미 진행 중 — 기존 작업을 기다립니다.")
                    else:
                        _status.write(f"**{selected_name}** 실시간 데이터 수집 중 (FDR / DART)...")

                    try:
                        result = _future.result(timeout=180)   # 최대 3분
                    except concurrent.futures.TimeoutError:
                        _status.update(label="시간 초과", state="error", expanded=True)
                        st.error(
                            "분석 시간이 초과되었습니다 (3분). "
                            "FDR/DART API 미응답 또는 모델 rate-limit 재시도 대기일 수 있습니다. "
                            "백그라운드 작업은 계속되며 완료 즉시 시연 캐시로 격리됩니다. "
                            "잠시 후 다시 시도하면 기존 작업 또는 캐시 결과를 사용합니다."
                        )

                if result is not None:
                    _status.write("LLM 신호 분석 완료!")

                    # 신규 결과는 워커가 반환하기 전에 이미 시연 폴더로 격리되어 있다.
                    _source = "batch" if _was_cached else ("demo" if _demo_cached else "new")

                    # 입력 검증(forward_verify) — get_price 호출이 있어 렌더링마다 돌면
                    # 위젯 조작 때마다 느려진다. 분석 시점에 1회만 수행해 결과를 저장
                    _status.write("입력 정보 검증 중 (현재가·정합성·리포트·DART)...")
                    try:
                        _vsummary, _vflags = verify_ticker(result)
                        _verr = None
                    except Exception as _ve:
                        _vsummary, _vflags, _verr = None, None, f"{type(_ve).__name__}: {_ve}"

                    _src_path = _batch_path if _source == "batch" else _demo_path
                    st.session_state["fw_result"] = result
                    st.session_state["fw_meta"] = {
                        "source":   _source,
                        "gen_time": (
                            datetime.fromtimestamp(os.path.getmtime(_src_path)).strftime("%H:%M")
                            if os.path.exists(_src_path) else None
                        ),
                        "summary":    _vsummary,
                        "flags":      _vflags,
                        "verify_err": _verr,
                    }
                    _status.update(label="분석 완료", state="complete", expanded=False)
            except Exception as _e:
                _status.update(label="오류 발생", state="error", expanded=True)
                st.error(f"**{type(_e).__name__}**: {_e}")

    fw = st.session_state.get("fw_result")
    if fw is None:
        st.info("종목·분석 조건·모델을 선택한 뒤 **🔍 분석하기** 버튼을 눌러주세요.")
        return

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

    # ── 1-b. 생성 출처 · 입력 검증 ─────────────────────
    # 캐시 재사용과 신규 호출이 화면상 동일해 "지금 호출한 결과인가"를
    # 구분할 수 없던 문제 보완. 검증은 forward_verify와 동일 기준.
    _meta = st.session_state.get("fw_meta")
    if _meta:
        _gt = _meta.get("gen_time")
        _src_txt = {
            "batch": "♻️ 당일 정식 배치 캐시",
            "demo":  "♻️ 당일 시연 캐시",
            "new":   "🆕 신규 LLM 호출",
        }.get(_meta.get("source"), "생성 출처 불명")
        if _gt:
            _src_txt += f" (오늘 {_gt} 생성)"
        if _meta.get("source") == "new":
            _src_txt += " · 평가 표본 제외(forward_demo)"

        _flags = _meta.get("flags")
        if _meta.get("verify_err"):
            _vf_txt = f"❔ 입력 검증 실패 ({_meta['verify_err']})"
        elif _flags is None:
            _vf_txt = "❔ 입력 검증 정보 없음"
        elif _flags:
            _vf_txt = f"⚠️ 입력 검증 플래그 {len(_flags)}건"
        else:
            _vs = _meta.get("summary") or {}
            _vf_txt = (
                f"✅ 입력 검증 통과 (현재가·ROE 정합성·52주 · "
                f"리포트 {_vs.get('reports', 0)}건 · DART {_vs.get('dart', '-')})"
            )

        st.caption(f"{_src_txt}  |  {_vf_txt}")
        if _flags:
            with st.expander(f"입력 검증 플래그 상세 ({len(_flags)}건)"):
                for _fl in _flags:
                    st.markdown(f"- {_fl}")

    st.divider()

    # ── 2. 신호 박스 ───────────────────────────────────
    st.markdown(signal_badge(fw["signal"]), unsafe_allow_html=True)
    st.markdown(f"**신뢰도**: {fw['confidence']}%")
    st.progress(fw["confidence"] / 100)
    # 재현성을 위해 temperature=0으로 고정한 결과 신뢰도가 좁은 대역에 몰린다.
    # 여러 종목을 눌러도 같은 값이 나오는 이유를 화면에서 먼저 밝혀 오해를 막는다.
    st.caption(
        "temperature=0은 재현성을 위한 설정이며, 신뢰도는 모델별로 일부 값에 집중되는 경향이 있습니다. "
        "모델 간 직접 비교나 주요 성과 판단에는 사용하지 않고 보조 정보로만 확인합니다."
    )

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
        col4.metric("시가총액", fmt_market_cap(ctx.get("market_cap")))

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
            # 조건은 리포트를 요구하는데 실제로는 빈 섹션이 들어갔다. 조용히 두면
            # 화면상 cond4인데 입력은 cond2에 가까운 상태가 드러나지 않는다
            st.warning(
                "최근 30일 이내 애널리스트 리포트가 없어 이 섹션이 비어 있습니다. "
                f"**{COND_LABELS[fw_cond]}**는 리포트를 입력에 포함하지만 이 종목은 해당 정보 없이 "
                "판단했으므로, 실질적으로는 재무 정보 기반 판단에 가깝습니다. "
                "커버리지가 낮은 종목에서 나타납니다."
            )

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

        col_a.metric(
            "매출 성장률 (YoY)",
            fmt_val(rev_growth, suffix="%"),
            delta=f"{rev_growth:+.1f}%" if (rev_growth is not None and not pd.isna(rev_growth)) else None,
        )
        col_b2.metric("영업이익률",  fmt_val(op_margin, suffix="%"))
        col_c2.metric("부채비율",   fmt_val(debt, suffix="%"))

        # 배당수익률은 표시하지 않는다. 사업연도말 기준가 산출이라 증권사 값과 평균 42% 벌어지고
        # (prove.md 각도 1), 애초에 LLM 컨텍스트에 넣지 않는 필드다. 화면에 두면 모델이 본 값으로
        # 오인되고, 값 자체가 이상해 설명 부담만 생긴다.

        st.divider()

    # ── 6-b. 수집 데이터 내려받기 ──────────────────────
    render_export_section(fw)

    st.divider()

    # ── 7. 백테스팅 성과 ───────────────────────────────
    st.subheader("📉 백테스팅 성과")

    bt_df = load_backtest_results(fw_cond, fw_model)

    if bt_df is None:
        # 4모델 × 5조건 전부 완료돼 정상 경로에서는 도달하지 않는다. 결과 파일이 없거나
        # 깨졌을 때 화면이 비는 대신 이유를 알리는 방어 분기.
        # 시연 중 관객에게 보일 화면이라 CLI 실행 명령은 넣지 않는다
        st.info(f"**{fw_model}**의 {fw_cond} 백테스트 결과 파일을 읽을 수 없어 과거 성과를 표시할 수 없습니다.")
        return

    ticker_df = get_ticker_backtest(bt_df, fw["ticker"])

    if ticker_df.empty:
        # 20종목 밖을 고르면 여기로 온다. 누락이 아니라 설계라는 것을 밝힌다
        st.info(
            f"**{fw['name']}** — 백테스트 검증 대상 20종목에 포함되지 않아 과거 성과 이력이 없습니다. "
            "백테스트는 2023-01~2025-12 대형주 20종목으로 방법을 검증하는 통제 실험이고, "
            "신호 생성은 같은 파이프라인으로 전 종목에 적용됩니다."
        )
        return

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

    st.caption(f"조건: **{COND_LABELS[fw_cond]}** | 모델: **{fw_model}** | 총 **{total}**개월 백테스트 이력")

    # 신호 성능 지표만 둔다. Buy/Sell/Neutral을 섞은 전체 평균은 종목의 기간 등락에 가까워
    # 옆 두 칸과 나란히 두면 세 번째 성능 지표로 읽힌다 — 아래 상세 통계에서 신호별로 본다
    col_bt1, col_bt2 = st.columns(2)
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

    # 신호별 상세 테이블
    with st.expander("신호별 상세 통계"):
        stats = (
            ticker_df.groupby("signal")["return_20d"]
            .agg(건수="count", 평균수익률="mean")
            .round(2)
        )
        stats["방향 적중률(%)"] = pd.NA
        for sig, g in ticker_df.groupby("signal")["return_20d"]:
            if sig == "Buy":
                stats.loc[sig, "방향 적중률(%)"] = round((g > 0).mean() * 100, 2)
            elif sig == "Sell":
                stats.loc[sig, "방향 적중률(%)"] = round((g < 0).mean() * 100, 2)
        st.dataframe(stats)

    # 시계열 차트 — signal로 색을 나눠야 "언제 무슨 신호였는지"가 보인다.
    # 선 하나로 그리면 수익률 곡선일 뿐 신호 정보가 사라진다
    with st.expander("신호별 수익률 추이"):
        chart_df = ticker_df[["signal_date", "return_20d", "signal"]].copy()
        # 문자열 그대로 두면 x축이 36개 범주로 잘려 나온다 — 시간축으로 변환
        chart_df["signal_date"] = pd.to_datetime(chart_df["signal_date"], errors="coerce")
        chart_df = chart_df.sort_values("signal_date")
        _signal_order = ["Buy", "Neutral", "Sell"]
        for _sig in _signal_order:
            chart_df[_sig] = chart_df["return_20d"].where(chart_df["signal"] == _sig)
        st.scatter_chart(
            chart_df,
            x="signal_date",
            y=_signal_order,
            color=[SIGNAL_STYLE[sig][1] for sig in _signal_order],
            x_label="신호일",
            y_label="20일 수익률 (%)",
            height=280,
        )
        st.caption("점 하나가 신호 한 건입니다. 색은 신호 종류, 세로축은 그 신호 이후 20거래일 수익률입니다.")

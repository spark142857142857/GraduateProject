"""탭 2 — 분석 프롬프트 생성 (LLM 호출 없음).

선택한 종목의 데이터를 지금 시점으로 수집해, 실험에서 LLM에 보낸 것과 같은 형식의
프롬프트를 만든다. 사용자는 이를 복사해 ChatGPT·Claude 등 원하는 LLM에 붙여 넣는다.
API 키도 호출 비용도 들지 않고, 앱이 지원하지 않는 회사의 모델에도 쓸 수 있다
(개별 분석 탭은 llm_experiment._provider의 접두어 분기에 묶여 있다).

**프롬프트는 실험 경로를 그대로 탄다.** forward_test.run_forward와 같은 순서로
get_today_context → FORWARD_BUILDER_MAP → build_prompt를 부른다. 옵션(판단 기간,
출력 형식, 섹션 임의 조합)을 두지 않은 것도 그래서다. 하나라도 바꾸면 실험에서 검증하지
않은 프롬프트가 되고, 이 탭의 가치가 "실험에 쓴 그 프롬프트"라는 데 있다.

예전 이름은 "종목 데이터 조회"였고 수집 지표 열람·내려받기가 주 기능이었다. 그 기능은
오른쪽 보조 열로 옮겨 그대로 남겼다. 한 번 수집한 ctx에서 프롬프트와 파일이 함께 나온다.

대상은 KRX 상장 보통주 전 종목이다. 현재 시점만 지원한다 — 임의 과거 날짜는 DART를
그 시점으로 되돌려야 하는데, 백테스트 파이프라인이 20종목에 대해서만 하는 일이라
전 종목으로 열면 틀린 시점의 데이터를 내보낼 위험이 있다.

**애널리스트 리포트는 프롬프트에만 들어가고 내려받는 파일에는 들어가지 않는다.**
cond3·cond4의 입력이라 프롬프트에서 빼면 조건이 성립하지 않는다(개별 분석 탭과 같은
처리). 리포트는 get_today_context 안에서 네이버 API로 오늘 기준 30일치가 채워진다
(shared._install_report_source). 파일에서 빼는 것은 지표 파일의 성격을 지키기 위해서다 —
한 행짜리 수치 표에 제목 목록을 섞으면 이어붙이기 좋은 형식이 깨진다.
"""

import json

import pandas as pd
import streamlit as st

from app_ui.shared import (
    COND_LABELS, UI_CONDS,
    check_dart_cache, check_trading_halt, load_krx_stocks,
    register_ticker,
)

# 리포트를 입력에 포함하는 조건 — 리포트 0건이면 실질 입력이 줄어든다는 안내가 필요하다
_REPORT_CONDS = ("cond3", "cond4")

# (ctx 키, 표시 라벨) — 라벨에 단위를 박아둔다. get_today_context는 금액을 원 단위
# 원시값으로 주므로(시가총액 1,350,490,358,448,000처럼) 단위를 안 적으면 읽을 수 없다.
EXPORT_FIELDS = [
    ("price",                "현재가(원)"),
    ("per",                  "PER"),
    ("pbr",                  "PBR"),
    ("roe",                  "ROE(%)"),
    ("market_cap",           "시가총액(원)"),
    ("high_52w",             "52주최고가(원)"),
    ("low_52w",              "52주최저가(원)"),
    ("price_position_52w",   "52주내위치(%)"),
    ("momentum_1m",          "1개월수익률(%)"),
    ("volume_change",        "거래량변화율(%)"),
    ("fiscal_period",        "실적기준"),
    ("report_name",          "보고서종류"),
    ("revenue",              "매출(원)"),
    ("revenue_yoy",          "매출증감률YoY(%)"),
    ("operating_income",     "영업이익(원)"),
    ("operating_income_yoy", "영업이익증감률YoY(%)"),
    ("operating_margin",     "영업이익률(%)"),
    ("net_income",           "순이익(원)"),
    ("debt_ratio",           "부채비율(%)"),
    ("operating_cashflow",   "영업현금흐름(원)"),
]

# 화면 미리보기에서 금액을 원 단위 그대로 두면 자릿수가 길어 읽히지 않는다.
# 파일에는 원시값을 넣고 화면에만 조 단위 표기를 쓴다.
_WON_KEYS = {"market_cap", "revenue", "operating_income", "net_income", "operating_cashflow"}

# 결측 안내용 묶음. 소형주에서는 이 두 묶음이 통째로 비는 일이 잦은데(실측: 시총 하위
# 3종목에서 20개 항목 중 11~12개 결측), 안내가 없으면 화면에 대시만 줄줄이 남아
# 앱이 고장 난 것처럼 보인다. 특히 DART는 보고서 종류·기간은 찾아 놓고 수치만 비어서
# "실적기준 2026 2분기"만 떠 있는 상태가 된다.
_DART_NUMERIC_KEYS = (
    "revenue", "operating_income", "net_income",
    "operating_margin", "debt_ratio", "operating_cashflow",
)
_VALUATION_KEYS = ("per", "pbr", "roe")


def fetch_context(ticker: str, name: str) -> dict:
    """실시간 수집. LLM은 부르지 않는다.

    register_ticker로 종목명·상장 시장을 src/ 레지스트리에 먼저 넣는다. 안 하면 20종목
    밖은 티커 코드가 회사명으로, 코스닥이 KOSPI로 프롬프트에 들어간다.

    리포트는 조건과 무관하게 항상 받는다(get_today_context 안에서 API 한 번). 수집 뒤
    조건을 바꿔도 다시 수집하지 않고 프롬프트만 새로 그리므로, cond3·cond4로 넘어갈 때
    리포트가 이미 있어야 한다.
    """
    register_ticker(ticker, name)
    from update import get_today_context
    ctx = get_today_context(ticker)
    # dividend_yield는 사업연도말 기준가로 산출해 증권사 값과 평균 42% 벌어진다
    # (prove.md 각도 1). 프롬프트에도 넣지 않는 필드이고 개별 분석 화면에서도 같은
    # 이유로 감춰 뒀는데, 데이터 파일로만 새어 나가면 기준이 어긋난다. 수집 직후 끊는다.
    ctx.pop("dividend_yield", None)
    return ctx


def build_prompt_text(ctx: dict, cond: str) -> str:
    """forward_test.run_forward와 같은 순서로 프롬프트를 만든다. 순서를 바꾸지 말 것."""
    from experiments import BLIND_CONDITIONS, EXPERIMENTS
    from forward_test import FORWARD_BUILDER_MAP
    from llm_experiment import build_prompt

    sections = [FORWARD_BUILDER_MAP[key](ctx) for key in EXPERIMENTS[cond]]
    return build_prompt(
        ctx["name"], ctx["price"], sections,
        ticker=ctx["ticker"], blind=cond in BLIND_CONDITIONS,
    )


def export_ctx(ctx: dict) -> dict:
    """내려받는 파일용 ctx. 리포트(모듈 docstring 참고)와 화면용 _prompt를 뺀다."""
    return {k: v for k, v in ctx.items() if k != "recent_reports" and not k.startswith("_")}


def build_frame(ctx: dict) -> pd.DataFrame:
    """지표를 한 행짜리 wide 표로. 여러 종목을 받아 그대로 이어붙일 수 있는 형태."""
    row = {"종목코드": ctx["ticker"], "종목명": ctx["name"], "기준일": ctx["date"]}
    # 결측은 빈 칸으로 둔다. 적자면 PER이 정의되지 않는 것처럼 실제로 없는 값이라,
    # 임의의 기본값으로 채우면 없는 값을 있는 것처럼 만든다.
    row.update({label: ctx.get(key) for key, label in EXPORT_FIELDS})
    return pd.DataFrame([row])


def build_markdown(ctx: dict) -> str:
    """사람이 읽는 브리핑."""
    lines = [
        f"# {ctx['name']} ({ctx['ticker']})",
        "",
        f"기준일: {ctx['date']}",
        "",
        "## 수집 지표",
        "",
        "| 항목 | 값 |",
        "|---|---|",
    ]
    for key, label in EXPORT_FIELDS:
        v = ctx.get(key)
        lines.append(f"| {label} | {'' if v is None or (isinstance(v, float) and pd.isna(v)) else v} |")

    lines += [
        "",
        "---",
        "",
        f"FinanceDataReader(시세·지표)와 DART 정기보고서(실적)를 {ctx['date']} 기준으로 "
        "수집한 값입니다. 금액은 원 단위 원시값입니다. 애널리스트 리포트는 포함하지 않습니다.",
    ]
    return "\n".join(lines)


def _fmt_cell(key: str, v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    if key in _WON_KEYS:
        return f"{v / 1e12:,.2f}조원"
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def _render_prompt(ctx: dict, cond: str) -> None:
    """메인 열 — 프롬프트 본문. 복사는 st.code가 우측 상단에 기본으로 다는 아이콘이 맡는다."""
    # 조건은 리포트를 요구하는데 실제로는 빈 섹션이 들어갔다. 조용히 두면 화면상 cond4인데
    # 프롬프트는 cond2에 가까운 상태가 드러나지 않는다. 프롬프트가 560px이라 아래에 두면
    # 스크롤해야 보이므로 위에 둔다
    if cond in _REPORT_CONDS and not ctx.get("recent_reports"):
        st.warning("최근 30일 안에 나온 애널리스트 리포트가 없어, 리포트 섹션 없이 만들어졌습니다.")

    st.code(ctx["_prompt"], language=None, wrap_lines=True, height=560)


def _fmt_market_cap_short(v) -> str:
    """좁은 열용. 1,674.96조원은 잘렸다. 100조 이상은 정수, 1조 미만은 억원."""
    if v is None or pd.isna(v):
        return "-"
    if v >= 1e14:
        return f"{v / 1e12:,.0f}조원"
    if v >= 1e12:
        return f"{v / 1e12:,.1f}조원"
    return f"{v / 1e8:,.0f}억원"


def _render_data(ctx: dict, cond: str) -> None:
    """보조 열 — 핵심 지표, 전체 지표, 내려받기."""
    st.caption(f"{ctx['date']} 기준")

    # 요약 칸에는 단위를 값에 붙인다. 전체 지표 표는 라벨에 단위가 있어 붙이지 않는다
    _price = ctx.get("price")
    _mom   = ctx.get("momentum_1m")
    # 한 줄에 하나씩 쌓는다. 2열로 두면 발표장 노트북 폭(열 폭 ~130px)에서 "286,5…"처럼
    # 값이 잘렸다(실측). 옆 프롬프트가 560px이라 세로로 쌓아도 자리가 남는다
    st.metric("현재가", f"{int(_price):,}원" if _price is not None else "-")
    st.metric("시가총액", _fmt_market_cap_short(ctx.get("market_cap")))
    st.metric("PER", _fmt_cell("per", ctx.get("per")))
    st.metric("1개월 수익률", f"{_mom:+.2f}%" if _mom is not None and not pd.isna(_mom) else "-")

    with st.expander("전체 지표"):
        rows = [{"항목": lb, "값": _fmt_cell(k, ctx.get(k))} for k, lb in EXPORT_FIELDS]
        st.dataframe(pd.DataFrame(rows).set_index("항목"), width="stretch", height=460)

    # 소형주에서 통째로 비는 두 묶음. 대시만 줄줄이 남으면 고장으로 읽히고, 프롬프트에
    # N/A로 들어간다는 것도 드러나지 않는다
    _missing = []
    if all(ctx.get(k) is None for k in _DART_NUMERIC_KEYS):
        _missing.append("DART 실적")
    if all(ctx.get(k) is None for k in _VALUATION_KEYS):
        _missing.append("PER·PBR·ROE")
    if _missing:
        st.warning(f"수집하지 못한 항목: {', '.join(_missing)}. 프롬프트에는 N/A로 들어갑니다.")

    # 버튼 네 개를 늘어놓지 않고 하나로 접는다. 복사가 주 동작이고 파일은 곁가지다.
    # 안쪽 버튼은 tertiary(테두리 없음)로 메뉴 항목처럼 보이게 한다 — 테두리 버튼을
    # 팝오버 카드 안에 쌓으면 카드 속 카드가 된다
    stem = f"{ctx['ticker']}_{ctx['name']}_{ctx['date']}"
    with st.popover("내려받기", icon=":material/download:", width="stretch"):
        st.download_button(
            "프롬프트 (TXT)",
            data=ctx["_prompt"].encode("utf-8"),
            file_name=f"{stem}_{cond}_prompt.txt",
            mime="text/plain",
            width="stretch",
            type="tertiary",
        )
        # utf-8-sig — 엑셀에서 한글 헤더가 깨지지 않게 (data/reports CSV와 같은 기준)
        st.download_button(
            "지표 (CSV)",
            data=build_frame(ctx).to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{stem}.csv",
            mime="text/csv",
            width="stretch",
            type="tertiary",
            help="한 행짜리 표. 금액은 원 단위 원시값입니다.",
        )
        st.download_button(
            "수집 원본 (JSON)",
            data=json.dumps(export_ctx(ctx), ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            file_name=f"{stem}.json",
            mime="application/json",
            width="stretch",
            type="tertiary",
        )
        st.download_button(
            "브리핑 (Markdown)",
            data=build_markdown(ctx).encode("utf-8"),
            file_name=f"{stem}.md",
            mime="text/markdown",
            width="stretch",
            type="tertiary",
        )


def render() -> None:
    stocks = load_krx_stocks()
    labels = [lb for lb, _ in stocks]
    label_to_ticker = dict(stocks)

    col_t, col_c, col_b = st.columns([4, 3, 2], vertical_alignment="bottom")
    # 시총 내림차순이라 index=0이 삼성전자다. 셀렉트박스는 타이핑으로 걸러지므로
    # 별도 검색창을 두지 않는다 — 2,759개를 스크롤할 일은 없다.
    label = col_t.selectbox("종목", labels, index=0)
    ticker = label_to_ticker[label]
    name = label.rsplit(" (", 1)[0]
    # 수집 뒤에 바꿔도 다시 수집하지 않는다. 같은 ctx로 프롬프트만 새로 그린다
    cond = col_c.selectbox(
        "분석 조건",
        options=UI_CONDS,
        format_func=lambda k: COND_LABELS[k] + (" — 권장" if k == "cond4" else ""),
        index=3,  # cond4 기본 — 개별 분석 탭과 같다
        key="prompt_cond",
    )
    fetch_btn = col_b.button("📝 프롬프트 생성", width="stretch", type="primary")

    # 개별 분석 탭과 가르는 한 가지만 적는다. 나머지는 화면이 말한다
    st.caption("LLM을 호출하지 않습니다. 만든 프롬프트를 복사해 원하는 LLM에 붙여 넣으세요.")

    if fetch_btn:
        st.session_state.pop("dl_ctx", None)
        try:
            with st.spinner(f"{name} 데이터 수집 중..."):
                warn = check_dart_cache()
                if warn:
                    st.warning(f"DART 초기화 경고: {warn}")
                st.session_state["dl_ctx"] = fetch_context(ticker, name)
        except Exception as e:
            st.error(f"수집 실패 — **{type(e).__name__}**: {e}")

    ctx = st.session_state.get("dl_ctx")
    # 종목을 바꾸고 아직 버튼을 안 눌렀으면 이전 종목의 결과를 보여주지 않는다.
    # 남겨 두면 셀렉트박스와 아래 내용이 서로 다른 종목을 가리킨다
    if ctx is None or ctx["ticker"] != ticker:
        return

    try:
        ctx["_prompt"] = build_prompt_text(ctx, cond)
    except Exception as e:
        st.error(f"프롬프트 생성 실패 — **{type(e).__name__}**: {e}")
        return

    # 거래정지 종목은 시세가 마지막 종가에 고정된다. 결측과 달리 겉보기에 멀쩡하고,
    # 모델도 그 0을 관측으로 읽는다(app_scope.md). 프롬프트보다 먼저 알린다.
    _halt = check_trading_halt(ctx["ticker"])
    if _halt:
        _hp = f"{int(_halt['price']):,}원" if _halt["price"] is not None else "직전 종가"
        st.warning(
            f"⛔ {_halt['days']}거래일째 거래량이 0입니다. 거래정지 종목으로 보이며, "
            f"시세 관련 값은 모두 {_hp}에 멈춘 값에서 나왔습니다."
        )

    col_main, col_side = st.columns([5, 2], gap="large")
    with col_main:
        _render_prompt(ctx, cond)
    with col_side:
        _render_data(ctx, cond)

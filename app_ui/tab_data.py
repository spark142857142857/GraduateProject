"""탭 2 — 종목 데이터 조회·내려받기 (LLM 호출 없음).

개별 분석 탭에서 분리해 나온 화면이다. 거기서는 데이터를 받으려면 먼저 신호 생성
버튼을 눌러야 했고, 그건 데이터만 필요한 사람에게 LLM 호출을 시키는 것이었다.

분리로 정확도도 같이 올라갔다. 개별 분석이 저장하는 `context_used`는 LLM 호출
**이후** 만들어지는 화면 표시 전용 축약본이라 52주 최고/최저가와 매출·영업이익·
순이익·영업현금흐름의 절대액이 빠져 있다. 여기서는 `get_today_context`를 직접 불러
파이프라인이 실제로 수집하는 값을 그대로 받는다(같은 함수를 forward_test도 쓴다).

대상은 KRX 상장 보통주 전 종목이다. 현재 시점만 지원한다 — 임의 과거 날짜는 DART를
그 시점으로 되돌려야 하는데, 백테스트 파이프라인이 20종목에 대해서만 하는 일이라
전 종목으로 열면 틀린 시점의 데이터를 내보낼 위험이 있다.

**애널리스트 리포트는 다루지 않는다.** 리포트만 갱신 기준이 종목에 따라 갈리기 때문이다.
ensure_reports는 백테스트 20종목을 건드리지 않는다(그 CSV는 crawl.py가 관리하는 실험
입력이라 앱이 30일치로 덮어쓰면 실험 데이터가 훼손된다). 그래서 20종목은 crawl.py가
마지막으로 돈 시점의 리포트가, 나머지는 오늘 받은 리포트가 나온다. 같은 화면에서 종목에
따라 수집 기준이 다른 셈이고, 제출·시연이 수집일보다 몇 달 뒤라 그 격차가 그대로 드러난다.
시세·재무·DART는 호출 시점에 새로 받으므로 이 문제가 없다. 기준이 흔들리는 항목 하나를
빼서 화면 전체의 기준을 "지금 수집한 값"으로 통일한다.
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st

from app_ui.shared import BACKTEST_TICKERS, TICKERS, check_dart_cache, load_krx_stocks

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


def fetch_context(ticker: str, name: str) -> dict:
    """실시간 수집. LLM은 부르지 않는다.

    get_today_context는 TICKERS에서 종목명을 역조회하고 못 찾으면 티커 코드를 이름으로
    쓴다. 화면 제목과 파일명에 "005490"이 찍히는 것을 막으려 이름을 주입한다
    (개별 분석 탭과 같은 이유이며, 20종목은 기존 항목이라 값이 바뀌지 않는다).

    ensure_reports는 부르지 않는다. 리포트를 화면에도 파일에도 넣지 않으므로 받을 이유가
    없고, 크롤링 한 번(실측 2.7~4.7초)이 빠져 조회가 그만큼 빨라진다. get_today_context는
    리포트 CSV가 없으면 빈 목록으로 넘어가므로(os.path.exists 가드) 안전하다.
    """
    TICKERS.setdefault(name, ticker)
    from update import get_today_context
    ctx = get_today_context(ticker)
    # 내보내지 않는 항목은 수집 직후 한 곳에서 끊는다. 화면·CSV·JSON·MD 어디로도
    # 흘러가지 않게 하려면 개별 출력부가 아니라 여기서 막아야 한다.
    #   · recent_reports — 디스크에 남아 있던 옛 리포트가 딸려 온다 (위 docstring 참고)
    #   · dividend_yield — 사업연도말 기준가로 산출해 증권사 값과 평균 42% 벌어진다
    #     (prove.md 각도 1). LLM 프롬프트에도 넣지 않는 필드이고 개별 분석 화면에서도
    #     같은 이유로 감춰 뒀는데, 데이터 파일로만 새어 나가면 기준이 어긋난다
    for _drop in ("recent_reports", "dividend_yield"):
        ctx.pop(_drop, None)
    return ctx


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


def render() -> None:
    stocks = load_krx_stocks()
    labels = [lb for lb, _ in stocks]
    label_to_ticker = dict(stocks)

    col_t, col_b = st.columns([5, 2], vertical_alignment="bottom")
    # 시총 내림차순이라 index=0이 삼성전자다. 셀렉트박스는 타이핑으로 걸러지므로
    # 별도 검색창을 두지 않는다 — 2,759개를 스크롤할 일은 없다.
    label = col_t.selectbox(
        "종목 검색",
        labels,
        index=0,
        help="종목명 또는 종목코드를 입력하면 목록이 걸러집니다.",
    )
    ticker = label_to_ticker[label]
    name = label.rsplit(" (", 1)[0]
    fetch_btn = col_b.button("📥 데이터 조회", width="stretch", type="primary")

    st.caption(
        "선택한 종목의 시세·기술지표·재무지표·DART 실적을 **현재 시점 기준**으로 수집합니다. "
        "**LLM을 호출하지 않아 API 비용이 들지 않습니다.** 대상은 백테스트 20종목이 아니라 "
        "KRX 상장 보통주 전 종목이며, 적자 종목의 PER처럼 정의되지 않는 값은 빈 칸으로 둡니다."
    )

    if fetch_btn:
        st.session_state.pop("dl_ctx", None)
        with st.status("수집 중...", expanded=True) as status:
            try:
                warn = check_dart_cache()
                if warn:
                    st.warning(f"DART 초기화 경고: {warn}")
                status.write(f"**{name}** 시세·재무·DART 수집 중...")
                st.session_state["dl_ctx"] = fetch_context(ticker, name)
                status.update(label="수집 완료", state="complete", expanded=False)
            except Exception as e:
                status.update(label="오류 발생", state="error", expanded=True)
                st.error(f"**{type(e).__name__}**: {e}")

    ctx = st.session_state.get("dl_ctx")
    if ctx is None:
        st.info("종목을 선택한 뒤 **📥 데이터 조회** 버튼을 눌러주세요.")
        return

    st.divider()
    st.subheader(f"{ctx['name']}  ({ctx['ticker']})")
    if ctx["ticker"] not in BACKTEST_TICKERS:
        st.caption("ℹ️ 백테스트 검증 대상 20종목 밖입니다. 수집 파이프라인은 동일합니다.")

    # ── 미리보기 ───────────────────────────────────────────
    st.markdown("**수집 지표**")
    rows = []
    for key, label_ in EXPORT_FIELDS:
        v = ctx.get(key)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            shown = "-"
        elif key in _WON_KEYS:
            shown = f"{v / 1e12:,.2f}조원"
        elif isinstance(v, float):
            shown = f"{v:,.2f}"
        else:
            shown = str(v)
        rows.append({"항목": label_, "값": shown})
    st.dataframe(pd.DataFrame(rows).set_index("항목"), width="stretch", height=460)
    st.caption(
        "화면은 읽기 쉽게 조원 단위로 줄여 보여줍니다. 내려받는 파일에는 원 단위 원시값이 들어갑니다. "
        "애널리스트 리포트는 포함하지 않습니다 — 종목에 따라 갱신 기준이 갈려 "
        "여기 표시된 다른 값들과 수집 시점이 어긋나기 때문입니다."
    )

    # ── 내려받기 ───────────────────────────────────────────
    st.divider()
    st.markdown("**내려받기**")
    stem = f"{ctx['ticker']}_{ctx['name']}_{ctx['date']}"
    col_csv, col_json, col_md = st.columns(3)

    # utf-8-sig — 엑셀에서 한글 헤더가 깨지지 않게 (data/reports CSV와 같은 기준)
    col_csv.download_button(
        "CSV",
        data=build_frame(ctx).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{stem}.csv",
        mime="text/csv",
        width="stretch",
        help="지표 한 행. 여러 종목을 받아 이어붙이기 좋은 형식입니다.",
    )
    col_json.download_button(
        "JSON",
        data=json.dumps(ctx, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
        file_name=f"{stem}.json",
        mime="application/json",
        width="stretch",
        help="수집 원본을 구조 그대로 담습니다.",
    )
    col_md.download_button(
        "Markdown",
        data=build_markdown(ctx).encode("utf-8"),
        file_name=f"{stem}.md",
        mime="text/markdown",
        width="stretch",
        help="사람이 읽는 브리핑 형식.",
    )

"""탭 2 — 백테스트 신호 매트릭스 (results/experiment 읽기 전용, API 호출 없음)."""

import pandas as pd
import streamlit as st

from app_ui.shared import (
    COND_LABELS, DEFAULT_MODEL, REPORT_CONDS, SIGNAL_STYLE, TICKERS,
    fmt_metric, list_backtest_models, load_backtest_results,
)


def signal_cell_style(v) -> str:
    """신호 매트릭스 셀 색상 — 'Buy 85%' 형태 문자열의 접두어로 판별."""
    if isinstance(v, str):
        for sig, (_, bg, fg) in SIGNAL_STYLE.items():
            if v.startswith(sig):
                return f"background-color:{bg};color:{fg}"
    return ""


def list_matrix_models() -> list[str]:
    """REPORT_CONDS 중 하나라도 백테스트 결과가 있는 모델 목록."""
    models: set[str] = set()
    for cond in REPORT_CONDS:
        models.update(list_backtest_models(cond))
    return sorted(models)


@st.cache_data(ttl=300, show_spinner=False)
def load_signal_matrix(model: str) -> pd.DataFrame:
    """조건별 백테스트 결과를 long DataFrame으로 합친다.

    탭2 매트릭스의 소스. forward 캐시를 쓰지 않는 이유는 ① 신호 생성을 2026-08-02로
    종료해 시간이 갈수록 낡은 날짜가 화면에 남고 ② forward를 앱에서 다루지 않기로 한
    결정(TODO "미채택 — forward 성과 탭")과 어긋나기 때문. 백테스트 기간(2023-01~2025-12)은
    설계상 고정이라 낡지 않고, 주력 근거를 원자료 수준에서 보여준다는 이점도 있다.
    """
    frames = []
    for cond in REPORT_CONDS:
        df = load_backtest_results(cond, model)
        if df is None or df.empty:
            continue
        keep = [c for c in ("ticker", "name", "signal_date", "signal", "confidence", "return_20d")
                if c in df.columns]
        d = df[keep].copy()
        d["cond"] = cond
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # 저장 형식이 int일 수 있어 zero-pad로 통일 (get_ticker_backtest와 같은 이유)
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out


def signal_hit(signal: str, ret: float | None) -> str:
    """신호 방향이 실제 20거래일 수익률과 맞았는지. Neutral은 방향이 없어 판정 제외."""
    if ret is None or pd.isna(ret):
        return ""
    if signal == "Buy":
        return " ✓" if ret > 0 else " ✗"
    if signal == "Sell":
        return " ✓" if ret < 0 else " ✗"
    return ""


def render() -> None:
    mx_models = list_matrix_models()
    if not mx_models:
        st.info(
            "백테스트 결과가 없습니다. "
            "`python src/experiment/llm_experiment.py --cond <조건> --model <모델명>` 실행 후 확인하세요."
        )
        return

    col_mm, col_md = st.columns(2)
    _midx = mx_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in mx_models else 0
    sel_mx_model = col_mm.selectbox("모델", mx_models, index=_midx, key="mx_model")

    sig_df = load_signal_matrix(sel_mx_model)
    if sig_df.empty:
        st.info(f"{sel_mx_model}의 백테스트 결과를 읽을 수 없습니다.")
        return

    # 최신 달이 기본 — 실험 종료 시점(2025-12)부터 거슬러 본다
    mx_dates = sorted(sig_df["signal_date"].astype(str).unique(), reverse=True)
    sel_mx_date = col_md.selectbox("신호일 (매월 첫 거래일)", mx_dates, index=0, key="mx_date")

    month_df = sig_df[sig_df["signal_date"].astype(str) == sel_mx_date]

    st.caption(
        f"백테스트 **{sel_mx_date}** 신호 · {month_df['ticker'].nunique()}종목 × "
        f"{month_df['cond'].nunique()}조건 (API 호출 없음). "
        "실험 기간 2023-01~2025-12의 매월 첫 거래일 중 하나를 골라 "
        "조건별 판단이 어떻게 갈렸는지 원자료로 확인할 수 있습니다."
    )

    # 신호 분포 요약 (보조 실험 조건은 제외 — REPORT_CONDS 주석 참고)
    dist = pd.crosstab(month_df["cond"], month_df["signal"])
    dist = dist.reindex(index=[c for c in REPORT_CONDS if c in dist.index],
                        columns=[s for s in ("Buy", "Neutral", "Sell") if s in dist.columns])
    dist.index = [COND_LABELS.get(c, c) for c in dist.index]
    st.markdown("**신호 분포 (조건 × 신호)**")
    st.dataframe(dist.fillna(0).astype(int), width="stretch")

    # 종목 × 조건 신호 매트릭스
    st.markdown("**종목별 신호 매트릭스**")
    _legend = "  ".join(
        f'<span style="background:{bg};color:{fg};padding:2px 10px;'
        f'border-radius:4px;font-size:0.85rem;">{sig}</span>'
        for sig, (_, bg, fg) in SIGNAL_STYLE.items()
    )
    st.markdown(
        f"{_legend}　셀은 `신호 신뢰도% 적중`입니다. "
        "✓/✗는 신호 방향이 실제 20거래일 수익률과 맞았는지이며, "
        "방향이 없는 Neutral은 판정하지 않습니다.",
        unsafe_allow_html=True,
    )

    mat_df = month_df.copy()
    mat_df["종목"] = mat_df["name"] + " (" + mat_df["ticker"] + ")"
    mat_df["cell"] = (
        mat_df["signal"] + " " + mat_df["confidence"].astype(int).astype(str) + "%"
        + [signal_hit(s, r) for s, r in zip(mat_df["signal"], mat_df["return_20d"])]
    )
    mat_df = mat_df.drop_duplicates(subset=["ticker", "cond"], keep="last")
    pivot = mat_df.pivot(index="종목", columns="cond", values="cell")
    pivot = pivot[[c for c in REPORT_CONDS if c in pivot.columns]]
    pivot.columns = [COND_LABELS.get(c, c) for c in pivot.columns]

    # 실제 20거래일 수익률은 (종목, 신호일)만으로 정해져 조건과 무관하다.
    # 조건별 셀에 반복해 넣으면 같은 값이 다섯 번 나오므로 열 하나로 분리한다
    actual = (
        month_df.drop_duplicates(subset=["ticker"])
        .assign(종목=lambda d: d["name"] + " (" + d["ticker"] + ")")
        .set_index("종목")["return_20d"]
    )

    # TICKERS 정의 순서로 행 정렬
    row_order = [f"{name} ({ticker})" for name, ticker in TICKERS.items() if f"{name} ({ticker})" in pivot.index]
    pivot = pivot.reindex(row_order)
    pivot["실제 20일 수익률"] = [
        fmt_metric(actual.get(idx), signed=True) + "%" if not pd.isna(actual.get(idx)) else "-"
        for idx in pivot.index
    ]
    pivot = pivot.fillna("-")
    st.dataframe(pivot.style.map(signal_cell_style), width="stretch", height=740)
    st.caption(
        "빈 칸(`-`)은 해당 조건에서 신호가 생성되지 않은 경우입니다. "
        "Claude Haiku는 컨텍스트가 없는 cond1에서 응답을 거절해 빈 칸이 나타납니다."
    )

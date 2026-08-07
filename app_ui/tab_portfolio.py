"""탭 4 — 포트폴리오 백테스트 (results/experiment 읽기 전용, API 호출 없음).

탭3이 "신호 하나의 평균 수익률"을 본다면 여기서는 그 신호대로 실제 운용했을 때의
누적 곡선과 낙폭을 본다. 같은 데이터라도 평균과 복리는 다른 그림을 그린다 —
신호를 적게 내는 모델은 평균이 좋아도 분산이 안 돼 포트폴리오가 흔들린다.

운용 규칙 (백테스트 데이터 구조에서 그대로 따라 나온다)
  · 매월 첫 거래일에 Buy 신호 종목을 동일가중 매수, 20거래일 보유 후 전량 청산
  · Buy가 0건인 달은 현금 보유(수익률 0)
  · Sell은 사용하지 않는다 — 개인투자자는 공매도가 사실상 불가하므로
    Sell을 "보유 회피" 신호로 보는 프로젝트 서술과 맞춘다 (롱온리)
"""

import numpy as np
import pandas as pd
import streamlit as st

from app_ui.shared import (
    COND_LABELS, DEFAULT_MODEL, REPORT_CONDS,
    list_matrix_models, load_signal_matrix,
)

# 한국 주식 왕복 거래비용 (거래세 + 수수료). 매월 전량 교체하므로 보유한 달마다 차감한다
ROUND_TRIP_COST = 0.25

# 월평균 보유 종목이 이 값 미만이면 분산이 안 돼 포트폴리오로 보기 어렵다
MIN_HOLDINGS = 3

# "전 종목"이라 쓰면 개별 분석 탭이 다루는 KRX 전 종목으로 읽힌다. 벤치마크의 모집단은
# 백테스트 유니버스인 20종목이므로 숫자로 못박는다
BENCH_LABEL = "시장참여 벤치마크 (20종목 동일가중)"


def monthly_returns(
    df: pd.DataFrame, months: list[str], cost: float, buy_only: bool = True
) -> tuple[pd.Series, pd.Series]:
    """월별 포트폴리오 수익률(%)과 그 달의 보유 종목 수.

    Buy 신호가 없는 달은 현금이므로 수익률 0이며 거래비용도 물지 않는다.
    """
    d = df[df["signal"] == "Buy"] if buy_only else df
    g = d.groupby("signal_date")["return_20d"].agg(["mean", "count"])
    held = g["count"].reindex(months).fillna(0).astype(int)
    ret = g["mean"].reindex(months).fillna(0.0) - np.where(held > 0, cost, 0.0)
    return ret, held


def summarize(ret: pd.Series, held: pd.Series) -> dict:
    """누적 곡선에서 성과·리스크 지표를 뽑는다.

    연환산은 관측 월수 기준 기하평균, 변동성·Sharpe는 월 수익률의 단순 연환산(×√12).
    Sharpe는 무위험수익률 0을 가정한다.
    """
    cum = (1 + ret / 100).cumprod()
    n_months = len(ret)
    total = (cum.iloc[-1] - 1) * 100
    annual = (cum.iloc[-1] ** (12 / n_months) - 1) * 100 if n_months else float("nan")
    mdd = ((cum / cum.cummax()) - 1).min() * 100
    sd = ret.std()
    return {
        "curve":   (cum - 1) * 100,
        "총수익":   total,
        "연환산":   annual,
        "MDD":     mdd,
        "변동성":   sd * np.sqrt(12),
        "Sharpe":  (ret.mean() / sd * np.sqrt(12)) if sd > 0 else float("nan"),
        "평균보유": held[held > 0].mean() if (held > 0).any() else 0.0,
        "현금개월": int((held == 0).sum()),
    }


def render() -> None:
    models = list_matrix_models()
    if not models:
        st.info(
            "백테스트 결과가 없습니다. "
            "`python src/experiment/llm_experiment.py --cond <조건> --model <모델명>` 실행 후 확인하세요."
        )
        return

    col_m, col_c = st.columns([3, 2])
    _idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    model = col_m.selectbox("모델", models, index=_idx, key="pf_model")
    col_c.markdown("<div style='height:1.75em'></div>", unsafe_allow_html=True)
    apply_cost = col_c.checkbox(
        f"거래비용 반영 (왕복 {ROUND_TRIP_COST}%)",
        value=True,
        help="한국 주식 거래세+수수료 기준. 매월 전량 교체하므로 보유한 달마다 차감합니다.",
    )
    cost = ROUND_TRIP_COST if apply_cost else 0.0

    sig_df = load_signal_matrix(model)
    if sig_df.empty:
        st.info(f"{model}의 백테스트 결과를 읽을 수 없습니다.")
        return

    months = sorted(sig_df["signal_date"].astype(str).unique())
    sig_df = sig_df.assign(signal_date=sig_df["signal_date"].astype(str))

    st.caption(
        f"매월 첫 거래일에 **Buy 신호 종목을 동일가중 매수**하고 20거래일 뒤 전량 청산하는 것을 "
        f"{len(months)}개월 반복했을 때의 누적 성과입니다. Buy가 없는 달은 현금이며, "
        "공매도는 개인투자자가 사실상 할 수 없으므로 **Sell은 사용하지 않습니다(롱온리)**."
    )

    # ── 곡선·지표 계산 ─────────────────────────────────────
    curves: dict[str, pd.Series] = {}
    rows: list[dict] = []
    thin: list[str] = []

    # 벤치마크 — 실험 20종목 동일가중. 조건과 무관하게 (종목, 신호일)이 같으므로 중복 제거로 얻는다.
    # Claude cond1처럼 응답 거절로 행이 빠진 조건이 있어도 나머지 조건이 채워준다
    bench_df = sig_df.drop_duplicates(subset=["ticker", "signal_date"])
    b_ret, b_held = monthly_returns(bench_df, months, cost, buy_only=False)
    b_stat = summarize(b_ret, b_held)
    curves[BENCH_LABEL] = b_stat["curve"]
    rows.append({"전략": BENCH_LABEL, **{k: v for k, v in b_stat.items() if k != "curve"}})

    for cond in REPORT_CONDS:
        sub = sig_df[sig_df["cond"] == cond]
        if sub.empty:
            continue
        label = COND_LABELS.get(cond, cond)
        ret, held = monthly_returns(sub, months, cost)
        stat = summarize(ret, held)
        if stat["평균보유"] < MIN_HOLDINGS:
            thin.append(f"{label} 월평균 {stat['평균보유']:.1f}종목·현금 {stat['현금개월']}개월")
            label += " ⚠️"
        curves[label] = stat["curve"]
        rows.append({"전략": label, **{k: v for k, v in stat.items() if k != "curve"}})

    # ── 누적 수익 곡선 ─────────────────────────────────────
    st.markdown("**누적 수익률 곡선**")
    chart_df = pd.DataFrame(curves)
    chart_df.index = pd.to_datetime(chart_df.index, errors="coerce")
    st.line_chart(chart_df, x_label="신호일", y_label="누적 수익률 (%)", height=380)

    # ── 성과·리스크 요약 ───────────────────────────────────
    st.markdown("**성과 · 리스크 요약**")
    summary = pd.DataFrame(rows).set_index("전략")
    summary.columns = [
        "총 수익률(%)", "연환산(%)", "MDD(%)", "연환산 변동성(%)", "Sharpe",
        "월평균 보유종목", "현금 보유 개월",
    ]
    st.dataframe(
        summary.style.format(
            {
                "총 수익률(%)": "{:+.1f}", "연환산(%)": "{:+.1f}", "MDD(%)": "{:.1f}",
                "연환산 변동성(%)": "{:.1f}", "Sharpe": "{:.2f}",
                "월평균 보유종목": "{:.1f}", "현금 보유 개월": "{:.0f}",
            },
            na_rep="-",
        ),
        width="stretch",
    )

    if thin:
        st.caption(
            f"⚠️ {', '.join(thin)} — 보유 종목이 너무 적어 분산이 되지 않습니다. "
            "포트폴리오 성과라기보다 소수 종목의 등락에 가까우므로 곡선을 성능으로 읽지 마십시오."
        )

    st.caption(
        f"**MDD**는 누적 곡선의 고점 대비 최대 낙폭입니다. 탭3의 수익률·Hit·Sharpe에는 없는 "
        f"리스크 축이라 함께 봐야 합니다. **연환산**은 {len(months)}개월 기하평균, "
        "**Sharpe**는 무위험수익률 0을 가정한 월 수익률의 단순 연환산(×√12)입니다."
    )

    with st.expander("이 곡선을 읽을 때 주의할 점"):
        st.markdown(
            "- **보유 구간이 정확히 맞물리지 않습니다.** 매월 첫 거래일 간격은 20~22거래일인데 "
            "보유는 20거래일 고정이라 달마다 며칠씩 어긋납니다. 근사치로 보십시오.\n"
            "- **단일 시장 국면입니다.** 2023-01~2025-12 대형주 20종목은 대체로 상승장이라 "
            "모든 곡선의 절대 수익률이 높게 나옵니다. 조건 간 비교와 벤치마크 대비 위치가 핵심이지 "
            "절대 수익률 자체가 아닙니다.\n"
            "- **벤치마크는 무기술 전략입니다.** 실험 대상 20종목을 매월 사서 들고 있는 것이라 "
            "종목 선별이 전혀 없습니다. 조건이 이 선을 넘지 못하면 선별이 값을 더하지 못했다는 뜻입니다.\n"
            "- **거래비용은 근사입니다.** 매월 전량 교체를 가정해 보유한 달마다 왕복 "
            f"{ROUND_TRIP_COST}%를 차감합니다. 실제로는 직전 달과 겹치는 종목은 교체 비용이 들지 않아 "
            "여기서 계산한 값이 다소 보수적입니다.\n"
            "- **슬리피지·유동성은 반영하지 않았습니다.** 대형주 20종목이라 영향은 작지만 0은 아닙니다."
        )

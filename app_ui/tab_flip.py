"""탭 5 — 조건 간 신호 전이 분석 (results/experiment 읽기 전용, API 호출 없음).

탭3의 유의성 검정은 조건별 Buy 신호를 **독립 표본**으로 놓고 비교한다. 그런데 조건들은
같은 (종목, 신호일)을 공유하므로 독립 가정이 성립하지 않는다(TODO A-2 방법론 한계).
여기서는 같은 (종목, 신호일)끼리 **짝지어** 컨텍스트를 추가했을 때 판단이 어떻게
바뀌었고 그 변화가 옳았는지를 본다.

중요 — 짝지은 표본에서 실제 20거래일 수익률은 조건과 무관하게 동일하다. (종목, 날짜)가
같으면 주가도 같기 때문이다. 따라서 수익률 자체를 짝지어 검정하면 차이가 항상 0이고,
검정 대상은 **신호가 만들어낸 결과**여야 한다. 두 가지를 나눠서 본다.

  · 부호검정 — 전환된 건들만 보고 판단이 좋아졌는지. 신호를 몇 개 내는지(노출량)에
    영향받지 않아 **판단의 질**을 직접 잰다. 주력 지표.
  · Wilcoxon — 롱온리로 운용했을 때 실현 수익의 순효과. 노출량이 섞여 있어 보조 지표다.
    상승장에서는 Buy를 많이 내기만 해도 값이 올라간다.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as sps

from app_ui.shared import (
    COND_LABELS, DEFAULT_MODEL, list_matrix_models, load_backtest_results,
)

SIGNALS = ["Buy", "Neutral", "Sell"]

# ablation 사슬에서 의미가 있는 쌍만 노출한다. 두 조건을 자유 선택하게 하면
# "cond1 → cond4_no_reports"처럼 해석이 곤란한 조합이 만들어진다
FLIP_PAIRS = [
    ("cond1", "cond2", "재무지표 추가"),
    ("cond2", "cond3", "리포트 추가"),
    ("cond3", "cond4", "DART 실적 추가"),
    ("cond4_no_reports", "cond4", "리포트 추가 (LOO ablation)"),
    ("cond2", "cond4_no_reports", "DART 실적 추가 (리포트 없이)"),
]


def signal_score(signal: pd.Series | np.ndarray, ret: np.ndarray) -> np.ndarray:
    """신호가 실제 방향과 맞았는지를 +1 / 0 / -1로 채점.

    Neutral은 방향을 주장하지 않으므로 0(중립). Buy·Sell은 맞으면 +1, 틀리면 -1.
    이 점수의 변화량으로 "전환이 개선인지 악화인지"를 판정한다.
    """
    s = np.asarray(signal)
    return np.where(s == "Buy", np.where(ret > 0, 1, -1),
                    np.where(s == "Sell", np.where(ret < 0, 1, -1), 0))


@st.cache_data(ttl=300, show_spinner=False)
def load_pair(cond_a: str, cond_b: str, model: str) -> pd.DataFrame:
    """두 조건을 (종목, 신호일)로 inner join한 짝 표본."""
    a = load_backtest_results(cond_a, model)
    b = load_backtest_results(cond_b, model)
    if a is None or b is None or a.empty or b.empty:
        return pd.DataFrame()
    a = a.assign(ticker=a["ticker"].astype(str).str.zfill(6))
    b = b.assign(ticker=b["ticker"].astype(str).str.zfill(6))
    cols = ["ticker", "name", "signal_date", "signal", "return_20d"]
    return a[cols].merge(b[cols], on=["ticker", "name", "signal_date"], suffixes=("_a", "_b"))


def render() -> None:
    models = list_matrix_models()
    if not models:
        st.info(
            "백테스트 결과가 없습니다. "
            "`python src/experiment/llm_experiment.py --cond <조건> --model <모델명>` 실행 후 확인하세요."
        )
        return

    col_m, col_p = st.columns([2, 3])
    _idx = models.index(DEFAULT_MODEL) if DEFAULT_MODEL in models else 0
    model = col_m.selectbox("모델", models, index=_idx, key="fl_model")
    pair_idx = col_p.selectbox(
        "비교할 컨텍스트 추가",
        options=range(len(FLIP_PAIRS)),
        format_func=lambda i: f"{FLIP_PAIRS[i][2]}  ({FLIP_PAIRS[i][0]} → {FLIP_PAIRS[i][1]})",
        index=3,  # 리포트 추가(LOO) 기본 — 보고서 핵심 비교
        key="fl_pair",
    )
    cond_a, cond_b, pair_label = FLIP_PAIRS[pair_idx]

    mg = load_pair(cond_a, cond_b, model)
    if mg.empty:
        st.info(f"{model}의 {cond_a} 또는 {cond_b} 결과가 없어 비교할 수 없습니다.")
        return

    label_a = COND_LABELS.get(cond_a, cond_a)
    label_b = COND_LABELS.get(cond_b, cond_b)
    ret = mg["return_20d_a"].to_numpy()   # (종목, 날짜)가 같으므로 조건 무관하게 동일
    changed = mg["signal_a"] != mg["signal_b"]
    n_pair, n_chg = len(mg), int(changed.sum())

    st.caption(
        f"**{label_a}** 에서 **{label_b}** 로 갈 때 무엇이 더해지는가 → **{pair_label}**. "
        f"같은 (종목, 신호일) {n_pair}쌍을 짝지어 비교합니다. "
        "탭3의 검정은 조건을 독립 표본으로 놓지만, 실제로는 같은 종목·같은 날짜를 공유하므로 "
        "독립 가정이 깨집니다. 이 탭이 그 한계를 짝 비교로 보완합니다."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("판단 유지", f"{n_pair - n_chg}건", delta=f"{(n_pair - n_chg) / n_pair * 100:.1f}%")
    c2.metric("판단 변경", f"{n_chg}건", delta=f"{n_chg / n_pair * 100:.1f}%")
    c3.metric(
        "Buy 신호 수 변화",
        f"{int((mg['signal_b'] == 'Buy').sum())}건",
        delta=f"{int((mg['signal_b'] == 'Buy').sum() - (mg['signal_a'] == 'Buy').sum()):+d}건",
        help="컨텍스트를 더했을 때 매수 신호가 늘었는지 줄었는지. 아래 Wilcoxon 해석에 필요합니다.",
    )

    # ── 전이표 ─────────────────────────────────────────────
    st.markdown(f"**신호 전이표** (행 = {label_a}, 열 = {label_b})")
    ct = (
        pd.crosstab(mg["signal_a"], mg["signal_b"])
        .reindex(index=SIGNALS, columns=SIGNALS)
        .fillna(0).astype(int)
    )
    st.dataframe(
        ct.style.apply(
            lambda s: ["background-color:#e2e3e5" if s.name == c else "" for c in s.index],
            axis=1,
        ),
        width="stretch",
    )
    st.caption("회색 대각선이 판단이 유지된 건수, 나머지가 컨텍스트를 더해 바뀐 건수입니다.")

    if n_chg == 0:
        st.info("판단이 바뀐 건이 없어 전환 분석을 할 수 없습니다.")
        return

    # ── 전환 상세 ──────────────────────────────────────────
    flips = mg[changed].copy()
    rf = flips["return_20d_a"].to_numpy()
    flips["delta"] = signal_score(flips["signal_b"], rf) - signal_score(flips["signal_a"], rf)

    st.markdown("**전환 유형별 결과**")
    detail = (
        flips.groupby([flips["signal_a"], flips["signal_b"]])
        .apply(lambda g: pd.Series({
            "건수": len(g),
            "실제 20일 수익률(%)": g["return_20d_a"].mean(),
            "개선": int((g["delta"] > 0).sum()),
            "악화": int((g["delta"] < 0).sum()),
        }), include_groups=False)
        .reset_index()
    )
    detail["전환"] = detail["signal_a"] + " → " + detail["signal_b"]
    detail = detail[["전환", "건수", "실제 20일 수익률(%)", "개선", "악화"]].set_index("전환")
    st.dataframe(
        detail.style.format({"건수": "{:.0f}", "실제 20일 수익률(%)": "{:+.2f}",
                             "개선": "{:.0f}", "악화": "{:.0f}"}),
        width="stretch",
    )
    st.caption(
        "실제 20일 수익률은 그 전환이 일어난 건들의 실제 주가 움직임입니다 "
        "(조건과 무관하게 종목·날짜로만 정해집니다). "
        "**개선·악화**는 바뀐 신호가 실제 방향과 더 맞게 됐는지로, "
        "맞으면 +1 · 틀리면 −1 · Neutral은 0으로 채점한 값의 변화입니다."
    )

    # ── 검정 ───────────────────────────────────────────────
    st.markdown("**짝 비교 검정**")
    up = int((flips["delta"] > 0).sum())
    dn = int((flips["delta"] < 0).sum())
    eq = int((flips["delta"] == 0).sum())

    long_a = np.where(mg["signal_a"] == "Buy", ret, 0.0)
    long_b = np.where(mg["signal_b"] == "Buy", ret, 0.0)

    col_s, col_w = st.columns(2)

    with col_s:
        st.markdown("**① 부호검정 — 판단의 질** (주력)")
        if up + dn > 0:
            p_sign = sps.binomtest(up, up + dn, 0.5).pvalue
            st.metric(
                f"개선 {up}건 vs 악화 {dn}건",
                f"p = {p_sign:.4f}",
                delta=f"{up - dn:+d}건",
                help="전환된 건들만 대상. 개선과 악화가 반반이면 컨텍스트가 판단을 바꾸기는 했어도 더 낫게 만들지는 못했다는 뜻입니다.",
            )
        else:
            st.caption(f"방향 판정이 가능한 전환이 없습니다 (무변 {eq}건).")
        st.caption(
            "**신호를 몇 개 내는지에 영향받지 않습니다.** 전환된 건만 보고 "
            "옳은 방향으로 바뀌었는지만 세기 때문입니다."
        )

    with col_w:
        st.markdown("**② Wilcoxon — 실현 수익의 순효과** (보조)")
        try:
            _, p_wil = sps.wilcoxon(long_b, long_a, zero_method="wilcox")
        except ValueError:
            p_wil = float("nan")
        st.metric(
            f"짝당 평균 실현수익 {long_a.mean():+.3f}% → {long_b.mean():+.3f}%",
            f"p = {p_wil:.4f}" if not np.isnan(p_wil) else "p = -",
            delta=f"{long_b.mean() - long_a.mean():+.3f}%p",
            help=(
                "Buy면 실제 수익률, 아니면 0으로 두고(롱온리) 짝지어 검정합니다. "
                "짝 하나가 (종목, 신호일) 한 건이라 Buy가 아닌 건의 0까지 평균에 들어갑니다. "
                "Buy만 동일가중으로 운용하는 포트폴리오 탭의 월수익률과는 축이 다르므로 "
                "숫자를 직접 견주지 마십시오."
            ),
        )
        st.caption(
            "⚠️ **노출량이 섞여 있습니다.** 실험 기간이 대체로 상승장이라 "
            "Buy를 많이 내기만 해도 이 값은 올라갑니다. 위의 Buy 신호 수 변화와 함께 보십시오. "
            "판단이 좋아졌는지는 ①로 판단합니다. "
            "짝 하나가 (종목, 신호일) 한 건이라 Buy가 아닌 건의 0도 평균에 포함되며, "
            "이 때문에 포트폴리오 탭의 월수익률보다 작게 나옵니다."
        )

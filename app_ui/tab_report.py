"""탭 3 — 백테스트 성과·모델 비교 (results/analysis 읽기 전용, API 호출 없음)."""

import os

import pandas as pd
import streamlit as st

from app_ui.shared import (
    ANALYSIS_DIR, COND_LABELS, DEFAULT_MODEL, REPORT_CONDS, SMALL_SAMPLE_N, fmt_metric,
)


def model_column_rows(models: list[str], per_row: int = 2):
    """모델 목록을 per_row개씩 끊어 (컬럼, 모델) 쌍으로 내보낸다.

    st.columns(len(models))로 한 줄에 몰면 gpt/claude 백테스트가 끝나 4모델이 됐을 때
    표가 읽을 수 없을 만큼 좁아진다. 줄을 나눠도 좌우 대조는 유지된다.
    """
    for i in range(0, len(models), per_row):
        cols = st.columns(per_row)
        for col, m in zip(cols, models[i:i + per_row]):
            yield col, m


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


def build_strategy_table(comp: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """all_comparison → '무기술 벤치마크 + 베이스라인 + 조건별 Buy' 단일 비교표.

    full_comparison.csv를 그대로 쓰면 조건 행이 다섯 줄 모두 소수점까지 같은 값이 된다.
    그 행은 Buy/Neutral/Sell을 합친 전 신호(720건)라 어느 조건이든 동일한
    (종목, 신호일) 집합이 되기 때문이다. 버그가 아니라 구조적 결과다.
    게다가 베이스라인 행은 Buy만 집계돼(compare.py가 베이스라인 신호를 전부 Buy로 처리)
    조건 행과 비교 축이 어긋나 있었다.

    그래서 여기서는 ① 조건을 **Buy 기준으로 통일**해 베이스라인과 축을 맞추고,
    ② 값이 같던 전 신호 행은 **'시장참여 벤치마크(무기술)' 한 줄**로 접는다.
    벤치마크가 한 줄로 서면 "cond Buy > 벤치마크"가 단순 상승장 편승이 아닌
    종목 선별 효과라는 논증이 화면에서 바로 읽힌다.

    Returns:
        (표, 소표본 경고 라벨, 벤치마크에서 제외된 조건 안내)
    """
    rows: list[dict] = []
    small: list[str] = []
    notes: list[str] = []

    def add(label: str, r: pd.Series | None, mark_small: bool = False) -> None:
        if r is None:
            rows.append({"전략": label, "신호수": "0", "평균수익률(%)": "신호 없음",
                         "Hit(%)": "-", "Sharpe": "-", "초과수익(%)": "-", "초과Hit(%)": "-"})
            return
        n = int(r["n"])
        flag = mark_small and n < SMALL_SAMPLE_N
        if flag:
            small.append(f"{label} {n}건")
        rows.append({
            "전략":          label + (" ⚠️" if flag else ""),
            "신호수":        f"{n:,}",
            "평균수익률(%)": fmt_metric(r["mean"], signed=True),
            "Hit(%)":        fmt_metric(r["hit_rate"], 1),
            "Sharpe":        fmt_metric(r["sharpe"]),
            "초과수익(%)":   fmt_metric(r.get("mean_excess"), signed=True),
            "초과Hit(%)":    fmt_metric(r.get("hit_rate_excess"), 1),
        })

    # ① 무기술 벤치마크 — 전 신호 행. 조건 간 동일하므로 표본이 가장 큰 것 하나만 쓴다.
    #    Claude cond1처럼 응답 거절로 n이 줄어든 조건은 부분집합이라 벤치마크가 아니다.
    total = comp[comp["signal"] == "전체"]
    if not total.empty:
        full_n = int(total["n"].max())
        add("시장참여 벤치마크 (무기술)", total[total["n"] == full_n].iloc[0])
        short = total[total["n"] < full_n]
        for _lb, _n in zip(short["label"], short["n"]):
            notes.append(f"{COND_LABELS.get(_lb, _lb)} {int(_n)}건")

    # ② 베이스라인 — compare.py가 신호 구분 없이 전부 Buy로 집계
    for key in ("Consensus", "GoldenCross"):
        r = comp[(comp["label"] == key) & (comp["signal"] == "Buy")]
        if not r.empty:
            add(COND_LABELS.get(key, key), r.iloc[0])

    # ③ LLM 조건 — Buy 기준으로 통일
    for cond in REPORT_CONDS:
        r = comp[(comp["label"] == cond) & (comp["signal"] == "Buy")]
        add(COND_LABELS.get(cond, cond), r.iloc[0] if not r.empty else None, mark_small=True)

    return pd.DataFrame(rows).set_index("전략"), small, notes


def localize_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """breakdown CSV의 원본 영문 컬럼·조건 코드를 화면 표기로 바꾼다.

    섹션 A·B·C는 모두 한글화하는데 이 표만 CSV 그대로 나와, 펼치면 앞 섹션과
    표기가 어긋났다. 컬럼 구성이 바뀌어도 죽지 않게 존재하는 것만 골라 바꾼다.
    """
    BD_RENAME = {
        "bucket": "구간", "cond": "조건", "signal": "신호", "n": "건수",
        "share_pct": "비중(%)",
        "abs_mean": "절대수익 평균(%)", "abs_median": "절대수익 중앙값(%)", "abs_hit": "절대 Hit(%)",
        "excess_mean": "초과수익 평균(%)", "excess_median": "초과수익 중앙값(%)",
        "excess_hit": "초과 Hit(%)",
        "bench_abs": "벤치 절대수익(%)", "bench_excess": "벤치 초과수익(%)",
    }
    out = df.copy()
    if "cond" in out.columns:
        out["cond"] = out["cond"].map(lambda k: COND_LABELS.get(k, k))
    return out[[c for c in BD_RENAME if c in out.columns]].rename(columns=BD_RENAME)


def render() -> None:
    an_models = list_analysis_models()
    if not an_models:
        st.info(
            "분석 결과가 없습니다. "
            "`python src/experiment/compare.py --all --model <모델명>` 실행 후 확인하세요."
        )
        return

    st.caption(
        f"백테스트 완료 모델 {len(an_models)}종: {', '.join(an_models)}. "
        "개별 종목 분석 탭은 이 외의 모델로도 신호를 생성할 수 있으나, 분석 산출물이 있는 모델만 이 탭에 표시됩니다."
    )

    # ── A. Buy 신호 성과 — 모델 비교 ────────────────────
    st.subheader("🎯 Buy 신호 성과 — 모델 비교 (20거래일)")
    BUY_RENAME = {
        "label": "조건", "n": "신호수", "mean": "평균수익률(%)", "hit_rate": "Hit(%)",
        "sharpe": "Sharpe", "mean_excess": "초과수익(%)", "hit_rate_excess": "초과Hit(%)",
        "conf_mean": "평균신뢰도",
    }
    _small_notes: list[str] = []
    for col, m in model_column_rows(an_models):
        with col:
            st.markdown(f"**{m}**")
            comp = load_analysis_csv(m, "all_comparison.csv")
            if comp is None:
                st.caption("해당 모델 결과 없음")
                continue
            buy = comp[(comp["signal"] == "Buy") & (comp["label"].isin(REPORT_CONDS))].copy()
            # 표본이 작은 조건은 평균이 크게 흔들린다 — 행에 ⚠️를 달고 아래에 한 줄로 모아 경고
            _small_labels = set()
            for _lb, _n in zip(buy["label"], buy["n"]):
                if _n < SMALL_SAMPLE_N:
                    _small_notes.append(f"{m} {COND_LABELS.get(_lb, _lb)} {int(_n)}건")
                    _small_labels.add(_lb)
            buy["label"] = buy["label"].map(
                lambda k: COND_LABELS.get(k, k) + (" ⚠️" if k in _small_labels else "")
            )
            # 구 스키마 CSV 방어 — 존재하는 컬럼만 선택 (섹션 B와 동일 패턴)
            buy = buy[[c for c in BUY_RENAME if c in buy.columns]].rename(columns=BUY_RENAME)
            # na_rep — Buy가 1건이면 Sharpe가 정의되지 않아 NaN이 그대로 노출됐다
            st.dataframe(
                buy.set_index("조건").style.format(precision=2, na_rep="-"),
                width="stretch",
            )

    if _small_notes:
        st.caption(
            f"⚠️ 매수 신호가 {SMALL_SAMPLE_N}건 미만인 조건({', '.join(_small_notes)})은 "
            "평균이 크게 흔들리므로 성능으로 해석하지 마십시오."
        )

    st.divider()

    # ── B. 벤치마크·베이스라인 대비 성과 ────────────────
    st.subheader("⚖️ 벤치마크·베이스라인 대비 성과 (Buy 신호, 20거래일)")
    st.caption(
        "**시장참여 벤치마크**는 실험 대상 20종목을 매월 사들여 20거래일 보유한 무기술 전략입니다. "
        "LLM 조건이 이 줄을 넘어야 단순 상승장 편승이 아닌 종목 선별 효과로 볼 수 있습니다. "
        "베이스라인은 컨센서스(애널리스트 투자의견)·골든크로스(기술분석)이며, "
        "모든 줄이 **매수 방향 신호 기준**이라 서로 비교됩니다."
    )
    _b_small: list[str] = []
    _b_notes: list[str] = []
    for col, m in model_column_rows(an_models):
        with col:
            st.markdown(f"**{m}**")
            comp = load_analysis_csv(m, "all_comparison.csv")
            if comp is None:
                st.caption("해당 모델 결과 없음")
                continue
            tbl, small, notes = build_strategy_table(comp)
            _b_small += [f"{m} {s}" for s in small]
            _b_notes += [f"{m} {n}" for n in notes]
            st.dataframe(tbl, width="stretch")

    if _b_small:
        st.caption(
            f"⚠️ 매수 신호가 {SMALL_SAMPLE_N}건 미만인 조건({', '.join(_b_small)})은 "
            "평균·Hit·Sharpe가 크게 흔들립니다. 성능으로 해석하지 마십시오 "
            "(신호 1건이면 Sharpe는 정의되지 않아 `-`로 표시됩니다)."
        )
    if _b_notes:
        st.caption(
            f"ℹ️ 벤치마크 산출에서 제외한 조건: {', '.join(_b_notes)}. "
            "모델이 응답을 거절해 전 신호 수가 줄어든 조건으로, 전체 표본의 부분집합이라 "
            "무기술 벤치마크로 쓸 수 없습니다."
        )

    st.divider()

    # ── C. 통계적 유의성 ────────────────────────────────
    st.subheader("📐 통계적 유의성 검정 (Buy 신호)")
    # 목록이 알파벳순이라 index=0이면 claude가 잡힌다. 다른 탭과 같이 앵커 모델을 기본으로
    _an_idx = an_models.index(DEFAULT_MODEL) if DEFAULT_MODEL in an_models else 0
    sel_an_model = st.selectbox("검정 결과 모델", an_models, index=_an_idx)
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

        # 원본 컬럼명(group_a, n_a, effect_size_type 등)이 그대로 노출되던 것을 한글화.
        # 섹션 A·B는 rename을 하는데 여기만 raw였다
        SIG_RENAME = {
            "group_a": "비교 대상", "group_b": "기준", "n_a": "n(대상)", "n_b": "n(기준)",
            "metric": "지표", "metric_type": "구분",
            "mean_a": "평균(대상)", "mean_b": "평균(기준)", "mean_diff": "평균차",
            "test": "검정", "statistic": "통계량", "p_value": "p값",
            "effect_size": "효과크기", "effect_size_type": "효과크기 종류",
            "significance": "유의",
        }
        METRIC_LABELS = {"return_20d": "20일 절대수익", "excess_return_20d": "20일 초과수익"}

        core = sig[sig["category"] == "core"].copy()
        core["group_a"] = core["group_a"].map(lambda k: COND_LABELS.get(k, k))
        core["group_b"] = core["group_b"].map(lambda k: COND_LABELS.get(k, k))
        core["metric"]  = core["metric"].map(lambda k: METRIC_LABELS.get(k, k))
        core = core[[c for c in SIG_RENAME if c in core.columns]].rename(columns=SIG_RENAME)
        styled = core.style.map(
            lambda v: "background-color:#d4edda" if isinstance(v, (int, float)) and v < 0.05 else "",
            subset=["p값"],
        )
        st.dataframe(styled, width="stretch", hide_index=True)
        st.caption("유의 수준: *** p<0.001, ** p<0.01, * p<0.05, . p<0.10, ns 비유의")
        # 보조 비교(category != core) 표는 미노출. 보고서가 다루지 않는 조합까지 나와
        # "이건 뭐냐"는 설명 부담만 생긴다. 핵심 비교만 보여준다.

    # ── D. 연도별·시장국면별 분석 ───────────────────────
    with st.expander("📅 연도별·시장국면별 분석 (breakdown)"):
        yearly = load_analysis_csv(sel_an_model, "breakdown_yearly.csv")
        regime = load_analysis_csv(sel_an_model, "breakdown_regime.csv")
        if yearly is not None:
            st.markdown("**연도별**")
            st.dataframe(localize_breakdown(yearly), width="stretch", hide_index=True)
        if regime is not None:
            st.markdown("**시장 국면별 (상승/하락)**")
            st.dataframe(localize_breakdown(regime), width="stretch", hide_index=True)
        if yearly is None and regime is None:
            st.caption("해당 모델 breakdown 결과 없음")

"""
실험 조건 간 통계적 유의성 검정 (추론통계)

compare.py가 기술통계(평균·Hit Rate·Sharpe)를 담당하는 반면,
이 스크립트는 "조건 간 차이가 우연인가, 유의미한가"를 검정한다.
  - Mann-Whitney U (비모수) + Cliff's delta (effect size)
  - Welch's t-test (모수)     + Cohen's d   (effect size)
Buy 신호만 대상, 20d 절대/초과 수익률 2개 metric.

데이터 로드는 compare.py의 로더(load_cond_data/load_baselines)를 재사용한다.

사용법:
  python src/experiment/significance.py --cond cond4   # cond1~cond4 + 베이스라인
  python src/experiment/significance.py --all          # 전체 cond + 베이스라인

저장 경로:
  results/analysis/{cond|all}_significance.csv
  results/analysis/latest/{cond|all}_significance.csv
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import get_analysis_dir, get_latest_analysis_dir
from experiments import EXPERIMENTS
from compare import load_cond_data, load_baselines, DEFAULT_MODEL

# --cond 옵션에서 사용할 순서 기준 목록 (experiments.py에서 자동 파생)
COND_ORDER = list(EXPERIMENTS.keys())


# ── effect size 헬퍼 ─────────────────────────────────────

def cliffs_delta(x: pd.Series, y: pd.Series) -> float:
    """Cliff's delta — 비모수 effect size (-1 ~ +1).

    Mann-Whitney와 짝을 이루는 effect size 지표.
    |delta| < 0.147: negligible, 0.147~0.33: small,
    0.33~0.474: medium, > 0.474: large
    """
    x = x.dropna().to_numpy()
    y = y.dropna().to_numpy()
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    nx, ny = len(x), len(y)
    greater = 0
    less    = 0
    for xi in x:
        greater += int((xi > y).sum())
        less    += int((xi < y).sum())
    return round((greater - less) / (nx * ny), 4)


def cohens_d(x: pd.Series, y: pd.Series) -> float:
    """Cohen's d — 모수 effect size (표준화 평균 차이).

    |d| < 0.2: negligible, 0.2~0.5: small,
    0.5~0.8: medium, > 0.8: large
    """
    x = x.dropna().to_numpy()
    y = y.dropna().to_numpy()
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return float("nan")
    return round((x.mean() - y.mean()) / pooled_sd, 4)


def significance_level(p: float) -> str:
    """p-value를 별표 레이블로 변환 (논문 표시 관행)."""
    if pd.isna(p):
        return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.10:  return "."
    return "ns"


# ── 검정 실행 ────────────────────────────────────────────

def run_significance_tests(
    cond_data: dict[str, pd.DataFrame],
    baseline_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """사전 정의된 비교 pair에 대해 Mann-Whitney + Welch's t-test 실행.

    Buy 신호만 대상, 20d 절대/초과 수익률 2개 metric.
    # TODO: paired test (signal flip analysis) is future work
    """
    # 각 pair는 독립적 연구 질문에 대응하므로 Bonferroni 등 다중비교 보정 미적용
    # (pair 간 귀무가설이 서로 다름. 동일 데이터 공유는 있으나 질문 자체가 독립적)
    PAIRS = [
        # ── 핵심 3개 ─────────────────────────────────────
        ("cond4", "cond1",            "core"),      # 컨텍스트 최대 vs 최소
        ("cond4", "GoldenCross",      "core"),      # LLM vs 기술분석
        ("cond4", "Consensus",        "core"),      # LLM vs 애널리스트
        # ── 보조: 컨텍스트 단계별 효과 ─────────────────
        ("cond2", "cond1",            "auxiliary"), # 재무지표 추가 효과
        ("cond3", "cond1",            "auxiliary"), # 재무+리포트 추가 효과
        # ── 보조: LOO ablation (reports marginal effect) ─
        ("cond4", "cond4_no_reports", "auxiliary"), # 리포트 순수 기여도
        # ── 보조: blind ablation ─────────────────────────
        ("cond4_blind", "cond4",      "auxiliary"), # blind ablation: 종목명 익명화 효과 측정
    ]
    METRICS = [
        ("return_20d",        "absolute"),
        ("excess_return_20d", "excess"),
    ]

    all_data = {**cond_data, **baseline_data}
    rows = []

    for group_a, group_b, category in PAIRS:
        if group_a not in all_data or group_b not in all_data:
            print(f"  [스킵] {group_a} vs {group_b}: 데이터 없음")
            continue

        df_a = all_data[group_a]
        df_b = all_data[group_b]
        buy_a = df_a[df_a["signal"] == "Buy"] if "signal" in df_a.columns else df_a
        buy_b = df_b[df_b["signal"] == "Buy"] if "signal" in df_b.columns else df_b

        for metric, metric_type in METRICS:
            if metric not in buy_a.columns or metric not in buy_b.columns:
                print(f"  [스킵] {metric}: {group_a} 또는 {group_b}에 컬럼 없음")
                continue

            x = buy_a[metric].dropna()
            y = buy_b[metric].dropna()

            if len(x) < 2 or len(y) < 2:
                print(f"  [스킵] {group_a} vs {group_b} ({metric}): 표본 부족")
                continue

            mean_a    = round(x.mean(), 4)
            mean_b    = round(y.mean(), 4)
            mean_diff = round(mean_a - mean_b, 4)

            try:
                mw_stat, mw_p = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
                cliff = cliffs_delta(x, y)
                rows.append({
                    "category": category, "group_a": group_a, "group_b": group_b,
                    "n_a": len(x), "n_b": len(y),
                    "metric": metric, "metric_type": metric_type,
                    "mean_a": mean_a, "mean_b": mean_b, "mean_diff": mean_diff,
                    "test": "mann_whitney",
                    "statistic":        round(float(mw_stat), 2),
                    "p_value":          round(float(mw_p), 4),
                    "effect_size":      cliff,
                    "effect_size_type": "cliffs_delta",
                    "significance":     significance_level(mw_p),
                })
            except Exception as e:
                print(f"  [오류] Mann-Whitney {group_a} vs {group_b}: {e}")

            try:
                t_stat, t_p = scipy_stats.ttest_ind(x, y, equal_var=False)
                d = cohens_d(x, y)
                rows.append({
                    "category": category, "group_a": group_a, "group_b": group_b,
                    "n_a": len(x), "n_b": len(y),
                    "metric": metric, "metric_type": metric_type,
                    "mean_a": mean_a, "mean_b": mean_b, "mean_diff": mean_diff,
                    "test": "welch_ttest",
                    "statistic":        round(float(t_stat), 4),
                    "p_value":          round(float(t_p), 4),
                    "effect_size":      d,
                    "effect_size_type": "cohens_d",
                    "significance":     significance_level(t_p),
                })
            except Exception as e:
                print(f"  [오류] Welch's t-test {group_a} vs {group_b}: {e}")

    return pd.DataFrame(rows)


def print_significance_tests(df: pd.DataFrame) -> None:
    """통계 검정 결과 콘솔 출력 (pair별 그룹화)."""
    if df.empty:
        print("  통계 검정 결과 없음")
        return

    print("\n" + "=" * 80)
    print("통계적 유의성 검정 (Buy 신호, two-sided)")
    print("=" * 80)
    print("유의 수준: *** p<0.001  ** p<0.01  * p<0.05  . p<0.10  ns otherwise")

    for category in ["core", "auxiliary"]:
        sub = df[df["category"] == category]
        if sub.empty:
            continue
        label = "핵심 비교" if category == "core" else "보조 비교"
        print(f"\n▶ {label}")

        for (ga, gb), grp in sub.groupby(["group_a", "group_b"], sort=False):
            print(f"\n  [{ga}] vs [{gb}]")
            for _, r in grp.iterrows():
                metric_label = "절대" if r["metric_type"] == "absolute" else "초과"
                print(
                    f"    {metric_label} {r['metric']:<20} "
                    f"n={r['n_a']}/{r['n_b']}  "
                    f"diff={r['mean_diff']:+.3f}%  "
                    f"{r['test']:<13} "
                    f"stat={r['statistic']:>10}  "
                    f"p={r['p_value']:.4f} {r['significance']:<3}  "
                    f"{r['effect_size_type']}={r['effect_size']}"
                )


# ── 메인 ─────────────────────────────────────────────────

def run(cond_target: str | None, is_all: bool, model: str = DEFAULT_MODEL) -> None:
    out_dir    = get_analysis_dir(model)
    latest_dir = get_latest_analysis_dir(model)

    if is_all:
        target_conds = COND_ORDER
        save_prefix  = "all"
    else:
        idx          = COND_ORDER.index(cond_target)
        target_conds = COND_ORDER[: idx + 1]
        save_prefix  = cond_target

    print(f"\n모델: {model} | 검정 대상: {target_conds}")
    print("결과 파일 로드 중...")

    cond_data     = load_cond_data(target_conds, model)
    baseline_data = load_baselines()

    if not cond_data:
        print("검정할 cond 결과 파일이 없습니다.")
        return

    sig_df = run_significance_tests(cond_data, baseline_data)
    print_significance_tests(sig_df)

    if not sig_df.empty:
        sig_fname = f"{save_prefix}_significance.csv"
        sig_path  = os.path.join(out_dir, sig_fname)
        sig_df.to_csv(sig_path, index=False, encoding="utf-8-sig")
        shutil.copy(sig_path, os.path.join(latest_dir, sig_fname))
        print(f"\n저장: {sig_fname}")

    print("\n검정 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 조건 간 통계적 유의성 검정")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cond", choices=COND_ORDER,
        help="검정 기준 cond (cond1부터 해당 cond까지 + 베이스라인 포함)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="전체 cond + 베이스라인 검정",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"분석할 모델 (기본값: {DEFAULT_MODEL})")
    args = parser.parse_args()
    run(
        cond_target = args.cond if not args.all else None,
        is_all      = args.all,
        model       = args.model,
    )

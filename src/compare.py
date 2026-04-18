"""
실험 조건 비교 분석 통합 스크립트

사용법:
  python src/compare.py --cond cond2           # cond1 + cond2 비교
  python src/compare.py --cond cond3           # cond1 + cond2 + cond3 비교
  python src/compare.py --cond cond3 --sector  # 섹터·종목별 분석 추가
  python src/compare.py --all                  # 전체 cond 자동 감지 + 섹터 포함

저장 경로:
  results/analysis/{cond}_comparison.csv
  results/analysis/{cond}_sector.csv   (--sector 또는 --all)
  results/analysis/{cond}_stock.csv    (--sector 또는 --all)
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import EXPERIMENT_DIR, KOSDAQ_TICKERS, get_latest_baseline_dir, get_analysis_dir, get_latest_analysis_dir
from experiments import EXPERIMENTS

# --cond 옵션에서 사용할 순서 기준 목록 (experiments.py에서 자동 파생)
COND_ORDER = list(EXPERIMENTS.keys())

# 전략별 표시 레이블
COND_LABELS = {
    "Consensus":          "컨센서스",
    "GoldenCross":        "골든크로스",
    "cond1":              "cond1 (종목명)",
    "cond2":              "cond2 (재무지표)",
    "cond3":              "cond3 (재무+리포트)",
    "cond4":              "cond4 (재무+리포트+DART)",
    "reports_only":       "리포트 단독",
    "dart_only":          "DART 단독",
    "cond4_no_reports":   "cond4 - reports (재무+DART)",
}

SECTORS = {
    "반도체":   ["005930", "000660"],
    "바이오":   ["207940", "068270", "196170"],
    "배터리":   ["373220", "006400", "247540"],
    "자동차":   ["005380", "000270"],
    "금융":     ["105560", "055550"],
    "IT플랫폼": ["035720", "035420"],
    "기타":     ["051910", "352820", "329180", "012450", "259960", "034020"],
}
# ── 통계 헬퍼 ─────────────────────────────────────────────

# Note: Sharpe는 관측치를 IID로 가정한 단순 연환산. 동일 종목의
# 월별 관측치 간 시계열 자기상관은 보정되지 않아 실제보다 과대추정
# 경향이 있음. 상세 한계는 논문 Limitations 섹션 참고.

def sharpe(series: pd.Series, hold_days: int = 20) -> float:
    """hold_days 수익률 → 연환산 Sharpe (무위험수익률 0 가정).

    연환산 승수: sqrt(252 / hold_days)
    - hold_days=5  → sqrt(50.4)  ≈ 7.10 (주간 기준)
    - hold_days=20 → sqrt(12.6)  ≈ 3.55 (≈월간 기준)

    Note: 관측치를 IID로 가정한 단순 연환산. 시계열 자기상관 및
    월별 샘플링 주기와 hold 구간 간 mismatch는 보정하지 않음.
    """
    s = series.dropna()
    if len(s) < 2 or s.std() == 0:
        return float("nan")
    annualize = np.sqrt(252 / hold_days)
    return round((s.mean() / s.std()) * annualize, 3)


def calc_stats(series: pd.Series, signal: str = "Buy", hold_days: int = 20) -> dict:
    """signal='Sell'이면 hit_rate = (s < 0) 비율 (하락이 성공), 나머지는 (s > 0)."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "hit_rate": float("nan"), "sharpe": float("nan")}
    hit = (s < 0).mean() if signal == "Sell" else (s > 0).mean()
    return {
        "n":        n,
        "mean":     round(float(s.mean()), 3),
        "hit_rate": round(float(hit * 100), 1),
        "sharpe":   sharpe(s, hold_days=hold_days),
    }


# ── 통계 검정 헬퍼 ───────────────────────────────────────

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


def run_significance_tests(
    cond_data: dict[str, pd.DataFrame],
    baseline_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """사전 정의된 비교 pair에 대해 Mann-Whitney + Welch's t-test 실행.

    Buy 신호만 대상, 20d 절대/초과 수익률 2개 metric.
    # TODO: paired test (signal flip analysis) is future work
    """
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


# ── 데이터 로드 ───────────────────────────────────────────

def _normalize_ret(df: pd.DataFrame) -> pd.DataFrame:
    """return_20d, excess_return_20d 컬럼 이름 통일."""
    if "return_20d" not in df.columns:
        for alt in ["return_pct", "return", "수익률"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "return_20d"})
                break
    if "excess_return_20d" not in df.columns:
        if "excess_return_pct" in df.columns:
            df = df.rename(columns={"excess_return_pct": "excess_return_20d"})
    return df


def load_cond_data(conds: list[str]) -> dict[str, pd.DataFrame]:
    """존재하는 cond 결과 파일만 로드."""
    loaded = {}
    for cond in conds:
        path = os.path.join(EXPERIMENT_DIR, cond, "latest", f"{cond}_results.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"ticker": str})
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df = _normalize_ret(df)
        loaded[cond] = df
        print(f"  로드: {cond} ({len(df)}행)")
    return loaded


def load_baselines() -> dict[str, pd.DataFrame]:
    """컨센서스·골든크로스 베이스라인 로드. 없으면 빈 dict."""
    try:
        baseline_dir = get_latest_baseline_dir()
    except Exception:
        return {}

    result = {}
    for label, fname in [("Consensus", "consensus_returns.csv"),
                         ("GoldenCross", "golden_returns.csv")]:
        path = os.path.join(baseline_dir, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"ticker": str})
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df = _normalize_ret(df)
        if "signal" not in df.columns:
            df["signal"] = "Buy"
        result[label] = df
        print(f"  로드: {label} ({len(df)}행)")
    return result


# ── 신호별 통계 행 생성 ────────────────────────────────────

def signal_rows(df: pd.DataFrame, label: str, has_confidence: bool = True,
                include_total: bool = True) -> list[dict]:
    """신호별(Buy/Neutral/Sell) + 전체 통계 딕셔너리 리스트 반환.

    include_total=False: 베이스라인처럼 단일 신호(전체=Buy)인 경우 중복 '전체' 행 생략.
    """
    rows = []
    has_5d         = "return_5d" in df.columns
    has_excess     = "excess_return_20d" in df.columns
    has_excess_5d  = "excess_return_5d" in df.columns
    _nan = {"mean": float("nan"), "hit_rate": float("nan"), "sharpe": float("nan")}

    groups = [
        ("Buy",     df[df["signal"] == "Buy"]),
        ("Neutral", df[df["signal"] == "Neutral"]),
        ("Sell",    df[df["signal"] == "Sell"]),
    ]
    if include_total:
        groups.append(("전체", df))
    for sig, g in groups:
        if sig != "전체" and len(g) == 0:
            continue
        s20  = calc_stats(g["return_20d"],        sig, hold_days=20)
        s5   = calc_stats(g["return_5d"],          sig, hold_days=5)  if has_5d        else _nan
        es20 = calc_stats(g["excess_return_20d"],  sig, hold_days=20) if has_excess    else _nan
        es5  = calc_stats(g["excess_return_5d"],   sig, hold_days=5)  if has_excess_5d else _nan
        row = {
            "label": label, "signal": sig,
            "n": s20["n"],
            "mean_5d":           s5["mean"],   "hit_rate_5d":           s5["hit_rate"],   "sharpe_5d":           s5["sharpe"],
            "mean":              s20["mean"],  "hit_rate":              s20["hit_rate"],  "sharpe":              s20["sharpe"],
            "mean_excess_5d":    es5["mean"],  "hit_rate_excess_5d":    es5["hit_rate"],  "sharpe_excess_5d":    es5["sharpe"],
            "mean_excess":       es20["mean"], "hit_rate_excess":       es20["hit_rate"], "sharpe_excess":       es20["sharpe"],
        }
        if has_confidence and "confidence" in g.columns and not g.empty:
            row["conf_mean"] = round(float(g["confidence"].mean()), 1)
            row["conf_std"]  = round(float(g["confidence"].std()), 1) if len(g) > 1 else float("nan")
        else:
            row["conf_mean"] = float("nan")
            row["conf_std"]  = float("nan")
        rows.append(row)
    return rows


# ── 전략 요약표 출력 (한 줄 = 한 전략) ──────────────────────

def print_full_comparison(baseline_data: dict, cond_data: dict) -> pd.DataFrame:
    """전략별 전체 통계를 한 줄씩 출력하고 DataFrame 반환."""
    SEP = "─" * 96
    FMT = "{:<22} {:>6} {:>12} {:>10} {:>8} {:>12} {:>10} {:>8}"
    print("\n" + SEP)
    print(FMT.format("전략", "신호수", "평균수익률", "Hit Rate", "Sharpe", "초과수익률", "초과Hit", "초과Sharpe"))
    print(SEP)

    rows = []
    for label, df in {**baseline_data, **cond_data}.items():
        s    = calc_stats(df["return_20d"], hold_days=20)
        has_excess = "excess_return_20d" in df.columns
        se   = calc_stats(df["excess_return_20d"], hold_days=20) if has_excess else {"mean": float("nan"), "hit_rate": float("nan"), "sharpe": float("nan")}
        name = COND_LABELS.get(label, label)
        mean  = f"{s['mean']:+.3f}%"   if not np.isnan(s["mean"])    else "N/A"
        hit   = f"{s['hit_rate']:.1f}%" if not np.isnan(s["hit_rate"]) else "N/A"
        shp   = f"{s['sharpe']:.3f}"    if not np.isnan(s["sharpe"])   else "N/A"
        emean = f"{se['mean']:+.3f}%"   if not np.isnan(se["mean"])    else "N/A"
        ehit  = f"{se['hit_rate']:.1f}%" if not np.isnan(se["hit_rate"]) else "N/A"
        eshp  = f"{se['sharpe']:.3f}"    if not np.isnan(se["sharpe"])   else "N/A"
        print(FMT.format(name, int(s["n"]), mean, hit, shp, emean, ehit, eshp))
        rows.append({
            "strategy": name, "n": s["n"],
            "mean_ret": s["mean"], "hit_rate": s["hit_rate"], "sharpe": s["sharpe"],
            "excess_mean": se["mean"], "excess_hit_rate": se["hit_rate"], "excess_sharpe": se["sharpe"],
        })
    print(SEP)
    return pd.DataFrame(rows)


# ── 비교표 출력 ───────────────────────────────────────────

def print_comparison(rows: list[dict]) -> None:
    # ── 테이블 1: 절대수익률 ──────────────────────────────────
    SEP = "─" * 104
    FMT = "{:<20} {:<8} {:>6} {:>10} {:>8} {:>10} {:>10} {:>8} {:>11}"
    print("\n[절대수익률]")
    print(SEP)
    print(FMT.format("전략", "신호", "신호수", "5d수익률", "5d Hit", "Sharpe(5d)", "20d수익률", "20d Hit", "Sharpe(20d)"))
    print(SEP)

    prev_label = None
    for r in rows:
        if prev_label and r["label"] != prev_label and r["signal"] == "Buy":
            print()
        prev_label = r["label"]

        m5   = f"{r['mean_5d']:+.3f}%"     if not np.isnan(r["mean_5d"])     else "N/A"
        h5   = f"{r['hit_rate_5d']:.1f}%"  if not np.isnan(r["hit_rate_5d"]) else "N/A"
        shp5 = f"{r['sharpe_5d']:.3f}"     if not np.isnan(r["sharpe_5d"])   else "N/A"
        m20  = f"{r['mean']:+.3f}%"        if not np.isnan(r["mean"])        else "N/A"
        h20  = f"{r['hit_rate']:.1f}%"     if not np.isnan(r["hit_rate"])    else "N/A"
        shp  = f"{r['sharpe']:.3f}"        if not np.isnan(r["sharpe"])      else "N/A"

        print(FMT.format(r["label"], r["signal"], int(r["n"]), m5, h5, shp5, m20, h20, shp))
        if r["signal"] == "전체":
            print(SEP)

    # ── 테이블 2: 초과수익률 (vs KOSPI/KOSDAQ) ────────────────
    SEP2 = "─" * 80
    FMT2 = "{:<20} {:<8} {:>6} {:>12} {:>10} {:>10} {:>10}"
    print("\n[초과수익률 — vs KOSPI/KOSDAQ]")
    print(SEP2)
    print(FMT2.format("전략", "신호", "신호수", "5d초과수익", "5d초과Hit", "20d초과수익", "20d초과Hit"))
    print(SEP2)

    prev_label = None
    for r in rows:
        if prev_label and r["label"] != prev_label and r["signal"] == "Buy":
            print()
        prev_label = r["label"]

        em5  = f"{r['mean_excess_5d']:+.3f}%"    if not np.isnan(r["mean_excess_5d"])    else "N/A"
        eh5  = f"{r['hit_rate_excess_5d']:.1f}%"  if not np.isnan(r["hit_rate_excess_5d"]) else "N/A"
        em20 = f"{r['mean_excess']:+.3f}%"        if not np.isnan(r["mean_excess"])        else "N/A"
        eh20 = f"{r['hit_rate_excess']:.1f}%"     if not np.isnan(r["hit_rate_excess"])    else "N/A"

        print(FMT2.format(r["label"], r["signal"], int(r["n"]), em5, eh5, em20, eh20))
        if r["signal"] == "전체":
            print(SEP2)


# ── 섹터별 분석 ───────────────────────────────────────────

def ticker_to_sector(ticker: str) -> str:
    for sector, tickers in SECTORS.items():
        if ticker in tickers:
            return sector
    return "기타"


def analysis_sector(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """섹터별 신호 분포 및 성과를 cond 간 비교."""
    print("\n" + "=" * 60)
    print("【섹터별 비교】 (신호별 분리)")
    print("=" * 60)

    conds = list(data.keys())
    rows = []

    for sector, tickers in SECTORS.items():
        for cond, df in data.items():
            sub = df[df["ticker"].isin(tickers)]
            has_excess = "excess_return_20d" in sub.columns
            for sig in ["Buy", "Neutral", "Sell"]:
                sig_sub = sub[sub["signal"] == sig]
                s  = calc_stats(sig_sub["return_20d"],        sig, hold_days=20)
                se = calc_stats(sig_sub["excess_return_20d"], sig, hold_days=20) if has_excess else {"mean": float("nan"), "hit_rate": float("nan"), "sharpe": float("nan")}
                rows.append({
                    "sector": sector, "cond": cond, "signal": sig,
                    "n": s["n"], "mean": s["mean"], "hit_rate": s["hit_rate"], "sharpe": s["sharpe"],
                    "mean_excess": se["mean"], "hit_rate_excess": se["hit_rate"], "sharpe_excess": se["sharpe"],
                })

    result = pd.DataFrame(rows)

    for sector in SECTORS:
        print(f"\n▶ {sector}")
        sec_df = result[result["sector"] == sector]
        for sig in ["Buy", "Neutral", "Sell"]:
            sig_df = sec_df[sec_df["signal"] == sig]
            if sig_df["n"].sum() == 0:
                continue
            print(f"   [{sig}]")
            for cond in conds:
                r = sig_df[sig_df["cond"] == cond]
                if r.empty or r.iloc[0]["n"] == 0:
                    continue
                r = r.iloc[0]
                print(f"     {cond}  n={int(r['n']):>3}  mean={r['mean']:>6.2f}%  hit={r['hit_rate']:>5.1f}%")

            # 인접 cond 간 Buy 신호 차이
            if sig == "Buy" and len(conds) >= 2:
                for i in range(1, len(conds)):
                    c_prev, c_cur = conds[i - 1], conds[i]
                    rp = sig_df[sig_df["cond"] == c_prev]
                    rc = sig_df[sig_df["cond"] == c_cur]
                    if rp.empty or rc.empty:
                        continue
                    rp, rc = rp.iloc[0], rc.iloc[0]
                    if rp["n"] > 0 and rc["n"] > 0:
                        dh = round(rc["hit_rate"] - rp["hit_rate"], 1)
                        dm = round(rc["mean"] - rp["mean"], 2)
                        print(f"     → {c_cur}-{c_prev}  "
                              f"Δhit={'+'if dh>=0 else ''}{dh}%  "
                              f"Δmean={'+'if dm>=0 else ''}{dm}%")
    return result


# ── 종목별 분석 ───────────────────────────────────────────

def analysis_stock(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """종목별 cond 간 Buy 신호 성과 비교.

    Note: 모든 cond가 동일 날짜를 평가하므로 전체 신호 평균은 cond에 무관하게
    항상 같다. Buy 신호만 필터링해야 cond별 의미 있는 차이가 드러난다.
    """
    print("\n" + "=" * 60)
    print("【종목별 비교】 (Buy 신호 기준)")
    print("=" * 60)

    rows = []
    for cond, df in data.items():
        has_excess = "excess_return_20d" in df.columns
        for ticker, grp in df.groupby("ticker"):
            ticker  = str(ticker).zfill(6)
            buy_grp = grp[grp["signal"] == "Buy"]
            s  = calc_stats(buy_grp["return_20d"],        signal="Buy", hold_days=20)
            se = calc_stats(buy_grp["excess_return_20d"], signal="Buy", hold_days=20) if has_excess else {"mean": float("nan"), "hit_rate": float("nan"), "sharpe": float("nan")}
            rows.append({
                "ticker":                  ticker,
                "name":                    grp["name"].iloc[0] if "name" in grp.columns else ticker,
                "market":                  "코스닥" if ticker in KOSDAQ_TICKERS else "코스피",
                "sector":                  ticker_to_sector(ticker),
                "cond":                    cond,
                "signal_filter":           "Buy",
                "n_buy":                   s["n"],
                "buy_mean_20d":            s["mean"],
                "buy_hit_rate_20d":        s["hit_rate"],
                "buy_sharpe_20d":          s["sharpe"],
                "buy_excess_mean_20d":     se["mean"],
                "buy_excess_hit_rate_20d": se["hit_rate"],
                "buy_excess_sharpe_20d":   se["sharpe"],
            })

    result = pd.DataFrame(rows)
    conds = list(data.keys())

    # cond1 Buy mean 기준 정렬 (없으면 첫 cond)
    sort_cond = "cond1" if "cond1" in conds else conds[0]
    tickers_sorted = (
        result[result["cond"] == sort_cond]
        .set_index("ticker")["buy_mean_20d"]
        .sort_values(ascending=False).index.tolist()
    )

    print(f"\n{'종목':<12} {'시장':<6} {'섹터':<8}", end="")
    for cond in conds:
        print(f"  {cond}(n_buy/mean/hit)", end="")
    print()
    print("-" * 100)

    for ticker in tickers_sorted:
        sub = result[result["ticker"] == ticker]
        print(f"{sub['name'].iloc[0]:<12} {sub['market'].iloc[0]:<6} {sub['sector'].iloc[0]:<8}", end="")
        for cond in conds:
            row = sub[sub["cond"] == cond]
            if row.empty or row.iloc[0]["n_buy"] == 0:
                print(f"  {'N/A':>22}", end="")
            else:
                r = row.iloc[0]
                print(f"  {int(r['n_buy']):>2}/{r['buy_mean_20d']:>6.2f}%/{r['buy_hit_rate_20d']:>4.1f}%", end="")
        print()

    if "cond1" in data:
        c1 = result[result["cond"] == "cond1"].sort_values("buy_mean_20d", ascending=False)
        print("\n▶ cond1 Buy 신호 평균 수익률 상위 5개 종목")
        print(c1.head(5)[["name", "market", "sector", "n_buy", "buy_mean_20d", "buy_hit_rate_20d"]].to_string(index=False))
        print("\n▶ cond1 Buy 신호 평균 수익률 하위 5개 종목")
        print(c1.tail(5)[["name", "market", "sector", "n_buy", "buy_mean_20d", "buy_hit_rate_20d"]].to_string(index=False))

    return result


# ── 메인 ─────────────────────────────────────────────────

def run(cond_target: str | None, include_sector: bool, is_all: bool) -> None:
    out_dir    = get_analysis_dir()
    latest_dir = get_latest_analysis_dir()

    if is_all:
        target_conds = COND_ORDER
        save_prefix  = "all"
    else:
        idx          = COND_ORDER.index(cond_target)
        target_conds = COND_ORDER[: idx + 1]
        save_prefix  = cond_target

    print(f"\n비교 대상: {target_conds}")
    print("결과 파일 로드 중...")

    cond_data     = load_cond_data(target_conds)
    baseline_data = load_baselines()

    if not cond_data:
        print("분석할 cond 결과 파일이 없습니다.")
        return

    # ── 신호별 비교표 ──────────────────────────────────────
    print("\n" + "=" * 80)
    print("실험 조건 비교 분석")
    print("=" * 80)

    all_rows = []
    for label, df in baseline_data.items():
        # 베이스라인은 signal="Buy" 일괄 부여 → 전체 행 = Buy 행 (중복 제거)
        all_rows.extend(signal_rows(df, label, has_confidence=False, include_total=False))
    for cond, df in cond_data.items():
        all_rows.extend(signal_rows(df, cond, has_confidence=True))

    print_comparison(all_rows)

    # 인접 cond 간 Buy 신호 차이 요약
    cond_labels = list(cond_data.keys())
    if len(cond_labels) >= 2:
        cmp_df = pd.DataFrame(all_rows)
        print("\n[Buy 신호 조건별 차이]")
        for i in range(1, len(cond_labels)):
            c_prev, c_cur = cond_labels[i - 1], cond_labels[i]
            rp = cmp_df[(cmp_df["label"] == c_prev) & (cmp_df["signal"] == "Buy")]
            rc = cmp_df[(cmp_df["label"] == c_cur)  & (cmp_df["signal"] == "Buy")]
            if rp.empty or rc.empty:
                continue
            d_ret = rc.iloc[0]["mean"]     - rp.iloc[0]["mean"]
            d_hit = rc.iloc[0]["hit_rate"] - rp.iloc[0]["hit_rate"]
            print(f"  {c_cur} - {c_prev}  Δmean={d_ret:+.3f}%  Δhit={d_hit:+.1f}%")

    fname    = f"{save_prefix}_comparison.csv"
    out_path = os.path.join(out_dir, fname)
    pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    shutil.copy(out_path, os.path.join(latest_dir, fname))
    print(f"\n저장: {out_path}")

    # ── 전략 요약표 (--all 시 전략별 한 줄 요약 + full_comparison.csv) ──
    if is_all:
        full_df = print_full_comparison(baseline_data, cond_data)
        full_fname = "full_comparison.csv"
        full_path  = os.path.join(out_dir, full_fname)
        full_df.to_csv(full_path, index=False, encoding="utf-8-sig")
        shutil.copy(full_path, os.path.join(latest_dir, full_fname))
        print(f"저장: {full_path}")

    # ── 섹터·종목 분석 ─────────────────────────────────────
    if include_sector or is_all:
        sector_df = analysis_sector(cond_data)
        stock_df  = analysis_stock(cond_data)

        sector_fname = f"{save_prefix}_sector.csv"
        stock_fname  = f"{save_prefix}_stock_buy.csv"   # Buy 신호 기준 집계임을 명시
        sector_path  = os.path.join(out_dir, sector_fname)
        stock_path   = os.path.join(out_dir, stock_fname)

        sector_df.to_csv(sector_path, index=False, encoding="utf-8-sig")
        stock_df.to_csv(stock_path,  index=False, encoding="utf-8-sig")
        shutil.copy(sector_path, os.path.join(latest_dir, sector_fname))
        shutil.copy(stock_path,  os.path.join(latest_dir, stock_fname))
        print(f"저장: {sector_fname}, {stock_fname} (Buy 신호 기준)")

    # ── 통계적 유의성 검정 ────────────────────────────────
    sig_df = run_significance_tests(cond_data, baseline_data)
    print_significance_tests(sig_df)

    if not sig_df.empty:
        sig_fname = f"{save_prefix}_significance.csv"
        sig_path  = os.path.join(out_dir, sig_fname)
        sig_df.to_csv(sig_path, index=False, encoding="utf-8-sig")
        shutil.copy(sig_path, os.path.join(latest_dir, sig_fname))
        print(f"\n저장: {sig_fname}")

    print("\n분석 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="실험 조건 비교 분석")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--cond", choices=COND_ORDER,
        help="비교 기준 cond (cond1부터 해당 cond까지 포함)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="전체 cond 자동 감지 + 섹터·종목 분석 포함",
    )
    parser.add_argument(
        "--sector", action="store_true",
        help="섹터·종목별 분석 추가 (--cond와 함께 사용)",
    )
    args = parser.parse_args()
    run(
        cond_target    = args.cond if not args.all else None,
        include_sector = args.sector,
        is_all         = args.all,
    )

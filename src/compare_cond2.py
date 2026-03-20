"""
cond1 vs cond2 비교 분석 스크립트

cond1: 종목명 + 현재가만 제공 (No Context)
cond2: 재무지표 + 기술지표 제공 (Financial Context)

비교 지표:
  - 신호 수 및 Buy/Neutral/Sell 분포
  - 20거래일 평균 수익률 (신호별 + 전체)
  - 20거래일 Hit Rate (신호별 + 전체)
  - Sharpe Ratio (Buy 신호 기준)
  - confidence 분포 (mean, std, min, max)

출력:
  - 터미널 비교표
  - results/experiment/cond2/latest/comparison_cond1_cond2.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import EXPERIMENT_DIR

# ── 경로 ──────────────────────────────────────────────────
COND1_PATH = os.path.join(EXPERIMENT_DIR, "cond1", "latest", "cond1_results.csv")
COND2_PATH = os.path.join(EXPERIMENT_DIR, "cond2", "latest", "cond2_results.csv")
OUT_PATH   = os.path.join(EXPERIMENT_DIR, "cond2", "latest", "comparison_cond1_cond2.csv")

HOLD = "return_20d"


def sharpe(returns: pd.Series) -> float:
    """
    연율화 Sharpe Ratio 계산 (무위험수익률 0 가정).

    Parameters
    ----------
    returns : pd.Series
        수익률 시계열 (%)

    Returns
    -------
    float
        Sharpe Ratio. 표준편차가 0이면 NaN.
    """
    if returns.std() == 0 or returns.empty:
        return float("nan")
    # 월별 신호 → 연 12회 거래 가정
    return (returns.mean() / returns.std()) * np.sqrt(12)


def signal_stats(df: pd.DataFrame, label: str) -> list[dict]:
    """
    신호별 + 전체 통계를 딕셔너리 리스트로 반환.

    Parameters
    ----------
    df : pd.DataFrame
        실험 결과 DataFrame (signal, return_20d, confidence 컬럼 필요)
    label : str
        실험 조건 레이블 (예: 'cond1', 'cond2')

    Returns
    -------
    list[dict]
        각 신호(Buy / Neutral / Sell / 전체)별 통계 딕셔너리
    """
    rows = []
    groups = [("Buy", df[df["signal"] == "Buy"]),
              ("Neutral", df[df["signal"] == "Neutral"]),
              ("Sell", df[df["signal"] == "Sell"]),
              ("전체", df)]

    for sig, g in groups:
        ret = g[HOLD].dropna()
        rows.append({
            "cond":      label,
            "signal":    sig,
            "n":         len(g),
            "mean_ret":  round(ret.mean(), 3) if not ret.empty else float("nan"),
            "hit_rate":  round((ret > 0).mean() * 100, 1) if not ret.empty else float("nan"),
            "sharpe":    round(sharpe(ret), 3),
            "conf_mean": round(g["confidence"].mean(), 1) if not g.empty else float("nan"),
            "conf_std":  round(g["confidence"].std(), 1)  if len(g) > 1 else float("nan"),
            "conf_min":  int(g["confidence"].min())        if not g.empty else float("nan"),
            "conf_max":  int(g["confidence"].max())        if not g.empty else float("nan"),
        })
    return rows


def print_table(df: pd.DataFrame) -> None:
    """비교표를 터미널에 출력."""
    SEP  = "─" * 90
    FMT  = "{:<8} {:<8} {:>6} {:>10} {:>10} {:>8} {:>10} {:>9} {:>7} {:>7}"
    HEAD = FMT.format(
        "조건", "신호", "건수",
        "평균수익률", "Hit Rate", "Sharpe",
        "conf평균", "conf_std", "최솟값", "최댓값"
    )

    print(SEP)
    print(HEAD)
    print(SEP)

    for _, r in df.iterrows():
        hit  = f"{r['hit_rate']:.1f}%" if not np.isnan(r['hit_rate']) else "N/A"
        shp  = f"{r['sharpe']:.3f}"   if not np.isnan(r['sharpe'])   else "N/A"
        mean = f"{r['mean_ret']:+.3f}%" if not np.isnan(r['mean_ret']) else "N/A"
        smin = int(r["conf_min"]) if not np.isnan(r["conf_min"]) else "N/A"
        smax = int(r["conf_max"]) if not np.isnan(r["conf_max"]) else "N/A"
        sstd = f"{r['conf_std']:.1f}" if not np.isnan(r["conf_std"]) else "N/A"
        smean = f"{r['conf_mean']:.1f}" if not np.isnan(r["conf_mean"]) else "N/A"
        print(FMT.format(
            r["cond"], r["signal"], int(r["n"]),
            mean, hit, shp,
            smean, sstd, smin, smax,
        ))
        if r["signal"] == "전체":
            print(SEP)

    # ── Buy 신호 기준 cond1 vs cond2 차이 강조 ───────────
    c1_buy = df[(df["cond"] == "cond1") & (df["signal"] == "Buy")].iloc[0]
    c2_buy = df[(df["cond"] == "cond2") & (df["signal"] == "Buy")].iloc[0]

    diff_ret = c2_buy["mean_ret"] - c1_buy["mean_ret"]
    diff_hit = c2_buy["hit_rate"] - c1_buy["hit_rate"]

    print("\n[Buy 신호 cond2 - cond1 차이]")
    print(f"  평균 수익률 : {diff_ret:+.3f}%p")
    print(f"  Hit Rate   : {diff_hit:+.1f}%p")
    print(f"  Sharpe     : {c1_buy['sharpe']:.3f} → {c2_buy['sharpe']:.3f}")


def run():
    """cond1, cond2 결과 로드 후 비교 분석 및 저장."""
    c1 = pd.read_csv(COND1_PATH, dtype={"ticker": str})
    c2 = pd.read_csv(COND2_PATH, dtype={"ticker": str})

    rows = signal_stats(c1, "cond1") + signal_stats(c2, "cond2")
    cmp_df = pd.DataFrame(rows)

    print("\n========== cond1 vs cond2 비교 분석 ==========\n")
    print("cond1: 종목명 + 현재가만 제공 (No Context)")
    print("cond2: 재무지표 + 기술지표 제공 (Financial Context)")
    print()
    print_table(cmp_df)

    cmp_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    run()

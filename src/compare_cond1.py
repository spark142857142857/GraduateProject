"""
cond1 vs 베이스라인 비교 분석
- 컨센서스, 골든크로스, cond1(LLM No Context)
- 신호 수, Buy/Neutral/Sell 분포, Hit Rate, 평균 수익률, Sharpe Ratio
- 결과: results/experiment/cond1/latest/comparison.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import get_price, calc_return, get_latest_baseline_dir, get_latest_experiment_dir

# utils 함수로 절대경로 생성 (하드코딩 제거)
BASELINE_DIR = get_latest_baseline_dir()
COND1_DIR    = get_latest_experiment_dir("cond1")
HOLD_SHORT   = 5
HOLD_LONG    = 20


# ── 5거래일 수익률 추가 ────────────────────────────────────
def add_return_5d(df: pd.DataFrame, ticker_col="ticker", date_col="signal_date") -> pd.DataFrame:
    """베이스라인 DataFrame에 return_5d 컬럼을 추가하여 반환."""
    price_cache: dict[str, pd.DataFrame] = {}
    r5 = []
    for _, row in df.iterrows():
        tk = str(row[ticker_col]).zfill(6)
        if tk not in price_cache:
            price_cache[tk] = get_price(tk)
        r5.append(calc_return(price_cache[tk], str(row[date_col]), HOLD_SHORT))
    df = df.copy()
    df["return_5d"] = r5
    return df


# ── 단일 시리즈 지표 계산 ──────────────────────────────────
def calc_metrics(series: pd.Series) -> dict:
    """수익률 시리즈에서 n, 평균수익률, Hit Rate, Sharpe Ratio 계산."""
    s = series.dropna()
    n = len(s)
    if n == 0:
        return {"n": 0, "mean": np.nan, "hit_rate": np.nan, "sharpe": np.nan}
    mean   = float(s.mean())
    std    = float(s.std(ddof=1))
    hit    = float((s > 0).mean() * 100)
    sharpe = mean / std if std > 0 else np.nan
    return {"n": n, "mean": round(mean, 4), "hit_rate": round(hit, 2),
            "sharpe": round(sharpe, 4) if not np.isnan(sharpe) else np.nan}


# ── 전략별 행 구성 ─────────────────────────────────────────
def make_rows(strategy: str, df: pd.DataFrame, signal_col: str | None = None) -> list[dict]:
    """전략명·신호 구분·보유기간별 지표 행 목록 반환. signal_col 없으면 전체 Buy로 간주."""
    rows = []

    # 신호 분포
    if signal_col:
        dist = df[signal_col].value_counts().to_dict()
    else:
        dist = {"Buy": len(df)}
    dist_str = " / ".join(
        f"{s}:{dist.get(s,0)}" for s in ["Buy", "Neutral", "Sell"] if dist.get(s, 0) > 0
    )

    signal_groups = [("All", df)]
    if signal_col:
        for sig in ["Buy", "Neutral", "Sell"]:
            if sig in dist:
                signal_groups.append((sig, df[df[signal_col] == sig]))

    for period, col in [("5d", "return_5d"), ("20d", "return_20d")]:
        if col not in df.columns:
            continue
        for sig_label, sub in signal_groups:
            m = calc_metrics(sub[col])
            rows.append({
                "strategy":    strategy,
                "signal_type": sig_label,
                "period":      period,
                "distribution": dist_str,
                "n_signals":   m["n"],
                "mean_ret":    m["mean"],
                "hit_rate":    m["hit_rate"],
                "sharpe":      m["sharpe"],
            })

    return rows


# ── 메인 ──────────────────────────────────────────────────
def run():
    # 1. 데이터 로드
    con = pd.read_csv(os.path.join(BASELINE_DIR, "consensus_returns.csv"), dtype={"ticker": str})
    gol = pd.read_csv(os.path.join(BASELINE_DIR, "golden_returns.csv"),    dtype={"ticker": str})
    c1  = pd.read_csv(os.path.join(COND1_DIR,    "cond1_results.csv"),     dtype={"ticker": str})

    # 2. 베이스라인: 5거래일 수익률 추가 + return_pct → return_20d
    print("베이스라인 5거래일 수익률 계산 중 (컨센서스)...")
    con = add_return_5d(con)
    con = con.rename(columns={"return_pct": "return_20d"})

    print("베이스라인 5거래일 수익률 계산 중 (골든크로스)...")
    gol = add_return_5d(gol)
    gol = gol.rename(columns={"return_pct": "return_20d"})

    # 3. 결측 제거
    con = con.dropna(subset=["return_20d"]).reset_index(drop=True)
    gol = gol.dropna(subset=["return_20d"]).reset_index(drop=True)
    c1  = c1.dropna(subset=["return_20d"]).reset_index(drop=True)

    # 4. 행 구성
    all_rows  = make_rows("Consensus",    con)
    all_rows += make_rows("GoldenCross",  gol)
    all_rows += make_rows("Cond1_LLM",   c1, signal_col="signal")

    result = pd.DataFrame(all_rows,
                          columns=["strategy", "signal_type", "period", "distribution",
                                   "n_signals", "mean_ret", "hit_rate", "sharpe"])

    # 5. 출력
    print("\n" + "="*70)
    print("전략 비교 - 20거래일 / 전체 신호(All)")
    print("="*70)
    summary_20 = result[(result["signal_type"] == "All") & (result["period"] == "20d")]
    print(summary_20.to_string(index=False))

    print("\n" + "="*70)
    print("전략 비교 - 5거래일 / 전체 신호(All)")
    print("="*70)
    summary_5 = result[(result["signal_type"] == "All") & (result["period"] == "5d")]
    print(summary_5.to_string(index=False))

    print("\n" + "="*70)
    print("Cond1(LLM) 신호별 상세")
    print("="*70)
    detail = result[result["strategy"] == "Cond1_LLM"]
    print(detail.to_string(index=False))

    # 6. 저장
    out_path = os.path.join(COND1_DIR, "comparison.csv")
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {out_path}")
    return result


if __name__ == "__main__":
    run()

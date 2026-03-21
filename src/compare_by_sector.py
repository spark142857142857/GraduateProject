"""
섹터별·종목별 실험 조건 비교 분석 스크립트

입력:
  - results/experiment/cond1/latest/cond1_results.csv
  - results/experiment/cond2/latest/cond2_results.csv
  - (cond3, cond4 추가 시 자동 인식)

출력 (터미널 + CSV):
  - results/analysis/overall_comparison.csv   : 전체 합산 비교
  - results/analysis/sector_comparison.csv    : 섹터별 비교
  - results/analysis/stock_comparison.csv     : 종목별 비교

실행: python src/compare_by_sector.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR, RESULTS_DIR

# ── 설정 ─────────────────────────────────────────────────
SECTORS = {
    "반도체":   ["005930", "000660"],
    "바이오":   ["207940", "068270", "196170"],
    "배터리":   ["373220", "006400", "247540"],
    "자동차":   ["005380", "000270"],
    "금융":     ["105560", "055550"],
    "IT플랫폼": ["035720", "035420"],
    "기타":     ["051910", "352820", "329180", "012450", "259960", "034020"],
}

KOSDAQ_TICKERS = {"247540", "196170"}  # 코스닥 종목

CONDS = ["cond1", "cond2", "cond3", "cond4"]  # 파일 없으면 자동 스킵

ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ── 유틸 ─────────────────────────────────────────────────
def sharpe(series: pd.Series) -> float:
    """월간 수익률 시리즈 → 연환산 Sharpe."""
    s = series.dropna()
    if len(s) < 2 or s.std() == 0:
        return float("nan")
    return round((s.mean() / s.std()) * np.sqrt(12), 3)


def hit_rate(series: pd.Series) -> float:
    s = series.dropna()
    if s.empty:
        return float("nan")
    return round((s > 0).sum() / len(s) * 100, 1)


def stats(df: pd.DataFrame) -> dict:
    """데이터프레임(ret 컬럼 필수) 기본 통계."""
    r = df["ret"].dropna()
    return {
        "n":        len(r),
        "mean":     round(r.mean(), 2) if not r.empty else float("nan"),
        "hit_rate": hit_rate(r),
        "sharpe":   sharpe(r),
    }


def load_results() -> dict[str, pd.DataFrame]:
    """존재하는 cond 결과 파일만 로드. {cond: DataFrame}"""
    loaded = {}
    for cond in CONDS:
        path = os.path.join(
            RESULTS_DIR, "experiment", cond, "latest", f"{cond}_results.csv"
        )
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype={"ticker": str})
        df["ticker"] = df["ticker"].astype(str).str.zfill(6)
        df["cond"] = cond
        loaded[cond] = df
        print(f"  로드: {cond} ({len(df)}행)")
    return loaded


def ticker_to_sector(ticker: str) -> str:
    for sector, tickers in SECTORS.items():
        if ticker in tickers:
            return sector
    return "기타"


# ── 분석 1: 전체 합산 비교 ───────────────────────────────
def analysis_overall(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("【분석 1】 전체 합산 비교")
    print("=" * 60)

    rows = []
    for cond, df in data.items():
        total = stats(df)
        row = {
            "cond":     cond,
            "total_n":  total["n"],
            "mean":     total["mean"],
            "hit_rate": total["hit_rate"],
            "sharpe":   total["sharpe"],
        }
        # 신호별 분포·성과
        for sig in ["Buy", "Neutral", "Sell"]:
            sub = df[df["signal"] == sig]
            s = stats(sub)
            row[f"{sig}_n"]        = len(sub)
            row[f"{sig}_mean"]     = s["mean"]
            row[f"{sig}_hit_rate"] = s["hit_rate"]
        rows.append(row)

    result = pd.DataFrame(rows)

    # 출력
    for _, r in result.iterrows():
        print(f"\n▶ {r['cond']}  (총 {r['total_n']}행)")
        print(f"   전체  mean={r['mean']:>6.2f}%  hit={r['hit_rate']:>5.1f}%  sharpe={r['sharpe']:.3f}")
        for sig in ["Buy", "Neutral", "Sell"]:
            n = int(r[f"{sig}_n"])
            if n == 0:
                continue
            print(
                f"   {sig:<7} n={n:>3}  mean={r[f'{sig}_mean']:>6.2f}%  "
                f"hit={r[f'{sig}_hit_rate']:>5.1f}%"
            )
    return result


# ── 분석 2: 섹터별 비교 ──────────────────────────────────
def analysis_sector(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """섹터별 신호 분포 및 신호별 성과를 cond 간 비교.

    전체 합산은 동일 날짜×종목을 공유하므로 return_20d 평균이 동일해진다.
    → 신호별(Buy/Neutral/Sell)로 분리해 실질적 차이를 확인.
    """
    print("\n" + "=" * 60)
    print("【분석 2】 섹터별 비교 (신호별 분리)")
    print("=" * 60)

    conds = list(data.keys())
    rows = []

    for sector, tickers in SECTORS.items():
        for cond, df in data.items():
            sub = df[df["ticker"].isin(tickers)]
            for sig in ["Buy", "Neutral", "Sell"]:
                sig_sub = sub[sub["signal"] == sig]
                s = stats(sig_sub)
                rows.append({
                    "sector":   sector,
                    "cond":     cond,
                    "signal":   sig,
                    "n":        len(sig_sub),
                    "mean":     s["mean"],
                    "hit_rate": s["hit_rate"],
                    "sharpe":   s["sharpe"],
                })

    result = pd.DataFrame(rows)

    # 출력: 섹터별 → 신호별 → cond 나란히
    for sector in SECTORS:
        print(f"\n▶ {sector}")
        sec_df = result[result["sector"] == sector]

        for sig in ["Buy", "Neutral", "Sell"]:
            sig_df = sec_df[sec_df["signal"] == sig]
            # 해당 신호를 낸 cond가 하나도 없으면 스킵
            if sig_df["n"].sum() == 0:
                continue

            print(f"   [{sig}]")
            for cond in conds:
                r = sig_df[sig_df["cond"] == cond]
                if r.empty or r.iloc[0]["n"] == 0:
                    continue
                r = r.iloc[0]
                print(
                    f"     {cond}  n={int(r['n']):>3}  "
                    f"mean={r['mean']:>6.2f}%  hit={r['hit_rate']:>5.1f}%"
                )

            # cond2 - cond1 Buy 신호 차이만 표시
            if sig == "Buy" and "cond1" in conds and "cond2" in conds:
                c1r = sig_df[sig_df["cond"] == "cond1"]
                c2r = sig_df[sig_df["cond"] == "cond2"]
                if not c1r.empty and not c2r.empty:
                    c1r, c2r = c1r.iloc[0], c2r.iloc[0]
                    if c1r["n"] > 0 and c2r["n"] > 0:
                        d_hit  = round(c2r["hit_rate"] - c1r["hit_rate"], 1)
                        d_mean = round(c2r["mean"]     - c1r["mean"],     2)
                        sh = "+" if d_hit  >= 0 else ""
                        sm = "+" if d_mean >= 0 else ""
                        print(f"     → cond2-cond1  Δhit={sh}{d_hit}%  Δmean={sm}{d_mean}%")

    return result


# ── 분석 3: 종목별 비교 ──────────────────────────────────
def analysis_stock(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("【분석 3】 종목별 비교")
    print("=" * 60)

    rows = []
    for cond, df in data.items():
        for ticker, grp in df.groupby("ticker"):
            ticker = str(ticker).zfill(6)   # zero-padding 보장
            s = stats(grp)
            name   = grp["name"].iloc[0] if "name" in grp.columns else ticker
            market = "코스닥" if ticker in KOSDAQ_TICKERS else "코스피"
            sector = ticker_to_sector(ticker)
            rows.append({
                "ticker":   ticker,
                "name":     name,
                "market":   market,
                "sector":   sector,
                "cond":     cond,
                "n":        s["n"],
                "mean":     s["mean"],
                "hit_rate": s["hit_rate"],
                "sharpe":   s["sharpe"],
            })

    result = pd.DataFrame(rows)

    # 종목별 출력
    tickers_sorted = (
        result.groupby("ticker")["mean"].mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    print(f"\n{'종목':<12} {'시장':<6} {'섹터':<8}", end="")
    for cond in data:
        print(f"  {cond}(mean/hit)", end="")
    print()
    print("-" * 70)

    for ticker in tickers_sorted:
        sub = result[result["ticker"] == ticker]
        name   = sub["name"].iloc[0]
        market = sub["market"].iloc[0]
        sector = sub["sector"].iloc[0]
        print(f"{name:<12} {market:<6} {sector:<8}", end="")
        for cond in data:
            row = sub[sub["cond"] == cond]
            if row.empty:
                print(f"  {'N/A':>16}", end="")
            else:
                r = row.iloc[0]
                print(f"  {r['mean']:>5.2f}% / {r['hit_rate']:>4.1f}%", end="")
        print()

    # 상위·하위 5개 (cond1 기준 mean)
    if "cond1" in data:
        c1 = result[result["cond"] == "cond1"].sort_values("mean", ascending=False)
        print("\n▶ cond1 기준 평균 수익률 상위 5개 종목")
        print(c1.head(5)[["name", "market", "sector", "mean", "hit_rate"]].to_string(index=False))
        print("\n▶ cond1 기준 평균 수익률 하위 5개 종목")
        print(c1.tail(5)[["name", "market", "sector", "mean", "hit_rate"]].to_string(index=False))

    return result


# ── 메인 ─────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("섹터별·종목별 실험 조건 비교 분석")
    print("=" * 60)

    print("\n결과 파일 로드 중...")
    data = load_results()
    if not data:
        print("분석할 결과 파일이 없습니다.")
        return

    # 수익률 컬럼 통일 → ret
    for cond, df in data.items():
        if "ret" not in df.columns:
            for alt in ["return_20d", "return_pct", "return", "수익률"]:
                if alt in df.columns:
                    df.rename(columns={alt: "ret"}, inplace=True)
                    break
        data[cond] = df

    # 분석 실행
    overall = analysis_overall(data)
    sector  = analysis_sector(data)
    stock   = analysis_stock(data)

    # CSV 저장
    overall.to_csv(os.path.join(ANALYSIS_DIR, "overall_comparison.csv"), index=False, encoding="utf-8-sig")
    sector.to_csv( os.path.join(ANALYSIS_DIR, "sector_comparison.csv"),  index=False, encoding="utf-8-sig")
    stock.to_csv(  os.path.join(ANALYSIS_DIR, "stock_comparison.csv"),   index=False, encoding="utf-8-sig")

    print("\n" + "=" * 60)
    print(f"분석 완료 → results/analysis/ 저장")
    print(f"  overall_comparison.csv")
    print(f"  sector_comparison.csv")
    print(f"  stock_comparison.csv")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

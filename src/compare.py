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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import EXPERIMENT_DIR, get_latest_baseline_dir, get_analysis_dir, get_latest_analysis_dir
from experiments import EXPERIMENTS

# --cond 옵션에서 사용할 순서 기준 목록 (experiments.py에서 자동 파생)
COND_ORDER = list(EXPERIMENTS.keys())

# 전략별 표시 레이블
COND_LABELS = {
    "Consensus":    "컨센서스",
    "GoldenCross":  "골든크로스",
    "cond1":        "cond1 (종목명)",
    "cond2":        "cond2 (재무지표)",
    "cond3":        "cond3 (재무+리포트)",
    "cond4":        "cond4 (재무+리포트+DART)",
    "reports_only": "리포트 단독",
    "dart_only":    "DART 단독",
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
KOSDAQ_TICKERS = {"247540", "196170"}


# ── 통계 헬퍼 ─────────────────────────────────────────────

def sharpe(series: pd.Series) -> float:
    """월간 수익률 → 연환산 Sharpe (무위험수익률 0 가정)."""
    s = series.dropna()
    if len(s) < 2 or s.std() == 0:
        return float("nan")
    return round((s.mean() / s.std()) * np.sqrt(12), 3)


def calc_stats(series: pd.Series, signal: str = "Buy") -> dict:
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
        "sharpe":   sharpe(s),
    }


# ── 데이터 로드 ───────────────────────────────────────────

def _normalize_ret(df: pd.DataFrame) -> pd.DataFrame:
    """return_20d 컬럼 이름 통일."""
    if "return_20d" not in df.columns:
        for alt in ["return_pct", "return", "수익률"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "return_20d"})
                break
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

def signal_rows(df: pd.DataFrame, label: str, has_confidence: bool = True) -> list[dict]:
    """신호별(Buy/Neutral/Sell) + 전체 통계 딕셔너리 리스트 반환."""
    rows = []
    has_5d = "return_5d" in df.columns
    groups = [
        ("Buy",     df[df["signal"] == "Buy"]),
        ("Neutral", df[df["signal"] == "Neutral"]),
        ("Sell",    df[df["signal"] == "Sell"]),
        ("전체",    df),
    ]
    for sig, g in groups:
        if sig != "전체" and len(g) == 0:
            continue
        s20 = calc_stats(g["return_20d"], sig)
        s5  = calc_stats(g["return_5d"], sig) if has_5d else {
            "mean": float("nan"), "hit_rate": float("nan"), "sharpe": float("nan"),
        }
        row = {
            "label": label, "signal": sig,
            "n": s20["n"],
            "mean_5d": s5["mean"], "hit_rate_5d": s5["hit_rate"], "sharpe_5d": s5["sharpe"],
            "mean": s20["mean"], "hit_rate": s20["hit_rate"], "sharpe": s20["sharpe"],
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
    SEP = "─" * 72
    FMT = "{:<22} {:>6} {:>12} {:>10} {:>8}"
    print("\n" + SEP)
    print(FMT.format("전략", "신호수", "평균수익률", "Hit Rate", "Sharpe"))
    print(SEP)

    rows = []
    for label, df in {**baseline_data, **cond_data}.items():
        s    = calc_stats(df["return_20d"])
        name = COND_LABELS.get(label, label)
        mean = f"{s['mean']:+.3f}%" if not np.isnan(s["mean"])     else "N/A"
        hit  = f"{s['hit_rate']:.1f}%" if not np.isnan(s["hit_rate"]) else "N/A"
        shp  = f"{s['sharpe']:.3f}"    if not np.isnan(s["sharpe"])   else "N/A"
        print(FMT.format(name, int(s["n"]), mean, hit, shp))
        rows.append({"strategy": name, "n": s["n"], "mean_ret": s["mean"],
                     "hit_rate": s["hit_rate"], "sharpe": s["sharpe"]})
    print(SEP)
    return pd.DataFrame(rows)


# ── 비교표 출력 ───────────────────────────────────────────

def print_comparison(rows: list[dict]) -> None:
    SEP = "─" * 104
    FMT = "{:<20} {:<8} {:>6} {:>10} {:>8} {:>10} {:>10} {:>8} {:>11}"
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
            for sig in ["Buy", "Neutral", "Sell"]:
                sig_sub = sub[sub["signal"] == sig]
                s = calc_stats(sig_sub["return_20d"], sig)
                rows.append({
                    "sector": sector, "cond": cond, "signal": sig,
                    "n": s["n"], "mean": s["mean"],
                    "hit_rate": s["hit_rate"], "sharpe": s["sharpe"],
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
        for ticker, grp in df.groupby("ticker"):
            ticker = str(ticker).zfill(6)
            buy_grp = grp[grp["signal"] == "Buy"]
            s = calc_stats(buy_grp["return_20d"], signal="Buy")
            rows.append({
                "ticker":          ticker,
                "name":            grp["name"].iloc[0] if "name" in grp.columns else ticker,
                "market":          "코스닥" if ticker in KOSDAQ_TICKERS else "코스피",
                "sector":          ticker_to_sector(ticker),
                "cond":            cond,
                "signal_filter":   "Buy",          # 집계 기준 명시
                "n_buy":           s["n"],
                "buy_mean_20d":    s["mean"],
                "buy_hit_rate_20d": s["hit_rate"],
                "buy_sharpe_20d":  s["sharpe"],
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
        all_rows.extend(signal_rows(df, label, has_confidence=False))
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

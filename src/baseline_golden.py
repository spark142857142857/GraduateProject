"""
대조군 B: 골든크로스 전략 (MA5 × MA20)
- 5일 이동평균이 20일 이동평균을 상향 돌파하는 시점에 매수 신호
- 신호 발생 후 hold_days 수익률 측정
- 결과: results/golden_returns.csv
"""

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import TICKERS, get_price, calc_return, ensure_dirs, get_baseline_dir, get_latest_baseline_dir

# ── 파라미터 ──────────────────────────────────────────────
MA_SHORT  = 5
MA_LONG   = 20
HOLD_DAYS = 20


def detect_golden_cross(df: pd.DataFrame) -> list[str]:
    """
    골든크로스 발생일 목록 반환.
    골든크로스: MA_SHORT가 MA_LONG을 하향에서 상향 돌파한 날 (전일 below → 당일 above).
    """
    df = df.copy()
    df["ma_short"] = df["Close"].rolling(MA_SHORT).mean()
    df["ma_long"]  = df["Close"].rolling(MA_LONG).mean()
    df["above"]    = df["ma_short"] > df["ma_long"]
    df["cross"]    = df["above"] & ~df["above"].shift(1).fillna(False)

    cross_dates = df.index[df["cross"]].strftime("%Y-%m-%d").tolist()
    return cross_dates


def run():
    ensure_dirs()
    all_results = []

    for name, ticker in TICKERS.items():
        price_df = get_price(ticker)
        if price_df.empty:
            print(f"[golden] {name}: 주가 없음, 스킵")
            continue

        cross_dates = detect_golden_cross(price_df)

        for sig_date in cross_dates:
            ret = calc_return(price_df, sig_date, HOLD_DAYS)
            if ret is None:
                continue

            future = price_df.loc[price_df.index >= sig_date]
            cur_price = future["Close"].iloc[0]

            all_results.append({
                "ticker":      ticker,
                "name":        name,
                "signal_date": sig_date,
                "cur_price":   cur_price,
                "return_pct":  round(ret, 2),
            })

        n = len([r for r in all_results if r["ticker"] == ticker])
        print(f"[golden] {name}: {n} 신호")

    result_df = pd.DataFrame(all_results)
    out_path  = os.path.join(get_baseline_dir(), "golden_returns.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    shutil.copy(out_path, os.path.join(get_latest_baseline_dir(), "golden_returns.csv"))
    print(f"\n저장 완료: {out_path}")
    print(result_df.describe())
    return result_df


def plot_signals(ticker: str, name: str):
    """특정 종목의 골든크로스 시각화"""
    price_df = get_price(ticker)
    if price_df.empty:
        return

    price_df["ma_short"] = price_df["Close"].rolling(MA_SHORT).mean()
    price_df["ma_long"]  = price_df["Close"].rolling(MA_LONG).mean()
    cross_dates = detect_golden_cross(price_df)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(price_df.index, price_df["Close"],    label="Close",  linewidth=1)
    ax.plot(price_df.index, price_df["ma_short"], label=f"MA{MA_SHORT}", linewidth=1)
    ax.plot(price_df.index, price_df["ma_long"],  label=f"MA{MA_LONG}",  linewidth=1)

    for d in cross_dates:
        ax.axvline(pd.Timestamp(d), color="red", alpha=0.4, linewidth=0.8)

    ax.set_title(f"{name} ({ticker}) Golden Cross")
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(get_baseline_dir(), f"golden_{ticker}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"차트 저장: {out_path}")


if __name__ == "__main__":
    run()

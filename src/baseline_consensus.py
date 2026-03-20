"""
대조군 A: 컨센서스 추종 전략
- 직전 N개 리포트의 평균 목표주가 vs 현재가 괴리율로 매수/매도 신호 생성
- 신호 발생 후 hold_days 수익률 측정
- 결과: results/consensus_returns.csv
"""

import os
import shutil
import pandas as pd
import numpy as np
from utils import TICKERS, get_price, calc_return, ensure_dirs, get_baseline_dir, get_latest_baseline_dir, load_analyst as _load_analyst

# ── 파라미터 ──────────────────────────────────────────────
N_REPORTS  = 3      # 컨센서스 산출에 사용할 최근 리포트 수
BUY_GAP    = 10.0   # 목표주가가 현재가보다 X% 이상 높으면 매수 신호 (%)
HOLD_DAYS  = 20     # 보유 기간 (거래일)


def load_analyst(ticker: str) -> pd.DataFrame | None:
    """utils.load_analyst 위임 (require_target_price=True: target_price NaN 행 제거)."""
    return _load_analyst(ticker, require_target_price=True)


def run():
    ensure_dirs()
    all_results = []

    for name, ticker in TICKERS.items():
        analyst = load_analyst(ticker)
        if analyst is None or analyst.empty:
            print(f"[consensus] {name}: 리포트 없음, 스킵")
            continue

        price_df = get_price(ticker)
        if price_df.empty:
            print(f"[consensus] {name}: 주가 없음, 스킵")
            continue

        for i in range(N_REPORTS - 1, len(analyst)):
            window   = analyst.iloc[i - N_REPORTS + 1 : i + 1]
            avg_tp   = window["target_price"].mean()
            sig_date = str(analyst.iloc[i]["date"].date())

            # 신호일 현재가
            future = price_df.loc[price_df.index > sig_date]
            if future.empty:
                continue
            cur_price = future["Close"].iloc[0]

            gap = (avg_tp - cur_price) / cur_price * 100
            # avg_tp가 NaN이면 gap도 NaN → 명시적으로 스킵 (NaN < BUY_GAP은 False라 통과되는 버그 방지)
            if pd.isna(gap) or gap < BUY_GAP:
                continue  # 매수 신호 없음

            ret = calc_return(price_df, sig_date, HOLD_DAYS)
            if ret is None:
                continue

            all_results.append({
                "ticker":       ticker,
                "name":         name,
                "signal_date":  sig_date,
                "avg_target":   round(avg_tp, 0),
                "cur_price":    cur_price,
                "gap_pct":      round(gap, 2),
                "return_pct":   round(ret, 2),
            })

        print(f"[consensus] {name}: {len([r for r in all_results if r['ticker']==ticker])} 신호")

    result_df = pd.DataFrame(all_results)
    out_path  = os.path.join(get_baseline_dir(), "consensus_returns.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    shutil.copy(out_path, os.path.join(get_latest_baseline_dir(), "consensus_returns.csv"))
    print(f"\n저장 완료: {out_path}")
    print(result_df.describe())
    return result_df


if __name__ == "__main__":
    run()

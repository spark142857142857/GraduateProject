"""
Forward Test 성숙 신호 평가

results/forward/{생성일}/{model}/{ticker}_{cond}.json 신호들을 스캔해,
신호 생성일 이후 실제 5/20거래일 수익률을 계산하고 적중 여부를 집계한다.
20거래일이 아직 안 지난 신호(미성숙)는 pending으로 분류.

일관성:
  - 진입 = 신호일 다음 첫 거래일 종가 (백테스트와 동일, calc_return 재사용)
  - 주말 생성 신호도 자동 처리 (calc_return이 신호일 이후 첫 거래일을 진입으로 잡음)
  - 적중: Buy는 수익률>0, Sell은 <0. Neutral은 방향성 없어 적중 평가 제외
  - 초과수익: 시장(KOSPI/KOSDAQ) 대비

주의: forward는 주간 반복 시 20일 보유구간이 겹쳐 표본이 독립이 아님 →
      실전 참고·추세 확인용. 유의성 검정은 백테스트(significance.py)가 담당.

사용법: python src/experiment/forward_eval.py
저장:   results/forward/evaluation.csv (전 신호 스냅샷) + 콘솔 요약
"""

import os
import sys
import json
import glob

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import FORWARD_DIR, get_price, calc_return, get_benchmark_price, calc_excess_return

HOLD_SHORT = 5
HOLD_LONG  = 20
DEFAULT_MODEL = "gemini-2.5-flash-lite"  # model 필드 없는 구 신호(앵커) 기본값


def _hit(ret: float | None, signal: str) -> bool | None:
    """Buy는 수익률>0, Sell은 <0이 적중. Neutral/미성숙은 None."""
    if ret is None or signal == "Neutral":
        return None
    return ret > 0 if signal == "Buy" else ret < 0


def evaluate() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(FORWARD_DIR, "*", "*", "*.json")))
    if not files:
        print("평가할 forward 신호가 없습니다. (results/forward/{날짜}/ 비어있음)")
        return pd.DataFrame()

    price_cache: dict[str, pd.DataFrame] = {}
    bench_cache: dict[str, pd.DataFrame] = {}
    rows = []

    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        ticker   = str(d["ticker"]).zfill(6)
        gen_date = d["date"]              # 신호 생성일 (주말이면 주말 날짜)
        signal   = d["signal"]

        if ticker not in price_cache:
            price_cache[ticker] = get_price(ticker, start="2022-12-01")
            bench_cache[ticker] = get_benchmark_price(ticker, start="2022-12-01")
        px, bench = price_cache[ticker], bench_cache[ticker]

        r5  = calc_return(px, gen_date, HOLD_SHORT)
        r20 = calc_return(px, gen_date, HOLD_LONG)
        e5  = calc_excess_return(px, bench, gen_date, HOLD_SHORT)
        e20 = calc_excess_return(px, bench, gen_date, HOLD_LONG)

        rows.append({
            "gen_date":   gen_date,
            "ticker":     ticker,
            "name":       d.get("name", ""),
            "model":      d.get("model", DEFAULT_MODEL),
            "cond":       d.get("cond", ""),
            "signal":     signal,
            "confidence": d.get("confidence"),
            "price":      d.get("price"),
            "return_5d":       round(r5, 2)  if r5  is not None else None,
            "return_20d":      round(r20, 2) if r20 is not None else None,
            "excess_5d":       round(e5, 2)  if e5  is not None else None,
            "excess_20d":      round(e20, 2) if e20 is not None else None,
            "status":          "matured" if r20 is not None else "pending",
            "hit_20d":         _hit(r20, signal),
            "hit_excess_20d":  _hit(e20, signal),
        })

    return pd.DataFrame(rows)


def _summarize(df: pd.DataFrame) -> None:
    total   = len(df)
    matured = df[df["status"] == "matured"]
    pending = total - len(matured)
    print("\n" + "=" * 68)
    print(f"Forward Test 평가  |  전체 {total}건  (성숙 {len(matured)} / 미성숙 {pending})")
    print("=" * 68)

    if matured.empty:
        print("아직 성숙된 신호(20거래일 경과)가 없습니다.")
        return

    def _block(title: str, group_col: str) -> None:
        print(f"\n▶ {title}")
        for key, g in matured.groupby(group_col):
            n = len(g)
            buy  = g[g["signal"] == "Buy"]
            sell = g[g["signal"] == "Sell"]
            neu  = g[g["signal"] == "Neutral"]
            # 적중률: Buy/Sell 방향 적중 (Neutral 제외). object dtype이라 bool로 캐스팅 후 평균
            directional = g[g["hit_20d"].notna()]
            hit = directional["hit_20d"].astype(bool).mean() * 100 if len(directional) else float("nan")
            exc = directional["hit_excess_20d"].astype(bool).mean() * 100 if len(directional) else float("nan")
            mret = g["return_20d"].mean()
            print(f"  {str(key):<22} n={n:>3}  "
                  f"Buy/Sell/Neu={len(buy)}/{len(sell)}/{len(neu)}  "
                  f"방향적중={hit:>5.1f}%  초과적중={exc:>5.1f}%  평균20d={mret:>+6.2f}%")

    _block("모델별", "model")
    _block("조건별", "cond")
    _block("신호별", "signal")

    if pending:
        print(f"\n※ 미성숙 {pending}건은 20거래일 경과 후 재실행 시 자동 평가됩니다.")


def main() -> None:
    df = evaluate()
    if df.empty:
        return
    out = os.path.join(FORWARD_DIR, "evaluation.csv")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    _summarize(df)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()

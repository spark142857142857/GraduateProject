"""
다축 분해 분석 — 연도별 / 시장 국면별

3년 총합이 국면 효과를 가리는 문제를 보완한다. 신호를
  (1) 연도(2023/2024/2025)
  (2) 시장 국면 — 각 신호의 벤치마크 20거래일 수익률 부호(>0 상승 / <0 하락)
로 분해해, 조건별 Buy/Sell 성과와 무기술 벤치마크를 함께 본다.

국면은 주가 재조회 없이 유도: 벤치 20d = return_20d - excess_return_20d.
(사후 실현 방향 기준 → 트레이딩 규칙이 아닌 진단용 분해)

보고 대상: cond1~4 + cond4_no_reports (reports_only/dart_only/cond4_blind는 보조 미완료).
breakdown은 기술통계(추세 확인)용. 유의성 검정은 significance.py의 pooled 결과 유지.

사용법: python src/experiment/breakdown.py
저장:   results/analysis/{날짜|latest}/breakdown_yearly.csv, breakdown_regime.csv
"""

import os
import sys
import shutil

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import get_analysis_dir, get_latest_analysis_dir
from compare import load_cond_data, calc_stats

REPORT_CONDS = ["cond1", "cond2", "cond3", "cond4", "cond4_no_reports"]

COND_SHORT = {
    "cond1": "cond1(종목명)",
    "cond2": "cond2(재무)",
    "cond3": "cond3(+리포트)",
    "cond4": "cond4(+DART)",
    "cond4_no_reports": "cond4_no_rep",
}


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """벤치 20d 수익률·국면·연도 컬럼 추가."""
    df = df.copy()
    df["bench_20d"] = df["return_20d"] - df["excess_return_20d"]
    df["regime"] = df["bench_20d"].apply(lambda x: "상승" if x > 0 else "하락")
    df["year"] = pd.to_datetime(df["signal_date"]).dt.year.astype(str)
    return df


def _bucket_benchmark(df_bucket: pd.DataFrame) -> tuple[float, float]:
    """버킷 내 전 신호(무기술) 평균 절대·초과 수익률.
    모든 조건이 동일 (ticker, date)를 평가하므로 조건 무관하게 동일."""
    return (round(df_bucket["return_20d"].mean(), 2),
            round(df_bucket["excess_return_20d"].mean(), 2))


def _cond_rows(cond: str, bucket: str, g: pd.DataFrame,
               bench_abs: float, bench_exc: float) -> list[dict]:
    """조건×버킷 → Buy/Sell 통계 행."""
    n_total = len(g)
    rows = []
    for sig in ["Buy", "Sell"]:
        sub = g[g["signal"] == sig]
        s_abs = calc_stats(sub["return_20d"], sig)
        s_exc = calc_stats(sub["excess_return_20d"], sig)
        # 중앙값 병기 — 소수 모멘텀 종목의 극단 수익률에 강건한 견고성 지표
        med_abs = round(sub["return_20d"].median(), 2) if len(sub) else float("nan")
        med_exc = round(sub["excess_return_20d"].median(), 2) if len(sub) else float("nan")
        rows.append({
            "bucket": bucket, "cond": cond, "signal": sig,
            "n": s_abs["n"],
            "share_pct": round(s_abs["n"] / n_total * 100, 1) if n_total else 0.0,
            "abs_mean": s_abs["mean"], "abs_median": med_abs, "abs_hit": s_abs["hit_rate"],
            "excess_mean": s_exc["mean"], "excess_median": med_exc, "excess_hit": s_exc["hit_rate"],
            "bench_abs": bench_abs, "bench_excess": bench_exc,
        })
    return rows


def run_axis(cond_data: dict[str, pd.DataFrame], axis_col: str,
             bucket_order: list[str], title: str) -> pd.DataFrame:
    """axis_col(year 또는 regime) 기준 분해 표 출력 + 행 반환."""
    print("\n" + "═" * 88)
    print(f"【 {title} 】  (Buy/Sell · 20거래일 · 무기술=전 신호 평균)")
    print("═" * 88)

    # 벤치마크는 조건 무관 → 첫 조건 데이터로 버킷별 산출
    ref = next(iter(cond_data.values()))
    all_rows = []

    for bucket in bucket_order:
        ref_b = ref[ref[axis_col] == bucket]
        if ref_b.empty:
            continue
        b_abs, b_exc = _bucket_benchmark(ref_b)
        print(f"\n▶ [{bucket}]  신호 {len(ref_b)}건 | 무기술 벤치: 절대 {b_abs:+.2f}% / 초과 {b_exc:+.2f}%")
        print(f"  {'조건':<14}{'Buy n':>6}{'Buy초과':>9}{'Buy초Hit':>9}  │{'Sell n':>7}{'Sell절Hit':>10}{'Sell초Hit':>10}{'Sell초中':>9}")
        print("  " + "-" * 83)
        for cond in REPORT_CONDS:
            if cond not in cond_data:
                continue
            g = cond_data[cond]
            gb = g[g[axis_col] == bucket]
            rows = _cond_rows(cond, bucket, gb, b_abs, b_exc)
            all_rows.extend(rows)
            buy = next(r for r in rows if r["signal"] == "Buy")
            sell = next(r for r in rows if r["signal"] == "Sell")
            def f(v, suf="%"):
                return "  N/A" if pd.isna(v) else f"{v:+.2f}{suf}" if suf == "%" else f"{v:.1f}"
            def h(v):
                return " N/A" if pd.isna(v) else f"{v:.1f}%"
            print(f"  {COND_SHORT.get(cond, cond):<14}"
                  f"{buy['n']:>6}{f(buy['excess_mean']):>9}{h(buy['excess_hit']):>9}  │"
                  f"{sell['n']:>7}{h(sell['abs_hit']):>10}{h(sell['excess_hit']):>10}{f(sell['excess_median']):>9}")

    return pd.DataFrame(all_rows)


def main():
    out_dir = get_analysis_dir()
    latest_dir = get_latest_analysis_dir()

    print("결과 로드 중...")
    cond_data = load_cond_data(REPORT_CONDS)
    if not cond_data:
        print("결과 파일이 없습니다.")
        return
    cond_data = {c: _enrich(df) for c, df in cond_data.items()}

    yearly = run_axis(cond_data, "year", ["2023", "2024", "2025"], "연도별")
    regime = run_axis(cond_data, "regime", ["상승", "하락"], "시장 국면별")

    for df, name in [(yearly, "breakdown_yearly"), (regime, "breakdown_regime")]:
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        shutil.copy(path, os.path.join(latest_dir, f"{name}.csv"))
    print(f"\n저장: breakdown_yearly.csv, breakdown_regime.csv → {latest_dir}")
    print("\nBuy초Hit=Buy 초과수익>0 비율 | Sell절Hit=하락(<0) 비율 | Sell초Hit=시장 언더퍼폼(<0) 비율")


if __name__ == "__main__":
    main()

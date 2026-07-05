"""
Forward Test 입력 정보 검증 — "그때 기준" 신선도·정합성 점검

forward_run_all 생성 직후 실행. 각 종목의 forward 컨텍스트(JSON의
context_used + price)가 생성 시점 기준으로 정확한지 확인한다.

검증 항목:
  - 현재가: JSON price == FDR 최신 종가(생성일 이하)  (신선도)
  - 내부 정합성: ROE ≈ PBR/PER
  - 52주 위치: [0, 100] 범위
  - 리포트 신선도: recent_reports 전부 생성일 기준 30일 이내
  - DART 값 유무: context_used에 DART 지표가 실렸는지 (forward는 인메모리 조회)

컨텍스트는 조건과 무관하게 동일하므로 **종목당 1회** 검증한다.
계산 로직(52주·모멘텀·ROE 등)은 백테스트 감사에서 이미 검증됨 —
여기선 "생성 시점 기준 신선도"에 집중.

사용법:
  python src/experiment/forward_verify.py                # 최신 생성일 폴더
  python src/experiment/forward_verify.py --date 2026-07-05
"""

import argparse
import os
import sys
import json
import glob
import re
from datetime import datetime

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import FORWARD_DIR, get_price

REPORT_WINDOW = 30  # context_builders.WINDOW_DAYS와 동일
PRICE_TOL     = 1.0  # 현재가 허용 오차(원)


def _latest_date_folder() -> str | None:
    dirs = [d for d in os.listdir(FORWARD_DIR)
            if os.path.isdir(os.path.join(FORWARD_DIR, d)) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", d)]
    return max(dirs) if dirs else None


def verify_ticker(d: dict) -> tuple[dict, list[str]]:
    """단일 종목 forward JSON → (요약, 플래그 리스트)."""
    ticker   = str(d["ticker"]).zfill(6)
    gen_date = d["date"]
    price    = d.get("price")
    ctx      = d.get("context_used", {})
    flags = []

    # 1. 현재가 신선도
    px = get_price(ticker, start="2022-12-01")
    prior = px.loc[px.index <= gen_date, "Close"] if not px.empty else pd.Series(dtype=float)
    fdr_close = float(prior.iloc[-1]) if not prior.empty else None
    if fdr_close is None:
        flags.append("FDR 가격 없음")
    elif price is None or abs(fdr_close - price) > PRICE_TOL:
        flags.append(f"현재가 불일치 JSON={price} vs FDR={fdr_close}")

    # 2. 내부 정합성 ROE ≈ PBR/PER
    per, pbr, roe = ctx.get("per"), ctx.get("pbr"), ctx.get("roe")
    if None not in (per, pbr, roe) and per not in (0, None):
        exp = round(pbr / per * 100, 1)
        if abs(exp - round(roe, 1)) > 0.3:
            flags.append(f"ROE 불일치 저장={roe} 기대(PBR/PER)={exp}")

    # 3. 52주 위치 범위
    pos = ctx.get("price_position_52w")
    if pos is not None and not (-0.1 <= pos <= 100.1):
        flags.append(f"52주 위치 범위밖 {pos}")

    # 4. 리포트 신선도 (30일 이내)
    gd = datetime.strptime(gen_date, "%Y-%m-%d")
    reports = ctx.get("recent_reports", []) or []
    for r in reports:
        rd = r.get("date")
        if not rd:
            continue
        try:
            delta = (gd - datetime.strptime(rd, "%Y-%m-%d")).days
            if delta < 0 or delta > REPORT_WINDOW:
                flags.append(f"리포트 날짜 범위밖 {rd} ({delta}일)")
        except ValueError:
            flags.append(f"리포트 날짜 파싱 실패 {rd}")

    # 5. DART 값 유무 — forward는 DART를 인메모리로 조회(CSV 미기록)하므로
    #    파일 날짜가 아닌 context_used에 DART 값이 실렸는지로 확인.
    #    은행 등 일부 종목은 매출/이익률이 없어도 부채비율은 있으므로 any()로 판정.
    dart_present = any(ctx.get(k) is not None for k in
                       ("operating_margin", "debt_ratio", "revenue_growth"))
    if not dart_present:
        flags.append("DART 값 없음 (인메모리 조회 실패 의심)")

    summary = {
        "ticker": ticker, "name": d.get("name", ""),
        "price": price, "per": per, "pbr": pbr, "roe": roe,
        "reports": len(reports), "dart": "OK" if dart_present else "없음",
        "flags": len(flags),
    }
    return summary, flags


def main(date: str | None) -> None:
    folder = date or _latest_date_folder()
    if folder is None:
        print("검증할 forward 생성일 폴더가 없습니다.")
        return
    files = sorted(glob.glob(os.path.join(FORWARD_DIR, folder, "*.json")))
    if not files:
        print(f"{folder} 폴더에 신호 JSON이 없습니다.")
        return

    # 컨텍스트는 조건 무관 동일 → 종목당 1개 JSON만
    by_ticker: dict[str, dict] = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        by_ticker.setdefault(str(d["ticker"]).zfill(6), d)

    print(f"Forward 입력 검증 | 생성일 {folder} | 종목 {len(by_ticker)}개\n")
    print(f"{'종목':<12}{'현재가':>10}{'PER':>7}{'PBR':>7}{'ROE':>7}{'리포트':>6}{'DART':>6}  판정")
    print("-" * 72)

    all_flags = []
    total_reports = 0
    for tk in sorted(by_ticker):
        s, flags = verify_ticker(by_ticker[tk])
        total_reports += s["reports"]
        verdict = "OK" if not flags else f"[!] {len(flags)}"
        per = s["per"] if s["per"] is not None else "-"
        pbr = s["pbr"] if s["pbr"] is not None else "-"
        roe = s["roe"] if s["roe"] is not None else "-"
        print(f"{s['name']:<12}{s['price'] or 0:>10,.0f}{str(per):>7}{str(pbr):>7}{str(roe):>7}"
              f"{s['reports']:>6}{s['dart']:>6}  {verdict}")
        for fl in flags:
            all_flags.append(f"  [{s['name']}] {fl}")

    print("-" * 78)
    if total_reports == 0 and len(by_ticker) > 1:
        print("\n[!] 전 종목 최근 30일 리포트 0건 — reports 데이터가 오래됐을 가능성.")
        print("  forward 전에 `python src/collect/crawl.py`로 리포트를 갱신하세요 (cond3/4 컨텍스트에 필요).")
    if all_flags:
        print(f"\n[!] 플래그 {len(all_flags)}건:")
        for fl in all_flags:
            print(fl)
    elif total_reports > 0:
        print("\n[OK] 전 종목 입력 정보 정상 (현재가·ROE정합성·52주·리포트·DART 값)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward 입력 정보 검증")
    parser.add_argument("--date", help="검증할 생성일 폴더 (기본: 최신)")
    args = parser.parse_args()
    main(args.date)

"""
월별 데이터 갱신 스크립트 (Forward Test용)

매월 1일 기준으로 최신 데이터를 갱신한다.
이번 달 신호 생성(llm_experiment_*.py) 전에 실행.

갱신 순서:
  1. 가격 데이터 갱신      (data/price/)       — 전 종목 최신 가격 추가
  2. 재무지표 갱신          (data/financials/)  — 이번 달 첫 거래일 행 추가
  3. 공시 갱신              (data/announcements/) — 이번 달 base_date 공시 추가
  4. 애널리스트 리포트 갱신 (data/reports/)     — 신규 리포트 append

이미 이번 달 데이터가 존재하는 항목은 스킵한다.

실행: python src/update.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import TICKERS, DATA_DIR, get_price, REPORTS_DIR

# collect_financials 핵심 함수·상수 재사용
from collect_financials import (
    get_dart_annual,
    applicable_fiscal_year,
    calc_52w,
    calc_momentum_volume,
    FINANCIALS_DIR,
)

# collect_announcements 핵심 함수 재사용
from collect_announcements import (
    load_corp_codes,
    collect_for_base_date,
    ANNOUNCEMENTS_DIR,
)

# crawl 핵심 함수 재사용
from crawl import fetch_reports


# ── 이번 달 첫 거래일 계산 ─────────────────────────────────
def get_this_month_first_trading_day() -> pd.Timestamp | None:
    """현재 연월 기준 첫 거래일 반환. 조회 실패 시 None."""
    today = datetime.today()
    ym_start = pd.Timestamp(today.year, today.month, 1)
    ym_end   = ym_start + pd.offsets.MonthEnd(0)

    ref_ticker = "005930"  # 삼성전자 기준
    try:
        price_df = fdr.DataReader(
            ref_ticker,
            ym_start.strftime("%Y-%m-%d"),
            ym_end.strftime("%Y-%m-%d"),
        )
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        if price_df.empty:
            return None
        return price_df.index[0]
    except Exception as e:
        print(f"  첫 거래일 조회 오류: {e}")
        return None


# ── 1단계: 가격 데이터 갱신 ───────────────────────────────
def update_price() -> int:
    """전 종목 가격 데이터 최신화. 갱신된 종목 수 반환.

    utils.get_price()의 캐시 로직이 신규 날짜만 append하므로
    그대로 호출하면 자동 갱신된다.
    """
    updated = 0
    for name, ticker in TICKERS.items():
        try:
            df = get_price(ticker)
            if df is not None and not df.empty:
                updated += 1
                print(f"  [{ticker}] {name}: 갱신 완료 (총 {len(df)}행)")
            else:
                print(f"  [{ticker}] {name}: 데이터 없음")
        except Exception as e:
            print(f"  [{ticker}] {name}: 오류 — {e}")
    return updated


# ── 2단계: 재무지표 갱신 ──────────────────────────────────
def update_financials(base_date: pd.Timestamp) -> int:
    """이번 달 첫 거래일 행이 없는 종목에 재무지표 행을 append.
    갱신된 종목 수 반환.
    """
    base_date_str = base_date.strftime("%Y-%m-%d")

    # 발행주식수 조회 (BPS 계산용)
    print("  발행주식수 조회 중 (FDR StockListing)...")
    listing = fdr.StockListing("KRX")
    shares_map: dict[str, int] = {}
    for _, row in listing.iterrows():
        if pd.isna(row.get("Stocks")):
            continue
        try:
            shares_map[row["Code"]] = int(
                str(row["Stocks"]).replace(",", "").split(".")[0]
            )
        except (ValueError, TypeError):
            pass

    updated = 0
    today_str = datetime.today().strftime("%Y-%m-%d")

    for name, ticker in TICKERS.items():
        out_path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")

        # 기존 CSV 로드
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path, dtype={"ticker": str})
            if base_date_str in df_existing["date"].values:
                print(f"  [{ticker}] {name}: {base_date_str} 이미 존재 — 스킵")
                continue
        else:
            df_existing = pd.DataFrame()

        # 가격 데이터 로드 (52주·기술 지표 계산용)
        try:
            price_df = fdr.DataReader(ticker, "2021-01-01", today_str)
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        except Exception as e:
            print(f"  [{ticker}] {name}: 가격 데이터 오류 — {e}")
            continue

        # 해당 거래일 종가 확인
        day_data = price_df.loc[price_df.index == base_date]
        if day_data.empty:
            print(f"  [{ticker}] {name}: {base_date_str} 거래일 없음 — 스킵")
            continue
        price = float(day_data["Close"].iloc[0])

        # DART 재무 데이터
        fy        = applicable_fiscal_year(base_date)
        dart_data = get_dart_annual(ticker, fy)
        eps       = dart_data["eps"]
        equity    = dart_data["equity"]
        shares    = shares_map.get(ticker, np.nan)

        # PER / PBR / ROE
        per = (
            round(price / eps, 2)
            if (not np.isnan(eps) and eps > 0)
            else np.nan
        )
        bps = (
            equity / shares
            if (
                not np.isnan(equity)
                and not np.isnan(shares)
                and shares > 0
            )
            else np.nan
        )
        pbr = (
            round(price / bps, 2)
            if (not np.isnan(bps) and bps > 0)
            else np.nan
        )
        roe = (
            round(pbr / per * 100, 2)
            if (
                not np.isnan(per)
                and not np.isnan(pbr)
                and per > 0
            )
            else np.nan
        )
        mktcap = round(price * shares) if not np.isnan(shares) else np.nan

        # 52주 고저가
        high52, low52 = calc_52w(price_df, base_date)
        if (
            not np.isnan(high52)
            and not np.isnan(low52)
            and (high52 - low52) > 0
        ):
            pos52 = round((price - low52) / (high52 - low52) * 100, 2)
        else:
            pos52 = np.nan

        # 기술 지표
        momentum, vol_change = calc_momentum_volume(price_df, base_date)

        new_row = pd.DataFrame([{
            "date":               base_date_str,
            "ticker":             str(ticker).zfill(6),
            "name":               name,
            "per":                per,
            "pbr":                pbr,
            "roe":                roe,
            "market_cap":         mktcap,
            "high_52w":           high52,
            "low_52w":            low52,
            "price_position_52w": pos52,
            "momentum_1m":        momentum,
            "volume_change":      vol_change,
        }])

        df_out = pd.concat([df_existing, new_row], ignore_index=True)
        df_out = df_out.sort_values("date").reset_index(drop=True)
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [{ticker}] {name}: {base_date_str} 행 추가 완료")
        updated += 1

    return updated


# ── 3단계: 공시 갱신 ──────────────────────────────────────
def update_announcements(base_date: pd.Timestamp) -> int:
    """이번 달 base_date 기준 직전 30일 공시를 수집하여 append.
    갱신된 종목 수 반환.
    """
    base_date_str = base_date.strftime("%Y-%m-%d")
    os.makedirs(ANNOUNCEMENTS_DIR, exist_ok=True)

    corp_code_map = load_corp_codes()
    updated = 0

    for name, ticker in TICKERS.items():
        out_path = os.path.join(ANNOUNCEMENTS_DIR, f"{ticker}.csv")

        # 이미 이번 달 base_date 공시 존재 여부 확인
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path, dtype={"ticker": str})
            if base_date_str in df_existing["base_date"].values:
                print(f"  [{ticker}] {name}: {base_date_str} 공시 이미 존재 — 스킵")
                continue
        else:
            df_existing = pd.DataFrame()

        corp_code = corp_code_map.get(ticker)
        if not corp_code:
            print(f"  [{ticker}] {name}: corp_code 없음 — 스킵")
            continue

        records = collect_for_base_date(corp_code, base_date_str)
        if not records:
            print(f"  [{ticker}] {name}: {base_date_str} 해당 공시 없음")
            continue

        rows = [
            {
                "base_date": base_date_str,
                "ticker":    ticker,
                "name":      name,
                "report_nm": rec["report_nm"],
                "rcept_dt":  rec["rcept_dt"],
            }
            for rec in records
        ]
        df_new = pd.DataFrame(
            rows, columns=["base_date", "ticker", "name", "report_nm", "rcept_dt"]
        )

        if not df_existing.empty:
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new

        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [{ticker}] {name}: {len(rows)}건 추가 → {out_path}")
        updated += 1

    return updated


# ── 4단계: 애널리스트 리포트 갱신 ────────────────────────
def update_reports() -> int:
    """기존 CSV의 마지막 날짜 이후 신규 리포트만 크롤링하여 append.
    추가된 총 건수 반환.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")
    total_new = 0

    for name, ticker in TICKERS.items():
        out_path = os.path.join(REPORTS_DIR, f"{ticker}.csv")

        # 기존 CSV에서 마지막 수집 날짜 확인
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path)
            if not df_existing.empty and "date" in df_existing.columns:
                last_date = (
                    pd.to_datetime(df_existing["date"]).max().strftime("%Y-%m-%d")
                )
            else:
                last_date = "2025-01-01"
        else:
            df_existing = pd.DataFrame()
            last_date = "2025-01-01"

        # 이미 오늘까지 수집됐으면 스킵
        if last_date >= today_str:
            print(f"  [{ticker}] {name}: 최신 리포트 이미 존재 — 스킵")
            continue

        # 신규 리포트 수집 후 last_date 이후 항목만 필터
        records = fetch_reports(ticker)
        new_records = [
            r for r in records
            if r.get("date") and r["date"] > last_date
        ]

        if not new_records:
            print(f"  [{ticker}] {name}: 신규 리포트 없음")
            continue

        df_new = pd.DataFrame(new_records)
        if not df_existing.empty:
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_out = df_new

        # 날짜순 정렬 + nid 기준 중복 제거
        df_out = (
            df_out.sort_values("date")
            .drop_duplicates(subset=["nid"])
            .reset_index(drop=True)
        )
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  [{ticker}] {name}: {len(new_records)}건 추가 → {out_path}")
        total_new += len(new_records)

    return total_new


# ── 메인 ─────────────────────────────────────────────────
def main():
    today = datetime.today()
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"데이터 갱신 시작: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"{sep}\n")

    # 이번 달 첫 거래일 계산
    base_date = get_this_month_first_trading_day()
    if base_date is None:
        print("이번 달 첫 거래일을 찾을 수 없습니다. 갱신 중단.")
        return
    print(f"이번 달 기준일: {base_date.strftime('%Y-%m-%d')}\n")

    # 1단계: 가격
    print("── 1단계: 가격 데이터 갱신 ──────────────────────────")
    price_count = update_price()
    print()

    # 2단계: 재무지표
    print("── 2단계: 재무지표 갱신 ─────────────────────────────")
    fin_count = update_financials(base_date)
    print()

    # 3단계: 공시
    print("── 3단계: 공시 갱신 ──────────────────────────────────")
    ann_count = update_announcements(base_date)
    print()

    # 4단계: 리포트
    print("── 4단계: 애널리스트 리포트 갱신 ────────────────────")
    report_new = update_reports()
    print()

    # 요약
    label = base_date.strftime("%Y-%m-%d")
    print(sep)
    print(f"{label} 기준 업데이트 완료")
    print(f"  가격:   {price_count}종목 갱신")
    print(f"  재무:   {fin_count}종목 갱신")
    print(f"  공시:   {ann_count}종목 갱신")
    print(f"  리포트: 신규 {report_new}건 추가")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()

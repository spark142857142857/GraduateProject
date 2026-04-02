"""
월별 데이터 갱신 스크립트 (Forward Test용)

매월 1일 기준으로 최신 데이터를 갱신한다.
이번 달 신호 생성(llm_experiment_*.py) 전에 실행.

갱신 순서:
  1. 가격 데이터 갱신      (data/price/)              - 전 종목 최신 가격 추가
  2. 재무지표 갱신          (data/financials/)         - 이번 달 첫 거래일 행 추가
  3. 애널리스트 리포트 갱신 (data/reports/)            - 신규 리포트 append
  4. DART 실적 갱신         (data/dart_fundamentals/) - 이번 달 첫 거래일 행 추가

이미 이번 달 데이터가 존재하는 항목은 스킵한다.

실행:
  python src/update.py                 → 전체 20개 갱신
  python src/update.py --ticker 005930 → 1개만 갱신
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import TICKERS, get_price, REPORTS_DIR

# collect_financials 핵심 함수·상수 재사용
from collect_financials import (
    get_dart_annual,
    applicable_fiscal_year,
    calc_52w,
    calc_momentum_volume,
    FINANCIALS_DIR,
)

# crawl 핵심 함수 재사용
from crawl import fetch_reports

# collect_dart_fundamentals 핵심 함수·상수 재사용
from collect_dart_fundamentals import (
    get_dart_annual as get_dart_fund_annual,
    get_dividend_yield,
    applicable_fiscal_year as dart_applicable_fy,
    DART_FUND_DIR,
)


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


# ── 공통 KRX 발행주식수 맵 로드 ──────────────────────────
_SHARES_MAP_CACHE: dict | None = None

def _load_shares_map() -> dict:
    """FDR StockListing에서 종목별 발행주식수 dict 반환. 첫 호출 시 로드 후 캐시."""
    global _SHARES_MAP_CACHE
    if _SHARES_MAP_CACHE is not None:
        return _SHARES_MAP_CACHE
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
    _SHARES_MAP_CACHE = shares_map
    return shares_map


# ── 단일 종목 갱신 헬퍼 ──────────────────────────────────

def _update_price_one(ticker: str, name: str) -> bool:
    """단일 종목 가격 갱신. 성공 시 True."""
    try:
        df = get_price(ticker)
        if df is not None and not df.empty:
            print(f"  [{ticker}] {name}: 가격 갱신 완료 (총 {len(df)}행)")
            return True
        print(f"  [{ticker}] {name}: 가격 데이터 없음")
        return False
    except Exception as e:
        print(f"  [{ticker}] {name}: 가격 오류 - {e}")
        return False


def _update_financials_one(
    ticker: str, name: str, base_date: pd.Timestamp, shares_map: dict
) -> bool:
    """단일 종목 재무지표 갱신. 갱신 시 True."""
    base_date_str = base_date.strftime("%Y-%m-%d")
    out_path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")
    today_str = datetime.today().strftime("%Y-%m-%d")

    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, dtype={"ticker": str})
        if base_date_str in df_existing["date"].values:
            print(f"  [{ticker}] {name}: {base_date_str} 이미 존재 - 스킵")
            return False
    else:
        df_existing = pd.DataFrame()

    try:
        price_df = fdr.DataReader(ticker, "2021-01-01", today_str)
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    except Exception as e:
        print(f"  [{ticker}] {name}: 가격 데이터 오류 - {e}")
        return False

    day_data = price_df.loc[price_df.index == base_date]
    if day_data.empty:
        print(f"  [{ticker}] {name}: {base_date_str} 거래일 없음 - 스킵")
        return False
    price = float(day_data["Close"].iloc[0])

    fy        = applicable_fiscal_year(base_date)
    dart_data = get_dart_annual(ticker, fy)
    eps       = dart_data["eps"]
    equity    = dart_data["equity"]
    shares    = shares_map.get(ticker, np.nan)

    per = round(price / eps, 2) if (not np.isnan(eps) and eps > 0) else np.nan
    bps = (
        equity / shares
        if (not np.isnan(equity) and not np.isnan(shares) and shares > 0)
        else np.nan
    )
    pbr = round(price / bps, 2) if (not np.isnan(bps) and bps > 0) else np.nan
    roe = (
        round(pbr / per * 100, 2)
        if (not np.isnan(per) and not np.isnan(pbr) and per > 0)
        else np.nan
    )
    mktcap = round(price * shares) if not np.isnan(shares) else np.nan

    high52, low52 = calc_52w(price_df, base_date)
    if not np.isnan(high52) and not np.isnan(low52) and (high52 - low52) > 0:
        pos52 = round((price - low52) / (high52 - low52) * 100, 2)
    else:
        pos52 = np.nan

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
    print(f"  [{ticker}] {name}: 재무지표 {base_date_str} 행 추가 완료")
    return True


def _update_reports_one(ticker: str, name: str) -> int:
    """단일 종목 리포트 갱신. 추가된 건수 반환."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    today_str = datetime.today().strftime("%Y-%m-%d")
    out_path = os.path.join(REPORTS_DIR, f"{ticker}.csv")

    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path)
        if not df_existing.empty and "date" in df_existing.columns:
            last_date = df_existing["date"].max()  # ISO 문자열 비교로 충분
        else:
            last_date = "2023-01-01"
    else:
        df_existing = pd.DataFrame()
        last_date = "2023-01-01"

    if last_date >= today_str:
        print(f"  [{ticker}] {name}: 최신 리포트 이미 존재 - 스킵")
        return 0

    try:
        records = fetch_reports(ticker, since_date=last_date)
    except Exception as e:
        print(f"  [{ticker}] {name}: 리포트 수집 오류 (네트워크?) - {e}")
        return 0

    if not records:
        print(f"  [{ticker}] {name}: 신규 리포트 없음")
        return 0

    df_new = pd.DataFrame(records)
    df_out = (
        pd.concat([df_existing, df_new], ignore_index=True)
        if not df_existing.empty
        else df_new
    )
    df_out = (
        df_out.sort_values("date")
        .drop_duplicates(subset=["nid"])
        .reset_index(drop=True)
    )
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [{ticker}] {name}: 리포트 {len(records)}건 추가")
    return len(records)


def _update_dart_one(ticker: str, name: str, base_date: pd.Timestamp) -> bool:
    """단일 종목 dart_fundamentals 갱신. 갱신 시 True.

    이번 달 기준일(base_date) 행이 없으면 해당 연도 DART 실적을 조회해 append.
    """
    base_date_str = base_date.strftime("%Y-%m-%d")
    out_path = os.path.join(DART_FUND_DIR, f"{ticker}.csv")

    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, dtype={"ticker": str})
        if base_date_str in df_existing["date"].values:
            print(f"  [{ticker}] {name}: DART {base_date_str} 이미 존재 - 스킵")
            return False
    else:
        df_existing = pd.DataFrame()

    fy   = dart_applicable_fy(base_date)
    data = get_dart_fund_annual(ticker, fy)

    revenue  = data["revenue"]
    oper_inc = data["operating_income"]
    net_inc  = data["net_income"]
    total_liab = data["total_liabilities"]
    total_eq   = data["total_equity"]
    oper_cf    = data["operating_cashflow"]

    oper_margin = (
        round(oper_inc / revenue * 100, 2)
        if not (np.isnan(oper_inc) or np.isnan(revenue) or revenue == 0)
        else np.nan
    )
    debt_ratio = (
        round(total_liab / total_eq * 100, 2)
        if not (np.isnan(total_liab) or np.isnan(total_eq) or total_eq == 0)
        else np.nan
    )

    def _yoy(curr, prev):
        if np.isnan(curr) or np.isnan(prev) or prev == 0:
            return np.nan
        return round((curr - prev) / abs(prev) * 100, 2)

    div_yield = get_dividend_yield(ticker, fy)

    new_row = pd.DataFrame([{
        "date":                 base_date_str,
        "ticker":               str(ticker).zfill(6),
        "name":                 name,
        "revenue":              revenue,
        "operating_income":     oper_inc,
        "net_income":           net_inc,
        "operating_margin":     oper_margin,
        "debt_ratio":           debt_ratio,
        "operating_cashflow":   oper_cf,
        "dividend_yield":       div_yield,
        "revenue_yoy":          _yoy(revenue,  data["revenue_prev"]),
        "operating_income_yoy": _yoy(oper_inc, data["operating_income_prev"]),
    }])

    df_out = pd.concat([df_existing, new_row], ignore_index=True)
    df_out = df_out.sort_values("date").reset_index(drop=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  [{ticker}] {name}: DART {base_date_str} 행 추가 완료")
    return True


# ── Public API ──────────────────────────────────────────

def get_today_context(ticker: str) -> dict:
    """오늘 날짜 기준 실시간 재무지표 수집 — 파일 저장 없음.

    forward_test.py 전용.
    data/financials/ 를 오염시키지 않고 dict 반환.

    수집 항목:
      - 현재가, PER/PBR/ROE/시가총액, 52주 고저/위치,
        1개월 모멘텀, 거래량 변화율
      - data/dart_fundamentals/ 최신 행 (매출·이익·부채비율 등)
      - data/reports/ 최근 30일 리포트

    Returns:
        Context dict (context_builders.build_*_from_dict 에 전달)
    """
    name      = next((n for n, t in TICKERS.items() if t == ticker), ticker)
    today_str = datetime.today().strftime("%Y-%m-%d")

    # ── 현재가 및 가격 이력 ─────────────────────────────
    full_price = fdr.DataReader(ticker, "2021-01-01", today_str)
    full_price.index = pd.to_datetime(full_price.index).tz_localize(None)
    if full_price.empty:
        raise ValueError(f"[{ticker}] 주가 데이터 없음")
    current_price = float(full_price["Close"].iloc[-1])
    today_ts = full_price.index[-1]

    # ── DART 재무 (PER/PBR/ROE용 EPS·자본) ──────────────
    fy        = applicable_fiscal_year(pd.Timestamp(today_str))
    dart_data = get_dart_annual(ticker, fy)
    eps       = dart_data["eps"]
    equity    = dart_data["equity"]

    # ── 발행주식수 (KRX 전체 목록 로드 - 단일 종목용으로는 과도하나
    #    발행주식수를 제공하는 다른 무료 API가 없어 불가피. ~2-3초 소요) ──
    shares = _load_shares_map().get(ticker, np.nan)

    # ── PER / PBR / ROE / 시가총액 ──────────────────────
    per    = round(current_price / eps, 2) if (not np.isnan(eps) and eps > 0) else np.nan
    bps    = (equity / shares
              if (not np.isnan(equity) and not np.isnan(shares) and shares > 0)
              else np.nan)
    pbr    = round(current_price / bps, 2) if (not np.isnan(bps) and bps > 0) else np.nan
    roe    = (round(pbr / per * 100, 2)
              if (not np.isnan(per) and not np.isnan(pbr) and per > 0)
              else np.nan)
    mktcap = round(current_price * shares) if not np.isnan(shares) else np.nan

    # ── 52주 고저 / 기술지표 ────────────────────────────
    high52, low52 = calc_52w(full_price, today_ts)
    if not np.isnan(high52) and not np.isnan(low52) and (high52 - low52) > 0:
        pos52 = round((current_price - low52) / (high52 - low52) * 100, 2)
    else:
        pos52 = np.nan
    momentum, vol_change = calc_momentum_volume(full_price, today_ts)

    # ── dart_fundamentals 최신 행 재활용 (3개월 이상 지났으면 갱신) ──
    dart_row: dict = {}
    dart_path = os.path.join(DART_FUND_DIR, f"{ticker}.csv")
    if os.path.exists(dart_path):
        df_dart = pd.read_csv(dart_path, dtype={"ticker": str})
        if not df_dart.empty:
            latest_dart_date = pd.to_datetime(df_dart["date"].max())
            days_stale = (pd.Timestamp(today_str) - latest_dart_date).days
            if days_stale > 90:
                print(f"  [{ticker}] DART 데이터 {days_stale}일 경과 -> 이번 달 행 추가 시도")
                base_date = get_this_month_first_trading_day()
                if base_date is not None:
                    _update_dart_one(ticker, name, base_date)
                    # 갱신 후 재로드
                    df_dart = pd.read_csv(dart_path, dtype={"ticker": str})
            dart_row = df_dart.sort_values("date").iloc[-1].to_dict()

    # ── 최근 리포트 (CSV 직접 읽기, 오늘 기준 30일) ─────
    # forward test는 오늘 포함, backtest(build_reports)는 전일까지 — 의도적 차이
    recent_reports: list = []
    rep_path = os.path.join(REPORTS_DIR, f"{ticker}.csv")
    if os.path.exists(rep_path):
        df_rep   = pd.read_csv(rep_path, parse_dates=["date"])
        end_dt   = pd.Timestamp(today_str)
        start_dt = end_dt - pd.Timedelta(days=30)
        sub = df_rep[(df_rep["date"] >= start_dt) & (df_rep["date"] <= end_dt)]
        sub = sub.sort_values("date", ascending=False).head(5)
        for _, r in sub.iterrows():
            tp = r.get("target_price")
            recent_reports.append({
                "date":         str(r["date"].date()),
                "title":        str(r["title"]),
                "target_price": None if pd.isna(tp) else int(tp),
            })

    def _v(val):
        """np.nan → None (JSON 직렬화 + pd.isna 양쪽 호환)."""
        try:
            return None if np.isnan(val) else val
        except (TypeError, ValueError):
            return val

    ctx = {
        # 메타
        "ticker":               ticker,
        "name":                 name,
        "date":                 today_str,
        "price":                current_price,
        # 재무지표
        "per":                  _v(per),
        "pbr":                  _v(pbr),
        "roe":                  _v(roe),
        "market_cap":           _v(mktcap),
        "high_52w":             _v(high52),
        "low_52w":              _v(low52),
        "price_position_52w":   _v(pos52),
        "momentum_1m":          _v(momentum),
        "volume_change":        _v(vol_change),
        # dart_fundamentals (최신 사업연도)
        "revenue":              _v(dart_row.get("revenue")),
        "operating_income":     _v(dart_row.get("operating_income")),
        "net_income":           _v(dart_row.get("net_income")),
        "operating_margin":     _v(dart_row.get("operating_margin")),
        "debt_ratio":           _v(dart_row.get("debt_ratio")),
        "operating_cashflow":   _v(dart_row.get("operating_cashflow")),
        "dividend_yield":       _v(dart_row.get("dividend_yield")),
        "revenue_yoy":          _v(dart_row.get("revenue_yoy")),
        "operating_income_yoy": _v(dart_row.get("operating_income_yoy")),
        # 최근 리포트
        "recent_reports":       recent_reports,
    }

    print(f"  [{ticker}] {name}: 실시간 지표 수집 완료 "
          f"(현재가={int(current_price):,}원, PER={ctx['per']})")
    return ctx


def update_single(ticker: str) -> None:
    """특정 종목 1개를 오늘 날짜 기준으로 갱신.

    forward_test.py 등 외부에서 호출 가능:
        from update import update_single
        update_single("005930")
    """
    # 종목명 역조회
    name = next((n for n, t in TICKERS.items() if t == ticker), ticker)

    print(f"\n[{ticker}] {name} 갱신 시작")

    base_date = get_this_month_first_trading_day()
    if base_date is None:
        print(f"  [{ticker}] 기준일 조회 실패 - 갱신 중단")
        return

    print(f"  기준일: {base_date.strftime('%Y-%m-%d')}")

    shares_map = _load_shares_map()

    _update_price_one(ticker, name)
    _update_financials_one(ticker, name, base_date, shares_map)
    _update_reports_one(ticker, name)
    _update_dart_one(ticker, name, base_date)

    print(f"[{ticker}] {name} 갱신 완료\n")


def run() -> None:
    """전체 20개 종목 일괄 갱신."""
    today = datetime.today()
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"데이터 갱신 시작: {today.strftime('%Y-%m-%d %H:%M')}")
    print(f"{sep}\n")

    base_date = get_this_month_first_trading_day()
    if base_date is None:
        print("이번 달 첫 거래일을 찾을 수 없습니다. 갱신 중단.")
        return
    print(f"이번 달 기준일: {base_date.strftime('%Y-%m-%d')}\n")

    # shares_map은 한 번만 로드
    shares_map = _load_shares_map()

    price_count = fin_count = report_new = dart_count = 0

    for name, ticker in TICKERS.items():
        if _update_price_one(ticker, name):
            price_count += 1
        if _update_financials_one(ticker, name, base_date, shares_map):
            fin_count += 1
        report_new += _update_reports_one(ticker, name)
        if _update_dart_one(ticker, name, base_date):
            dart_count += 1

    label = base_date.strftime("%Y-%m-%d")
    print(f"\n{sep}")
    print(f"{label} 기준 업데이트 완료")
    print(f"  가격:   {price_count}종목 갱신")
    print(f"  재무:   {fin_count}종목 갱신")
    print(f"  리포트: 신규 {report_new}건 추가")
    print(f"  DART:   {dart_count}종목 갱신")
    print(f"{sep}\n")


# ── 진입점 ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="주식 데이터 갱신")
    parser.add_argument("--ticker", type=str, default=None, help="단일 종목 코드 (예: 005930)")
    args = parser.parse_args()

    if args.ticker:
        update_single(args.ticker)
    else:
        run()

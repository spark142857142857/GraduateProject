"""
DART 기본 재무 데이터 수집 스크립트 (cond4용)

DART 연간 사업보고서에서 핵심 재무 지표를 수집한다.

수집 항목:
  - 손익계산서 : 매출, 영업이익, 순이익
  - 재무상태표 : 부채총계, 자본총계 → 부채비율
  - 현금흐름표 : 영업활동현금흐름
  - 배당       : 배당수익률 (현재 NaN, 추후 구현)

Look-ahead Bias 방지 (collect_financials.py 와 동일):
  - 1~3월 → 전전년도 사업보고서 (3/31 이전은 전년도 미공시)
  - 4~12월 → 전년도 사업보고서

금액 단위: DART 원본 그대로 저장 (대부분의 대형 상장사 기준 백만원)
  → 조원 환산은 context_builders.py 에서 처리

저장 경로: data/dart_fundamentals/{ticker}.csv

실행: python src/collect_dart_fundamentals.py
"""

import os
import sys
import time
import warnings
import requests
from datetime import datetime

import numpy as np
import pandas as pd
import OpenDartReader as odr
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import TICKERS, DATA_DIR
# 월별 첫 거래일 목록·적용 회계연도 결정 함수 재사용 (코드 중복 방지)
from collect_financials import applicable_fiscal_year, get_monthly_first_days

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────
DART_API_KEY  = os.environ.get("DARTS_API_KEY", "")
DART_FUND_DIR = os.path.join(DATA_DIR, "dart_fundamentals")
REQ_DELAY     = 0.3   # DART API 요청 간 딜레이 (초)

os.makedirs(DART_FUND_DIR, exist_ok=True)
try:
    dart = odr(api_key=DART_API_KEY)
except Exception:
    dart = None  # 키 없음 → get_dart_annual() 내부 try/except가 NaN 반환


def _current_ym() -> str:
    """오늘 날짜 기준 'YYYY-MM' 반환."""
    today = datetime.today()
    return f"{today.year}-{today.month:02d}"


# ── 계정과목명 후보 (우선순위 순) ─────────────────────────
# DART 보고서마다 계정명이 상이해 fallback 목록 사용
REVENUE_NAMES      = ["매출액", "수익(매출액)", "영업수익", "매출"]
OPER_INC_NAMES     = ["영업이익", "영업이익(손실)"]
NET_INC_NAMES      = ["당기순이익", "당기순이익(손실)"]
TOTAL_LIAB_NAMES   = ["부채총계"]
TOTAL_EQUITY_NAMES = ["자본총계"]
OPER_CF_NAMES      = ["영업활동 현금흐름", "영업활동현금흐름", "영업활동으로 인한 현금흐름"]

# ── DART 조회 캐시 ────────────────────────────────────────
_cache: dict[tuple, dict] = {}


def _get_amount(df: pd.DataFrame, names: list[str],
                col: str = "thstrm_amount") -> float:
    """계정과목명 우선순위에 따라 금액 추출. 없으면 np.nan."""
    for name in names:
        rows = df[df["account_nm"] == name]
        if not rows.empty:
            val = rows[col].iloc[0]
            if pd.notna(val) and str(val).strip() not in ("", "-", "−"):
                try:
                    return float(str(val).replace(",", ""))
                except ValueError:
                    pass
    return np.nan


def _yoy(curr: float, prev: float) -> float:
    """전년比 변화율 (%). 분모 0 또는 NaN이면 np.nan."""
    if np.isnan(curr) or np.isnan(prev) or prev == 0:
        return np.nan
    return round((curr - prev) / abs(prev) * 100, 2)


def get_dart_annual(ticker: str, fiscal_year: int) -> dict:
    """
    DART 사업보고서(연간)에서 핵심 재무 지표 추출.
    결과를 캐시해 동일 (ticker, fiscal_year) 재호출 시 API 생략.
    """
    key = (ticker, fiscal_year)
    if key in _cache:
        return _cache[key]

    result = {
        "revenue":               np.nan,
        "revenue_prev":          np.nan,
        "operating_income":      np.nan,
        "operating_income_prev": np.nan,
        "net_income":            np.nan,
        "total_liabilities":     np.nan,
        "total_equity":          np.nan,
        "operating_cashflow":    np.nan,
    }

    try:
        time.sleep(REQ_DELAY)
        df = dart.finstate_all(ticker, fiscal_year, "11011")   # 사업보고서
        if df is None or df.empty:
            _cache[key] = result
            return result

        # 재무제표 구분별 분리 (연결 재무제표 우선)
        is_df = df[df["sj_div"].isin(["IS", "CIS"])]   # 손익계산서
        bs_df = df[df["sj_div"] == "BS"]                # 재무상태표
        cf_df = df[df["sj_div"] == "CF"]                # 현금흐름표

        # 당기(thstrm) + 전기(frmtrm) 함께 추출 → YoY 계산용
        result["revenue"]               = _get_amount(is_df, REVENUE_NAMES,      "thstrm_amount")
        result["revenue_prev"]          = _get_amount(is_df, REVENUE_NAMES,      "frmtrm_amount")
        result["operating_income"]      = _get_amount(is_df, OPER_INC_NAMES,     "thstrm_amount")
        result["operating_income_prev"] = _get_amount(is_df, OPER_INC_NAMES,     "frmtrm_amount")
        result["net_income"]            = _get_amount(is_df, NET_INC_NAMES,      "thstrm_amount")
        result["total_liabilities"]     = _get_amount(bs_df, TOTAL_LIAB_NAMES)
        result["total_equity"]          = _get_amount(bs_df, TOTAL_EQUITY_NAMES)
        result["operating_cashflow"]    = _get_amount(cf_df, OPER_CF_NAMES)

    except Exception:
        pass   # 조회 실패 → NaN 유지

    _cache[key] = result
    return result


# ── 종목 처리 ──────────────────────────────────────────────

def process_ticker(name: str, ticker: str) -> pd.DataFrame | None:
    """종목별 DART 실적 수집. 기존 CSV가 있으면 누락 월만 추가(append-only).

    동적 END_YM(_current_ym())을 사용하므로 매월 실행 시 자동 확장.
    """
    out_path = os.path.join(DART_FUND_DIR, f"{ticker}.csv")

    # 기존 날짜 셋 로드 (있으면 누락 월만 처리)
    existing_dates: set[str] = set()
    df_existing = pd.DataFrame()
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, dtype={"ticker": str})
        existing_dates = set(df_existing["date"].astype(str).tolist())

    # 오늘 기준 월까지 동적 생성
    monthly_dates = get_monthly_first_days(ticker, end_ym=_current_ym())
    if not monthly_dates:
        tqdm.write(f"  [{ticker}] 월별 거래일 없음")
        return None

    # 누락된 월만 필터링
    missing = [d for d in monthly_dates if d.strftime("%Y-%m-%d") not in existing_dates]
    if not missing:
        tqdm.write(f"  [{ticker}] 이미 최신 ({len(df_existing)}행) → 스킵")
        return None

    tqdm.write(f"  [{ticker}] {len(missing)}개월 신규 추가 예정")

    rows = []
    for date in tqdm(missing, desc=f"  {name}({ticker})", leave=False):
        fy   = applicable_fiscal_year(date)
        data = get_dart_annual(ticker, fy)

        revenue    = data["revenue"]
        oper_inc   = data["operating_income"]
        net_inc    = data["net_income"]
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

        rows.append({
            "date":                 date.strftime("%Y-%m-%d"),
            "ticker":               str(ticker).zfill(6),
            "name":                 name,
            "revenue":              revenue,
            "operating_income":     oper_inc,
            "net_income":           net_inc,
            "operating_margin":     oper_margin,
            "debt_ratio":           debt_ratio,
            "operating_cashflow":   oper_cf,
            "dividend_yield":       np.nan,
            "revenue_yoy":          _yoy(revenue,  data["revenue_prev"]),
            "operating_income_yoy": _yoy(oper_inc, data["operating_income_prev"]),
        })

    if not rows:
        return None

    df_new = pd.DataFrame(rows)
    df_out = (
        pd.concat([df_existing, df_new], ignore_index=True)
        if not df_existing.empty else df_new
    )
    df_out = df_out.sort_values("date").reset_index(drop=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"  [{ticker}] +{len(rows)}행 추가 → 총 {len(df_out)}행")
    return df_new


# ── 배당수익률 수집 ──────────────────────────────────────

_div_cache: dict[tuple, float] = {}

def get_dividend_yield(ticker: str, fiscal_year: int) -> float:
    """
    DART /api/alotMatter.json 에서 보통주 배당수익률(%) 추출.
    데이터 없으면 np.nan 반환.
    """
    key = (ticker, fiscal_year)
    if key in _div_cache:
        return _div_cache[key]

    result = np.nan
    try:
        corp_code = dart.find_corp_code(ticker)
        if not corp_code:
            _div_cache[key] = result
            return result

        time.sleep(REQ_DELAY)
        url = "https://opendart.fss.or.kr/api/alotMatter.json"
        params = {
            "crtfc_key": DART_API_KEY,
            "corp_code":  corp_code,
            "bsns_year":  str(fiscal_year),
            "reprt_code": "11011",
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data.get("status") != "000" or "list" not in data:
            _div_cache[key] = result
            return result

        for row in data["list"]:
            # 보통주 배당수익률 우선 추출
            se = str(row.get("se", ""))
            knd = str(row.get("stock_knd", ""))
            if "배당수익률" in se and "보통주" in knd:
                val = str(row.get("thstrm", "")).replace(",", "").strip()
                if val not in ("", "-", "−"):
                    try:
                        result = float(val)
                        break
                    except ValueError:
                        pass

    except Exception:
        pass

    _div_cache[key] = result
    return result


# ── 누락 컬럼 업데이트 ──────────────────────────────────

def update_missing_columns() -> None:
    """
    기존 dart_fundamentals CSV의 operating_cashflow, dividend_yield 컬럼이
    전부 NaN인 경우 재수집하여 업데이트.
    운영현금흐름: OPER_CF_NAMES 확장(공백 포함)으로 재시도.
    배당수익률: DART alotMatter API 직접 호출.
    """
    csv_files = [f for f in os.listdir(DART_FUND_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("업데이트할 파일 없음")
        return

    for fname in tqdm(csv_files, desc="CF+배당 업데이트"):
        path = os.path.join(DART_FUND_DIR, fname)
        df = pd.read_csv(path, dtype={"ticker": str})
        ticker = fname.replace(".csv", "")

        need_cf  = df["operating_cashflow"].isna().all()
        need_div = df["dividend_yield"].isna().all()

        if not need_cf and not need_div:
            tqdm.write(f"  [{ticker}] 이미 완료 → 스킵")
            continue

        # 고유 회계연도 목록
        fiscal_years = df["date"].apply(
            lambda d: applicable_fiscal_year(pd.Timestamp(d))
        ).unique()

        cf_map:  dict[int, float] = {}
        div_map: dict[int, float] = {}

        for fy in fiscal_years:
            if need_cf:
                data = get_dart_annual(ticker, fy)   # 캐시 재활용
                cf_map[fy] = data["operating_cashflow"]

            if need_div:
                div_map[fy] = get_dividend_yield(ticker, fy)

        if need_cf:
            df["operating_cashflow"] = df["date"].apply(
                lambda d: cf_map.get(applicable_fiscal_year(pd.Timestamp(d)), np.nan)
            )
        if need_div:
            df["dividend_yield"] = df["date"].apply(
                lambda d: div_map.get(applicable_fiscal_year(pd.Timestamp(d)), np.nan)
            )

        df.to_csv(path, index=False, encoding="utf-8-sig")
        cf_ok  = df["operating_cashflow"].notna().sum()
        div_ok = df["dividend_yield"].notna().sum()
        total_rows = len(df)
        tqdm.write(f"  [{ticker}] CF={cf_ok}/{total_rows}  배당={div_ok}/{total_rows}")


# ── 메인 ──────────────────────────────────────────────────

def run():
    for name, ticker in tqdm(TICKERS.items(), desc="전체 종목"):
        process_ticker(name, ticker)
    print("\n누락 컬럼(CF·배당) 업데이트 중...")
    update_missing_columns()
    print("collect_dart_fundamentals 완료")


if __name__ == "__main__":
    run()

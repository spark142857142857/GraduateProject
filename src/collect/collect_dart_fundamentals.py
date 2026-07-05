"""
DART 기본 재무 데이터 수집 스크립트 (cond4용)

DART 정기보고서(분기/반기/사업)에서 핵심 재무 지표를 수집한다.
신호일 기준 "가장 최근 공시된 보고서"를 사용하며, 손익 지표는 **단일분기**
기준(그 분기 3개월치 + 전년 동분기比 YoY)이다. 사업보고서 구간(4~5월)만
연간 실적을 쓴다 (사업보고서는 분기 분해가 없으므로).

수집 항목:
  - 손익계산서 : 매출, 영업이익, 순이익 (단일분기, 연간 구간은 연간)
  - 재무상태표 : 부채총계, 자본총계 → 부채비율 (분기말 스냅샷)
  - 현금흐름표 : 영업활동현금흐름 (분기 보고서는 누적치)
  - 배당       : 배당수익률 (연간 기준 — 배당은 연 단위 개념)

Look-ahead Bias 방지 (applicable_dart_period, 자본시장법 §160 제출기한):
  - 분기·반기보고서: 기간 경과 후 45일 이내 공시
  - 사업보고서: 사업연도 종료 후 90일 이내 (3/31)
  - 신호일 기준 제출기한이 지난(경계일 당일 제외) 가장 최근 보고서만 사용

금액 단위: DART 원본 그대로 저장 (원/full KRW)
  → 조원·억원 환산은 context_builders.py 에서 처리

저장 경로: data/dart_fundamentals/{ticker}.csv

실행: python src/collect_dart_fundamentals.py
"""

import os
import sys
import time
import warnings
import requests

import numpy as np
import pandas as pd
from opendartreader import OpenDartReader as odr
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))  # src/ — utils
sys.path.insert(0, _here)                       # src/collect/ — collect_financials
from utils import TICKERS, DATA_DIR
# 월별 첫 거래일 목록·적용 회계연도 결정 함수 재사용 (코드 중복 방지)
from collect_financials import applicable_fiscal_year, get_monthly_first_days, END_YM

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


# ── 계정과목명 후보 (우선순위 순) ─────────────────────────
# DART 보고서마다 계정명이 상이해 fallback 목록 사용
REVENUE_NAMES      = ["매출액", "수익(매출액)", "영업수익", "매출"]
# 영업이익: 분기 보고서는 공백·손익 변형 다수 (전 종목 스캔으로 수집)
OPER_INC_NAMES     = ["영업이익", "영업이익(손실)", "영업이익 (손실)", "영업손익"]
# 순이익: 보고서 유형(당기/반기/분기)×표기(이익/손실/손익) 변형 다수 (전 종목 스캔으로 수집)
NET_INC_NAMES      = [
    "당기순이익", "당기순이익(손실)", "당기순손익", "당기순손실", "연결당기순이익",
    "반기순이익", "반기순이익(손실)", "반기순손익", "반기순손실", "연결반기순이익",
    "분기순이익", "분기순이익(손실)", "분기순손익", "분기순손실", "연결분기순이익",
]
TOTAL_LIAB_NAMES   = ["부채총계", "부채 총계"]   # 분기 보고서는 공백 포함 "부채 총계" 사용하는 종목 있음
# 자본총계: 공백 표기 및 "OO말자본"(기말/당기말/반기말/분기말) 변형 대응
TOTAL_EQUITY_NAMES = ["자본총계", "자본 총계", "기말자본", "당기말자본", "반기말자본", "분기말자본"]
# (부채+자본) 총계 — 자본총계 라인이 없는 보고서에서 자본 폴백(총계−부채)에 사용
GRAND_TOTAL_NAMES  = ["부채와자본총계", "부채와 자본총계", "자본과부채총계", "자본 및 부채 총계",
                      "부채 및 자본총계", "부채및자본총계", "자본및부채총계"]
OPER_CF_NAMES      = ["영업활동 현금흐름", "영업활동현금흐름", "영업활동으로 인한 현금흐름",
                      "영업활동 순현금흐름", "영업활동으로 인한 순현금흐름",
                      "영업활동에서 창출된 현금흐름", "영업활동으로부터 창출된 현금흐름"]

# reprt_code → 단일분기 표시 정보. DART 정기보고서 코드:
#   11013=1분기, 11012=반기(손익 thstrm=Q2 단일), 11014=3분기, 11011=사업(연간)
REPRT_INFO = {
    "11013": {"quarter_label": "1분기", "report_name": "분기보고서"},
    "11012": {"quarter_label": "2분기", "report_name": "반기보고서"},
    "11014": {"quarter_label": "3분기", "report_name": "분기보고서"},
    "11011": {"quarter_label": "연간",  "report_name": "사업보고서"},
}


def applicable_dart_period(date: pd.Timestamp) -> tuple[int, str]:
    """date 기준 가장 최근 공시된 정기보고서의 (회계연도, reprt_code) 반환.

    제출기한(자본시장법 §160): 분기·반기 45일, 사업보고서 90일(3/31).
    경계일 당일은 미공시로 간주(제출기한 경과 후에만 사용) — 연간 로직과 동일.
    구간(연 Y 기준):
      ~ 3/31   → 전년 3분기보고서 (11014, 공시 Y-1.11.14)
      ~ 5/15   → 전년 사업보고서   (11011, 공시 Y.3.31)
      ~ 8/14   → 당해 1분기보고서  (11013, 공시 Y.5.15)
      ~ 11/14  → 당해 반기보고서   (11012, 공시 Y.8.14)
      그 외     → 당해 3분기보고서 (11014, 공시 Y.11.14)
    """
    y = date.year
    if date <= pd.Timestamp(y, 3, 31):
        return y - 1, "11014"
    if date <= pd.Timestamp(y, 5, 15):
        return y - 1, "11011"
    if date <= pd.Timestamp(y, 8, 14):
        return y, "11013"
    if date <= pd.Timestamp(y, 11, 14):
        return y, "11012"
    return y, "11014"

# ── DART 조회 캐시 ────────────────────────────────────────
_cache: dict[tuple, dict] = {}


def _get_amount(df: pd.DataFrame, names: list[str],
                col: str = "thstrm_amount") -> float:
    """계정과목명 우선순위에 따라 금액 추출. 없으면 np.nan."""
    if col not in df.columns:   # 분기 보고서는 연간과 컬럼 구성이 달라(frmtrm_q_amount 등) 방어
        return np.nan
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
    return round((curr - prev) / abs(prev) * 100, 2)  # 전년 손실(음수) 분모의 부호 오류 방지


def get_dart_annual(ticker: str, fiscal_year: int, reprt_code: str = "11011") -> dict:
    """
    DART 정기보고서에서 핵심 재무 지표 추출. (이름은 하위호환 유지 — 분기/반기도 처리)
    결과를 캐시해 동일 (ticker, fiscal_year, reprt_code) 재호출 시 API 생략.

    손익(IS)은 당기(thstrm)를 사용하되, 전기 컬럼이 보고서 유형마다 다르다:
      - 사업보고서(11011): thstrm=연간, 전기=frmtrm_amount(전년 연간)
      - 분기/반기(11013/12/14): thstrm=단일분기(그 분기 3개월),
        전기=frmtrm_q_amount(전년 동분기 3개월). frmtrm_amount는 비어있음.
    재무상태표(BS)는 분기말 스냅샷이라 thstrm_amount로 동일하게 처리.
    """
    key = (ticker, fiscal_year, reprt_code)
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

    # 손익 전기 컬럼: 연간은 전년 연간, 분기/반기는 전년 동분기(frmtrm_q_amount)
    prev_col = "frmtrm_amount" if reprt_code == "11011" else "frmtrm_q_amount"

    try:
        time.sleep(REQ_DELAY)
        df = dart.finstate_all(ticker, fiscal_year, reprt_code)
        if df is None or df.empty:
            _cache[key] = result
            return result

        # 재무제표 구분별 분리 (연결 재무제표 우선)
        is_df = df[df["sj_div"].isin(["IS", "CIS"])]   # IS(개별)/CIS(포괄) 모두 포함. finstate_all은 연결재무제표 우선 반환
        bs_df = df[df["sj_div"] == "BS"]                # 재무상태표
        cf_df = df[df["sj_div"] == "CF"]                # 현금흐름표

        # 당기(thstrm) + 전기(prev_col) 함께 추출 → YoY 계산용
        result["revenue"]               = _get_amount(is_df, REVENUE_NAMES,      "thstrm_amount")
        result["revenue_prev"]          = _get_amount(is_df, REVENUE_NAMES,      prev_col)
        result["operating_income"]      = _get_amount(is_df, OPER_INC_NAMES,     "thstrm_amount")
        result["operating_income_prev"] = _get_amount(is_df, OPER_INC_NAMES,     prev_col)
        result["net_income"]            = _get_amount(is_df, NET_INC_NAMES,      "thstrm_amount")
        result["total_liabilities"]     = _get_amount(bs_df, TOTAL_LIAB_NAMES)
        result["total_equity"]          = _get_amount(bs_df, TOTAL_EQUITY_NAMES)
        # 자본총계 라인이 없고 (부채+자본)총계만 있는 보고서 폴백: 자본 = 총계 − 부채 (회계 항등식)
        if np.isnan(result["total_equity"]) and not np.isnan(result["total_liabilities"]):
            grand = _get_amount(bs_df, GRAND_TOTAL_NAMES)
            if not np.isnan(grand):
                result["total_equity"] = grand - result["total_liabilities"]
        result["operating_cashflow"]    = _get_amount(cf_df, OPER_CF_NAMES)

    except Exception:
        pass   # 조회 실패 → NaN 유지

    _cache[key] = result
    return result


# ── 종목 처리 ──────────────────────────────────────────────

def process_ticker(name: str, ticker: str) -> pd.DataFrame | None:
    """종목별 DART 실적 수집. 기존 CSV가 있으면 누락 월만 추가(append-only)."""
    out_path = os.path.join(DART_FUND_DIR, f"{ticker}.csv")

    # 기존 날짜 셋 로드 (있으면 누락 월만 처리)
    existing_dates: set[str] = set()
    df_existing = pd.DataFrame()
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, dtype={"ticker": str})
        existing_dates = set(df_existing["date"].astype(str).tolist())

    monthly_dates = get_monthly_first_days(ticker, end_ym=END_YM)
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
        fy, reprt = applicable_dart_period(date)
        data      = get_dart_annual(ticker, fy, reprt)
        info      = REPRT_INFO[reprt]

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
            "fiscal_period":        f"{fy} {info['quarter_label']}",  # 예: "2024 2분기" / "2024 연간"
            "report_name":          info["report_name"],
            "revenue":              revenue,
            "operating_income":     oper_inc,
            "net_income":           net_inc,
            "operating_margin":     oper_margin,
            "debt_ratio":           debt_ratio,
            "operating_cashflow":   oper_cf,
            # 배당수익률은 연 단위 개념 → 손익의 분기 회계연도가 아닌 연간 기준(applicable_fiscal_year)으로 조회
            "dividend_yield":       get_dividend_yield(ticker, applicable_fiscal_year(date)),
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
    # CF·배당은 process_ticker에서 인라인으로 채움 (분기 기준). update_missing_columns는
    # 연간 기준 백필이라 분기 데이터와 불일치 → 호출하지 않음 (함수는 하위호환용으로 정의만 유지).
    print("collect_dart_fundamentals 완료")


if __name__ == "__main__":
    run()

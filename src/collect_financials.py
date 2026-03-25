"""
재무지표 수집 스크립트 (cond2용)

수집 기준: 2023-01 ~ 2025-12, 월별 첫 거래일 (36개월 × 20종목 = 최대 720행)

데이터 소스:
  - DART OpenAPI : EPS (기본주당이익), 자본총계 → PER, PBR, ROE 계산
  - FinanceDataReader : 월별 종가, 52주 고저가, 모멘텀, 거래량 변화
  - FDR StockListing : 발행주식수 (BPS 계산용)

컬럼:
  date, ticker, name,
  per, pbr, roe, market_cap, high_52w, low_52w, price_position_52w,
  momentum_1m, volume_change

  momentum_1m   : (현재가 - 21 거래일 전 종가) / 21 거래일 전 종가 × 100
  volume_change : (최근 20 거래일 평균 거래량 - 직전 20 거래일 평균) / 직전 × 100

실행: python src/collect_financials.py
"""

import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import OpenDartReader as odr
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import TICKERS, DATA_DIR

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────
# DARTS_API_KEY는 .env 파일 또는 환경변수에서 반드시 설정 필요
DART_API_KEY   = os.environ["DARTS_API_KEY"]
START_YM       = "2023-01"
END_YM         = "2025-12"
FINANCIALS_DIR = os.path.join(DATA_DIR, "financials")   # utils.DATA_DIR 기반 절대경로
REQ_DELAY      = 0.3   # DART API 요청 간 딜레이(초)
WEEKS_52       = 252   # 52주 = 252 거래일

os.makedirs(FINANCIALS_DIR, exist_ok=True)

dart = odr(api_key=DART_API_KEY)


# ── 월별 첫 거래일 목록 생성 ──────────────────────────────
def get_monthly_first_days(ticker: str, end_ym: str | None = None) -> list[pd.Timestamp]:
    """START_YM ~ end_ym(기본: END_YM) 각 월의 첫 거래일 반환."""
    _end = end_ym or END_YM
    try:
        # end_ym이 오늘 이후일 수 있으므로 충분히 넉넉한 end date 사용
        price_df = fdr.DataReader(ticker, "2022-12-01", "2027-12-31")
    except Exception:
        return []

    price_df.index = pd.to_datetime(price_df.index).tz_localize(None)

    months = pd.period_range(START_YM, _end, freq="M")
    first_days = []
    for m in months:
        month_start = m.to_timestamp()
        month_end   = (m + 1).to_timestamp() - pd.Timedelta(days=1)
        in_month = price_df.loc[(price_df.index >= month_start) &
                                (price_df.index <= month_end)]
        if not in_month.empty:
            first_days.append(in_month.index[0])
    return first_days


# ── DART 연간보고서 EPS·자본총계 캐시 ──────────────────────
_dart_cache: dict[tuple, dict] = {}

def get_dart_annual(ticker: str, fiscal_year: int) -> dict:
    """EPS, 자본총계를 DART 사업보고서에서 가져옴. 결과 캐시."""
    key = (ticker, fiscal_year)
    if key in _dart_cache:
        return _dart_cache[key]

    result = {"eps": np.nan, "equity": np.nan}
    try:
        time.sleep(REQ_DELAY)
        df = dart.finstate_all(ticker, fiscal_year, "11011")  # 사업보고서
        if df is None or df.empty:
            _dart_cache[key] = result
            return result

        # EPS: 계정과목명이 회사/연도마다 다름 → 우선순위 순서로 fallback
        # 1) 기본주당이익  2) 연속영업기본주당손익  3) 기본주당(넓은 검색)
        # 4) 보통주 기본 및 희석주당이익(손실)  5) 주당이익(가장 넓은 검색)
        eps_candidates = [
            df[df["account_nm"] == "기본주당이익"],
            df[df["account_nm"].str.contains("연속영업기본주당손익", na=False)],
            df[df["account_nm"].str.contains("기본주당이익", na=False)],
            df[df["account_nm"].str.contains("기본주당", na=False)],
            df[df["account_nm"].str.contains("주당이익", na=False)],
        ]
        eps_row = next((c for c in eps_candidates if not c.empty), pd.DataFrame())
        if not eps_row.empty:
            val = eps_row["thstrm_amount"].iloc[0]
            if val and str(val).strip() not in ("", "-", "−"):
                try:
                    result["eps"] = float(str(val).replace(",", ""))
                except ValueError:
                    pass

        # 자본총계 (연결 재무상태표, 지배기업+비지배 합계)
        bs = df[df["sj_div"] == "BS"]
        eq_row = bs[bs["account_nm"] == "자본총계"]
        if not eq_row.empty:
            val = eq_row["thstrm_amount"].iloc[0]
            if val and str(val).strip() not in ("", "-", "−"):
                result["equity"] = float(str(val).replace(",", ""))

    except Exception as e:
        pass  # 조회 실패 → NaN 유지

    _dart_cache[key] = result
    return result


# ── 적용할 회계연도 결정 (look-ahead bias 방지) ────────────
def applicable_fiscal_year(date: pd.Timestamp) -> int:
    """date 기준 가장 최근 공시된 사업보고서 회계연도 반환.

    사업보고서 공시 기한: 사업연도 종료 후 90일 이내 → 매년 3월 31일
    따라서:
      - 1~3월 → 전전년도 사업보고서 (작년 보고서 미공시)
      - 4~12월 → 전년도 사업보고서 공시 완료
    """
    cutoff = pd.Timestamp(date.year, 3, 31)
    if date <= cutoff:
        return date.year - 2
    return date.year - 1


# ── 52주 고저가 계산 ──────────────────────────────────────
def calc_52w(price_df: pd.DataFrame, date: pd.Timestamp) -> tuple:
    """date 포함 과거 252 거래일의 종가 최고·최저 반환."""
    hist = price_df.loc[price_df.index <= date].tail(WEEKS_52)
    if len(hist) < 20:            # 데이터 부족
        return np.nan, np.nan
    return float(hist["Close"].max()), float(hist["Close"].min())


# ── 종목 처리 ──────────────────────────────────────────────
def process_ticker(name: str, ticker: str, shares_map: dict) -> pd.DataFrame | None:
    out_path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")
    if os.path.exists(out_path):
        tqdm.write(f"  [{ticker}] 이미 존재 → 스킵")
        return None

    # 가격 데이터 로드 (52주 계산을 위해 2021까지)
    try:
        price_df = fdr.DataReader(ticker, "2021-01-01", "2026-01-31")
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    except Exception as e:
        tqdm.write(f"  [{ticker}] 가격 데이터 오류: {e}")
        return None

    monthly_dates = get_monthly_first_days(ticker)
    if not monthly_dates:
        tqdm.write(f"  [{ticker}] 월별 거래일 없음")
        return None

    shares = shares_map.get(ticker, np.nan)

    rows = []
    for date in tqdm(monthly_dates, desc=f"  {name}({ticker})", leave=False):
        fy = applicable_fiscal_year(date)

        # 종가
        day_data = price_df.loc[price_df.index == date]
        if day_data.empty:
            continue
        price = float(day_data["Close"].iloc[0])

        # DART 재무 데이터
        dart_data = get_dart_annual(ticker, fy)
        eps    = dart_data["eps"]
        equity = dart_data["equity"]

        # PER / PBR / ROE
        per = round(price / eps, 2)  if (not np.isnan(eps)    and eps > 0) else np.nan
        bps = equity / shares        if (not np.isnan(equity) and not np.isnan(shares)
                                         and shares > 0) else np.nan
        pbr = round(price / bps, 2)  if (not np.isnan(bps)   and bps > 0) else np.nan
        roe = round(pbr / per * 100, 2) if (not np.isnan(per) and not np.isnan(pbr)
                                             and per > 0) else np.nan

        # 시가총액
        mktcap = round(price * shares) if not np.isnan(shares) else np.nan

        # 52주 고저
        high52, low52 = calc_52w(price_df, date)
        if not np.isnan(high52) and not np.isnan(low52) and (high52 - low52) > 0:
            pos52 = round((price - low52) / (high52 - low52) * 100, 2)
        else:
            pos52 = np.nan

        rows.append({
            "date":              date.strftime("%Y-%m-%d"),
            "ticker":            str(ticker).zfill(6),
            "name":              name,
            "per":               per,
            "pbr":               pbr,
            "roe":               roe,
            "market_cap":        mktcap,
            "high_52w":          high52,
            "low_52w":           low52,
            "price_position_52w": pos52,
        })

    if not rows:
        return None

    result = pd.DataFrame(rows)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"  [{ticker}] {len(result)}행 저장 → {out_path}")
    return result


# ── 기술 지표 계산 (단일 날짜) ────────────────────────────
def calc_momentum_volume(price_df: pd.DataFrame, date: pd.Timestamp) -> tuple:
    """date 기준 momentum_1m, volume_change 반환.

    momentum_1m   : iloc[-1] vs iloc[-22] (21 거래일 전) 종가 수익률 (%)
    volume_change : iloc[-20:] vs iloc[-40:-20] 평균 거래량 변화율 (%)
    데이터 부족 시 (np.nan, np.nan) 반환.
    """
    hist = price_df.loc[price_df.index <= date]

    # momentum_1m: 최소 22개 행 필요 (현재 + 21 거래일 전)
    if len(hist) < 22:
        return np.nan, np.nan

    price_now  = float(hist["Close"].iloc[-1])
    price_prev = float(hist["Close"].iloc[-22])
    momentum   = round((price_now - price_prev) / price_prev * 100, 4) \
                 if price_prev != 0 else np.nan

    # volume_change: 최소 40개 행 필요
    if len(hist) < 40:
        return momentum, np.nan

    vol_recent = float(hist["Volume"].iloc[-20:].mean())
    vol_prior  = float(hist["Volume"].iloc[-40:-20].mean())
    vol_change = round((vol_recent - vol_prior) / vol_prior * 100, 4) \
                 if vol_prior != 0 else np.nan

    return momentum, vol_change


# ── 기존 CSV에 기술 지표 컬럼 추가 ───────────────────────
def add_technical_indicators():
    """data/financials/{ticker}.csv 에 momentum_1m, volume_change 컬럼 추가.

    이미 두 컬럼이 모두 존재하는 파일은 스킵한다.
    가격 데이터는 process_ticker() 와 동일 범위(2021-01-01 ~ 2026-01-31)로 조회.
    """
    csv_files = [f for f in os.listdir(FINANCIALS_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("  추가할 financials CSV 없음")
        return

    for fname in tqdm(csv_files, desc="기술 지표 추가"):
        out_path = os.path.join(FINANCIALS_DIR, fname)
        df = pd.read_csv(out_path, dtype={"ticker": str})

        # 이미 두 컬럼 모두 존재하면 스킵
        if "momentum_1m" in df.columns and "volume_change" in df.columns:
            tqdm.write(f"  [{fname}] 컬럼 이미 존재 → 스킵")
            continue

        ticker = fname.replace(".csv", "")

        try:
            price_df = fdr.DataReader(ticker, "2021-01-01", "2026-01-31")
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        except Exception as e:
            tqdm.write(f"  [{ticker}] 가격 데이터 오류: {e}")
            df["momentum_1m"]  = np.nan
            df["volume_change"] = np.nan
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            continue

        mom_list = []
        vol_list = []
        for date_str in df["date"]:
            date = pd.Timestamp(date_str)
            mom, vol = calc_momentum_volume(price_df, date)
            mom_list.append(mom)
            vol_list.append(vol)

        df["momentum_1m"]   = mom_list
        df["volume_change"] = vol_list
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"  [{ticker}] momentum_1m / volume_change 추가 완료")


# ── 메인 ──────────────────────────────────────────────────
def run():
    print("발행주식수 조회 중 (FDR StockListing)...")
    listing = fdr.StockListing("KRX")
    # int() 변환 전 쉼표 제거 + 소수점 절삭 (FDR이 문자열 또는 float으로 반환할 수 있음)
    shares_map = {}
    for _, row in listing.iterrows():
        if pd.isna(row.get("Stocks")):
            continue
        try:
            shares_map[row["Code"]] = int(str(row["Stocks"]).replace(",", "").split(".")[0])
        except (ValueError, TypeError):
            pass

    tickers = list(TICKERS.items())   # [(name, ticker), ...]
    all_rows = []

    for name, ticker in tqdm(tickers, desc="전체 종목"):
        result = process_ticker(name, ticker, shares_map)
        if result is not None:
            all_rows.append(result)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        total = sum(len(r) for r in all_rows)
        print(f"\n완료: {total}행 수집 ({len(all_rows)}개 종목)")
    else:
        print("\n새로 수집된 데이터 없음 (모두 스킵됨)")

    print("\n기술 지표 (momentum_1m, volume_change) 추가 중...")
    add_technical_indicators()


if __name__ == "__main__":
    run()

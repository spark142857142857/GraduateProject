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
from opendartreader import OpenDartReader as odr
from dotenv import load_dotenv
from tqdm import tqdm

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import TICKERS, DATA_DIR

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────
# DARTS_API_KEY는 .env 파일 또는 환경변수에서 반드시 설정 필요
DART_API_KEY   = os.environ.get("DARTS_API_KEY", "")
START_YM       = "2023-01"
END_YM         = "2025-12"
FINANCIALS_DIR = os.path.join(DATA_DIR, "financials")   # utils.DATA_DIR 기반 절대경로
REQ_DELAY      = 0.3   # DART API 요청 간 딜레이(초)
WEEKS_52       = 252   # 52주 = 252 거래일

os.makedirs(FINANCIALS_DIR, exist_ok=True)

try:
    dart = odr(api_key=DART_API_KEY)
except Exception:
    dart = None  # 키 없음 → get_dart_annual() 내부 try/except가 NaN 반환


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


# ── DART 조회 캐시 ────────────────────────────────────────
_dart_cache: dict[tuple, dict] = {}              # (ticker, fy) → {eps, equity} 결과
_finstate_cache: dict[tuple, pd.DataFrame] = {}  # (ticker, fy, reprt) → finstate_all 원본


def _dart_finstate(ticker: str, fiscal_year: int, reprt_code: str):
    """finstate_all 원본을 (ticker, fy, reprt)별로 캐시해 재조회 방지. 실패 시 None."""
    key = (ticker, fiscal_year, reprt_code)
    if key in _finstate_cache:
        return _finstate_cache[key]
    df = None
    if dart is not None:
        try:
            time.sleep(REQ_DELAY)
            df = dart.finstate_all(ticker, fiscal_year, reprt_code)
        except Exception:
            df = None
    _finstate_cache[key] = df
    return df


def _extract_eps(df) -> float:
    """finstate_all 결과에서 기본주당이익 thstrm_amount 추출. 계정명 fallback.

    계정과목명이 회사/연도마다 다름 → 우선순위 순서로 fallback:
      1) 기본주당이익  2) 연속영업기본주당손익  3) 기본주당이익(넓은 검색)
      4) 기본주당(손실 등 접미)  5) 주당이익(가장 넓은 검색)
    """
    if df is None or df.empty:
        return np.nan
    eps_candidates = [
        df[df["account_nm"] == "기본주당이익"],
        df[df["account_nm"].str.contains("연속영업기본주당손익", na=False)],
        df[df["account_nm"].str.contains("기본주당이익", na=False)],
        df[df["account_nm"].str.contains("기본주당", na=False)],
        df[df["account_nm"].str.contains("주당이익", na=False)],
    ]
    eps_row = next((c for c in eps_candidates if not c.empty), pd.DataFrame())
    if eps_row.empty:
        return np.nan
    val = eps_row["thstrm_amount"].iloc[0]
    if val and str(val).strip() not in ("", "-", "−"):
        try:
            return float(str(val).replace(",", ""))
        except ValueError:
            pass
    return np.nan


def get_dart_annual(ticker: str, fiscal_year: int) -> dict:
    """연간 EPS·자본총계를 DART 사업보고서(11011)에서 가져옴. 결과 캐시.

    PER은 forward/backtest 모두 get_ttm_eps(TTM)로 산출하므로, 이 함수의 eps는
    TTM 연간 구간용·fallback이고 주 용도는 자본총계(PBR용) 제공이다.
    """
    key = (ticker, fiscal_year)
    if key in _dart_cache:
        return _dart_cache[key]

    result = {"eps": np.nan, "equity": np.nan}
    try:
        df = _dart_finstate(ticker, fiscal_year, "11011")  # 사업보고서
        if df is None or df.empty:
            _dart_cache[key] = result
            return result

        result["eps"] = _extract_eps(df)

        # 자본총계 (연결 재무상태표). 계정명이 회사·업종마다 상이:
        # "자본총계" / "자본 총계"(공백) / "기말자본"(은행) 순으로 fallback
        bs = df[df["sj_div"] == "BS"]
        eq_candidates = [bs[bs["account_nm"] == en] for en in ("자본총계", "자본 총계", "기말자본")]
        eq_row = next((c for c in eq_candidates if not c.empty), pd.DataFrame())
        if not eq_row.empty:
            val = eq_row["thstrm_amount"].iloc[0]
            if val and str(val).strip() not in ("", "-", "−"):
                result["equity"] = float(str(val).replace(",", ""))

    except Exception:
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


# ── TTM(최근 4분기) EPS ────────────────────────────────────
# 단일분기 EPS 조회용 보고서 코드 (DART 분기/반기 thstrm = 단일분기 3개월)
_Q_REPRT = {1: "11013", 2: "11012", 3: "11014"}


def _single_q_eps(ticker: str, fiscal_year: int, quarter: int) -> float:
    """fiscal_year년 quarter분기(1~3)의 단일분기 기본주당이익."""
    return _extract_eps(_dart_finstate(ticker, fiscal_year, _Q_REPRT[quarter]))


def get_ttm_eps(ticker: str, date: pd.Timestamp) -> float:
    """date 시점 가장 최근 공시 기준 최근 4개 분기(TTM) 기본주당이익 합.

    시장(네이버·토스 등) PER과 정합하도록 연간 EPS 대신 TTM EPS를 사용한다.
    DART 분기/반기 thstrm은 단일분기(3개월)이므로 누적은 단일분기 합산으로 구한다.

      (fy, reprt) = applicable_dart_period(date)
      - 사업보고서(11011)      → TTM = 연간 EPS(fy)
      - 분기/반기(11013/12/14) → TTM = 연간(fy-1)
                                       + Σ(fy년 단일 Q1..q)
                                       − Σ(fy-1년 단일 Q1..q)

    한 분기라도 결측이면 np.nan 반환(연간 fallback은 호출부 판단).
    """
    # 지연 import: collect_dart_fundamentals ↔ collect_financials 순환 회피
    from collect_dart_fundamentals import applicable_dart_period

    fy, reprt = applicable_dart_period(pd.Timestamp(date))
    if reprt == "11011":                       # 사업보고서 구간 → 연간 EPS가 곧 TTM
        return get_dart_annual(ticker, fy)["eps"]

    q = {"11013": 1, "11012": 2, "11014": 3}[reprt]
    prev_annual = get_dart_annual(ticker, fy - 1)["eps"]
    if np.isnan(prev_annual):
        return np.nan

    cur_ytd = prev_ytd = 0.0
    for qq in range(1, q + 1):
        cur  = _single_q_eps(ticker, fy,     qq)
        prev = _single_q_eps(ticker, fy - 1, qq)
        if np.isnan(cur) or np.isnan(prev):
            return np.nan
        cur_ytd  += cur
        prev_ytd += prev

    return prev_annual + cur_ytd - prev_ytd


def get_per_eps(ticker: str, date: pd.Timestamp) -> float:
    """PER용 EPS: TTM 우선, 분기 XBRL 미비로 TTM 산출 불가 시 연간 EPS로 대체.

    - get_ttm_eps가 유한값(음수 포함)을 반환 → 그대로 사용
      (트레일링 적자 TTM≤0은 대체하지 않음; 호출부 eps>0 아니면 PER=NaN)
    - np.nan(분기 데이터 공백) → 최근 사업보고서 연간 EPS로 대체
      (은행 등 오래된 분기보고서가 DART finstate_all에 없는 구간 커버)
    """
    eps = get_ttm_eps(ticker, date)
    if np.isnan(eps):
        eps = get_dart_annual(ticker, applicable_fiscal_year(date))["eps"]
    return eps


# ── 52주 고저가 계산 ──────────────────────────────────────
def calc_52w(price_df: pd.DataFrame, date: pd.Timestamp) -> tuple:
    """date 포함 과거 252 거래일의 종가 최고·최저 반환."""
    hist = price_df.loc[price_df.index <= date].tail(WEEKS_52)
    if len(hist) < 20:  # 20행 미만은 신뢰 불가로 NaN 반환. 첫 신호일(2023-01) 기준 ~500거래일 확보되어 실제 미충족
        return np.nan, np.nan
    return float(hist["Close"].max()), float(hist["Close"].min())


# ── 종목 처리 ──────────────────────────────────────────────
def process_ticker(name: str, ticker: str, shares_map: dict) -> pd.DataFrame | None:
    out_path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")

    # 기존 날짜 셋 로드 (있으면 누락 월만 처리)
    existing_dates: set[str] = set()
    df_existing = pd.DataFrame()
    if os.path.exists(out_path):
        df_existing = pd.read_csv(out_path, dtype={"ticker": str})
        existing_dates = set(df_existing["date"].astype(str).tolist())

    # 가격 데이터 로드 (52주 계산을 위해 2021까지)
    try:
        price_df = fdr.DataReader(ticker, "2021-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))  # 52주 고저가(252거래일)·모멘텀 계산을 위해 실험 시작(2023-01)보다 2년 앞당겨 로드
        price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
    except Exception as e:
        tqdm.write(f"  [{ticker}] 가격 데이터 오류: {e}")
        return None

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

    shares = shares_map.get(ticker, np.nan)

    rows = []
    for date in tqdm(missing, desc=f"  {name}({ticker})", leave=False):
        fy = applicable_fiscal_year(date)

        # 종가
        day_data = price_df.loc[price_df.index == date]
        if day_data.empty:
            continue
        price = float(day_data["Close"].iloc[0])

        # DART 재무 데이터 — PER은 TTM EPS(공백 시 연간 대체), 자본(PBR)은 연간 유지
        dart_data = get_dart_annual(ticker, fy)
        eps    = get_per_eps(ticker, date)
        equity = dart_data["equity"]

        # PER / PBR / ROE
        per = round(price / eps, 2)  if (not np.isnan(eps)    and eps > 0) else np.nan
        bps = equity / shares        if (not np.isnan(equity) and not np.isnan(shares)
                                         and shares > 0) else np.nan
        pbr = round(price / bps, 2)  if (not np.isnan(bps)   and bps > 0) else np.nan
        roe = round(pbr / per * 100, 2) if (not np.isnan(per) and not np.isnan(pbr)  # ROE = PBR/PER × 100 (DuPont identity). DART에서 직접 미제공하므로 산출
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

    df_new = pd.DataFrame(rows)
    df_out = (
        pd.concat([df_existing, df_new], ignore_index=True)
        if not df_existing.empty else df_new
    )
    df_out = df_out.sort_values("date").reset_index(drop=True)
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    tqdm.write(f"  [{ticker}] +{len(rows)}행 추가 → 총 {len(df_out)}행")
    return df_new


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
    가격 데이터는 process_ticker() 와 동일 범위(2021-01-01 ~ 오늘)로 조회.
    """
    csv_files = [f for f in os.listdir(FINANCIALS_DIR) if f.endswith(".csv")]
    if not csv_files:
        print("  추가할 financials CSV 없음")
        return

    for fname in tqdm(csv_files, desc="기술 지표 추가"):
        out_path = os.path.join(FINANCIALS_DIR, fname)
        df = pd.read_csv(out_path, dtype={"ticker": str})

        # 이미 두 컬럼이 존재하고 모든 행이 채워져 있으면 스킵
        if ("momentum_1m" in df.columns and "volume_change" in df.columns
                and df["momentum_1m"].notna().all() and df["volume_change"].notna().all()):
            tqdm.write(f"  [{fname}] 모든 행 이미 채워짐 → 스킵")
            continue

        ticker = fname.replace(".csv", "")

        try:
            price_df = fdr.DataReader(ticker, "2021-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))
            price_df.index = pd.to_datetime(price_df.index).tz_localize(None)
        except Exception as e:
            tqdm.write(f"  [{ticker}] 가격 데이터 오류: {e}")
            df["momentum_1m"]  = np.nan
            df["volume_change"] = np.nan
            df.to_csv(out_path, index=False, encoding="utf-8-sig")
            continue

        mom_list = df["momentum_1m"].tolist() if "momentum_1m" in df.columns else [np.nan] * len(df)
        vol_list = df["volume_change"].tolist() if "volume_change" in df.columns else [np.nan] * len(df)

        for i, date_str in enumerate(df["date"]):
            if not pd.isna(mom_list[i]) and not pd.isna(vol_list[i]):
                continue
            date = pd.Timestamp(date_str)
            mom, vol = calc_momentum_volume(price_df, date)
            mom_list[i] = mom
            vol_list[i] = vol

        df["momentum_1m"]   = mom_list
        df["volume_change"] = vol_list
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"  [{ticker}] momentum_1m / volume_change 갱신 완료")


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
        total = sum(len(r) for r in all_rows)
        print(f"\n완료: {total}행 수집 ({len(all_rows)}개 종목)")
    else:
        print("\n새로 수집된 데이터 없음 (모두 스킵됨)")

    print("\n기술 지표 (momentum_1m, volume_change) 추가 중...")
    add_technical_indicators()


if __name__ == "__main__":
    run()

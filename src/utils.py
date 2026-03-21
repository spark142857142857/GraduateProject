"""
공통 유틸리티 함수
"""

import os
import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime

# ── 분석 설정 ─────────────────────────────────────────────
START_DATE = "2025-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")

TICKERS = {
    "삼성전자":      "005930",
    "SK하이닉스":    "000660",
    "삼성바이오로직스": "207940",
    "셀트리온":      "068270",
    "LG에너지솔루션": "373220",
    "삼성SDI":       "006400",
    "현대차":        "005380",
    "기아":          "000270",
    "KB금융":        "105560",
    "신한지주":      "055550",
    "카카오":        "035720",
    "네이버":        "035420",
    "LG화학":        "051910",
    "HYBE":          "352820",
    "HD현대중공업":  "329180",
    "한화에어로스페이스": "012450",
    "크래프톤":      "259960",
    "에코프로비엠":  "247540",
    "알테오젠":      "196170",
    "두산에너빌리티": "034020",
}

# ── 경로 ──────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
PRICE_DIR   = os.path.join(DATA_DIR, "price")
RESULTS_DIR  = os.path.join(BASE_DIR, "results")
BASELINE_DIR    = os.path.join(RESULTS_DIR, "baseline")
EXPERIMENT_DIR  = os.path.join(RESULTS_DIR, "experiment")


def get_price(ticker: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
    """
    주가 데이터 반환 (캐시 우선, 자동 업데이트).
    - 캐시 없음: 전체 구간 API 호출 후 저장
    - 캐시 있음: 마지막 날짜 이후 데이터만 추가 fetch 후 append 저장
    - API 오류 시 캐시 데이터만 반환 (캐시 없으면 빈 DataFrame)
    columns: Open, High, Low, Close, Volume, Change
    """
    cache_path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    today = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        df = df.set_index("Date")
        last_date = df.index.max().strftime("%Y-%m-%d")

        if last_date < today:
            try:
                new = fdr.DataReader(ticker, last_date, today)
                if not new.empty:
                    new["Change"] = (new["Change"] * 100).round(2)
                    new = new[~new.index.isin(df.index)]  # 중복 제거
                    if not new.empty:
                        df = pd.concat([df, new]).sort_index()
                        df.index.name = "Date"
                        df.reset_index().to_csv(cache_path, index=False)
            except Exception:
                pass  # 업데이트 실패 시 기존 캐시 데이터로 계속 진행
    else:
        try:
            df = fdr.DataReader(ticker, start, today)
            df["Change"] = (df["Change"] * 100).round(2)
            df.index.name = "Date"
            os.makedirs(PRICE_DIR, exist_ok=True)
            df.reset_index().to_csv(cache_path, index=False)
        except Exception:
            return pd.DataFrame()  # API 오류 시 빈 DataFrame 반환

    mask = (df.index >= start) & (df.index <= end)
    return df.loc[mask].copy()


def load_analyst(ticker: str, require_target_price: bool = False) -> pd.DataFrame | None:
    """
    애널리스트 리포트 CSV 로드.

    Parameters
    ----------
    ticker : str
        종목 코드 (6자리)
    require_target_price : bool
        True이면 target_price가 NaN인 행도 제거 (컨센서스 전략 등에서 사용)

    Returns
    -------
    pd.DataFrame | None
        date 기준 오름차순 정렬된 DataFrame. 파일 없으면 None.
    """
    path = os.path.join(REPORTS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    subset = ["target_price", "date"] if require_target_price else ["date"]
    df = df.dropna(subset=subset)
    return df.sort_values("date").reset_index(drop=True)


def calc_return(df: pd.DataFrame, entry_date: str, hold_days: int = 20) -> float | None:
    """
    entry_date 이후 첫 거래일 종가 기준으로 hold_days 후 수익률(%) 반환.
    데이터 부족 시 None 반환.
    """
    df = df.sort_index()
    future = df.loc[df.index > entry_date]

    if len(future) < hold_days + 1:
        return None

    price_in  = future["Close"].iloc[0]
    price_out = future["Close"].iloc[hold_days]
    return (price_out - price_in) / price_in * 100


def get_baseline_dir() -> str:
    """실행 날짜 기준 baseline 저장 폴더 반환 및 생성."""
    dated = os.path.join(BASELINE_DIR, datetime.now().strftime("%Y%m%d"))
    os.makedirs(dated, exist_ok=True)
    return dated


def get_latest_baseline_dir() -> str:
    """baseline/latest/ 폴더 반환 및 생성."""
    latest = os.path.join(BASELINE_DIR, "latest")
    os.makedirs(latest, exist_ok=True)
    return latest


def get_experiment_dir(cond: str) -> str:
    """실행 날짜 기준 experiment/{cond}/ 저장 폴더 반환 및 생성."""
    dated = os.path.join(EXPERIMENT_DIR, cond, datetime.now().strftime("%Y%m%d"))
    os.makedirs(dated, exist_ok=True)
    return dated


def get_latest_experiment_dir(cond: str) -> str:
    """experiment/{cond}/latest/ 폴더 반환 및 생성."""
    latest = os.path.join(EXPERIMENT_DIR, cond, "latest")
    os.makedirs(latest, exist_ok=True)
    return latest


def ensure_dirs():
    for d in [REPORTS_DIR, PRICE_DIR, RESULTS_DIR, BASELINE_DIR, EXPERIMENT_DIR]:
        os.makedirs(d, exist_ok=True)

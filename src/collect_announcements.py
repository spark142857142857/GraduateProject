"""
DART 공시 제목 수집 스크립트 (cond3용)

수집 기준:
  - 종목: utils.TICKERS 기준 20개
  - 날짜: data/financials/{ticker}.csv의 date 컬럼 (월별 첫 거래일, 최대 36행)
  - 각 base_date 기준 직전 30일 공시 수집

수집 공시 유형 (report_nm 텍스트 필터):
  1. 잠정실적       : "잠정" 포함  ← DART 실제명 "(잠정)실적"에 맞춰 수정
  2. 자사주 관련    : "자기주식취득결정" 또는 "자기주식처분결정" 포함
  3. 주요 이벤트    : "합병결정" 또는 "분할결정" 포함
  4. 풍문·보도 해명 : "풍문또는보도에대한해명" 포함

데이터 소스:
  - DART OpenAPI (list.json)
  - 법인코드: docs_cache/opendartreader_corp_codes_*.pkl

저장 경로: data/announcements/{ticker}.csv
  컬럼: base_date, ticker, name, report_nm, rcept_dt

실행: python src/collect_announcements.py
"""

import os
import sys
import time
import pickle
import glob
import requests
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import TICKERS, DATA_DIR

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────
DART_API_KEY       = os.environ["DARTS_API_KEY"]
API_URL            = "https://opendart.fss.or.kr/api/list.json"
WINDOW_DAYS        = 30      # base_date 기준 직전 수집 기간
REQ_DELAY          = 0.3     # API 요청 간 딜레이(초)
PAGE_COUNT         = 100     # 페이지당 최대 건수

FINANCIALS_DIR     = os.path.join(DATA_DIR, "financials")
ANNOUNCEMENTS_DIR  = os.path.join(DATA_DIR, "announcements")
DOCS_CACHE_DIR     = os.path.join(os.path.dirname(DATA_DIR), "docs_cache")

os.makedirs(ANNOUNCEMENTS_DIR, exist_ok=True)

# ── 공시 유형 필터 키워드 ─────────────────────────────────
FILTER_KEYWORDS = [
    "잠정",            # 연결/별도 재무제표기준영업(잠정)실적(공정공시) 등
    "자기주식취득결정",
    "자기주식처분결정",
    "합병결정",
    "분할결정",
    "풍문또는보도에대한해명",
]


def load_corp_codes() -> dict[str, str]:
    """
    docs_cache/ 내 opendartreader pkl 파일에서 ticker → corp_code 매핑 반환.

    Returns
    -------
    dict[str, str]
        {stock_code: corp_code} 형태의 딕셔너리
    """
    pattern = os.path.join(DOCS_CACHE_DIR, "opendartreader_corp_codes_*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"corp_code 캐시 파일을 찾을 수 없습니다: {pattern}")

    with open(files[-1], "rb") as f:
        df = pickle.load(f)

    mapping = (
        df[df["stock_code"].isin(TICKERS.values())]
        .set_index("stock_code")["corp_code"]
        .to_dict()
    )
    return mapping


def load_base_dates(ticker: str) -> list[str]:
    """
    data/financials/{ticker}.csv에서 base_date 목록 로드.

    Returns
    -------
    list[str]
        'YYYY-MM-DD' 형식 날짜 문자열 목록
    """
    path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path, usecols=["date"])
    return df["date"].dropna().tolist()


def is_target(report_nm: str) -> bool:
    """수집 대상 공시인지 report_nm 키워드로 판별."""
    return any(kw in report_nm for kw in FILTER_KEYWORDS)


def fetch_dart_page(corp_code: str, bgn_de: str, end_de: str, page_no: int) -> dict:
    """
    DART list.json API 단일 페이지 요청.

    Parameters
    ----------
    corp_code : str
        DART 법인코드 (8자리)
    bgn_de : str
        조회 시작일 (YYYYMMDD)
    end_de : str
        조회 종료일 (YYYYMMDD)
    page_no : int
        페이지 번호 (1부터)

    Returns
    -------
    dict
        API 응답 JSON
    """
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code":  corp_code,
        "bgn_de":     bgn_de,
        "end_de":     end_de,
        "page_no":    page_no,
        "page_count": PAGE_COUNT,
    }
    resp = requests.get(API_URL, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def collect_for_base_date(corp_code: str, base_date: str) -> list[dict]:
    """
    base_date 기준 직전 WINDOW_DAYS일 공시 수집 (페이지네이션 포함).

    Parameters
    ----------
    corp_code : str
        DART 법인코드
    base_date : str
        기준일 ('YYYY-MM-DD')

    Returns
    -------
    list[dict]
        수집된 공시 레코드 리스트 (report_nm, rcept_dt 포함)
    """
    end_dt   = pd.to_datetime(base_date) - timedelta(days=1)
    start_dt = end_dt - timedelta(days=WINDOW_DAYS - 1)
    bgn_de   = start_dt.strftime("%Y%m%d")
    end_de   = end_dt.strftime("%Y%m%d")

    results = []
    page_no = 1

    while True:
        time.sleep(REQ_DELAY)
        data = fetch_dart_page(corp_code, bgn_de, end_de, page_no)

        if data.get("status") != "000":
            # 조회 결과 없음(013) 또는 기타 오류
            break

        for item in data.get("list", []):
            if is_target(item.get("report_nm", "")):
                results.append({
                    "report_nm": item["report_nm"],
                    "rcept_dt":  item["rcept_dt"],
                })

        total_page = data.get("total_page", 1)
        if page_no >= int(total_page):
            break
        page_no += 1

    return results


def run():
    """전체 종목 × base_date 공시 수집 실행."""
    corp_code_map = load_corp_codes()

    name_by_ticker = {v: k for k, v in TICKERS.items()}

    for name, ticker in tqdm(TICKERS.items(), desc="공시 수집"):
        out_path = os.path.join(ANNOUNCEMENTS_DIR, f"{ticker}.csv")
        if os.path.exists(out_path):
            tqdm.write(f"  [{ticker}] 이미 존재 — 스킵")
            continue

        corp_code = corp_code_map.get(ticker)
        if not corp_code:
            tqdm.write(f"  [{ticker}] corp_code 없음 — 스킵")
            continue

        base_dates = load_base_dates(ticker)
        if not base_dates:
            tqdm.write(f"  [{ticker}] financials 파일 없음 — 스킵")
            continue

        rows = []
        for base_date in base_dates:
            records = collect_for_base_date(corp_code, base_date)
            for rec in records:
                rows.append({
                    "base_date": base_date,
                    "ticker":    ticker,
                    "name":      name,
                    "report_nm": rec["report_nm"],
                    "rcept_dt":  rec["rcept_dt"],
                })

        if not rows:
            tqdm.write(f"  [{ticker}] 수집된 공시 없음 — 파일 저장 스킵")
            continue

        df_out = pd.DataFrame(rows, columns=["base_date", "ticker", "name", "report_nm", "rcept_dt"])
        df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
        tqdm.write(f"  [{ticker}] {len(rows)}건 저장 → {out_path}")


if __name__ == "__main__":
    run()

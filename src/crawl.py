"""
애널리스트 리포트 크롤러
- 네이버 금융 종목 리포트 페이지에서 목표주가·투자의견·날짜 수집
- 결과: data/analyst/{ticker}.csv

수정사항:
  - URL 파라미터: searchType (대문자 T)
  - 날짜 파싱: YY.MM.DD → YYYY-MM-DD
  - 목표주가: 상세 페이지 em.money
  - 투자의견: 상세 페이지 em.coment
"""

import time
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from utils import TICKERS, REPORTS_DIR, START_DATE, END_DATE, ensure_dirs

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
BASE_URL = "https://finance.naver.com/research/"


def parse_date(raw: str) -> str | None:
    """'26.03.12' → '2026-03-12' 변환"""
    try:
        yy, mm, dd = raw.strip().split(".")
        year = f"20{yy}"
        return f"{year}-{mm}-{dd}"
    except Exception:
        return None


def fetch_detail(nid: str, ticker: str) -> tuple[int | None, str | None]:
    """상세 페이지에서 목표주가·투자의견 추출"""
    url = (
        f"{BASE_URL}company_read.naver"
        f"?nid={nid}&page=1&searchType=itemCode&itemCode={ticker}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        price_tag   = soup.select_one("em.money")
        opinion_tag = soup.select_one("em.coment")

        price_text   = price_tag.get_text(strip=True)   if price_tag   else None
        opinion_text = opinion_tag.get_text(strip=True) if opinion_tag else None

        if price_text in (None, "없음", ""):
            target_price = None
        else:
            target_price = int(price_text.replace(",", ""))

        if opinion_text in (None, "없음", ""):
            opinion = None
        else:
            opinion = opinion_text

        return target_price, opinion
    except Exception:
        return None, None


def fetch_reports(ticker: str, max_pages: int = 15) -> list[dict]:
    """종목별 리포트 목록 수집"""
    list_url = (
        f"{BASE_URL}company_list.naver"
        f"?searchType=itemCode&itemCode={ticker}&page={{page}}"
    )
    records = []

    for page in range(1, max_pages + 1):
        url  = list_url.format(page=page)
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.encoding = "euc-kr"
        soup = BeautifulSoup(resp.text, "html.parser")

        table = soup.select_one("table.type_1")
        if not table:
            break

        rows = [r for r in table.find_all("tr") if len(r.find_all("td")) == 6]
        if not rows:
            break

        found_in_range = False
        stop_crawl     = False

        for row in rows:
            cols      = row.find_all("td")
            date_raw  = cols[4].get_text(strip=True)
            date_str  = parse_date(date_raw)

            if date_str is None:
                continue
            if date_str > END_DATE:
                continue
            if date_str < START_DATE:
                stop_crawl = True
                break

            found_in_range = True
            title_tag = cols[1].find("a")
            if not title_tag:
                continue

            title = title_tag.get_text(strip=True)
            firm  = cols[2].get_text(strip=True)
            href  = title_tag.get("href", "")
            nid   = ""
            for part in href.split("&"):
                if "nid=" in part:
                    nid = part.split("nid=")[-1]
                    break

            target_price, opinion = fetch_detail(nid, ticker)
            time.sleep(0.3)

            records.append({
                "date":         date_str,
                "title":        title,
                "firm":         firm,
                "target_price": target_price,
                "opinion":      opinion,
                "nid":          nid,
            })

        if stop_crawl or not found_in_range:
            break
        time.sleep(0.5)

    return records


def run():
    ensure_dirs()
    for name, ticker in TICKERS.items():
        print(f"[crawl] {name} ({ticker}) ...", end=" ", flush=True)
        records  = fetch_reports(ticker)
        df       = pd.DataFrame(records)
        out_path = os.path.join(REPORTS_DIR, f"{ticker}.csv")
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        has_tp = df["target_price"].notna().sum() if not df.empty else 0
        print(f"{len(df)} rows (목표주가 {has_tp}건) → {out_path}")


if __name__ == "__main__":
    run()

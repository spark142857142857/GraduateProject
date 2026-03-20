"""
조건 2 (재무+기술지표 컨텍스트) LLM 백테스팅 — B 방식

- 종목명, 현재가, PER/PBR/ROE, 시가총액, 52주 고저가, 모멘텀, 거래량 변화를 LLM에 제공
- LLM이 직접 Buy / Sell / Neutral 판단 및 confidence 반환 (score percentile 방식 제거)
- 기준일: data/financials/{ticker}.csv의 date 컬럼 (월별 첫 거래일, 36개월)
- 20거래일 후 수익률 측정
- 결과: results/experiment/cond2/{날짜}/ + latest/
"""

import os
import sys
import json
import time
import shutil
import re

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    TICKERS, EXPERIMENT_DIR, DATA_DIR,
    get_price, calc_return,
    get_experiment_dir, get_latest_experiment_dir,
)

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────
COND          = "cond2"
MODEL         = "gemini-2.5-flash-lite"
HOLD_LONG     = 20       # 보유 기간 (거래일)
REQ_DELAY     = 0.5      # 요청 간 딜레이 (초)

FINANCIALS_DIR = os.path.join(DATA_DIR, "financials")
CKPT_PATH      = os.path.join(EXPERIMENT_DIR, COND, "checkpoint.csv")

PROMPT_TMPL = """\
당신은 주식 투자 분석가입니다.
아래 재무지표와 기술지표를 바탕으로 이 종목의 향후 20거래일 투자 방향을 판단해주세요.

[종목 정보]
종목명: {name}
현재가: {price}원

[재무지표]
PER: {per} (해당없음 시 적자기업)
PBR: {pbr}
ROE: {roe}%
시가총액: {market_cap}조원

[기술지표]
52주 최고가: {high_52w}원
52주 최저가: {low_52w}원
52주 내 현재 위치: {price_position_52w}%
최근 1개월 수익률: {momentum_1m}%
거래량 변화율: {volume_change}%

[판단 기준]
- Buy    : 향후 20거래일 내 시장 대비 상승 예상
- Sell   : 향후 20거래일 내 시장 대비 하락 예상
- Neutral: 방향성 불분명하거나 판단 근거 부족

다음 JSON 형식으로만 답변하세요. 다른 텍스트는 절대 포함하지 마세요.
{{
  "signal": "Buy" 또는 "Sell" 또는 "Neutral",
  "confidence": 0~100 사이 정수 (판단에 대한 확신도),
  "reasons": [
    "한 문장",
    "한 문장",
    "한 문장"
  ]
}}"""


# ── 헬퍼 ──────────────────────────────────────────────────

def fmt(value, decimals: int = 2, na_str: str = "N/A") -> str:
    """숫자를 문자열로 변환. NaN이면 na_str 반환."""
    if pd.isna(value):
        return na_str
    return f"{value:.{decimals}f}"


def fmt_price(value) -> str:
    """가격을 천단위 콤마 정수 문자열로 변환."""
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


def load_financials(ticker: str) -> pd.DataFrame | None:
    """
    data/financials/{ticker}.csv 로드.

    Returns
    -------
    pd.DataFrame | None
        date 컬럼 포함 DataFrame. 파일 없으면 None.
    """
    path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_checkpoint() -> pd.DataFrame:
    """
    체크포인트 CSV 로드. 없으면 빈 DataFrame 반환.
    구버전(score 컬럼) 체크포인트는 스키마 불일치로 무시하고 새로 시작.
    """
    COLS = ["ticker", "name", "signal_date", "price",
            "per", "pbr", "roe", "market_cap",
            "signal", "confidence", "reasons"]
    if os.path.exists(CKPT_PATH):
        df = pd.read_csv(CKPT_PATH, dtype={"ticker": str})
        if "signal" in df.columns and "confidence" in df.columns:
            return df
        # 구버전 체크포인트 — 무시하고 초기화
        print(f"[{COND}] 체크포인트 스키마 변경 감지 → 초기화")
    return pd.DataFrame(columns=COLS)


def save_checkpoint(df: pd.DataFrame) -> None:
    """체크포인트 저장."""
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
    df.to_csv(CKPT_PATH, index=False, encoding="utf-8-sig")


def call_llm(
    client: genai.Client,
    name: str,
    price: float,
    per, pbr, roe, market_cap,
    high_52w, low_52w, price_position_52w,
    momentum_1m, volume_change,
) -> tuple[str, int, list[str]]:
    """
    Gemini API 호출 후 signal, confidence, reasons 반환.

    Parameters
    ----------
    client : genai.Client
    name : str
        종목명
    price : float
        현재가 (base_date 다음 거래일 종가)
    per, pbr, roe, market_cap : float | NaN
        재무지표 (financials 행 값)
    high_52w, low_52w, price_position_52w : float | NaN
        52주 가격 지표
    momentum_1m, volume_change : float | NaN
        기술지표

    Returns
    -------
    tuple[str, int, list[str]]
        (signal, confidence, reasons)
        signal: "Buy" | "Sell" | "Neutral"
    """
    prompt = PROMPT_TMPL.format(
        name=name,
        price=fmt_price(price),
        per=fmt(per, 1, na_str="해당없음(적자)"),
        pbr=fmt(pbr, 2),
        roe=fmt(roe, 1),
        market_cap=fmt(market_cap / 1e12, 1) if not pd.isna(market_cap) else "N/A",
        high_52w=fmt_price(high_52w),
        low_52w=fmt_price(low_52w),
        price_position_52w=fmt(price_position_52w, 1),
        momentum_1m=fmt(momentum_1m, 2),
        volume_change=fmt(volume_change, 2),
    )

    resp = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=genai_types.GenerateContentConfig(temperature=0.3),
    )
    text = resp.text.strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 파싱 실패: {text[:200]}")
    data = json.loads(match.group())

    signal     = str(data["signal"]).strip()
    confidence = int(data["confidence"])
    reasons    = data.get("reasons", [])

    if signal not in ("Buy", "Sell", "Neutral"):
        raise ValueError(f"signal 값 오류: {signal}")

    return signal, confidence, reasons


# ── 메인 ──────────────────────────────────────────────────
def run():
    """전체 종목 × base_date LLM 실험 실행."""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ckpt_df = load_checkpoint()
    done    = set(zip(ckpt_df["ticker"].astype(str),
                      ckpt_df["signal_date"].astype(str)))

    print(f"[{COND}] 체크포인트 로드: {len(ckpt_df)}건 기처리\n")

    done_counts = (
        ckpt_df.drop_duplicates(subset=["ticker", "signal_date"])
               .groupby("ticker").size()
    )

    total = 0

    # ── LLM 호출 루프 ─────────────────────────────────────
    for name, ticker in TICKERS.items():
        fin_df = load_financials(ticker)
        if fin_df is None or fin_df.empty:
            print(f"[{COND}] {name}: financials 파일 없음, 스킵")
            continue

        if done_counts.get(ticker, 0) >= len(fin_df):
            print(f"[{COND}] {name}: 완료됨 ({len(fin_df)}건), 스킵")
            continue

        price_df = get_price(ticker)
        if price_df.empty:
            print(f"[{COND}] {name}: 주가 없음, 스킵")
            continue

        ticker_new = 0
        for _, row in fin_df.iterrows():
            sig_date = str(row["date"].date())

            if (ticker, sig_date) in done:
                continue

            # base_date 다음 거래일 종가 (Look-ahead Bias 방지)
            future = price_df.loc[price_df.index > sig_date]
            if future.empty:
                continue
            cur_price = future["Close"].iloc[0]

            # 429/503 등 일시적 오류 시 최대 3회 재시도
            # 429: retry-after 헤더값 or 최소 30초 대기 / 기타: 지수 백오프
            for attempt in range(3):
                try:
                    signal, confidence, reasons = call_llm(
                        client, name, cur_price,
                        per=row.get("per"),
                        pbr=row.get("pbr"),
                        roe=row.get("roe"),
                        market_cap=row.get("market_cap"),
                        high_52w=row.get("high_52w"),
                        low_52w=row.get("low_52w"),
                        price_position_52w=row.get("price_position_52w"),
                        momentum_1m=row.get("momentum_1m"),
                        volume_change=row.get("volume_change"),
                    )
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = any(k in err_str for k in ("429", "TooManyRequests", "ResourceExhausted", "rate limit"))
                    if is_rate_limit:
                        # retry-after 헤더 파싱 시도
                        retry_after = 30
                        import re as _re
                        m = _re.search(r"retry.after['\"]?\s*[:\s]+(\d+)", err_str, _re.IGNORECASE)
                        if m:
                            retry_after = max(30, int(m.group(1)))
                        print(f"  429 Rate Limit ({name} {sig_date}) [{attempt+1}/3] → {retry_after}s 대기")
                        time.sleep(retry_after)
                    else:
                        wait = 2 ** (attempt + 1)
                        print(f"  LLM 오류 ({name} {sig_date}) [{attempt+1}/3]: {e} → {wait}s 대기")
                        time.sleep(wait)
            else:
                print(f"  {name} {sig_date}: 3회 실패, 스킵")
                continue

            record = {
                "ticker":      ticker,
                "name":        name,
                "signal_date": sig_date,
                "price":       cur_price,
                "per":         row.get("per"),
                "pbr":         row.get("pbr"),
                "roe":         row.get("roe"),
                "market_cap":  row.get("market_cap"),
                "signal":      signal,
                "confidence":  confidence,
                "reasons":     json.dumps(reasons, ensure_ascii=False),
            }

            ckpt_df = pd.concat([ckpt_df, pd.DataFrame([record])], ignore_index=True)
            save_checkpoint(ckpt_df)
            done.add((ticker, sig_date))
            total      += 1
            ticker_new += 1

            print(f"  [{total:4d}] {name} {sig_date}  가격={int(cur_price):,}  signal={signal}  confidence={confidence}")
            time.sleep(REQ_DELAY)

        print(f"[{COND}] {name} 완료: 신규 {ticker_new}건\n")

    if ckpt_df.empty:
        print("처리된 데이터가 없습니다.")
        return

    # ── LLM 응답의 signal 직접 사용 (percentile 분류 없음) ────
    result_df = ckpt_df.copy()

    # ── 수익률 계산 ───────────────────────────────────────
    print("수익률 계산 중...")
    price_cache: dict[str, pd.DataFrame] = {}

    ret20_list = []
    for _, row in result_df.iterrows():
        tk = row["ticker"]
        if tk not in price_cache:
            price_cache[tk] = get_price(tk)
        pdf = price_cache[tk]
        ret20_list.append(calc_return(pdf, row["signal_date"], HOLD_LONG))

    result_df = result_df.copy()
    result_df["return_20d"] = ret20_list
    result_df = result_df.dropna(subset=["return_20d"]).reset_index(drop=True)

    # ── 저장 ──────────────────────────────────────────────
    out_dir    = get_experiment_dir(COND)
    latest_dir = get_latest_experiment_dir(COND)
    fname      = f"{COND}_results.csv"
    out_path   = os.path.join(out_dir, fname)

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    shutil.copy(out_path, os.path.join(latest_dir, fname))

    print(f"\n저장 완료: {out_path}")
    print(f"전체 신호: {len(result_df)}개")
    print("\n[신호별 20거래일 수익률 요약]")
    print(result_df.groupby("signal")["return_20d"].agg(
        count="count",
        mean="mean",
        hit_rate=lambda x: (x > 0).mean() * 100,
    ).round(2))

    return result_df


if __name__ == "__main__":
    run()

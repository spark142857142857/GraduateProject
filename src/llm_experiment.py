"""
LLM 백테스팅 — 통합 실험 스크립트

사용법:
  python src/llm_experiment.py --cond cond1          # No Context
  python src/llm_experiment.py --cond cond2          # 재무지표
  python src/llm_experiment.py --cond cond3          # 재무지표 + 리포트
  python src/llm_experiment.py --cond cond1 --test   # 삼성전자 1건만 테스트

실험 조합: experiments.py 참고
컨텍스트 빌더: context_builders.py 참고
저장 경로: results/experiment/{cond}/
"""

import argparse
import json
import os
import re
import shutil
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    TICKERS, EXPERIMENT_DIR,
    get_price, calc_return,
    get_experiment_dir, get_latest_experiment_dir,
)
from context_builders import build_financials, build_reports, build_dart_fundamentals
from experiments import EXPERIMENTS

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────
MODEL     = "gemini-2.5-flash-lite"
HOLD_SHORT = 5   # 5거래일
HOLD_LONG  = 20  # 20거래일
REQ_DELAY  = 0.5

BUILDER_MAP = {
    "financials":         build_financials,
    "reports":            build_reports,
    "dart_fundamentals":  build_dart_fundamentals,
}

CKPT_COLS = ["ticker", "name", "signal_date", "price", "signal", "confidence", "reasons"]


# ── 프롬프트 빌더 ──────────────────────────────────────────

def build_prompt(name: str, price: float, context_sections: list[str]) -> str:
    """종목명·현재가·컨텍스트 섹션을 조합해 LLM 프롬프트 생성."""
    intro = (
        "아래 정보를 바탕으로 이 종목의 향후 20거래일 투자 방향을 판단해주세요."
        if context_sections else
        "아래 종목의 향후 20거래일 투자 방향을 판단해주세요."
    )

    parts = [
        f"당신은 주식 투자 분석가입니다.\n{intro}",
        f"\n[종목 정보]\n종목명: {name}\n현재가: {int(price):,}원",
    ]

    for section in context_sections:
        if section:
            parts.append(f"\n{section}")

    parts.append(
        "\n[판단 기준]\n"
        "- Buy    : 향후 20거래일 내 시장 대비 상승 예상\n"
        "- Sell   : 향후 20거래일 내 시장 대비 하락 예상\n"
        "- Neutral: 방향성 불분명하거나 판단 근거 부족\n"
        "\n다음 JSON 형식으로만 답변하세요. 다른 텍스트는 절대 포함하지 마세요.\n"
        "{\n"
        '  "signal": "Buy" 또는 "Sell" 또는 "Neutral",\n'
        '  "confidence": 0~100 사이 정수 (판단에 대한 확신도),\n'
        '  "reasons": [\n'
        '    "한 문장",\n'
        '    "한 문장",\n'
        '    "한 문장"\n'
        "  ]\n"
        "}"
    )

    return "\n".join(parts)


# ── 헬퍼 ──────────────────────────────────────────────────

def load_financials_dates(ticker: str) -> pd.DataFrame | None:
    """data/financials/{ticker}.csv 로드 (date 컬럼 포함)."""
    from utils import DATA_DIR
    path = os.path.join(DATA_DIR, "financials", f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_checkpoint(cond: str) -> pd.DataFrame:
    """체크포인트 로드. 스키마 불일치 시 초기화."""
    path = os.path.join(EXPERIMENT_DIR, cond, "checkpoint.csv")
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"ticker": str})
        if "signal" in df.columns and "confidence" in df.columns:
            return df
        print(f"[{cond}] 체크포인트 스키마 변경 감지 → 초기화")
    return pd.DataFrame(columns=CKPT_COLS)


def save_checkpoint(df: pd.DataFrame, cond: str) -> None:
    path = os.path.join(EXPERIMENT_DIR, cond, "checkpoint.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def call_llm(client: genai.Client, prompt: str) -> tuple[str, int, list[str]]:
    """Gemini API 호출 후 (signal, confidence, reasons) 반환."""
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

def run(cond: str, test: bool = False):
    """전체 종목 × base_date LLM 실험 실행."""
    contexts = EXPERIMENTS[cond]
    client   = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    ckpt_df = load_checkpoint(cond)
    # 테스트 모드: 기존 체크포인트 무시하고 첫 1건만 실행
    if test:
        done        = set()
        done_counts = pd.Series(dtype=int)
    else:
        done = set(zip(ckpt_df["ticker"].astype(str), ckpt_df["signal_date"].astype(str)))
        done_counts = (
            ckpt_df.drop_duplicates(subset=["ticker", "signal_date"])
                   .groupby("ticker").size()
        )

    print(f"[{cond}] 컨텍스트: {contexts if contexts else '없음 (No Context)'}")
    print(f"[{cond}] 체크포인트 로드: {len(ckpt_df)}건 기처리\n")

    total = 0

    for name, ticker in TICKERS.items():
        if test and ticker != "005930":
            continue

        fin_df = load_financials_dates(ticker)
        if fin_df is None or fin_df.empty:
            print(f"[{cond}] {name}: financials 파일 없음, 스킵")
            continue

        if not test and done_counts.get(ticker, 0) >= len(fin_df):
            print(f"[{cond}] {name}: 완료됨 ({len(fin_df)}건), 스킵")
            continue

        price_df = get_price(ticker, start="2022-12-01")
        if price_df.empty:
            print(f"[{cond}] {name}: 주가 없음, 스킵")
            continue

        ticker_new = 0
        for _, row in fin_df.iterrows():
            sig_date = str(row["date"].date())

            if (ticker, sig_date) in done:
                continue

            future = price_df.loc[price_df.index > sig_date]
            if future.empty:
                continue
            cur_price = future["Close"].iloc[0]

            # 컨텍스트 섹션 빌드
            context_sections = [
                BUILDER_MAP[ctx](ticker, sig_date)
                for ctx in contexts
            ]

            prompt = build_prompt(name, cur_price, context_sections)

            if test:
                print("=" * 60)
                print(f"[테스트] {cond} | {name} ({ticker}) | {sig_date}")
                print("=" * 60)
                print(prompt)
                print("=" * 60)

            # LLM 호출 (최대 3회 재시도)
            for attempt in range(3):
                try:
                    signal, confidence, reasons = call_llm(client, prompt)
                    break
                except Exception as e:
                    err_str = str(e)
                    is_rate_limit = any(k in err_str for k in
                        ("429", "TooManyRequests", "ResourceExhausted", "rate limit"))
                    if is_rate_limit:
                        retry_after = 30
                        m = re.search(r"retry.after['\"]?\s*[:\s]+(\d+)", err_str, re.IGNORECASE)
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
                "signal":      signal,
                "confidence":  confidence,
                "reasons":     json.dumps(reasons, ensure_ascii=False),
            }

            ckpt_df = pd.concat([ckpt_df, pd.DataFrame([record])], ignore_index=True)
            save_checkpoint(ckpt_df, cond)
            done.add((ticker, sig_date))
            total      += 1
            ticker_new += 1

            print(f"  [{total:4d}] {name} {sig_date}  가격={int(cur_price):,}  signal={signal}  confidence={confidence}")

            if test:
                print(f"\n[테스트 완료] signal={signal}, confidence={confidence}")
                return

            time.sleep(REQ_DELAY)

        print(f"[{cond}] {name} 완료: 신규 {ticker_new}건\n")

    if ckpt_df.empty:
        print("처리된 데이터가 없습니다.")
        return

    # ── 수익률 계산 ───────────────────────────────────────
    print("수익률 계산 중...")
    price_cache: dict[str, pd.DataFrame] = {}
    ret5_list  = []
    ret20_list = []
    for _, row in ckpt_df.iterrows():
        tk = row["ticker"]
        if tk not in price_cache:
            price_cache[tk] = get_price(tk, start="2022-12-01")
        ret5_list.append(calc_return(price_cache[tk], row["signal_date"], HOLD_SHORT))
        ret20_list.append(calc_return(price_cache[tk], row["signal_date"], HOLD_LONG))

    result_df = ckpt_df.copy()
    result_df["return_5d"]  = ret5_list
    result_df["return_20d"] = ret20_list
    result_df = result_df.dropna(subset=["return_20d"]).reset_index(drop=True)

    # ── 저장 ──────────────────────────────────────────────
    out_dir    = get_experiment_dir(cond)
    latest_dir = get_latest_experiment_dir(cond)
    fname      = f"{cond}_results.csv"
    out_path   = os.path.join(out_dir, fname)

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    shutil.copy(out_path, os.path.join(latest_dir, fname))

    print(f"\n저장 완료: {out_path}")
    print(f"전체 신호: {len(result_df)}개")
    print("\n[신호별 5거래일 수익률 요약]")
    print(result_df.groupby("signal")["return_5d"].agg(
        count="count",
        mean="mean",
        hit_rate=lambda x: (x > 0).mean() * 100,
    ).round(2))
    print("\n[신호별 20거래일 수익률 요약]")
    print(result_df.groupby("signal")["return_20d"].agg(
        count="count",
        mean="mean",
        hit_rate=lambda x: (x > 0).mean() * 100,
    ).round(2))

    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM 백테스팅 실험")
    parser.add_argument(
        "--cond", default="cond1",
        choices=list(EXPERIMENTS.keys()),
        help="실험 조건 (기본값: cond1)",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="삼성전자 첫 1건만 테스트 (프롬프트 출력 + API 호출 1건)",
    )
    args = parser.parse_args()
    run(cond=args.cond, test=args.test)

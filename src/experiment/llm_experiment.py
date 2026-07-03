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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from utils import (
    TICKERS, KOSDAQ_TICKERS, EXPERIMENT_DIR, EXPERIMENT_END,
    get_price, calc_return, get_benchmark_price, calc_excess_return,
    get_experiment_dir, get_latest_experiment_dir,
)
from context_builders import build_financials, build_reports, build_dart_fundamentals
from experiments import EXPERIMENTS, BLIND_CONDITIONS
from prompt import ROLE, CONFIDENCE_GUIDE, build_criteria

load_dotenv(override=True)

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

def build_prompt(name: str, price: float, context_sections: list[str], ticker: str = "", blind: bool = False) -> str:
    """종목명·현재가·컨텍스트 섹션을 조합해 LLM 프롬프트 생성."""
    market_name = "KOSDAQ" if ticker in KOSDAQ_TICKERS else "KOSPI"

    # 종목 식별 정보를 제거해 LLM 사전학습 기반 편향을 차단하는 ablation 조건
    display_name = "종목 A" if blind else name
    market_line  = "" if blind else f"\n상장 시장: {market_name}"

    intro = (
        "아래 정보를 바탕으로 이 종목의 향후 20거래일 투자 방향을 판단해주세요."
        if context_sections else
        "아래 종목의 향후 20거래일 투자 방향을 판단해주세요."
    )

    parts = [
        f"{ROLE}\n{intro}",
        f"\n[종목 정보]\n종목명: {display_name}\n현재가: {int(price):,}원{market_line}",
    ]

    for section in context_sections:
        if section:
            parts.append(f"\n{section}")

    parts.append(
        f"\n{build_criteria()}\n"
        f"\n{CONFIDENCE_GUIDE}\n"
        "\n다음 JSON 형식으로만 답변하세요. 다른 텍스트는 절대 포함하지 마세요.\n"
        "{\n"
        '  "signal": "Buy" 또는 "Sell" 또는 "Neutral",\n'
        '  "confidence": 0~100 사이 정수 (판단에 대한 확신도),\n'
        '  "reasons": [\n'
        '    "제공된 데이터를 근거로 든 한 문장",\n'
        '    "제공된 데이터를 근거로 든 한 문장",\n'
        '    "제공된 데이터를 근거로 든 한 문장"\n'
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
            return df.drop_duplicates(subset=["ticker", "signal_date"]).reset_index(drop=True)
        print(f"[{cond}] 체크포인트 스키마 변경 감지 → 초기화")
    return pd.DataFrame(columns=CKPT_COLS)


def save_checkpoint(df: pd.DataFrame, cond: str) -> None:
    path = os.path.join(EXPERIMENT_DIR, cond, "checkpoint.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ── LLM 클라이언트 (provider별 lazy 캐시 — 쓰는 provider의 키만 필요) ──
_clients: dict[str, object] = {}

def _gemini_client():
    if "gemini" not in _clients:
        _clients["gemini"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _clients["gemini"]

def _openai_client():
    if "openai" not in _clients:
        from openai import OpenAI
        _clients["openai"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _clients["openai"]

def _anthropic_client():
    if "anthropic" not in _clients:
        from anthropic import Anthropic
        _clients["anthropic"] = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _clients["anthropic"]


def _raw_text(prompt: str, model: str) -> str:
    """model 접두어로 provider 분기해 프롬프트 → 원문 응답 텍스트 (temperature=0.0)."""
    if model.startswith(("gemini", "gemma")):
        resp = _gemini_client().models.generate_content(
            model=model, contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.0),
        )
        return resp.text or ""
    if model.startswith("gpt"):
        resp = _openai_client().chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""
    if model.startswith("claude"):
        resp = _anthropic_client().messages.create(
            model=model, max_tokens=1024, temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text or ""
    raise ValueError(f"provider 미매핑 모델: {model}")


def call_llm(prompt: str, model: str = MODEL) -> tuple[str, int, list[str]]:
    """provider 분기 후 (signal, confidence, reasons) 반환. JSON 파싱·검증은 공용."""
    text = _raw_text(prompt, model).strip()

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        raise ValueError(f"JSON 파싱 실패: {text[:200]}")
    data = json.loads(match.group())

    signal     = str(data["signal"]).strip()
    confidence = max(0, min(100, int(data["confidence"])))
    reasons    = data.get("reasons") or []  # None 반환 시 빈 리스트로 대체

    if signal not in ("Buy", "Sell", "Neutral"):
        raise ValueError(f"signal 값 오류: {signal}")

    return signal, confidence, reasons


# ── 메인 ──────────────────────────────────────────────────

def run(cond: str, test: bool = False, model: str = MODEL):
    """전체 종목 × base_date LLM 실험 실행."""
    contexts = EXPERIMENTS[cond]

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

    is_blind = cond in BLIND_CONDITIONS  # 종목 식별 정보 익명화 여부 (experiments.py의 BLIND_CONDITIONS 참조)
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

        price_df = get_price(ticker, start="2022-12-01")  # 2022-12-01부터 로드: 52주 고저가·모멘텀 계산에 실험 시작일(2023-01) 이전 데이터 필요
        if price_df.empty:
            print(f"[{cond}] {name}: 주가 없음, 스킵")
            continue

        ticker_new = 0
        for _, row in fin_df.iterrows():
            sig_date = str(row["date"].date())

            if sig_date > EXPERIMENT_END:
                continue  # 실험 기간(2023-01~2025-12) 밖 행은 백테스팅 대상에서 제외

            if (ticker, sig_date) in done:
                continue

            if price_df.loc[price_df.index > sig_date].empty:
                continue

            past = price_df.loc[price_df.index <= sig_date]
            if past.empty:
                continue
            cur_price = past["Close"].iloc[-1]  # 신호일 당일 종가 (재무지표 기준 시점과 일치)

            # 컨텍스트 섹션 빌드
            context_sections = [
                BUILDER_MAP[ctx](ticker, sig_date)
                for ctx in contexts
            ]

            prompt = build_prompt(name, cur_price, context_sections, ticker=ticker, blind=is_blind)

            if test:
                print("=" * 60)
                print(f"[테스트] {cond} | {name} ({ticker}) | {sig_date}")
                print("=" * 60)
                print(prompt)
                print("=" * 60)

            # LLM 호출 (최대 3회 재시도)
            for attempt in range(3):
                try:
                    signal, confidence, reasons = call_llm(prompt, model)
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

            if test:
                print(f"\n[테스트 완료] signal={signal}, confidence={confidence}")
                return  # 테스트 모드는 실제 체크포인트를 오염시키지 않도록 저장 전에 종료

            ckpt_df = pd.concat([ckpt_df, pd.DataFrame([record])], ignore_index=True)
            save_checkpoint(ckpt_df, cond)
            done.add((ticker, sig_date))
            total      += 1
            ticker_new += 1

            print(f"  [{total:4d}] {name} {sig_date}  가격={int(cur_price):,}  signal={signal}  confidence={confidence}")

            time.sleep(REQ_DELAY)

        print(f"[{cond}] {name} 완료: 신규 {ticker_new}건\n")

    if ckpt_df.empty:
        print("처리된 데이터가 없습니다.")
        return

    # ── 수익률 계산 ───────────────────────────────────────
    print("수익률 계산 중...")
    price_cache: dict[str, pd.DataFrame] = {}
    bench_cache: dict[str, pd.DataFrame] = {}
    ret5_list     = []
    ret20_list    = []
    excess5_list  = []
    excess20_list = []
    for _, row in ckpt_df.iterrows():
        tk = row["ticker"]
        if tk not in price_cache:
            price_cache[tk] = get_price(tk, start="2022-12-01")  # 2022-12-01부터 로드: 52주 고저가·모멘텀 계산에 실험 시작일(2023-01) 이전 데이터 필요
        if tk not in bench_cache:
            bench_cache[tk] = get_benchmark_price(tk, start="2022-12-01")
        stock_df = price_cache[tk]
        bench_df = bench_cache[tk]
        sig_date = row["signal_date"]
        ret5_list.append(calc_return(stock_df, sig_date, HOLD_SHORT))
        ret20_list.append(calc_return(stock_df, sig_date, HOLD_LONG))
        excess5_list.append(calc_excess_return(stock_df, bench_df, sig_date, HOLD_SHORT))
        excess20_list.append(calc_excess_return(stock_df, bench_df, sig_date, HOLD_LONG))

    result_df = ckpt_df.copy()
    result_df["return_5d"]         = ret5_list
    result_df["return_20d"]        = ret20_list
    result_df["excess_return_5d"]  = excess5_list
    result_df["excess_return_20d"] = excess20_list
    result_df = result_df.dropna(subset=["return_20d"]).reset_index(drop=True)  # 백테스팅 말미(2025-12 등) 신호는 20거래일 매도가 미확정이므로 자동 제외

    # ── 저장 ──────────────────────────────────────────────
    out_dir    = get_experiment_dir(cond)
    latest_dir = get_latest_experiment_dir(cond)
    fname      = f"{cond}_results.csv"
    out_path   = os.path.join(out_dir, fname)

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    shutil.copy(out_path, os.path.join(latest_dir, fname))

    print(f"\n저장 완료: {out_path}")
    print(f"전체 신호: {len(result_df)}개")
    def _summary(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """신호별 수익률 요약. Sell은 (< 0) 기준 히트율."""
        stats = df.groupby("signal")[col].agg(count="count", mean="mean").round(2)
        for sig, g in df.groupby("signal")[col]:
            hr = (g < 0).mean() if sig == "Sell" else (g > 0).mean()
            stats.loc[sig, "hit_rate"] = round(hr * 100, 2)
        return stats

    print("\n[신호별 5거래일 수익률 요약 (절대)]")
    print(_summary(result_df, "return_5d"))
    print("\n[신호별 5거래일 초과수익률 요약 (vs 시장)]")
    print(_summary(result_df, "excess_return_5d"))
    print("\n[신호별 20거래일 수익률 요약 (절대)]")
    print(_summary(result_df, "return_20d"))
    print("\n[신호별 20거래일 초과수익률 요약 (vs 시장)]")
    print(_summary(result_df, "excess_return_20d"))

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
    parser.add_argument(
        "--model", default=MODEL,
        help=f"사용 모델 (기본값: {MODEL}). gemini-*/gemma-*/gpt-*/claude-* 접두어로 provider 분기",
    )
    args = parser.parse_args()
    run(cond=args.cond, test=args.test, model=args.model)

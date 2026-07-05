"""
Forward Test — 오늘 날짜 기준 단일 종목 LLM 신호 생성

사용법:
  python src/forward_test.py --ticker 005930
  python src/forward_test.py --ticker 005930 --cond cond3

결과 저장: results/forward/{날짜}/{model}/{ticker}_{cond}.json
같은 날 · 같은 종목 · 같은 cond · 같은 model이면 캐시 반환.

설계 원칙:
  - get_today_context()로 오늘 기준 실시간 지표 수집
  - data/financials/ 파일 비오염 (백테스팅 데이터 보존)
  - context_date = 오늘 날짜 (월 첫 거래일 아님)
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

from dotenv import load_dotenv

_here = os.path.dirname(os.path.abspath(__file__))
_src  = os.path.join(_here, "..")
sys.path.insert(0, os.path.join(_src, "collect"))  # src/collect/ — update (지연 import)
sys.path.insert(0, _here)                           # src/experiment/ — llm_experiment
sys.path.insert(0, _src)                            # src/ — utils, context_builders, experiments

from utils import TICKERS, FORWARD_DIR
from context_builders import (
    build_financials_from_dict,
    build_reports_from_dict,
    build_dart_fundamentals_from_dict,
)
from experiments import EXPERIMENTS, BLIND_CONDITIONS
from llm_experiment import build_prompt, call_llm, preflight, MODEL

load_dotenv()

# dict 기반 빌더 맵 (forward_test 전용)
FORWARD_BUILDER_MAP = {
    "financials":        build_financials_from_dict,
    "reports":           build_reports_from_dict,
    "dart_fundamentals": build_dart_fundamentals_from_dict,
}


# ── 메인 ──────────────────────────────────────────────────
def run_forward(ticker: str, cond: str = "cond4", model: str = MODEL) -> dict:
    """오늘 날짜 기준 단일 종목 LLM 신호 생성.

    1. 캐시 확인 (당일 동일 ticker+cond+model → 즉시 반환)
    2. get_today_context(ticker) → 실시간 지표 수집 (파일 비저장)
    3. dict 기반 context_builders로 프롬프트 생성
    4. LLM 호출 (call_llm — model 접두어로 provider 분기)
    5. JSON 저장 및 반환

    Returns:
        {
          "ticker", "name", "date", "price",
          "signal", "confidence", "reasons",
          "cond", "model", "context_used"
        }
    """
    today_str  = datetime.today().strftime("%Y-%m-%d")
    cache_dir  = os.path.join(FORWARD_DIR, today_str, model)
    cache_path = os.path.join(cache_dir, f"{ticker}_{cond}.json")  # 당일 동일 ticker+cond+model 캐시 반환 — 중복 호출 방지

    # 1. 캐시
    if os.path.exists(cache_path):
        print(f"[{ticker}] 캐시 사용: {cache_path}")
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    preflight(model)  # 캐시 미스 → 비싼 데이터 수집 전에 모델·키 검증

    # 2. 실시간 지표 수집
    print(f"[{ticker}] 실시간 데이터 수집 중...")
    from update import get_today_context  # 함수 내 지연 import — update.py와의 순환 의존성 방지
    ctx = get_today_context(ticker)

    name          = ctx["name"]
    current_price = ctx["price"]

    # 3. 프롬프트 생성 (dict 기반 빌더)
    context_sections = [
        FORWARD_BUILDER_MAP[key](ctx)
        for key in EXPERIMENTS[cond]
    ]
    is_blind = cond in BLIND_CONDITIONS
    prompt = build_prompt(name, current_price, context_sections, ticker=ticker, blind=is_blind)

    # 4. LLM 호출 (최대 3회 재시도)
    print(f"[{ticker}] LLM 호출 중 (cond={cond}, model={model})...")

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
                print(f"  429 Rate Limit [{attempt+1}/3] -> {retry_after}s 대기")
                time.sleep(retry_after)
            else:
                wait = 2 ** (attempt + 1)
                print(f"  LLM 오류 [{attempt + 1}/3]: {e} -> {wait}s 대기")
                time.sleep(wait)
    else:
        raise RuntimeError(f"[{ticker}] LLM 3회 호출 실패")

    # context_used: get_today_context 반환값에서 UI 표시용 필드만 추출
    context_used = {
        "per":               ctx.get("per"),
        "pbr":               ctx.get("pbr"),
        "roe":               ctx.get("roe"),
        "market_cap":        round(ctx["market_cap"] / 1e12, 1) if ctx.get("market_cap") else None,
        "momentum_1m":       ctx.get("momentum_1m"),
        "volume_change":     ctx.get("volume_change"),
        "price_position_52w": ctx.get("price_position_52w"),
        "revenue_growth":    ctx.get("revenue_yoy"),
        "operating_margin":  ctx.get("operating_margin"),
        "debt_ratio":        ctx.get("debt_ratio"),
        "dividend_yield":    ctx.get("dividend_yield"),
        "recent_reports":    ctx.get("recent_reports", []),
    }

    result = {
        "ticker":       ticker,
        "name":         name,
        "date":         today_str,
        "context_date": today_str,   # 백테스팅은 월 첫 거래일, forward test는 오늘 날짜 기준 — 의도적 차이
        "price":        current_price,
        "signal":       signal,
        "confidence":   confidence,
        "reasons":      reasons,
        "cond":         cond,
        "model":        model,
        "context_used": context_used,
    }

    # 5. 저장
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[{ticker}] 결과 저장: {cache_path}")

    return result


# ── CLI ───────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forward Test - 오늘 날짜 LLM 신호 생성")
    parser.add_argument("--ticker", type=str, required=True, help="종목 코드 (예: 005930)")
    parser.add_argument(
        "--cond", type=str, default="cond4",
        choices=list(EXPERIMENTS.keys()),
        help="분석 조건 (기본값: cond4)",
    )
    parser.add_argument("--model", type=str, default=MODEL, help=f"사용 모델 (기본값: {MODEL})")
    args = parser.parse_args()

    result = run_forward(args.ticker, args.cond, args.model)
    print(f"\n{'=' * 50}")
    print(f"종목: {result['name']} ({result['ticker']})")
    print(f"날짜: {result['date']}  현재가: {int(result['price']):,}원")
    print(f"신호: {result['signal']}  신뢰도: {result['confidence']}%")
    print("근거:")
    for r in result["reasons"]:
        print(f"  - {r}")

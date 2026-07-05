"""
Forward Test 일괄 실행 — 전 종목 × 메인 5조건

매주 (일요일 저녁) 실행. 20종목 각각에 대해 cond1~4 + cond4_no_reports
forward 신호를 생성한다. forward_test.run_forward를 반복 호출하며,
당일·동일 ticker·동일 cond는 캐시 반환이라 중단 후 재실행해도 안전(멱등).

기본 모델은 llm_experiment.MODEL (백테스트와 동일 = gemini-2.5-flash-lite).
--model로 다른 모델 지정 가능 (call_llm이 접두어로 provider 분기).

사용법:
  python src/experiment/forward_run_all.py                    # 20종목 × 5조건 (앵커)
  python src/experiment/forward_run_all.py --cond cond4       # 단일 조건만
  python src/experiment/forward_run_all.py --model claude-haiku-4-5   # 다른 모델
저장: results/forward/{오늘}/{model}/{ticker}_{cond}.json (run_forward가 처리)
"""

import argparse
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, ".."))  # src/ — utils
sys.path.insert(0, _here)                        # src/experiment — forward_test

from utils import TICKERS
from forward_test import run_forward
from llm_experiment import MODEL, preflight

MAIN_CONDS = ["cond1", "cond2", "cond3", "cond4", "cond4_no_reports"]
REQ_DELAY  = 0.3


def run_all(conds: list[str], model: str = MODEL) -> None:
    preflight(model)  # 루프 밖 1회 — 모델·키 오류를 100콜 전에 즉시 (per-combo try/except 회피)
    total = len(TICKERS) * len(conds)
    print(f"Forward 일괄 실행 | 모델={model} | {len(TICKERS)}종목 × {len(conds)}조건 = {total}건\n")

    done = 0
    fails = []
    for name, ticker in TICKERS.items():
        for cond in conds:
            done += 1
            try:
                r = run_forward(ticker, cond, model)
                print(f"[{done:3d}/{total}] {name:<12} {cond:<16} → {r['signal']} ({r['confidence']}%)")
            except Exception as e:
                fails.append((name, cond, str(e)))
                print(f"[{done:3d}/{total}] {name:<12} {cond:<16} → 실패: {e}")
            time.sleep(REQ_DELAY)

    print(f"\n완료: {total - len(fails)}/{total} 성공")
    if fails:
        print("실패 목록:")
        for nm, c, e in fails:
            print(f"  {nm} {c}: {e[:80]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="전 종목 forward test 일괄 실행")
    parser.add_argument(
        "--cond", choices=MAIN_CONDS,
        help="단일 조건만 실행 (기본값: 메인 5조건 전부)",
    )
    parser.add_argument("--model", default=MODEL, help=f"사용 모델 (기본값: {MODEL})")
    args = parser.parse_args()
    run_all([args.cond] if args.cond else MAIN_CONDS, args.model)

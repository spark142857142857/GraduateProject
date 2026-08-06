"""탭 공용 상수와 로더.

여기 두는 기준은 "둘 이상의 탭이 쓰는가"다. 한 탭만 쓰는 헬퍼는 해당 탭 모듈에
둔다(예: signal_badge는 tab_analyze, signal_cell_style은 tab_matrix).
"""

import os

import pandas as pd
import streamlit as st

from utils import TICKERS, EXPERIMENT_DIR, FORWARD_DIR, ANALYSIS_DIR
from compare import COND_LABELS, DEFAULT_MODEL

__all__ = [
    "TICKERS", "EXPERIMENT_DIR", "FORWARD_DIR", "ANALYSIS_DIR",
    "COND_LABELS", "DEFAULT_MODEL",
    "UI_CONDS", "REPORT_CONDS", "SMALL_SAMPLE_N", "UI_MODELS",
    "SIGNAL_STYLE", "FORWARD_DEMO_DIR",
    "load_backtest_results", "list_backtest_models", "fmt_metric",
]


# ── 상수 ──────────────────────────────────────────────────
# 연구용 조건(cond4_no_reports, cond4_blind)은 개별 분석 UI 미노출 — 사용자 편의 조건만 표시
UI_CONDS = ["cond1", "cond2", "cond3", "cond4"]

# 화면에 노출하는 조건 — 보고서가 다루는 5개로 한정한다.
# COND_ORDER(= EXPERIMENTS 전체)를 그대로 쓰면 보조 실험(reports_only·dart_only·cond4_blind)을
# 실행한 뒤부터 결과가 화면에 섞여 나온다. 미완료 조건이 성과표에 뜨면 설명 부담만 생긴다.
REPORT_CONDS = ["cond1", "cond2", "cond3", "cond4", "cond4_no_reports"]

# 조건별 Buy 표본이 이 값 미만이면 평균이 크게 흔들려 성능으로 읽으면 안 된다 (cond1이 대표적)
SMALL_SAMPLE_N = 30

# 개별 분석용 모델 목록 — 앵커(DEFAULT_MODEL)/gemma가 우선(앞 배치), gpt/claude는 별도 키·호출 비용 필요
UI_MODELS = [DEFAULT_MODEL, "gemma-4-31b-it", "gpt-5.4-mini", "claude-haiku-4-5"]

SIGNAL_STYLE = {
    "Buy":     ("초록", "#d4edda", "#155724"),
    "Sell":    ("빨강", "#f8d7da", "#721c24"),
    "Neutral": ("회색", "#e2e3e5", "#383d41"),
}

# 앱 시연으로 생성된 신호의 격리 경로 — 정식 주간 배치(results/forward/)와 분리한다.
# forward_eval.py가 results/forward/*/*/*.json만 훑으므로, 형제 폴더에 두면
# 평가 표본에서 자동 제외된다(임의 시점·임의 종목 클릭이 통계에 섞이는 것을 차단).
FORWARD_DEMO_DIR = os.path.join(os.path.dirname(FORWARD_DIR), "forward_demo")


# ── 공용 로더 ─────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)  # 백테스팅 결과 5분 캐시 — 빈번한 파일 재로드 방지
def load_backtest_results(cond: str, model: str) -> pd.DataFrame | None:
    """results/experiment/{cond}/{model}/latest/{cond}_results.csv 로드."""
    path = os.path.join(EXPERIMENT_DIR, cond, model, "latest", f"{cond}_results.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, dtype={"ticker": str})
    except Exception:
        return None


def list_backtest_models(cond: str) -> list[str]:
    """해당 조건의 백테스트 결과가 실제로 존재하는 모델 목록.

    forward는 UI_MODELS 전 모델로 신호를 만들 수 있으나 백테스트는 일부만
    완료된 상태라, "결과 없음" 안내에서 어떤 모델이 가능한지 함께 제시한다.
    """
    base = os.path.join(EXPERIMENT_DIR, cond)
    if not os.path.isdir(base):
        return []
    return sorted(
        m for m in os.listdir(base)
        if os.path.exists(os.path.join(base, m, "latest", f"{cond}_results.csv"))
    )


def fmt_metric(v, decimals: int = 2, signed: bool = False) -> str:
    """표 셀 포맷. 결측(n=1의 Sharpe 등)은 'nan'이 아니라 '-'로 낸다."""
    if v is None or pd.isna(v):
        return "-"
    return f"{v:+.{decimals}f}" if signed else f"{v:.{decimals}f}"

"""탭 공용 상수와 로더.

여기 두는 기준은 "둘 이상의 탭이 쓰는가"다. 한 탭만 쓰는 헬퍼는 해당 탭 모듈에
둔다(예: signal_badge는 tab_analyze, signal_cell_style은 tab_matrix).
"""

import os

import pandas as pd
import streamlit as st

from utils import TICKERS, EXPERIMENT_DIR, FORWARD_DIR, ANALYSIS_DIR, REPORTS_DIR
from compare import COND_LABELS, DEFAULT_MODEL

__all__ = [
    "TICKERS", "EXPERIMENT_DIR", "FORWARD_DIR", "ANALYSIS_DIR", "REPORTS_DIR",
    "COND_LABELS", "DEFAULT_MODEL",
    "UI_CONDS", "REPORT_CONDS", "SMALL_SAMPLE_N", "UI_MODELS",
    "SIGNAL_STYLE", "FORWARD_DEMO_DIR",
    "load_backtest_results", "list_backtest_models", "fmt_metric",
    "list_matrix_models", "load_signal_matrix",
]


# ── 상수 ──────────────────────────────────────────────────
# 연구용 조건(cond4_no_reports)은 개별 분석 UI 미노출 — 사용자 편의 조건만 표시
UI_CONDS = ["cond1", "cond2", "cond3", "cond4"]

# 화면에 노출하는 조건 — 보고서가 다루는 5개로 한정한다.
# COND_ORDER(= EXPERIMENTS 전체)를 그대로 쓰면 보조 실험(reports_only·dart_only)을
# 실행한 뒤부터 결과가 화면에 섞여 나온다. 미완료 조건이 성과표에 뜨면 설명 부담만 생긴다.
REPORT_CONDS = ["cond1", "cond2", "cond3", "cond4", "cond4_no_reports"]

# 조건별 Buy 표본이 이 값 미만이면 평균이 크게 흔들려 성능으로 읽으면 안 된다 (cond1이 대표적)
SMALL_SAMPLE_N = 30

# 개별 분석용 모델 목록 — 앵커(DEFAULT_MODEL)/gemma가 우선(앞 배치), gpt/claude는 별도 키·호출 비용 필요
UI_MODELS = [DEFAULT_MODEL, "gemma-4-31b-it", "gpt-5.4-mini", "claude-haiku-4-5"]

# 신호별 (배경색, 글자색) — 배지·매트릭스 셀·범례·산점도 색이 모두 여기서 나온다
SIGNAL_STYLE = {
    "Buy":     ("#d4edda", "#155724"),
    "Sell":    ("#f8d7da", "#721c24"),
    "Neutral": ("#e2e3e5", "#383d41"),
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

    폴더가 아니라 결과 CSV의 존재로 판정한다. 실행이 중단돼 빈 모델 폴더만 남은
    경우 셀렉트박스에 뜨면 안 되기 때문.
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


def list_matrix_models() -> list[str]:
    """REPORT_CONDS 중 하나라도 백테스트 결과가 있는 모델 목록."""
    models: set[str] = set()
    for cond in REPORT_CONDS:
        models.update(list_backtest_models(cond))
    return sorted(models)


@st.cache_data(ttl=300, show_spinner=False)
def load_signal_matrix(model: str) -> pd.DataFrame:
    """조건별 백테스트 결과를 long DataFrame으로 합친다.

    탭1(신호 매트릭스)과 탭4(포트폴리오)가 공유하는 소스. forward 캐시를 쓰지 않는
    이유는 ① 신호 생성을 2026-08-02로 종료해 시간이 갈수록 낡은 날짜가 화면에 남고
    ② forward를 앱에서 다루지 않기로 한 결정(TODO "미채택 — forward 성과 탭")과
    어긋나기 때문. 백테스트 기간(2023-01~2025-12)은 설계상 고정이라 낡지 않고,
    주력 근거를 원자료 수준에서 보여준다는 이점도 있다.
    """
    frames = []
    for cond in REPORT_CONDS:
        df = load_backtest_results(cond, model)
        if df is None or df.empty:
            continue
        keep = [c for c in ("ticker", "name", "signal_date", "signal", "confidence", "return_20d")
                if c in df.columns]
        d = df[keep].copy()
        d["cond"] = cond
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # 저장 형식이 int일 수 있어 zero-pad로 통일 (get_ticker_backtest와 같은 이유)
    out["ticker"] = out["ticker"].astype(str).str.zfill(6)
    return out

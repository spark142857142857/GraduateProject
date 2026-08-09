"""탭 공용 상수와 로더.

여기 두는 기준은 "둘 이상의 탭이 쓰는가"다. 한 탭만 쓰는 헬퍼는 해당 탭 모듈에
둔다(예: signal_badge는 tab_analyze, signal_cell_style은 tab_matrix).
"""

import glob
import os
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

from app_ui import ROOT_DIR
from utils import TICKERS, EXPERIMENT_DIR, FORWARD_DIR, ANALYSIS_DIR, REPORTS_DIR
from compare import COND_LABELS, DEFAULT_MODEL

__all__ = [
    "TICKERS", "EXPERIMENT_DIR", "FORWARD_DIR", "ANALYSIS_DIR", "REPORTS_DIR",
    "COND_LABELS", "DEFAULT_MODEL", "BACKTEST_TICKERS",
    "UI_CONDS", "REPORT_CONDS", "SMALL_SAMPLE_N", "UI_MODELS",
    "SIGNAL_STYLE", "FORWARD_DEMO_DIR",
    "load_backtest_results", "list_backtest_models", "fmt_metric",
    "list_matrix_models", "load_signal_matrix",
    "load_krx_stocks", "ensure_reports", "check_dart_cache",
]


# ── 상수 ──────────────────────────────────────────────────
# 백테스트 대상 20종목의 티커 — import 시점에 고정한다.
# tab_analyze가 20종목 밖을 분석할 때 종목명을 TICKERS에 주입하므로(그 이름이 LLM
# 프롬프트에 들어가야 한다) TICKERS는 실행 중에 커진다. "실험 대상 종목인가" 판정에
# TICKERS를 그대로 쓰면 한 번 분석한 종목이 20종목처럼 취급돼 버린다.
BACKTEST_TICKERS = frozenset(TICKERS.values())

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


# ── 실시간 수집 공용 (개별 분석 · 종목 데이터 조회) ───────
@st.cache_resource
def check_dart_cache() -> str:
    """DART corp_codes pkl 캐시 유효성 점검.

    오늘 날짜 캐시가 없거나 읽기 실패 시 구 캐시를 삭제하고
    OpenDartReader 초기화로 재생성(법인코드 약 11MB 다운로드).

    호출 시점 주의: 앱 시작 시가 아니라 실제 수집이 필요한 버튼에서 호출한다.
    날짜가 바뀐 첫 실행이면 재생성에 수십 초가 걸리는데, DART가 필요 없는
    탭(캐시 읽기 전용)까지 그 대기에 묶이기 때문.
    @st.cache_resource라 프로세스당 1회만 수행된다.

    Returns:
        "" : 정상
        str: 오류 메시지 (재생성 실패 시)
    """
    docs_cache = os.path.join(ROOT_DIR, "docs_cache")
    today_fn = os.path.join(
        docs_cache,
        f"opendartreader_corp_codes_{datetime.today().strftime('%Y%m%d')}.pkl",
    )

    # 오늘 날짜 캐시가 있으면 읽기 테스트
    if os.path.exists(today_fn):
        try:
            pd.read_pickle(today_fn)
            return ""  # 정상
        except Exception:
            pass  # 호환 불가 → 아래에서 삭제 후 재생성

    # 구 캐시(오늘 것 포함) 전체 삭제
    for old in glob.glob(os.path.join(docs_cache, "opendartreader_corp_codes_*.pkl")):
        try:
            os.remove(old)
        except OSError:
            pass

    # OpenDartReader 재초기화 → 캐시 자동 재생성
    try:
        from opendartreader import OpenDartReader as _odr
        dart_key = os.environ.get("DARTS_API_KEY", "")
        if not dart_key:
            return "DARTS_API_KEY 환경변수가 설정되지 않았습니다."
        _odr(api_key=dart_key)
        return ""
    except Exception as e:
        return f"DART 캐시 재생성 실패: {e}"


@st.cache_data(ttl=3600, show_spinner=False)  # 상장 목록은 하루 단위로만 바뀐다
def load_krx_stocks() -> list[tuple[str, str]]:
    """KRX 상장 종목 (표시 라벨, 티커) 목록. 시가총액 내림차순.

    시총 순으로 정렬하는 이유는 셀렉트박스에서 검색 없이 훑을 때 아는 이름이
    먼저 나와야 하기 때문. 라벨에 티커를 붙여 동명 종목이 겹치지 않게 한다.

    우선주는 제외한다. 우선주는 별도 종목코드를 갖지만 DART 재무제표는 보통주 기준
    하나뿐이라 EPS가 매칭되지 않아 PER이 비고(삼성전자우처럼 시총 100조가 넘어도
    마찬가지다), 발행주식수도 보통주 기준이라 시가총액이 어긋난다. 분석이 성립하지
    않는 종목을 목록에 두면 시연에서 빈 화면을 고르게 된다.

    판별은 KRX 종목코드 규약을 쓴다 — 보통주는 끝자리가 0이고 우선주는 5/7/9/K/L 등이다.
    이름 규칙(`...우`로 끝남)은 성우·이오플로우·에코글로우 같은 보통주를 잘못 걸러낸다.
    실측에서 코드 규칙으로 걸린 113개는 전부 이름에도 '우'가 들어가 오탐이 없었다.

    조회 실패 시 백테스트 20종목으로 폴백한다 — 네트워크가 없어도 시연은 되어야 한다.
    """
    try:
        import FinanceDataReader as fdr
        df = fdr.StockListing("KRX").dropna(subset=["Code", "Name"])
        df = df[df["Code"].str[-1] == "0"]
        if "Marcap" in df.columns:
            df = df.sort_values("Marcap", ascending=False, na_position="last")
        out = [(f"{n} ({c})", c) for c, n in zip(df["Code"], df["Name"])]
        if out:
            return out
    except Exception:
        pass
    return [(f"{n} ({t})", t) for n, t in TICKERS.items()]


def ensure_reports(ticker: str) -> None:
    """20종목 밖 종목의 리포트 캐시를 당일 기준으로 확보한다.

    get_today_context가 data/reports/{ticker}.csv를 직접 읽으므로, 파일이 없으면
    리포트가 빈 채로 돌아간다. 30일치만 받는 이유는 그 창만 쓰기 때문이다
    (백테스트용 전체 이력은 crawl.py 담당).

    **파일이 있어도 오늘 받은 것이 아니면 다시 받는다.** 예전에 받아둔 파일은 그때의
    30일 창이라, 시간이 지나면 get_today_context가 보는 창(오늘 기준 30일)과 겹치지
    않아 리포트가 있는 종목이 "리포트 없음"으로 나온다. 제출·시연이 수집일보다 몇 달
    뒤라 실제로 발생하는 경로다. 판정은 파일 수정시각으로 하며(마지막 리포트 날짜로
    하면 원래 리포트가 뜸한 종목을 매번 다시 받게 된다) 하루 1회로 제한된다.

    백테스트 20종목은 건드리지 않는다. 그 CSV는 실험 입력이고 crawl.py가 전체 이력을
    관리하는 파일이라, 앱이 30일치로 덮어쓰면 실험 데이터를 훼손한다.

    실패해도 예외를 올리지 않는다 — 리포트는 없으면 없는 대로 성립하고,
    화면에서 "리포트 없음"으로 안내된다.
    """
    # TICKERS가 아니라 BACKTEST_TICKERS로 판정한다. 분석 시 종목명을 TICKERS에
    # 주입하므로, TICKERS로 보면 한 번 조회한 종목이 20종목으로 취급돼 이후 영영
    # 리포트를 받지 않는다
    if ticker in BACKTEST_TICKERS:
        return

    path = os.path.join(REPORTS_DIR, f"{ticker}.csv")
    today = datetime.today().date()
    if os.path.exists(path) and datetime.fromtimestamp(os.path.getmtime(path)).date() == today:
        return

    try:
        from crawl import fetch_reports
        since = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
        recs = fetch_reports(ticker, since_date=since, max_pages=3)
        if recs:
            os.makedirs(REPORTS_DIR, exist_ok=True)
            pd.DataFrame(recs).to_csv(path, index=False, encoding="utf-8-sig")
        elif os.path.exists(path):
            # 30일 내 리포트가 없는데 옛 파일이 남아 있으면 그 옛 행이 계속 읽힌다.
            # 빈 CSV를 쓰면 get_today_context의 read_csv가 터지므로 파일을 지운다
            os.remove(path)
    except Exception:
        pass

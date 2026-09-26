"""탭 공용 상수와 로더.

여기 두는 기준은 "둘 이상의 탭이 쓰는가"다. 한 탭만 쓰는 헬퍼는 해당 탭 모듈에
둔다(예: signal_badge는 tab_analyze, signal_cell_style은 tab_matrix).
"""

import glob
import re
import os
from datetime import datetime

import pandas as pd
import streamlit as st

from app_ui import ROOT_DIR
from utils import TICKERS, KOSDAQ_TICKERS, EXPERIMENT_DIR, FORWARD_DIR, ANALYSIS_DIR, REPORTS_DIR, PRICE_DIR
from compare import COND_LABELS, DEFAULT_MODEL

__all__ = [
    "TICKERS", "EXPERIMENT_DIR", "FORWARD_DIR", "ANALYSIS_DIR", "REPORTS_DIR",
    "COND_LABELS", "DEFAULT_MODEL", "BACKTEST_TICKERS",
    "UI_CONDS", "REPORT_CONDS", "SMALL_SAMPLE_N", "load_ui_models", "check_model_id",
    "explain_model_error",
    "SIGNAL_STYLE", "FORWARD_DEMO_DIR",
    "load_backtest_results", "list_backtest_models", "fmt_metric",
    "list_matrix_models", "load_signal_matrix",
    "load_krx_stocks", "register_ticker", "fetch_recent_reports", "check_dart_cache",
    "check_trading_halt", "HALT_MIN_DAYS",
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

# 개별 분석용 모델 목록은 config/models.toml에서 읽는다(load_ui_models). 새 모델이 나올 때마다
# 소스를 고치지 않게 하려는 것이다. 파일이 없거나 깨졌을 때만 아래 4모델로 폴백한다.
MODELS_CONFIG = os.path.join(ROOT_DIR, "config", "models.toml")

# llm_experiment._provider가 받는 접두어. src/ 동결이라 여기에 없는 회사는 설정으로도 못 넣는다.
# 목록에서 미리 걸러야 분석하기를 누른 뒤에야 ValueError를 보는 일이 없다
_SUPPORTED_PREFIXES = ("gemini", "gemma", "gpt", "claude")

_PAID_NOTE = "유료 API라 호출마다 비용이 듭니다."
_FALLBACK_MODELS = [
    {"id": DEFAULT_MODEL,      "label": DEFAULT_MODEL,      "note": "", "verified": True},
    {"id": "gemma-4-31b-it",   "label": "gemma-4-31b-it",
     "note": "gemma는 응답이 최대 3분 걸릴 수 있습니다.", "verified": True},
    {"id": "gpt-5.4-mini",     "label": "gpt-5.4-mini",     "note": _PAID_NOTE, "verified": True},
    {"id": "claude-haiku-4-5", "label": "claude-haiku-4-5", "note": _PAID_NOTE, "verified": True},
]

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
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def check_model_id(mid: str) -> str | None:
    """직접 입력한 모델 ID 검사. 문제가 없으면 None, 있으면 화면에 낼 이유.

    ID는 호출에만 쓰이는 게 아니라 results/forward_demo/{날짜}/{모델}/ 경로가 된다.
    슬래시나 공백이 들어가면 경로가 깨지므로 영문·숫자·.-_만 받는다.
    """
    if not mid:
        return "모델 ID를 입력해 주세요."
    if not _MODEL_ID_RE.match(mid):
        return "모델 ID에는 영문, 숫자, 점(.), 하이픈(-), 밑줄(_)만 쓸 수 있습니다."
    if not mid.startswith(_SUPPORTED_PREFIXES):
        return "gemini, gemma, gpt, claude로 시작하는 모델만 호출할 수 있습니다."
    return None


def explain_model_error(e: Exception) -> str | None:
    """호출 실패 중 모델 탓인 것을 사용자 말로 바꾼다. 모르는 오류는 None(원문을 그대로 낸다).

    설정에 새 모델을 넣거나 직접 입력하면 가장 먼저 부딪히는 두 가지다(실측 2026-09-26).
    """
    msg = str(e)
    # 크레딧 소진은 모델이 아니라 계정 문제다. Gemini는 무료 모델(gemma)까지 프로젝트 단위로
    # 막는다(실측 2026-09-26: 402 "prepayment credits are depleted"). 모델을 바꿔도 안 되므로
    # 다른 원인보다 먼저 가려 "다른 모델로 해 보라"는 쪽으로 읽히지 않게 한다
    if "402" in msg or "credits are depleted" in msg or "insufficient_quota" in msg or "credit balance" in msg:
        return ("API 크레딧이 소진되어 호출할 수 없습니다. 해당 회사 콘솔에서 결제 상태를 확인해 주세요. "
                "같은 키를 쓰는 모델은 모두 막혀 있습니다.")
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate limit" in msg.lower():
        return "호출 한도를 넘었습니다. 잠시 후 다시 시도해 주세요."
    if "temperature" in msg:
        # gpt-5.6-luna: "Only the default (1) value is supported"
        return ("이 모델은 temperature=0을 지원하지 않아 쓸 수 없습니다. "
                "실험과 같은 조건(temperature=0 고정)으로만 호출하기 때문입니다.")
    if "not found" in msg.lower() or "does not exist" in msg.lower() or "404" in msg:
        return "해당 회사에 없는 모델 ID입니다. 철자를 확인해 주세요."
    return None


def load_ui_models() -> tuple[list[dict], list[str]]:
    """config/models.toml → (모델 항목 목록, 화면에 알릴 경고 목록).

    각 항목은 {"id", "label", "note", "verified"}. 캐시하지 않고 매번 읽는다 — 몇 줄짜리
    파일이라 비용이 없고, 그래야 파일에 한 줄 추가한 것이 재시작 없이 목록에 뜬다.

    **앱이 죽으면 안 된다.** 파일이 없거나 TOML 문법이 깨졌거나 쓸 만한 항목이 하나도
    없으면 4모델로 폴백하고 경고만 돌려준다. 항목 단위의 문제(id 없음, 미지원 접두어,
    중복)는 그 항목만 빼고 이유를 알린다. enabled=false는 의도한 숨김이라 알리지 않는다.
    """
    import tomllib

    warnings: list[str] = []
    try:
        with open(MODELS_CONFIG, "rb") as f:
            entries = tomllib.load(f).get("model", [])
    except FileNotFoundError:
        return _FALLBACK_MODELS, ["모델 설정 파일(config/models.toml)이 없어 기본 4모델을 표시합니다."]
    except (tomllib.TOMLDecodeError, OSError) as e:
        return _FALLBACK_MODELS, [f"모델 설정 파일을 읽지 못해 기본 4모델을 표시합니다 ({e})."]

    models: list[dict] = []
    seen: set[str] = set()
    for e in entries if isinstance(entries, list) else []:
        if not isinstance(e, dict) or not e.get("enabled", True):
            continue
        mid = str(e.get("id", "")).strip()
        if not mid:
            warnings.append("id가 없는 항목을 건너뛰었습니다.")
            continue
        if not mid.startswith(_SUPPORTED_PREFIXES):
            warnings.append(
                f"{mid}은(는) 앱이 호출할 수 없는 모델이라 목록에서 뺐습니다 "
                "(gemini, gemma, gpt, claude로 시작하는 모델만 지원)."
            )
            continue
        if mid in seen:
            continue
        seen.add(mid)
        models.append({
            "id":       mid,
            "label":    str(e.get("label") or mid),
            "note":     str(e.get("note") or ""),
            "verified": bool(e.get("verified", False)),
        })

    if not models:
        return _FALLBACK_MODELS, warnings + ["모델 설정 파일에 쓸 수 있는 항목이 없어 기본 4모델을 표시합니다."]
    return models, warnings


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


# ── 실시간 수집 공용 (개별 분석 · 분석 프롬프트 생성) ─────
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
def _krx_listing() -> pd.DataFrame | None:
    """KRX 상장 목록 원본. 종목 목록과 코스닥 판별이 한 번의 조회를 나눠 쓴다."""
    try:
        import FinanceDataReader as fdr
        return fdr.StockListing("KRX").dropna(subset=["Code", "Name"])
    except Exception:
        return None


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
    df = _krx_listing()
    if df is not None:
        df = df[df["Code"].str[-1] == "0"]
        if "Marcap" in df.columns:
            df = df.sort_values("Marcap", ascending=False, na_position="last")
        out = [(f"{n} ({c})", c) for c, n in zip(df["Code"], df["Name"])]
        if out:
            return out
    return [(f"{n} ({t})", t) for n, t in TICKERS.items()]


def register_ticker(ticker: str, name: str) -> None:
    """20종목 밖 종목을 src/의 레지스트리에 주입해 프롬프트가 올바르게 만들어지게 한다.

    src/는 코드 동결이라 호출 전에 모듈 전역을 채우는 방식으로 푼다.

    - 종목명: get_today_context는 TICKERS에서 이름을 역조회하고 못 찾으면 티커 코드를
      이름으로 쓴다. 그 이름이 프롬프트에 들어가므로(cond1은 종목명이 입력의 전부다)
      "005490"을 회사명으로 받게 된다.
    - 상장 시장: build_prompt는 KOSDAQ_TICKERS(20종목 안의 두 개)만 보고 시장을 적어,
      20종목 밖 코스닥 종목이 전부 "상장 시장: KOSPI"로 나갔다. KRX 목록의 Market으로
      판별해 채운다. KOSDAQ GLOBAL도 코스닥이라 접두어로 본다.

    백테스트 20종목은 건드리지 않는다. 실험 때와 같은 프롬프트가 나와야 하기 때문이다.
    같은 set 객체를 llm_experiment가 import해 쓰므로 여기서 add하면 그쪽에도 보인다.
    """
    if ticker in BACKTEST_TICKERS:
        return
    TICKERS.setdefault(name, ticker)
    df = _krx_listing()
    if df is None or "Market" not in df.columns:
        return
    market = df.loc[df["Code"] == ticker, "Market"]
    if not market.empty and str(market.iloc[0]).startswith("KOSDAQ"):
        KOSDAQ_TICKERS.add(ticker)


# 네이버 금융 리서치의 JSON API. 2026년 하반기 네이버가 리서치 페이지를 finance.naver.com에서
# stock.naver.com으로 옮기면서 옛 목록 페이지가 새 주소로 넘어가게 됐고, crawl.py가 찾던
# table.type_1이 사라져 전 종목이 0건으로 나왔다(삼성전자 포함). 새 페이지는 화면을 스크립트로
# 그리며 이 API에서 데이터를 받는다. 목표주가와 투자의견이 목록에 같이 있어 상세 페이지를
# 따로 부를 필요가 없다. size는 최대 10이다(초과 시 400) — 프롬프트는 5건만 쓴다.
_NAVER_RESEARCH_API = "https://stock.naver.com/api/stockSecurity/researches/v2/company/by-items"
_REPORT_WINDOW_DAYS = 30   # context_builders.WINDOW_DAYS와 같은 값
_REPORT_MAX = 5            # get_today_context의 head(5)와 같은 값


def fetch_recent_reports(ticker: str, today: str | None = None) -> list[dict] | None:
    """오늘 기준 30일 이내 리포트 최대 5건. get_today_context의 recent_reports와 같은 형식.

    None은 조회 실패, []는 30일 안에 리포트가 없다는 뜻이다. 호출측은 None일 때 기존 값을
    그대로 둔다.
    """
    import requests
    try:
        resp = requests.get(
            _NAVER_RESEARCH_API,
            params={"itemCodes": ticker, "size": 10},
            headers={"User-Agent": "Mozilla/5.0", "Referer": "https://stock.naver.com/"},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get(ticker, [])
    except Exception:
        return None

    end = pd.Timestamp(today or datetime.today().date())
    start = end - pd.Timedelta(days=_REPORT_WINDOW_DAYS)
    out = []
    for it in items:
        d = pd.to_datetime(it.get("writeDate"), errors="coerce")
        if pd.isna(d) or not (start <= d <= end):
            continue
        try:
            tp = int(float(it["goalPrice"])) if it.get("goalPrice") else None
        except (TypeError, ValueError):
            tp = None
        out.append({"date": str(d.date()), "title": str(it.get("title", "")).strip(), "target_price": tp})
    out.sort(key=lambda r: r["date"], reverse=True)
    return out[:_REPORT_MAX]


def _install_report_source() -> None:
    """get_today_context의 리포트를 CSV 대신 네이버 API에서 채우도록 감싼다.

    src/는 코드 동결이라 crawl.py를 고칠 수 없고, get_today_context는 data/reports/{ticker}.csv를
    직접 읽는다. 호출 후 ctx["recent_reports"]만 바꿔 끼운다. 모듈 속성을 바꾸므로 함수 안에서
    `from update import get_today_context`를 하는 forward_test·tab_data 양쪽에 다 적용된다.

    **파일은 하나도 쓰지 않는다.** 예전 방식(ensure_reports)은 20종목 밖 CSV를 앱이 받아 썼고,
    20종목은 실험 입력이라 손대지 못해 crawl.py가 마지막으로 돈 시점(2026-07-31)의 리포트가
    남았다. 오늘 기준 30일 창에 안 걸려 삼성전자도 "리포트 없음"이었다. 파일을 거치지 않으니
    20종목도 실험 데이터를 건드리지 않고 최신 리포트를 받는다.

    앱 프로세스 안에서만 바뀐다. 주간 배치(forward_run_all 등)는 별도 프로세스라 영향이 없다.
    API가 실패하면 원래 값(CSV 기준)을 그대로 둔다.
    """
    import functools
    import update
    if getattr(update.get_today_context, "_app_report_source", False):
        return
    original = update.get_today_context

    @functools.wraps(original)
    def get_today_context(ticker: str) -> dict:
        ctx = original(ticker)
        recs = fetch_recent_reports(ticker, ctx.get("date"))
        if recs is not None:
            ctx["recent_reports"] = recs
        return ctx

    get_today_context._app_report_source = True
    update.get_today_context = get_today_context


_install_report_source()


# 거래정지·상장폐지 판정에 쓰는 최소 연속일수. 유동성이 낮은 종목은 하루 이틀쯤
# 거래량 0이 나올 수 있어 단발성은 걸러야 한다. 5거래일(약 일주일) 연속이면
# 일시적 소강이 아니라 거래 자체가 막힌 상태로 본다.
HALT_MIN_DAYS = 5


def check_trading_halt(ticker: str, min_days: int = HALT_MIN_DAYS) -> dict | None:
    """시세 캐시 끝에서 거래량 0이 연속되는지 확인한다.

    거래정지·상장폐지 종목은 FDR이 마지막 종가를 그대로 반복해서 돌려준다.
    이노벡스(279060)가 2026-01-26부터 종가 115원·거래량 0으로 142거래일 고정된
    것이 그 예다. 이 값이 그대로 컨텍스트에 들어가면 모멘텀·거래량 변화율이
    전부 0이 되고, LLM은 그것을 "변동성이 없다"는 실제 관측으로 읽는다.

    시세 캐시만 읽고 API는 호출하지 않는다. 캐시가 없으면(한 번도 조회한 적 없는
    종목) 판정할 근거가 없으므로 None을 돌려준다 — 분석·조회를 한 번 거치면
    캐시가 생기므로 다음 렌더에서 자연히 잡힌다.

    Returns:
        None: 캐시 없음 / 읽기 실패 / 정상 거래 중
        dict: {"days": 연속 거래량 0 일수,
               "last_traded": 마지막으로 거래가 있었던 날짜(str) 또는 None,
               "price": 고정된 종가(float) 또는 None}
    """
    path = os.path.join(PRICE_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path, usecols=["Date", "Close", "Volume"])
    except Exception:
        return None  # 형식이 다르거나 깨진 캐시는 판정하지 않는다
    if df.empty:
        return None

    vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
    zero_tail = 0
    for v in reversed(vol.tolist()):
        if v != 0:
            break
        zero_tail += 1

    if zero_tail < min_days:
        return None

    traded = df.iloc[: len(df) - zero_tail]
    last_close = pd.to_numeric(df["Close"], errors="coerce").iloc[-1]
    return {
        "days": zero_tail,
        "last_traded": str(traded["Date"].iloc[-1]) if not traded.empty else None,
        "price": None if pd.isna(last_close) else float(last_close),
    }

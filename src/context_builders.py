"""
프롬프트 컨텍스트 빌더

각 함수: (ticker: str, date: str) → str
  - 해당 date 기준 LLM 프롬프트에 삽입할 섹션 텍스트 반환
  - 데이터 없거나 해당 없으면 빈 문자열 반환
"""

import os
import sys

import pandas as pd
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import DATA_DIR

FINANCIALS_DIR    = os.path.join(DATA_DIR, "financials")
REPORTS_DIR       = os.path.join(DATA_DIR, "reports")
DART_FUND_DIR     = os.path.join(DATA_DIR, "dart_fundamentals")

WINDOW_DAYS = 30


# ── 포맷 헬퍼 ─────────────────────────────────────────────

def _fmt(value, decimals: int = 2, na_str: str = "N/A") -> str:
    if pd.isna(value):
        return na_str
    return f"{value:.{decimals}f}"


def _fmt_price(value) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{int(value):,}"


# ── 빌더 함수 ─────────────────────────────────────────────

def build_financials(ticker: str, date: str) -> str:
    """
    data/financials/{ticker}.csv에서 date 행 로드 후 프롬프트 텍스트 반환.
    NaN → "N/A" 또는 "해당없음(적자)" 처리.
    """
    path = os.path.join(FINANCIALS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return ""

    df = pd.read_csv(path, dtype={"ticker": str}, parse_dates=["date"])
    row_df = df[df["date"].dt.strftime("%Y-%m-%d") == date]
    if row_df.empty:
        return ""
    row = row_df.iloc[0]

    market_cap = row.get("market_cap")
    market_cap_str = _fmt(market_cap / 1e12, 1) if not pd.isna(market_cap) else "N/A"

    return "\n".join([
        "[재무지표]",
        f"PER: {_fmt(row.get('per'), 1, na_str='해당없음(적자)')}",
        f"PBR: {_fmt(row.get('pbr'), 2)}",
        f"ROE: {_fmt(row.get('roe'), 1)}%",
        f"시가총액: {market_cap_str}조원",
        "",
        "[기술지표]",
        f"52주 최고가: {_fmt_price(row.get('high_52w'))}원",
        f"52주 최저가: {_fmt_price(row.get('low_52w'))}원",
        f"52주 내 현재 위치: {_fmt(row.get('price_position_52w'), 1)}%",
        f"최근 1개월 수익률: {_fmt(row.get('momentum_1m'), 2)}%",
        f"거래량 변화율: {_fmt(row.get('volume_change'), 2)}%",
    ])


def build_reports(ticker: str, date: str) -> str:
    """
    data/reports/{ticker}.csv에서 date 기준 직전 30일 리포트 최대 5건 반환.
    없으면 빈 문자열 반환.
    """
    path = os.path.join(REPORTS_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return ""

    df = pd.read_csv(path, parse_dates=["date"])
    end_dt   = pd.to_datetime(date) - timedelta(days=1)
    start_dt = end_dt - timedelta(days=WINDOW_DAYS - 1)

    sub = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]
    sub = sub.sort_values("date", ascending=False).head(5)

    if sub.empty:
        return ""

    lines = ["[애널리스트 리포트 (최근 30일, 최대 5건)]"]
    for _, r in sub.iterrows():
        tp = r.get("target_price")
        tp_str = f"목표주가 {int(tp):,}원" if not pd.isna(tp) else "목표주가 없음"
        lines.append(f"- {r['date'].strftime('%Y-%m-%d')} | {r['title']} ({tp_str})")

    return "\n".join(lines)



# ── dict 기반 빌더 (forward_test 전용) ────────────────────
# get_today_context() 반환 dict를 받아 동일 포맷의 섹션 텍스트 생성.
# CSV 파일을 읽지 않으므로 data/financials/ 를 오염시키지 않는다.

def build_financials_from_dict(ctx: dict) -> str:
    """ctx['per'/'pbr'/...] → [재무지표] 섹션 텍스트."""
    mc = ctx.get("market_cap")
    mc_str = _fmt(mc / 1e12, 1) if (mc is not None and not pd.isna(mc)) else "N/A"
    return "\n".join([
        "[재무지표]",
        f"PER: {_fmt(ctx.get('per'), 1, na_str='해당없음(적자)')}",
        f"PBR: {_fmt(ctx.get('pbr'), 2)}",
        f"ROE: {_fmt(ctx.get('roe'), 1)}%",
        f"시가총액: {mc_str}조원",
        "",
        "[기술지표]",
        f"52주 최고가: {_fmt_price(ctx.get('high_52w'))}원",
        f"52주 최저가: {_fmt_price(ctx.get('low_52w'))}원",
        f"52주 내 현재 위치: {_fmt(ctx.get('price_position_52w'), 1)}%",
        f"최근 1개월 수익률: {_fmt(ctx.get('momentum_1m'), 2)}%",
        f"거래량 변화율: {_fmt(ctx.get('volume_change'), 2)}%",
    ])


def build_reports_from_dict(ctx: dict) -> str:
    """ctx['recent_reports'] → [애널리스트 리포트] 섹션 텍스트."""
    reports = ctx.get("recent_reports", [])
    if not reports:
        return ""
    lines = ["[애널리스트 리포트 (최근 30일, 최대 5건)]"]
    for r in reports:
        tp     = r.get("target_price")
        tp_str = f"목표주가 {int(tp):,}원" if tp else "목표주가 없음"
        lines.append(f"- {r.get('date', '')} | {r['title']} ({tp_str})")
    return "\n".join(lines)


def build_dart_fundamentals_from_dict(ctx: dict) -> str:
    """ctx['revenue'/'operating_income'/...] → [분기 실적] 섹션 텍스트."""
    def to_trillion(val) -> str:
        if val is None or pd.isna(val):
            return "N/A"
        t    = val / 1_000_000_000_000
        sign = "+" if t > 0 else ""
        return f"{sign}{t:.1f}조원"

    def fmt_yoy(val) -> str:
        if val is None or pd.isna(val):
            return ""
        return f" (전년比 {float(val):+.1f}%)"

    return "\n".join([
        "[분기 실적]",
        f"매출: {to_trillion(ctx.get('revenue'))}{fmt_yoy(ctx.get('revenue_yoy'))}",
        f"영업이익: {to_trillion(ctx.get('operating_income'))}{fmt_yoy(ctx.get('operating_income_yoy'))}",
        f"영업이익률: {_fmt(ctx.get('operating_margin'), 1)}%",
        f"순이익: {to_trillion(ctx.get('net_income'))}",
        "",
        "[재무 안정성]",
        f"부채비율: {_fmt(ctx.get('debt_ratio'), 1)}%",
        f"영업현금흐름: {to_trillion(ctx.get('operating_cashflow'))}",
        "",
        "[주주환원]",
        f"배당수익률: {_fmt(ctx.get('dividend_yield'), 1)}%",
    ])


def build_dart_fundamentals(ticker: str, date: str) -> str:
    """
    data/dart_fundamentals/{ticker}.csv에서 date 행 로드 후 프롬프트 텍스트 반환.
    금액은 조원 단위로 변환 (DART 원본값 백만원 ÷ 1,000,000).
    없는 항목은 "N/A" 처리.
    """
    path = os.path.join(DART_FUND_DIR, f"{ticker}.csv")
    if not os.path.exists(path):
        return ""

    df = pd.read_csv(path, dtype={"ticker": str})
    row_df = df[df["date"] == date]
    if row_df.empty:
        return ""
    row = row_df.iloc[0]

    def to_trillion(val) -> str:
        """원(KRW) 단위 값을 조원 문자열로 변환."""
        if pd.isna(val):
            return "N/A"
        t = val / 1_000_000_000_000   # 원 → 조원
        sign = "+" if t > 0 else ""
        return f"{sign}{t:.1f}조원"

    def fmt_yoy(val) -> str:
        """전년比 변화율 포맷. NaN이면 빈 문자열."""
        if pd.isna(val):
            return ""
        return f" (전년比 {val:+.1f}%)"

    revenue    = row.get("revenue")
    oper_inc   = row.get("operating_income")
    net_inc    = row.get("net_income")
    oper_mgn   = row.get("operating_margin")
    debt_ratio = row.get("debt_ratio")
    oper_cf    = row.get("operating_cashflow")
    div_yield  = row.get("dividend_yield")
    rev_yoy    = row.get("revenue_yoy")
    op_yoy     = row.get("operating_income_yoy")

    return "\n".join([
        "[분기 실적]",
        f"매출: {to_trillion(revenue)}{fmt_yoy(rev_yoy)}",
        f"영업이익: {to_trillion(oper_inc)}{fmt_yoy(op_yoy)}",
        f"영업이익률: {_fmt(oper_mgn, 1)}%",
        f"순이익: {to_trillion(net_inc)}",
        "",
        "[재무 안정성]",
        f"부채비율: {_fmt(debt_ratio, 1)}%",
        f"영업현금흐름: {to_trillion(oper_cf)}",
        "",
        "[주주환원]",
        f"배당수익률: {_fmt(div_yield, 1)}%",
    ])

"""
실험 조합 정의

EXPERIMENTS[cond] = 사용할 컨텍스트 빌더 이름 목록
  - "financials"    → build_financials()
  - "reports"       → build_reports()
"""

EXPERIMENTS = {
    # 메인 Ablation Study
    "cond1": [],
    "cond2": ["financials"],
    "cond3": ["financials", "reports"],
    "cond4": ["financials", "reports", "dart_fundamentals"],

    # 단독 실험 (컨텍스트별 독립 기여도 측정)
    "reports_only": ["reports"],
    "dart_only":    ["dart_fundamentals"],

    # Leave-One-Out ablation (cond4 - 단일 카테고리)
    "cond4_no_reports": ["financials", "dart_fundamentals"],
}

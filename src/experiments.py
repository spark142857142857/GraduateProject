"""
실험 조합 정의

EXPERIMENTS[cond] = 사용할 컨텍스트 빌더 이름 목록
  - "financials"    → build_financials()
  - "reports"       → build_reports()
  - "announcements" → build_announcements()
"""

EXPERIMENTS = {
    "cond1":                [],
    "cond2":                ["financials"],
    "cond3":                ["financials", "reports"],
    "cond3_reports_only":   ["reports"],
}

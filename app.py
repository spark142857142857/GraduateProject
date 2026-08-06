"""LLM 기반 주식 투자 신호 시스템 — Streamlit 앱 진입점.

실행: streamlit run app.py

탭 구성:
  ① 개별 종목 분석 — 종목·조건·모델 선택 후 실시간 신호 생성 (당일 캐시)
  ② 백테스트 신호 매트릭스 — 특정 신호일의 20종목×5조건 판단과 실제 결과 (API 호출 없음)
  ③ 백테스트 성과·모델 비교 — results/analysis 기반 모델별 성과·유의성 (API 호출 없음)
  ④ 포트폴리오 백테스트 — 신호대로 운용했을 때의 누적 곡선·MDD (API 호출 없음)
  ⑤ 조건 간 신호 전이 — 컨텍스트 추가로 판단이 어떻게·옳게 바뀌었나 (API 호출 없음)

②는 forward 캐시가 아니라 results/experiment(백테스트)를 읽는다. forward는 신호 생성을
2026-08-02로 종료해 시간이 갈수록 낡은 날짜가 화면에 남고, 앱에서 forward를 다루지 않기로
한 결정과도 어긋나기 때문. 백테스트 기간(2023-01~2025-12)은 설계상 고정이라 낡지 않는다.

구현은 app_ui/ 아래 탭별 모듈에 있다. app_ui를 import하는 시점에 sys.path·.env
부트스트랩이 먼저 실행된다(app_ui/__init__.py 참고).
"""

import streamlit as st

from app_ui import tab_analyze, tab_flip, tab_matrix, tab_portfolio, tab_report

# ── 페이지 설정 ────────────────────────────────────────────
# 다른 st 명령보다 먼저 호출해야 한다. 위 import는 함수 정의와 데코레이터뿐이라
# 렌더링을 일으키지 않으므로 이 순서로 안전하다.
st.set_page_config(
    page_title="LLM 주식 신호 시스템",
    page_icon="📈",
    layout="wide",
)

st.title("📈 LLM 기반 주식 투자 신호 시스템")
st.caption("멀티모델 LLM 기반 | 향후 20거래일 방향성 예측")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 개별 종목 분석",
    "📋 백테스트 신호 매트릭스",
    "📊 백테스트 성과·모델 비교",
    "💼 포트폴리오 백테스트",
    "🔀 조건 간 신호 전이",
])

with tab1:
    tab_analyze.render()

with tab2:
    tab_matrix.render()

with tab3:
    tab_report.render()

with tab4:
    tab_portfolio.render()

with tab5:
    tab_flip.render()

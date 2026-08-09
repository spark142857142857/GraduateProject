"""LLM 기반 주식 투자 신호 시스템 — Streamlit 앱 진입점.

실행: streamlit run app.py

탭 구성:
  ① 개별 종목 분석 — 종목·조건·모델 선택 후 실시간 신호 생성 (LLM 호출)
  ② 종목 데이터 조회 — 수집 지표 열람·내려받기 (LLM 호출 없음)
  ③ 포트폴리오 백테스트 — 신호대로 운용했을 때의 누적 곡선·MDD (API 호출 없음)

①②가 시스템, ③이 검증이다. 시스템 쪽을 앞에 묶었다. 검증 결과는 보고서와 발표
자료(포스터·영상)가 주로 맡으므로 앱에는 ③만 남겼다.

**마운트에서 제외한 탭** (파일은 app_ui/에 그대로 있다)
  · tab_matrix — 백테스트 신호 매트릭스 (20종목×5조건 원자료)
  · tab_report — 백테스트 성과·모델 비교 (표 11개)
  · tab_flip   — 조건 간 신호 전이 (짝 비교 검정)

셋 다 인터랙션이 값을 더하지 않아 정지 화면 한 장으로 대체 가능하다고 보고 뺐다. 판단
근거와 각 탭이 담던 내용의 행선지는 docs/app_scope.md에 있다. 되돌리려면 아래 import와
st.tabs 목록에 다시 넣기만 하면 된다 — 모듈은 삭제하지 않았다.

구현은 app_ui/ 아래 탭별 모듈에 있다. app_ui를 import하는 시점에 sys.path·.env
부트스트랩이 먼저 실행된다(app_ui/__init__.py 참고).
"""

import streamlit as st

from app_ui import tab_analyze, tab_data, tab_portfolio

# ── 페이지 설정 ────────────────────────────────────────────
# 다른 st 명령보다 먼저 호출해야 한다. 위 import는 함수 정의와 데코레이터뿐이라
# 렌더링을 일으키지 않으므로 이 순서로 안전하다.
st.set_page_config(
    page_title="LLM 주식 신호 시스템",
    page_icon="📈",
    layout="wide",
)

# 여백 두 곳만 줄인다. 테마(config.toml)로는 지정할 수 없는 값이라 CSS로 처리한다.
#   · 본문 상단 패딩 96px — 제목·캡션과 합쳐 720p 화면에서 탭이 나오기까지 284px을
#     썼다. 탭이 첫 화면에 안 보이면 시연에서 앱 구조가 한 번에 안 읽힌다.
#   · divider 상하 32px씩 64px — 개별 분석 탭에만 6개라 결과 화면에서 384px이
#     순수 여백이었다. 구분선은 그대로 두고 간격만 줄인다.
# hr은 st.divider()가 stMarkdownContainer 안에 내는 것뿐이라 스코프가 안전하다.
st.markdown(
    """
    <style>
      [data-testid="stMainBlockContainer"] { padding-top: 3rem; }
      [data-testid="stMarkdownContainer"] hr { margin: 1.25rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 LLM 기반 주식 투자 신호 시스템")
st.caption("멀티모델 LLM 기반 | 향후 20거래일 방향성 예측")
st.caption(
    "⚠️ 본 화면은 졸업작품 연구 결과물이며, 여기 표시되는 신호는 투자 조언이 아닙니다. "
    "과거 성과는 미래 수익을 보장하지 않으며, 투자 판단과 그 결과에 대한 책임은 이용자 본인에게 있습니다."
)

tab1, tab2, tab3 = st.tabs([
    "🔍 개별 종목 분석",
    "📥 종목 데이터 조회",
    "💼 포트폴리오 백테스트",
])

with tab1:
    tab_analyze.render()

with tab2:
    tab_data.render()

with tab3:
    tab_portfolio.render()

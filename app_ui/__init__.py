"""Streamlit 앱 UI 모듈.

이 패키지를 import하면 아래 부트스트랩이 먼저 실행된다. shared 이하 모듈이
`from utils import ...`처럼 src/ 아래 모듈을 직접 참조하므로, import 순서와
무관하게 sys.path와 .env가 준비되어 있어야 한다.
"""

import os
import sys

from dotenv import load_dotenv

# .env를 앱 시작 시점에 로드 — DARTS_API_KEY 등을 읽는 코드가 모두
# 버튼 클릭 시 lazy import/실행이라, 여기서 미리 넣어두지 않으면 키를 못 찾는다
load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
sys.path.insert(0, os.path.join(_SRC, "collect"))
sys.path.insert(0, os.path.join(_SRC, "experiment"))
sys.path.insert(0, _SRC)

# 앱 루트 — docs_cache 등 저장소 최상단 기준 경로가 필요한 곳에서 쓴다
ROOT_DIR = _ROOT

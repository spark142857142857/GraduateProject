# 앱 남은 작업 인수인계 (2026-09-20)

> 다른 PC에서 Codex로 이어서 작업하기 위한 문서다. 중간보고서는 제출이 끝났고
> 실험도 종료됐다. **남은 일은 앱뿐이다.**

## 0. 먼저 읽을 것

1. [README.md](../README.md) — "프로젝트 구조", "Streamlit 대시보드" 절
2. [app_scope.md](app_scope.md) — 앱에 무엇을 넣고 무엇을 뺐는지, 그 판단 근거
3. `app_context_for_gpt.md` — **이 저장소에 없다**(개인 문서라 push 대상이 아니다).
   없어도 된다. 2026-07-29 스냅샷이라 탭 구성과 파일 구조가 이미 현재와 다르고,
   거기서 아직 유효한 것은 제약 조항뿐인데 그건 아래 1절에 옮겨 놨다.

보고서 계열 문서는 `docs/private/`에 모여 있고 **저장소에 올라가지 않는다**
(개인 제출물이라 push 대상이 아니다). 이 작업과 무관하니 없어도 된다.
`docs/experiments_log.md`는 저장소에 있으나 역시 앱 작업과는 무관하다.

## 1. 깨면 안 되는 제약

1. **`src/` 전체 수정 금지.** 백테스트와 forward의 재현성 때문에 동결 상태다.
   프롬프트(`prompt.py`, `context_builders.py`), `llm_experiment.py`,
   `forward_test.py`는 특히 손대지 않는다. 앱에서 필요한 동작은
   **호출 후 후처리**로 푼다.
2. **temperature=0.0 고정.** 실험 전체가 이 값으로 돌아갔다.
3. **연구 데이터 오염 금지.** 앱이 만든 신호는 `results/forward_demo/`로 보낸다
   (`demote_to_demo`). `results/forward/{날짜}/`는 주간 정식 배치 전용이며,
   `forward_eval.py`가 그 경로를 통째로 훑어 평가 표본으로 삼는다.
4. **streamlit 1.58** — `use_container_width`는 폐기됐다. `width="stretch"`.
5. **LLM 호출은 사용자 사전 확인 없이 하지 말 것.** 호출이 필요하면 먼저 묻고,
   승인받으면 **gemma**로 부른다(무료, rate-limit만 감수).
6. 보고서 문서는 건드리지 않는다. 이 작업은 앱 전용이다.

## 2. 현재 상태 (실측 확인 2026-09-20)

탭 3개가 마운트돼 있다.

| 탭 | 모듈 | LLM 호출 |
|---|---|---|
| ① 개별 종목 분석 | `app_ui/tab_analyze.py` | **있음** |
| ② 분석 프롬프트 생성 | `app_ui/tab_data.py` | 없음 |
| ③ 포트폴리오 백테스트 | `app_ui/tab_portfolio.py` | 없음 |

`tab_matrix.py`, `tab_report.py`, `tab_flip.py`는 파일은 있으나 마운트에서
빠져 있다. 되살리려면 `app.py`의 import와 `st.tabs` 목록에 다시 넣으면 된다.
뺀 이유는 [app_scope.md](app_scope.md)에 있다.

**2026-09-26 진행분** — ② 를 "분석 프롬프트 생성"으로 피벗했고, 리포트 수집원을 네이버
JSON API로 바꿨으며(옛 크롤러는 네이버 개편으로 전 종목 0건이었다), 세 탭의 화면 문구를
정리했다(작업 2의 일부). 내용은 [app_scope.md](app_scope.md) 끝의 두 절과 `TODO.md` 앱-14에 있다.
**남은 주 작업은 아래 작업 1이다.**

**모델 선택 UI는 이미 있다.** `tab_analyze.py:137`의 selectbox가
`shared.py:49`의 `UI_MODELS`를 읽는다. 4모델 전부 화면에서 고를 수 있고,
provider 분기도 접두어 기준으로 동작한다. 그러니 아래 작업 1은
"새로 만들기"가 아니라 **"하드코딩된 목록을 설정으로 빼기"**다.

## 3. 작업 1 — 모델 목록을 설정으로 분리 (주 작업)

### 왜

지금은 모델 목록이 파이썬 코드 안에 리터럴로 박혀 있다.

```python
# app_ui/shared.py:49
UI_MODELS = [DEFAULT_MODEL, "gemma-4-31b-it", "gpt-5.4-mini", "claude-haiku-4-5"]
```

새 모델이 나오면 소스를 고쳐야 한다. 모델이 계속 갱신되는 동안 작품이 낡지
않게 하려는 것이 이 작업의 목적이다. **설정 파일에 한 줄 추가하면 화면 목록에
뜨는** 구조로 바꾼다.

### 어떻게

- 설정 파일을 하나 만든다(예: `config/models.yaml` 또는 `config/models.json`).
  포맷은 구현자 판단에 맡기되, 새 의존성을 들이지 않는 쪽을 권한다.
- 한 항목이 가질 필드: 모델 ID, 화면 표시 이름, 안내 문구(선택), 노출 여부.
  지금 `tab_analyze.py:147~152`가 접두어로 분기해 뿌리는 caption 세 종류를
  이 필드로 옮길 수 있다.
- `shared.py`는 파일을 읽어 `UI_MODELS`를 만든다. 파일이 없거나 깨졌으면
  **현재의 4모델로 폴백**하고 화면에 조용히 경고를 띄운다. 앱이 죽으면 안 된다.
- `DEFAULT_MODEL`(= `compare.DEFAULT_MODEL`, `gemini-2.5-flash-lite`)은
  앵커 모델이라 분석·비교 코드 전반이 참조한다. **이 상수는 건드리지 말 것.**
  설정 파일의 첫 항목이 기본 선택이 되게만 하고, `DEFAULT_MODEL` 자체를
  설정에서 덮어쓰려 하지 않는다.

### provider 한계를 문서에 남길 것

`src/experiment/llm_experiment.py:153`의 `_provider()`가 접두어로만 분기한다.

```python
if model.startswith(("gemini", "gemma")): return "gemini"
if model.startswith("gpt"):                return "openai"
if model.startswith("claude"):             return "anthropic"
raise ValueError(...)
```

즉 **같은 회사의 새 모델은 설정에 한 줄로 추가되지만, 새 회사의 모델은
안 된다**(`src/` 동결이라 분기를 늘릴 수 없다). 이건 고칠 문제가 아니라
알려진 경계다. 설정 파일 주석과 README에 그렇게 적어 둔다.
설정에 미지원 접두어가 들어오면 앱이 목록에서 걸러 내고 이유를 보여 주는 편이
좋다. `preflight()`가 던지는 에러를 분석 버튼 누른 뒤에 보는 것보다 낫다.

### 목록은 골라서 넣는다 (2026-09-26 합의)

선택 가능한 모델을 전부 넣지 않는다. 호출 경로가 `src/`에 묶여 있어 목록에 떠도 돌지 않는
모델이 생긴다.

- `_raw_text()`가 temperature=0.0을 고정으로 넘긴다. 일부 reasoning 모델은 이를 거부한다
- `_provider()`가 접두어로만 갈라 `o3`·`o4-mini`처럼 `gpt`로 시작하지 않는 모델은 못 넣는다
- 새 모델은 넣기 전에 한 번 실제로 불러 봐야 하고, gpt·claude는 유료라 사전 확인이 필요하다

그래서 **연구 검증 4모델 + 회사별 최신 1~2개**만 넣고, 나머지는 `enabled: false`로 두거나
아예 넣지 않는다. 설정 항목에 연구 검증 여부 필드(예: `verified`)를 두어 화면에서 "백테스트
없음"을 구분한다. "아무 모델이나 쓰고 싶다"는 요구는 ② 프롬프트 복사가 받는다. 어떤 모델 ID가
있는지는 각 회사의 모델 목록 API로 확인한다(생성 호출이 아니라 비용 없음).

### 하지 말 것

- `src/` 수정. 위 제약 1.
- 모델을 더 돌려 결과를 새로 만드는 일. 실험은 끝났다.
- 종목 유니버스나 보유 기간을 넓히는 일. 하지 않기로 확정된 사안이다.

## 4. 작업 2 — `app.py` 디자인 개선 (후순위)

`TODO.md` C-3의 마지막 항목이며 "최후 — 발표/데모용"으로 적혀 있다.
**2026-09-26 화면 문구 정리는 끝났다**(반복 안내·내부 용어·남는 상태 상자·구분선 제거,
[app_scope.md](app_scope.md) "화면 문구 정리 기준"). 같은 기준을 지키며 작업 1 뒤에 손댄다. 범위를 미리 넓히지 말고, 발표 형식이 정해진 다음에
필요한 만큼만 한다.

## 5. 환경

- Windows, 파이썬 의존성은 `uv` 관리
- 실행: 프로젝트 루트에서 `streamlit run app.py`
- **`.env`는 git에 없다**(gitignore). 다른 PC에서는 API 키를 새로 넣어야 한다.
  `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.
  gemma까지만 쓸 거면 `GEMINI_API_KEY` 하나로 충분하다.
- `app_ui/*.py`를 고치면 **서버를 재시작해야 반영된다.** 브라우저 새로고침으로는
  안 된다.
- 저장소: `https://github.com/spark142857142857/GraduateProject.git`

## 6. 다 됐다고 말하기 전에

화면을 띄워서 눈으로 본 다음에 말한다. 코드가 그럴듯하다는 것과 화면이
제대로 나온다는 것은 다르다. LLM 호출이 필요한 확인은 먼저 사용자에게 묻고,
승인되면 gemma로 한 번만 부른다.

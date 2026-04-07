# 실험 변수 정리

LLM 기반 주식 투자 신호 시스템에서 수정 가능한 실험 변수 목록입니다.

---

## 1. 실험 조건 (Ablation Study)

**파일:** `src/experiments.py`

```python
EXPERIMENTS = {
    "cond1": [],                                              # 종목명만 (No Context)
    "cond2": ["financials"],                                  # + 재무지표
    "cond3": ["financials", "reports"],                       # + 애널리스트 리포트
    "cond4": ["financials", "reports", "dart_fundamentals"],  # + DART 연간 실적

    "reports_only": ["reports"],            # 리포트 단독 기여도 측정
    "dart_only":    ["dart_fundamentals"],  # DART 단독 기여도 측정
}
```

- 컨텍스트 조합 변경, 새로운 cond 추가 가능
- 사용 가능한 컨텍스트 빌더: `"financials"` / `"reports"` / `"dart_fundamentals"`

---

## 2. LLM 파라미터

**파일:** `src/llm_experiment.py`

| 변수 | 현재값 | 설명 |
|---|---|---|
| `MODEL` | `"gemini-2.5-flash-lite"` | 사용할 Gemini 모델명 |
| `temperature` | `0.3` | 샘플링 온도. 0이면 결정적(재현 가능), 높을수록 다양한 출력 |
| `HOLD_SHORT` | `5` | 단기 수익률 측정 기간 (거래일) |
| `HOLD_LONG` | `20` | 장기 수익률 측정 기간 (거래일) |
| `REQ_DELAY` | `0.5` | Gemini API 호출 간 대기 시간 (초) |

> **논문 주의:** `temperature > 0`이면 동일 프롬프트에도 실행마다 결과가 달라집니다.
> 완전한 재현성이 필요하면 `temperature=0`으로 설정하세요.

---

## 3. 프롬프트

**파일:** `src/llm_experiment.py` — `build_prompt()` 함수

수정 가능한 요소:

- **예측 기간 문구:** `"향후 20거래일 투자 방향을 판단해주세요."` (숫자 변경 시 `HOLD_LONG`과 함께 맞출 것)
- **신호 판단 기준 설명:** Buy / Sell / Neutral 각 정의 문구
- **출력 형식:** JSON 스키마 (signal / confidence / reasons)

---

## 4. 백테스팅 기간

**파일:** `src/collect_financials.py`

| 변수 | 현재값 | 설명 |
|---|---|---|
| `START_YM` | `"2023-01"` | 재무 데이터 수집 시작 월 |
| `END_YM` | `"2025-12"` | 재무 데이터 수집 종료 월 (백테스트 범위) |

**파일:** `src/utils.py`

| 변수 | 현재값 | 설명 |
|---|---|---|
| `START_DATE` | `"2023-01-01"` | 주가 데이터 수집 시작일 |

> `START_YM`과 `START_DATE`는 같은 시점을 가리키도록 맞춰야 합니다.

---

## 5. 베이스라인 파라미터

### 컨센서스 전략

**파일:** `src/baseline_consensus.py`

| 변수 | 현재값 | 설명 |
|---|---|---|
| `N_REPORTS` | `3` | 컨센서스 산출에 사용할 최근 리포트 수 |
| `BUY_GAP` | `10.0` | 목표주가 괴리율 Buy 임계값 (%). 현재가 대비 목표주가가 이 값 이상 높으면 Buy |
| `HOLD_DAYS` | `20` | 신호 발생 후 보유 기간 (거래일) |

### 골든크로스 전략

**파일:** `src/baseline_golden.py`

| 변수 | 현재값 | 설명 |
|---|---|---|
| `MA_SHORT` | `5` | 단기 이동평균 기간 (일) |
| `MA_LONG` | `20` | 장기 이동평균 기간 (일) |
| `HOLD_DAYS` | `20` | 신호 발생 후 보유 기간 (거래일) |

---

## 6. 분석 대상 종목

**파일:** `src/utils.py`

```python
TICKERS = {
    "삼성전자": "005930",
    "SK하이닉스": "000660",
    # ... 총 20종목
}
```

종목 추가/제거로 실험 범위를 조정할 수 있습니다.
`compare.py`의 `SECTORS` 딕셔너리도 함께 수정해야 섹터 분석이 정확하게 동작합니다.

---

## 7. 데이터 수집 파라미터

| 변수 | 현재값 | 파일 | 설명 |
|---|---|---|---|
| `REQ_DELAY` | `0.3` | `collect_financials.py` | DART API 요청 간 대기 시간 (초) |
| `REQ_DELAY` | `0.3` | `collect_dart_fundamentals.py` | DART API 요청 간 대기 시간 (초) |
| `max_pages` | `100` | `crawl.py` | 종목당 크롤링 최대 페이지 수 |
| `WEEKS_52` | `252` | `collect_financials.py` | 52주 고저가 계산 기준 거래일 수 |

---

## 논문 핵심 변수 요약

| 우선도 | 변수 | 위치 | 논문 기여 |
|---|---|---|---|
| ★★★ | `cond1~4` | `experiments.py` | Ablation Study 핵심 — 컨텍스트 추가 효과 |
| ★★★ | `temperature` | `llm_experiment.py` | 재현성에 직접 영향 |
| ★★★ | `MODEL` | `llm_experiment.py` | 실험 조건의 핵심 명세 |
| ★★ | `HOLD_LONG` | `llm_experiment.py` | 예측 기간 설정 (20거래일이 적절한가?) |
| ★★ | `BUY_GAP` | `baseline_consensus.py` | 컨센서스 임계값 민감도 분석 가능 |
| ★ | `N_REPORTS` | `baseline_consensus.py` | 컨센서스 윈도우 크기 |
| ★ | `MA_SHORT / MA_LONG` | `baseline_golden.py` | 골든크로스 파라미터 |

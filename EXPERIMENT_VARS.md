# 실험 변수 정리

LLM 기반 주식 투자 신호 시스템에서 수정 가능한 실험 변수 목록입니다.

---

## 1. 실험 조건 (Ablation Study)

**파일:** `src/experiments.py`

```python
EXPERIMENTS = {
    # 메인 Ablation Study
    "cond1": [],                                              # 종목명만 (No Context)
    "cond2": ["financials"],                                  # + 재무지표
    "cond3": ["financials", "reports"],                       # + 애널리스트 리포트
    "cond4": ["financials", "reports", "dart_fundamentals"],  # + DART 연간 실적

    # 단독 기여도 측정
    "reports_only": ["reports"],            # 리포트 단독 기여도 측정
    "dart_only":    ["dart_fundamentals"],  # DART 단독 기여도 측정

    # Leave-One-Out ablation
    "cond4_no_reports": ["financials", "dart_fundamentals"],  # cond4에서 리포트만 제거
}
```

- 컨텍스트 조합 변경, 새로운 cond 추가 가능
- 사용 가능한 컨텍스트 빌더: `"financials"` / `"reports"` / `"dart_fundamentals"`
- 단독 기여도(`reports_only`, `dart_only`)는 "그 카테고리만 있을 때", LOO(`cond4_no_reports`)는 "전체에서 그 카테고리만 뺐을 때"로 해석이 다름

### 각 cond별 LLM 프롬프트 실제 전달 항목

| cond | 섹션 | 항목 |
|------|------|------|
| cond1 | 없음 | 종목명, 현재가만 전달 |
| cond2 | `[재무지표]` | PER, PBR, ROE, 시가총액 |
|       | `[기술지표]` | 52주 최고가/최저가, 52주 내 현재 위치, 1개월 수익률, 거래량 변화율 |
| cond3 | ↑ + `[애널리스트 리포트]` | 날짜, 리포트 제목, 목표주가 (최근 30일 / 최대 5건, 신호일 전일까지) |
| cond4 | ↑ + `[연간 실적 (DART 사업보고서)]` | 매출, 영업이익, 영업이익률, 순이익 (매출·영업이익 전년比 포함) |
|       | `[재무 안정성]` | 부채비율, 영업현금흐름 |
| cond4_no_reports | `[재무지표]` + `[기술지표]` + `[연간 실적]` + `[재무 안정성]` | cond4에서 애널리스트 리포트만 제거 |

> **설계 주의사항**
> - cond3: 해당 월에 리포트가 없으면 리포트 섹션이 프롬프트에서 누락되어 cond2와 동일하게 동작함
> - cond4 YoY 항목(`revenue_yoy`, `operating_income_yoy`): 설계서 기준 항목에 추가로 포함됨 (의도적)
> - **배당수익률은 LLM 프롬프트에 포함되지 않음**: 단기(5/20거래일) 방향 예측에 무관한 지표로 판단되어 의도적으로 제외. `data/dart_fundamentals/` CSV에는 수집되나 프롬프트에 삽입되지 않으며, Streamlit UI(`app.py`)에서만 참고용으로 표시됨.

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

### Sharpe Ratio 계산

**파일:** `src/compare.py` — `sharpe()` 함수

- 연환산 승수: `√(252 / hold_days)`
  - `hold_days=5`: √(252/5) ≈ 7.10
  - `hold_days=20`: √(252/20) ≈ 3.55
- 가정: 관측치를 IID로 간주 (시계열 자기상관 미보정)

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

### 벤치마크 지수 (초과수익률 계산용)

**파일:** `src/utils.py`

| 변수 | 값 | 설명 |
|---|---|---|
| `KOSPI_INDEX` | `"KS11"` | KOSPI 지수 코드 (FinanceDataReader) |
| `KOSDAQ_INDEX` | `"KQ11"` | KOSDAQ 지수 코드 |
| `KOSDAQ_TICKERS` | `{"247540", "196170"}` | 에코프로비엠, 알테오젠 (KOSDAQ 상장, 분석 대상) |

- `get_benchmark_price(ticker)`: 종목의 상장 시장에 맞는 벤치마크 지수 주가 반환
- `calc_excess_return(stock_df, bench_df, date, hold_days)`: 종목 수익률 − 벤치마크 수익률

---

## 7. 데이터 수집 파라미터

| 변수 | 현재값 | 파일 | 설명 |
|---|---|---|---|
| `REQ_DELAY` | `0.3` | `collect_financials.py` | DART API 요청 간 대기 시간 (초) |
| `REQ_DELAY` | `0.3` | `collect_dart_fundamentals.py` | DART API 요청 간 대기 시간 (초) |
| `max_pages` | `100` | `crawl.py` | 종목당 크롤링 최대 페이지 수 |
| `WEEKS_52` | `252` | `collect_financials.py` | 52주 고저가 계산 기준 거래일 수 |

---

## 8. 통계 검정

**파일:** `src/compare.py` — `run_significance_tests()` 함수

### 검정 대상 Pair

| 구분 | Group A | Group B | 목적 |
|---|---|---|---|
| Core | cond4 | cond1 | 컨텍스트 최대 vs 최소 |
| Core | cond4 | GoldenCross | LLM vs 기술분석 |
| Core | cond4 | Consensus | LLM vs 애널리스트 |
| Auxiliary | cond2 | cond1 | 재무지표의 기여 |
| Auxiliary | cond3 | cond1 | 재무+리포트의 기여 |
| Auxiliary | cond4 | cond4_no_reports | 리포트의 marginal 기여 |

### 검정 방법

- **Mann-Whitney U test** (primary): 비모수, fat-tail robust
- **Welch's t-test** (secondary): 모수, 분포 가정 위반 탐지용
- **Effect size**: Cliff's delta (MW용), Cohen's d (t-test용)

### 대상 지표

- Buy 신호만
- `return_20d` (절대), `excess_return_20d` (초과) 2개 metric

### 유의 수준 표기

| 기호 | 의미 |
|---|---|
| `***` | p < 0.001 |
| `**` | p < 0.01 |
| `*` | p < 0.05 |
| `.` | p < 0.10 |
| `ns` | 유의하지 않음 |

### 다중 비교 보정

각 pair는 독립적 연구 질문에 대응하므로 Bonferroni 등 보정 미적용.

---

## 논문 핵심 변수 요약

| 우선도 | 변수 | 위치 | 논문 기여 |
|---|---|---|---|
| ★★★ | `cond1~4 + cond4_no_reports` | `experiments.py` | Ablation Study 핵심 — 컨텍스트 추가/LOO 효과 |
| ★★★ | `temperature` | `llm_experiment.py` | 재현성에 직접 영향 |
| ★★★ | `MODEL` | `llm_experiment.py` | 실험 조건의 핵심 명세 |
| ★★ | `HOLD_LONG` | `llm_experiment.py` | 예측 기간 설정 (20거래일이 적절한가?) |
| ★★ | `KOSPI_INDEX / KOSDAQ_INDEX` | `utils.py` | 초과수익률 계산 기준 |
| ★★ | `BUY_GAP` | `baseline_consensus.py` | 컨센서스 임계값 민감도 분석 가능 |
| ★ | `N_REPORTS` | `baseline_consensus.py` | 컨센서스 윈도우 크기 |
| ★ | `MA_SHORT / MA_LONG` | `baseline_golden.py` | 골든크로스 파라미터 |
| ★ | 통계 검정 pair 구성 | `compare.py` | `run_significance_tests()` 내 `PAIRS` 리스트 |

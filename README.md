# LLM 기반 주식 투자 신호 생성 시스템

LLM에 제공하는 컨텍스트 유형에 따라 투자 신호 품질이 달라지는지 검증하는 **Ablation Study** 프레임워크.  
백테스팅 실험 외에 오늘 날짜 기준 실시간 신호를 생성하는 **Forward Test**와 **Streamlit 대시보드**를 포함한다.

---

## 개요

### 연구 질문

> "LLM에 더 많은 재무 컨텍스트를 제공할수록 투자 신호의 수익률·정확도가 향상되는가?"

### 전체 흐름

```
데이터 수집 → LLM 백테스팅 (cond1~4) → 성과 비교 (compare.py)
                                        ↓
              오늘 기준 실시간 신호 (Forward Test) → Streamlit 대시보드
```

---

## 실험 설계 (Ablation Study)

4가지 조건으로 동일한 LLM에 서로 다른 컨텍스트를 제공하고 성과를 비교한다.

| 조건 | 추가 컨텍스트 | 세부 항목 |
|------|-------------|-----------|
| **cond1** | 없음 | 종목명 + 현재가만 제공 (No Context) |
| **cond2** | 재무 + 기술지표 | PER / PBR / ROE / 시가총액 / 52주 위치 / 1개월 수익률 / 거래량 변화율 |
| **cond3** | + 애널리스트 리포트 | 리포트 제목 / 목표주가 (최근 30일, 최대 5건) |
| **cond4** | + DART 연간 실적 | 매출 / 영업이익 / 영업이익률 / 순이익 (전년比) / 부채비율 / 영업현금흐름 |

### 공통 조건

| 항목 | 값 |
|------|----|
| 실험 기간 | 2023-01 ~ 2025-12 (36개월) |
| 평가 시점 | 매월 첫 거래일 |
| 대상 종목 | KOSPI / KOSDAQ 대형주 20개 |
| LLM | Gemini 2.5 Flash-Lite (temperature=0.3) |
| 신호 | Buy / Neutral / Sell |
| 수익률 측정 | 신호일 +1 거래일 매수 → 5 / 20거래일 후 종가 |
| 대조군 | 컨센서스 추종 / 골든크로스 (MA5×MA20) |

### 결과

실험 결과 및 분석 일지는 [docs/experiments_log.md](docs/experiments_log.md) 참고.

---

## 프로젝트 구조

```
stock_analysis/
├── app.py                           # Streamlit 대시보드
├── src/
│   ├── utils.py                     # 공통 유틸 (TICKERS, 경로, 주가 캐시, 수익률 계산)
│   ├── experiments.py               # 실험 조건 정의 (cond1~cond4)
│   ├── context_builders.py          # LLM 프롬프트용 컨텍스트 섹션 빌더
│   │
│   ├── crawl.py                     # 네이버금융 애널리스트 리포트 크롤링 (증분)
│   ├── collect_financials.py        # DART + FDR 재무/기술지표 수집
│   ├── collect_dart_fundamentals.py # DART 사업보고서 연간 실적 수집
│   ├── update.py                    # Forward Test용 실시간 데이터 수집
│   │
│   ├── baseline_consensus.py        # 대조군 A: 컨센서스 추종 전략
│   ├── baseline_golden.py           # 대조군 B: 골든크로스 전략
│   ├── llm_experiment.py            # LLM 백테스팅 (체크포인트 재개 지원)
│   ├── compare.py                   # 조건 간 성과 비교 분석
│   └── forward_test.py              # Forward Test (오늘 기준 신호 생성)
│
├── data/                            # 수집 데이터 (gitignore)
│   ├── financials/                  # 재무 + 기술지표 CSV
│   ├── price/                       # 주가 캐시 CSV
│   ├── reports/                     # 애널리스트 리포트 CSV
│   └── dart_fundamentals/           # DART 연간 실적 CSV
├── results/                         # 실험 결과 (gitignore)
│   ├── baseline/                    # 대조군 수익률
│   ├── experiment/cond{1-4}/        # LLM 실험 결과 (체크포인트 포함)
│   ├── analysis/                    # 비교 분석 CSV
│   └── forward/                     # Forward Test 결과 JSON
├── docs/
│   └── experiments_log.md           # 실험 일지
├── docs_cache/                      # DART API 법인코드 캐시 (gitignore)
├── EXPERIMENT_VARS.md               # 실험 변수 정리
├── requirements.txt
└── .env                             # 환경변수 (gitignore)
```

---

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
```

`.env` 파일을 생성하고 API 키를 입력한다.

```
DARTS_API_KEY=your_dart_api_key
GEMINI_API_KEY=your_gemini_api_key
```

### 2. 데이터 수집

```bash
# 애널리스트 리포트 크롤링 (네이버금융, 증분 업데이트)
python src/crawl.py

# 재무/기술지표 수집 (DART + FinanceDataReader)
python src/collect_financials.py

# DART 연간 실적 수집 (cond4용)
python src/collect_dart_fundamentals.py
```

### 3. 베이스라인 실험 (대조군)

```bash
python src/baseline_consensus.py   # 컨센서스 추종
python src/baseline_golden.py      # 골든크로스 (MA5×MA20)
```

### 4. LLM 백테스팅

```bash
python src/llm_experiment.py --cond cond1
python src/llm_experiment.py --cond cond2
python src/llm_experiment.py --cond cond3
python src/llm_experiment.py --cond cond4
```

- 중단 후 재개 가능: 완료된 (ticker, signal_date) 쌍은 자동 스킵  
- 단일 종목 테스트 (프롬프트 출력 확인): `--test` 플래그 추가

```bash
python src/llm_experiment.py --cond cond4 --test
```

### 5. 성과 비교 분석

```bash
# cond1~4 전체 + 섹터·종목별 분석
python src/compare.py --all

# 특정 조건까지만 비교
python src/compare.py --cond cond3

# 섹터·종목 분석 포함
python src/compare.py --cond cond4 --sector
```

결과 파일은 `results/analysis/{날짜}/` 및 `results/analysis/latest/`에 저장된다.

| 파일 | 내용 |
|------|------|
| `all_comparison.csv` | 신호별(Buy/Neutral/Sell/전체) × 조건별 수익률·Hit Rate·Sharpe |
| `full_comparison.csv` | 전략별 한 줄 요약 (대조군 포함) |
| `all_sector.csv` | 섹터별 × 조건별 성과 |
| `all_stock_buy.csv` | 종목별 × 조건별 Buy 신호 성과 |

### 6. Forward Test

오늘 날짜 기준으로 단일 종목의 LLM 신호를 실시간 생성한다.  
백테스팅 데이터(`data/financials/`)를 오염시키지 않도록 별도 수집 경로를 사용한다.

```bash
# 삼성전자, cond4 (기본값)
python src/forward_test.py --ticker 005930

# 조건 지정
python src/forward_test.py --ticker 035420 --cond cond3
```

- 결과는 `results/forward/{날짜}/{ticker}_{cond}.json`에 저장  
- 당일 동일 ticker + cond는 캐시에서 즉시 반환 (API 재호출 없음)

### 7. Streamlit 대시보드

```bash
streamlit run app.py
```

사이드바에서 종목과 분석 조건(cond1~4)을 선택한 뒤 **분석하기** 버튼을 누르면 Forward Test가 실행되고 결과가 표시된다.

| 섹션 | 표시 조건 | 내용 |
|------|----------|------|
| 신호 배지 | 항상 | Buy / Neutral / Sell + 신뢰도(%) |
| 투자 근거 | 항상 | LLM이 생성한 판단 근거 |
| 재무지표 | cond2 이상 | PER / PBR / ROE / 시가총액 / 52주 위치 등 |
| 애널리스트 리포트 | cond3 이상 | 최근 30일 리포트 제목 / 목표주가 |
| 연간 실적 | cond4 | 매출 성장률 / 영업이익률 / 부채비율 |
| 백테스팅 성과 | 항상 | 해당 종목의 과거 신호 수익률 참고 |

---

## 환경변수 (.env)

```
DARTS_API_KEY=your_dart_api_key      # DART OpenAPI 키 (https://opendart.fss.or.kr)
GEMINI_API_KEY=your_gemini_api_key   # Google Gemini API 키
```

---

## 실험 변수

조정 가능한 모든 실험 변수(LLM 모델, 온도, 보유 기간 등)는 [EXPERIMENT_VARS.md](EXPERIMENT_VARS.md) 참고.

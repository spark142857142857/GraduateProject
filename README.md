# LLM 기반 주식 투자 신호 생성 시스템

LLM(대형 언어 모델)에 제공하는 컨텍스트 유형에 따라 투자 신호 품질이 달라지는지 검증하는 Ablation Study.

## 개요

4가지 조건(cond1~cond4)으로 LLM 투자 신호를 생성하고, 20거래일 후 수익률로 성과를 비교한다.

| 조건 | 컨텍스트 | 설명 |
|------|----------|------|
| cond1 | 없음 | 종목명 + 현재가만 제공 (No Context) |
| cond2 | 재무+기술지표 | PER / PBR / ROE / 모멘텀 / 거래량 추가 |
| cond3 | + 애널리스트 리포트 | 증권사 투자의견 및 목표주가 추가 |
| cond4 | + DART 연간 실적 | 사업보고서 기반 매출 / 영업이익 / 부채비율 추가 |

- 실험 기간: 2023-01 ~ 2025-12 (월별 첫 거래일, 36개월)
- 대상 종목: KOSPI / KOSDAQ 대형주 20개
- LLM: Gemini 2.5 Flash-Lite (temperature=0.3)
- 신호: Buy / Neutral / Sell (LLM 직접 판단)
- 수익률 측정: 신호일 기준 +1 거래일 매수 → 5 / 20거래일 후

## 프로젝트 구조

```
stock_analysis/
├── src/
│   ├── utils.py                     # 공통 유틸 (TICKERS, 경로, 주가 캐시, 수익률 계산)
│   ├── crawl.py                     # 네이버금융 애널리스트 리포트 크롤링 (증분)
│   ├── update.py                    # Forward Test용 실시간 데이터 갱신
│   ├── collect_financials.py        # DART + FDR 재무/기술지표 수집
│   ├── collect_dart_fundamentals.py # DART 사업보고서 연간 실적 수집 (cond4용)
│   ├── context_builders.py          # LLM 프롬프트용 컨텍스트 섹션 빌더
│   ├── experiments.py               # 실험 조건 정의 (cond1~cond4)
│   ├── baseline_consensus.py        # 대조군 A: 컨센서스 추종 전략
│   ├── baseline_golden.py           # 대조군 B: 골든크로스 전략 (MA5 × MA20)
│   ├── llm_experiment.py            # LLM 백테스팅 실험 (체크포인트 재개 지원)
│   ├── compare.py                   # 실험 조건 간 성과 비교 분석
│   └── forward_test.py              # Forward Test (오늘 날짜 기준 신호 생성)
├── app.py                           # Streamlit 대시보드
├── data/
│   ├── financials/   # 재무+기술지표 CSV (gitignore)
│   ├── price/        # 주가 캐시 CSV (gitignore)
│   └── reports/      # 애널리스트 리포트 CSV (gitignore)
├── results/          # 실험 결과 (gitignore)
├── docs_cache/       # DART API 법인코드 캐시 (gitignore)
├── EXPERIMENT_VARS.md  # 실험 변수 정리
├── requirements.txt
├── TODO.md
└── .env              # 환경변수 (gitignore)
```

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt

# .env 파일 생성 후 API 키 입력
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

### 3. 베이스라인 실험

```bash
python src/baseline_consensus.py
python src/baseline_golden.py
```

### 4. LLM 백테스팅

```bash
# cond1~cond4 순차 실행 (중단 후 재개 가능)
python src/llm_experiment.py --cond cond1
python src/llm_experiment.py --cond cond2
python src/llm_experiment.py --cond cond3
python src/llm_experiment.py --cond cond4

# 단일 종목 테스트 (삼성전자 1건, 프롬프트 출력)
python src/llm_experiment.py --cond cond4 --test
```

### 5. 결과 비교 분석

```bash
# cond1~cond4 전체 + 섹터 분석
python src/compare.py --all

# 특정 조건까지 비교 (cond1 포함)
python src/compare.py --cond cond3

# 섹터·종목 분석 추가
python src/compare.py --cond cond4 --sector
```

### 6. Forward Test (실시간)

```bash
# 오늘 날짜 기준 단일 종목 신호 생성
python src/forward_test.py --ticker 005930
python src/forward_test.py --ticker 005930 --cond cond3

# Streamlit 대시보드 실행
streamlit run app.py
```

## 환경변수 (.env)

```
DARTS_API_KEY=your_dart_api_key      # DART OpenAPI 키 (https://opendart.fss.or.kr)
GEMINI_API_KEY=your_gemini_api_key   # Google Gemini API 키
```

## 실험 변수

조정 가능한 모든 실험 변수는 [EXPERIMENT_VARS.md](EXPERIMENT_VARS.md)를 참고하세요.

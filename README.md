# LLM 기반 주식 투자 신호 생성 시스템

LLM(대형 언어 모델)에 제공하는 컨텍스트 유형에 따라 투자 신호 품질이 달라지는지 검증하는 Ablation Study.

## 개요

4가지 조건(cond1~cond4)으로 LLM 투자 신호를 생성하고, 20거래일 후 수익률로 성과를 비교한다.

| 조건 | 컨텍스트 | 설명 |
|------|----------|------|
| cond1 | 없음 | 종목명 + 현재가만 제공 |
| cond2 | 재무+기술지표 | PER/PBR/ROE/모멘텀/거래량 추가 |
| cond3 | 공시 정보 | DART 주요 공시 제목 추가 |
| cond4 | 애널리스트 리포트 | 증권사 투자의견 추가 |

- 실험 기간: 2023-01 ~ 2025-12 (월별 첫 거래일, 36개월)
- 대상 종목: KOSPI 대형주 20개
- LLM: Gemini 2.5 Flash-Lite
- 신호: Buy / Neutral / Sell (LLM 직접 판단)
- 수익률 측정: base_date 기준 +1 거래일 매수 → 20거래일 후

## 프로젝트 구조

```
stock_analysis/
├── src/
│   ├── utils.py                  # 공통 유틸 (TICKERS, 경로, 주가 캐시)
│   ├── collect_financials.py     # DART + FDR 재무/기술지표 수집
│   ├── collect_announcements.py  # DART 공시 제목 수집 (cond3용)
│   ├── baseline_consensus.py     # 컨센서스 기반 베이스라인
│   ├── baseline_golden.py        # 골든크로스 기반 베이스라인
│   ├── llm_experiment.py         # cond1 LLM 실험
│   ├── llm_experiment_cond2.py   # cond2 LLM 실험
│   ├── compare_baselines.py      # 베이스라인 비교 분석
│   ├── compare_cond1.py          # cond1 분석
│   └── compare_cond2.py          # cond1 vs cond2 비교 분석
├── data/
│   ├── financials/   # 재무+기술지표 CSV (gitignore)
│   ├── price/        # 주가 캐시 CSV (gitignore)
│   ├── reports/      # 애널리스트 리포트 CSV (gitignore)
│   └── announcements/ # DART 공시 CSV (gitignore)
├── results/          # 실험 결과 (gitignore)
├── docs_cache/       # DART API 법인코드 캐시 (gitignore)
├── requirements.txt
├── TODO.md
└── .env.example      # 환경변수 예시
```

## 실행 방법

### 1. 환경 설정

```bash
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 입력
```

### 2. 데이터 수집

```bash
# 재무/기술지표 수집 (DART + FinanceDataReader)
python src/collect_financials.py

# DART 공시 수집 (cond3용)
python src/collect_announcements.py
```

### 3. 베이스라인 실험

```bash
python src/baseline_consensus.py
python src/baseline_golden.py
python src/compare_baselines.py
```

### 4. LLM 실험

```bash
# cond1, cond2 동시 실행 가능
python src/llm_experiment.py
python src/llm_experiment_cond2.py

# 비교 분석
python src/compare_cond2.py
```

## 필요 패키지

```
requests
beautifulsoup4
pandas
finance-datareader
matplotlib
numpy
google-genai
groq
python-dotenv
opendartreader
tqdm
```

## 환경변수 (.env)

```
DARTS_API_KEY=your_dart_api_key
GEMINI_API_KEY=your_gemini_api_key
```

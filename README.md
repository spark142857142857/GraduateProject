# LLM 기반 주식 투자 신호 생성 시스템

LLM을 활용해 한국 주식 종목의 **매수/매도 신호를 생성**하고, 백테스팅으로 그 유효성을 검증하는 시스템.  
백테스팅 실험 외에 오늘 날짜 기준 실시간 신호를 생성하는 **Forward Test**와 **Streamlit 대시보드**를 포함한다.

---

## 개요

### 연구 질문

> **메인**: LLM이 유효한 주식 투자 신호(Buy/Sell)를 생성할 수 있는가?  
> **서브**: 어떤 재무 컨텍스트 조합을 제공할 때 신호 품질이 최적화되는가?

### 전체 흐름

```
데이터 수집 → LLM 백테스팅 (cond1~4) → 성과 비교 (compare.py)
                                        ↓
              오늘 기준 실시간 신호 (Forward Test) → Streamlit 대시보드
```

---

## 실험 설계

LLM에 제공하는 재무 컨텍스트 조합을 달리하며 최적 구성을 탐색한다. 동일 LLM에 서로 다른 컨텍스트를 제공하고 성과를 비교하는 Ablation Study 방식으로 설계됐다.

| 조건 | 추가 컨텍스트 | 세부 항목 |
|------|-------------|-----------|
| **cond1** | 없음 | 종목명 + 현재가만 제공 (No Context) |
| **cond2** | 재무 + 기술지표 | PER / PBR / ROE / 시가총액 / 52주 위치 / 1개월 수익률 / 거래량 변화율 |
| **cond3** | + 애널리스트 리포트 | 리포트 제목 / 목표주가 (최근 30일, 최대 5건) |
| **cond4** | + DART 연간 실적 | 매출 / 영업이익 / 영업이익률 / 순이익 (전년比) / 부채비율 / 영업현금흐름 |
| cond4_no_reports | cond4에서 리포트 제거 (LOO ablation) | 재무지표 + DART 실적. 리포트의 marginal effect 측정용 |
| cond4_blind | 재무지표 + DART (종목명 익명화) | LLM 사전학습 편향 측정 |

### 공통 조건

| 항목 | 값 |
|------|----|
| 실험 기간 | 2023-01 ~ 2025-12 (36개월) |
| 평가 시점 | 매월 첫 거래일 |
| 대상 종목 | KOSPI / KOSDAQ 대형주 20개 |
| LLM | Gemini 2.5 Flash-Lite (temperature=0.0) |
| 신호 | Buy / Neutral / Sell (**절대 방향** 예측 — 단일 종목 데이터만 받는 LLM에 시장 대비 예측은 ill-posed) |
| 수익률 측정 | 신호일 +1 거래일 매수 → 5 / 20거래일 후 종가 |
| 대조군 | 컨센서스 추종 / 골든크로스 (MA5×MA20) |
| 평가 지표 | 절대 수익률(return_20d) + **시장 대비 초과 수익률**(excess_return_20d, 알파) |
| 벤치마크 | KOSPI(KS11) / KOSDAQ(KQ11) — 상장 시장별 분리 |

### 결과

실험 결과 및 분석 일지는 [docs/experiments_log.md](docs/experiments_log.md) 참고.

---

## 프로젝트 구조

```
stock_analysis/
├── app.py                           # Streamlit 대시보드
├── src/
│   ├── utils.py                     # 공통 유틸 (TICKERS, 경로, 주가 캐시, 수익률 계산)
│   ├── experiments.py               # 실험 조건 정의 (cond1~cond4 + ablation)
│   ├── context_builders.py          # LLM 프롬프트용 컨텍스트 섹션 빌더
│   ├── prompt.py                    # LLM 프롬프트 구성 요소 (역할·판단기준·confidence)
│   │
│   ├── collect/                     # 데이터 수집
│   │   ├── crawl.py                 # 네이버금융 애널리스트 리포트 크롤링 (증분)
│   │   ├── collect_financials.py    # DART + FDR 재무/기술지표 수집
│   │   ├── collect_dart_fundamentals.py  # DART 사업보고서 연간 실적 수집
│   │   └── update.py               # Forward Test용 실시간 데이터 수집
│   │
│   └── experiment/                  # 실험 실행 및 분석
│       ├── baseline_consensus.py    # 대조군 A: 컨센서스 추종 전략
│       ├── baseline_golden.py       # 대조군 B: 골든크로스 전략
│       ├── llm_experiment.py        # LLM 백테스팅 (체크포인트 재개 지원)
│       ├── compare.py               # 기술통계 비교 (평균·Hit·Sharpe, 섹터·종목)
│       ├── significance.py          # 추론통계 (유의성 검정: Mann-Whitney·Welch·effect size)
│       ├── breakdown.py             # 다축 분해 (연도별·시장국면별, mean+median 병기)
│       ├── forward_test.py          # Forward Test (오늘 기준 단일 종목 신호 생성)
│       ├── forward_run_all.py       # Forward 일괄 실행 (전 종목 × 5조건, 주간 반복)
│       ├── forward_verify.py        # Forward 입력 정보 신선도·정합성 검증
│       └── forward_eval.py          # Forward 성숙 신호 평가 (신호일 이후 실제 수익률·적중)
│
├── data/                            # 수집 데이터
│   ├── financials/                  # 재무 + 기술지표 CSV
│   ├── price/                       # 주가 캐시 CSV
│   ├── reports/                     # 애널리스트 리포트 CSV
│   └── dart_fundamentals/           # DART 연간 실적 CSV
├── results/                         # 실험 결과
│   ├── baseline/                    # 대조군 수익률
│   ├── experiment/cond{1-4}/        # LLM 실험 결과 (체크포인트 포함)
│   ├── analysis/                    # 비교 분석 CSV
│   └── forward/                     # Forward Test 결과 JSON
├── docs/
│   └── experiments_log.md           # 실험 일지
├── docs_cache/                      # DART API 법인코드 캐시 (gitignore)
├── EXPERIMENT_VARS.md               # 실험 변수 정리
├── pyproject.toml                   # 의존성 정의 (uv)
├── uv.lock                          # 정확한 버전 고정 파일
└── .env                             # 환경변수 (gitignore)
```

---

## 실행 방법

### 1. 환경 설정

[uv](https://docs.astral.sh/uv/) 설치 후 의존성을 한 번에 설치한다.

```bash
# uv 설치 (최초 1회)
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 + 의존성 설치
uv sync
```

`.env` 파일을 생성하고 API 키를 입력한다.

```
DARTS_API_KEY=your_dart_api_key
GEMINI_API_KEY=your_gemini_api_key
# 멀티모델 비교 시에만 추가 (쓰는 provider의 키만 필요 — lazy)
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 2. 데이터 수집

```bash
# 애널리스트 리포트 크롤링 (네이버금융, 증분 업데이트)
python src/collect/crawl.py

# 재무/기술지표 수집 (DART + FinanceDataReader)
python src/collect/collect_financials.py

# DART 연간 실적 수집 (cond4용)
python src/collect/collect_dart_fundamentals.py
```

### 3. 베이스라인 실험 (대조군)

```bash
python src/experiment/baseline_consensus.py   # 컨센서스 추종
python src/experiment/baseline_golden.py      # 골든크로스 (MA5×MA20)
```

### 4. LLM 백테스팅

```bash
python src/experiment/llm_experiment.py --cond cond1
python src/experiment/llm_experiment.py --cond cond2
python src/experiment/llm_experiment.py --cond cond3
python src/experiment/llm_experiment.py --cond cond4
python src/experiment/llm_experiment.py --cond cond4_no_reports
```

- 중단 후 재개 가능: 완료된 (ticker, signal_date) 쌍은 자동 스킵  
- 단일 종목 테스트 (프롬프트 출력 확인): `--test` 플래그 추가

```bash
python src/experiment/llm_experiment.py --cond cond4 --test
```

### 5. 성과 비교 분석

기술통계는 `compare.py`, 추론통계(유의성 검정)는 `significance.py`, 다축 분해(연도·국면)는 `breakdown.py`로 분리돼 있다. 3년 총합은 국면 효과를 가릴 수 있어 breakdown으로 보완한다.

```bash
# 기술통계 — cond1~4 전체 + 섹터·종목별 분석
python src/experiment/compare.py --all

# 특정 조건까지만 비교
python src/experiment/compare.py --cond cond3

# 섹터·종목 분석 포함
python src/experiment/compare.py --cond cond4 --sector

# 추론통계 — 조건 간 차이의 유의성 검정 (별도 실행)
python src/experiment/significance.py --all

# 다축 분해 — 연도별 / 시장 국면별 (mean+median 병기)
python src/experiment/breakdown.py
```

**멀티모델**: 백테스팅·분석 모두 `--model`로 모델 지정 (기본값 앵커 `gemini-2.5-flash-lite`). 모델 접두어로 provider 자동 분기 (gemini/gemma→Google, gpt→OpenAI, claude→Anthropic). 교차 제공사는 `.env`에 `OPENAI_API_KEY`·`ANTHROPIC_API_KEY` 필요 (쓰는 provider만). 예: `python src/experiment/llm_experiment.py --cond cond4 --model claude-haiku-4-5`.

결과는 모델별로 분리 저장된다:
- 실험: `results/experiment/{cond}/{model}/{latest|날짜}/`
- 분석: `results/analysis/{model}/{latest|날짜}/`

| 파일 | 생성 | 내용 |
|------|------|------|
| `all_comparison.csv` | compare.py | 신호별(Buy/Neutral/Sell/전체) × 조건별 수익률·Hit Rate·Sharpe |
| `full_comparison.csv` | compare.py | 전략별 한 줄 요약 (대조군 포함) |
| `all_sector.csv` | compare.py | 섹터별 × 조건별 성과 |
| `all_stock_buy.csv` | compare.py | 종목별 × 조건별 Buy 신호 성과 |
| `all_significance.csv` | significance.py | 통계적 유의성 검정 결과 (Mann-Whitney, Welch's t-test, effect size) |
| `breakdown_yearly.csv` | breakdown.py | 연도별(2023~25) × 조건별 Buy/Sell 성과 (mean+median) |
| `breakdown_regime.csv` | breakdown.py | 시장 국면별(상승/하락) × 조건별 Buy/Sell 성과 |

### 6. Forward Test

오늘 날짜 기준으로 LLM 신호를 실시간 생성하고, 일정 기간 후 실제 수익률로 검증한다. 백테스팅과 **동일 모델·프롬프트**로 일관성을 유지한다 (앵커: gemini-2.5-flash-lite).

**주간 워크플로우 (일요일 저녁 권장):**

```bash
python src/collect/crawl.py                # ① 애널리스트 리포트 최신화 (증분) — cond3/4에 필요
python src/experiment/forward_run_all.py   # ② 전 종목 × 5조건 신호 생성 (DART는 자동 갱신)
python src/experiment/forward_verify.py    # ③ 넣은 정보 신선도·정합성 점검 (현재가·ROE·리포트·DART)
# (4주 뒤부터) python src/experiment/forward_eval.py   # ④ 성숙분 실제 수익률·적중 평가
```

단일 종목 테스트: `python src/experiment/forward_test.py --ticker 005930 --cond cond3`

- 신호 저장: `results/forward/{날짜}/{model}/{ticker}_{cond}.json` (당일 동일 ticker+cond+model은 캐시 반환)
- 평가 저장: `results/forward/evaluation.csv` — 미성숙(20거래일 미경과) 신호는 pending
- 주간 반복 시 20거래일 보유구간이 겹쳐 표본이 독립이 아니므로 **실전 참고용** (유의성 검정은 백테스트가 담당)
- ⚠️ **리포트는 자동 갱신 안 됨** → forward 전 반드시 `crawl.py` 실행 (안 하면 cond3/4가 빈 리포트). `forward_verify.py`가 리포트 0건 시 경고.
- ✅ **백테스트 데이터 비오염**: forward는 재무지표·DART를 모두 **인메모리로 계산**하며 `data/financials/`·`data/dart_fundamentals/`에 쓰지 않는다 → 백테스트 데이터(2023-2025)가 순수하게 유지됨. 리포트만 `crawl.py`로 증분 갱신(백테스트는 신호일 30일 창으로 필터하여 무영향).

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
DARTS_API_KEY=your_dart_api_key           # DART OpenAPI 키 (https://opendart.fss.or.kr)
GEMINI_API_KEY=your_gemini_api_key        # Google Gemini/Gemma API 키 (앵커 모델)
OPENAI_API_KEY=your_openai_api_key        # (선택) gpt-* 모델 사용 시
ANTHROPIC_API_KEY=your_anthropic_api_key  # (선택) claude-* 모델 사용 시
```

> 모델명 접두어로 provider가 정해진다: `gemini-*`/`gemma-*`→Google, `gpt-*`→OpenAI, `claude-*`→Anthropic. 그 외 접두어는 지원하지 않는다(즉시 에러). 키는 **실제 사용하는 provider의 것만** 있으면 된다.

---

## 실험 변수

조정 가능한 모든 실험 변수(LLM 모델, 온도, 보유 기간 등)는 [EXPERIMENT_VARS.md](EXPERIMENT_VARS.md) 참고.

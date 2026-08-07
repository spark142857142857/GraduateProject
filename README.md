# LLM 기반 주식 투자 신호 생성 시스템

LLM을 활용해 한국 주식 종목의 **매수/매도 신호를 생성**하고, 백테스팅으로 그 유효성을 검증하는 시스템.  
백테스팅 실험 외에 오늘 날짜 기준 실시간 신호를 생성하는 **Forward Test**와 **Streamlit 대시보드**를 포함한다.

**목차**: [개요](#개요) · [실험 설계](#실험-설계) · [프로젝트 구조](#프로젝트-구조) · [실행 방법](#실행-방법) · [환경변수](#환경변수-env) · [실험 변수](#실험-변수)

---

## 개요

### 연구 질문

> **메인**: LLM이 유효한 주식 투자 신호(Buy/Sell)를 생성할 수 있는가?  
> **서브**: 어떤 재무 컨텍스트 조합을 제공할 때 신호 품질이 최적화되는가?

### 전체 흐름

```
데이터 수집 → LLM 백테스팅 (4모델 × 5조건) → 성과 비교·유의성 검정 (compare / significance / breakdown)
                                             ↓
              오늘 기준 실시간 신호 (Forward Test) → Streamlit 대시보드 (탭 5개)
```

---

## 실험 설계

LLM에 제공하는 재무 컨텍스트 조합을 달리하며 최적 구성을 탐색한다. 동일 LLM에 서로 다른 컨텍스트를 제공하고 성과를 비교하는 Ablation Study 방식으로 설계됐다.

| 조건 | 추가 컨텍스트 | 세부 항목 |
|------|-------------|-----------|
| **cond1** | 없음 | 종목명 + 현재가만 제공 (No Context) |
| **cond2** | 재무 + 기술지표 | PER / PBR / ROE / 시가총액 / 52주 위치 / 1개월 수익률 / 거래량 변화율 |
| **cond3** | + 애널리스트 리포트 | 리포트 제목 / 목표주가 (최근 30일, 최대 5건) |
| **cond4** | + DART 분기 실적 | 매출 / 영업이익 / 영업이익률 / 순이익 (전년동기比) / 부채비율 / 영업현금흐름 (최근 정기보고서 = 단일분기, 4~5월만 연간) |
| cond4_no_reports | cond4에서 리포트 제거 (LOO ablation) | 재무지표 + DART 실적. 리포트의 marginal effect 측정용 |

> ~~cond4_blind (종목명 익명화)~~는 **미실행**. 종목명을 가려도 시가총액·현재가로 부분 식별이
> 가능해 완전한 blind가 성립하지 않는다. 사전학습 편향 검증은 대신 **cond1 신호의 종목 편중
> 분석**으로 수행했다 ([experiments_log.md](docs/experiments_log.md) "cond1 사전학습 편향 검증").

### 공통 조건

| 항목 | 값 |
|------|----|
| 실험 기간 | 2023-01 ~ 2025-12 (36개월) |
| 평가 시점 | 매월 첫 거래일 |
| 대상 종목 | KOSPI / KOSDAQ 대형주 20개 |
| LLM | **4모델** (temperature=0.0) — Gemini 2.5 Flash-Lite(앵커) / Gemma 4 31B / GPT-5.4-mini / Claude Haiku 4.5 |
| 신호 | Buy / Neutral / Sell (**절대 방향** 예측 — 단일 종목 데이터만 받는 LLM에 시장 대비 예측은 ill-posed) |
| 수익률 측정 | 신호일 +1 거래일 매수 → 5 / 20거래일 후 종가 |
| 대조군 | 컨센서스 추종 / 골든크로스 (MA5×MA20) |
| 평가 지표 | 절대 수익률(return_20d) + **시장 대비 초과 수익률**(excess_return_20d, 알파) |
| 벤치마크 | KOSPI(KS11) / KOSDAQ(KQ11) — 상장 시장별 분리 |

### 결과 요약

4모델 × 5조건 백테스트(모델당 3,600콜) 완료 기준.

- **LLM > 컨센서스**: cond4 Buy가 애널리스트 컨센서스를 **4모델 중 3모델에서 유의하게** 상회
  (gemini p=0.013\*, gemma p=0.002\*\*, claude p=0.019\*). GPT만 비유의(p=0.121)이나 평균 차이는
  +0.69%p로 방향이 같으며, Buy 표본이 115건으로 작아 검정력이 부족한 것으로 해석된다.
- **애널리스트 리포트는 신호를 늘리되 품질에 기여하지 않는다**: cond4 vs cond4_no_reports가
  4모델 전부 비유의. 짝 비교(부호검정)에서도 4모델 전부 비유의로 재확인.
  cond2→cond3에서 Buy 건수가 2~4배 늘면서 평균 수익률은 오히려 하락한다.
- **컨텍스트는 사전학습 편향을 밀어낸다**: 종목명만 준 cond1에서 Buy 대상 종목의 실제
  성과순위가 2.7위(20종목 중)로 극단적으로 쏠리지만, 재무 데이터를 주면 cond4에서
  10.3위(무작위 기대값 10.5)로 수렴한다(p=6.9×10⁻¹⁰).
- **신호 수준과 포트폴리오 수준은 다른 그림**: cond4는 컨센서스를 이기지만, 실제로 운용하면
  어느 모델도 "전 종목 동일가중 보유"(+188.6%)를 넘지 못한다. 다만 낙폭은 줄여준다
  (gemini cond4 MDD −6.4% vs 벤치마크 −7.6%).

상세 일지는 [docs/experiments_log.md](docs/experiments_log.md),
Forward Test 운영 이력은 [docs/forward_log.md](docs/forward_log.md),
남은 작업·한계는 [docs/TODO.md](docs/TODO.md) 참고.

---

## 프로젝트 구조

```
stock_analysis/
├── app.py                           # Streamlit 진입점 (page_config·탭 배치만)
├── app_ui/                          # 대시보드 구현 (탭별 분리)
│   ├── __init__.py                  # 부트스트랩 (sys.path·.env·ROOT_DIR)
│   ├── shared.py                    # 탭 공용 상수·로더 (둘 이상의 탭이 쓰는 것만)
│   ├── tab_matrix.py                # ① 백테스트 신호 매트릭스
│   ├── tab_analyze.py               # ② 개별 종목 분석 (실시간 신호 생성, 유일하게 API 사용)
│   ├── tab_report.py                # ③ 백테스트 성과·모델 비교
│   ├── tab_portfolio.py             # ④ 포트폴리오 백테스트 (누적 곡선·MDD)
│   └── tab_flip.py                  # ⑤ 조건 간 신호 전이 (짝 비교 검정)
├── src/
│   ├── utils.py                     # 공통 유틸 (TICKERS, 경로, 주가 캐시, 수익률 계산)
│   ├── experiments.py               # 실험 조건 정의 (cond1~cond4 + ablation)
│   ├── context_builders.py          # LLM 프롬프트용 컨텍스트 섹션 빌더
│   ├── prompt.py                    # LLM 프롬프트 구성 요소 (역할·판단기준·confidence)
│   │
│   ├── collect/                     # 데이터 수집
│   │   ├── crawl.py                 # 네이버금융 애널리스트 리포트 크롤링 (증분)
│   │   ├── collect_financials.py    # DART + FDR 재무/기술지표 수집
│   │   ├── collect_dart_fundamentals.py  # DART 정기보고서 분기 실적 수집 (최근 공시)
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
│   └── dart_fundamentals/           # DART 분기 실적 CSV (최근 정기보고서)
├── results/                         # 실험 결과
│   ├── baseline/                    # 대조군 수익률
│   ├── experiment/cond{1-4}/        # LLM 실험 결과 (체크포인트 포함)
│   ├── analysis/                    # 비교 분석 CSV
│   └── forward/                     # Forward Test 결과 JSON
├── docs/
│   ├── experiments_log.md           # 실험 일지 (백테스트)
│   └── forward_log.md               # Forward Test 운영 일지
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

```bash
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

# DART 분기 실적 수집 (cond4용, 최근 정기보고서)
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

탭 5개로 구성된다. **API를 호출하는 것은 탭②의 "분석하기" 버튼 하나뿐**이며, 나머지 네 탭은
`results/` 아래 저장된 결과를 읽기만 한다(즉시 표시, 비용 없음).

| 탭 | 내용 | API |
|----|------|-----|
| ① 백테스트 신호 매트릭스 | 특정 신호일의 20종목 × 5조건 판단을 한 화면에. 실제 20거래일 수익률과 방향 적중(✓/✗) 병기 | 없음 |
| ② 개별 종목 분석 | **KRX 상장 보통주 전 종목(약 2,760개)** 중 하나를 골라 오늘 기준 신호 생성. 신호 배지·투자 근거·재무지표(cond2+)·리포트(cond3+)·DART 실적(cond4)·해당 종목 백테스트 성과 | **호출** |
| ③ 백테스트 성과·모델 비교 | 4모델 Buy 성과, 무기술 벤치마크·베이스라인 대비, 유의성 검정, 연도별·국면별 분해 | 없음 |
| ④ 포트폴리오 백테스트 | 신호대로 운용했을 때의 누적 곡선 + MDD + 연환산 변동성. 거래비용 왕복 0.25% 토글 | 없음 |
| ⑤ 조건 간 신호 전이 | 컨텍스트 추가로 판단이 어떻게 바뀌었나. 3×3 전이표 + 부호검정·Wilcoxon(짝 비교) | 없음 |

> 탭②에서 생성된 신호는 `results/forward_demo/`로 격리되어 정식 평가 표본(`forward_eval.py`)에
> 섞이지 않는다. 임의 시점·임의 종목 클릭이 통계를 오염시키는 것을 차단하기 위함이다.
>
> 탭①은 forward 캐시가 아니라 백테스트 결과를 읽는다. forward는 신호 생성을 2026-08-02로
> 종료해 시간이 지나면 낡은 날짜가 화면에 남지만, 백테스트 기간(2023-01~2025-12)은 설계상
> 고정이라 낡지 않는다.
>
> 매트릭스를 첫 탭에 둔 이유는 개별 분석이 버튼을 누르기 전까지 빈 화면이기 때문이다. 앱을
> 열자마자 20종목 × 5조건이 채워진 화면이 나오고, 비용이 드는 탭이 첫 화면에서 눌리지 않는다.
>
> 화면 상단에는 투자 조언이 아니라는 고지를 상시 표시한다.
>
> 우선주는 목록에서 제외한다. 별도 종목코드를 갖지만 DART 재무제표는 보통주 기준
> 하나뿐이라 EPS가 매칭되지 않아 PER이 비고(삼성전자우처럼 시총 100조가 넘어도
> 마찬가지), 발행주식수도 보통주 기준이라 시가총액이 어긋난다. 판별은 KRX 종목코드
> 규약(보통주 끝자리 0)을 쓴다. 60종목 실측에서 "우리만 PER이 없는" 7건 중 5건이
> 우선주였다.
>
> **탭②의 대상은 백테스트 20종목이 아니라 KRX 상장 보통주 전 종목이다.** 백테스트와 forward는
> 방법을 검증하는 통제 실험이고, 분석 자체는 임의 종목에 적용되어야 하기 때문이다. 다만
> 20종목 밖에서는 ① 과거 성과 이력을 붙일 수 없고 ② 애널리스트 리포트 커버리지가 낮아
> cond3·cond4의 리포트 섹션이 비는 경우가 있다. 둘 다 화면에서 명시한다. 리포트가 없는
> 종목은 최근 30일치를 그 자리에서 수집하며(`crawl.fetch_reports`), 그래도 0건이면
> "이 조건은 리포트를 포함하지만 해당 정보 없이 판단했다"고 경고를 띄운다.

---

## 환경변수 (.env)

```bash
DARTS_API_KEY=your_dart_api_key           # DART OpenAPI 키 (https://opendart.fss.or.kr)
GEMINI_API_KEY=your_gemini_api_key        # Google Gemini/Gemma API 키 (앵커 모델)
OPENAI_API_KEY=your_openai_api_key        # (선택) gpt-* 모델 사용 시
ANTHROPIC_API_KEY=your_anthropic_api_key  # (선택) claude-* 모델 사용 시
```

> 모델명 접두어로 provider가 정해진다: `gemini-*`/`gemma-*`→Google, `gpt-*`→OpenAI, `claude-*`→Anthropic. 그 외 접두어는 지원하지 않는다(즉시 에러). 키는 **실제 사용하는 provider의 것만** 있으면 된다.

---

## 실험 변수

조정 가능한 모든 실험 변수(LLM 모델, 온도, 보유 기간 등)는 [EXPERIMENT_VARS.md](EXPERIMENT_VARS.md) 참고.

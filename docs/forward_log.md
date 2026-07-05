# Forward Test 운영 일지

매주(일요일 저녁) 실행하는 Forward Test의 실행 이력·이슈·특이사항을 기록한다.
통계적 유의성 검정은 백테스트(`experiments_log.md`)가 담당하며, 이 로그는
운영상 발견 사항(모델별 실패 패턴, 데이터 소스 변경 영향 등)을 추적하는 목적.

---

## 2026-07-05 (첫 실행)

### 실행 내역

- **모델**: gemini-2.5-flash-lite(앵커) · gemma-4-31b-it · gpt-5.4-mini · claude-haiku-4-5 (4종)
- **조건**: cond1 ~ cond4 + cond4_no_reports (5조건) × 20종목 = 100건/모델
- **순서**: `crawl.py`(리포트 최신화) → 모델별 `forward_run_all.py` → `forward_verify.py`
- **저장 구조**: `results/forward/{날짜}/{model}/{ticker}_{cond}.json` (같은 날 평면 구조 → 모델별 하위폴더로 재정리, 커밋 `a529666`)

### 결과

| 모델 | 성공 | 비고 |
|------|------|------|
| gemini-2.5-flash-lite | 100/100 | — |
| gemma-4-31b-it | 100/100 | 무료 tier rate-limit로 소요 시간 김 (재시도로 커버됨) |
| gpt-5.4-mini | 100/100 | temperature=0 거부 없음 |
| claude-haiku-4-5 | 96/100 | **cond1 4건 실패** (아래 이슈 참고) |

### ⚠️ 알려진 이슈: Claude Haiku의 cond1(무컨텍스트) 거절

**증상**: 삼성SDI·LG화학·HYBE·알테오젠 4종목의 cond1(종목명+현재가만 제공)에서 `call_llm`이 JSON 파싱 실패로 3회 재시도 후 실패. 원인은 API 오류가 아니라 **Claude가 "판단할 데이터가 부족하다"는 산문 거절 응답**을 반환한 것 (JSON 형식을 안 지킴).

**해석**: cond1은 원래 "컨텍스트 없음" 베이스라인으로 설계된 조건 — 정보 부족이 의도된 설정인데, Claude Haiku가 다른 모델(Gemini/GPT)과 달리 이 조건에서 20% 확률(4/20)로 형식을 어기며 거절함. 모델별 행동 차이로 판단, 코드 버그 아님.

**처리**: 재시도 없이 결측 처리 (사용자 결정). forward_eval 집계 시 해당 4건은 자연히 신호 없음으로 반영됨.

**향후 참고**: 매주 반복 시 Claude cond1에서 유사 거절이 재현되는지 관찰 필요. 반복적으로 발생하면 프롬프트 레벨 대응(예: "정보가 부족해도 반드시 JSON으로 최선의 추정 응답" 문구 추가) 검토 여지 있으나, 현재는 **동결된 프롬프트를 건드리지 않기로** 결정.

---

## 2026-07-05 (같은 날, 분기 DART 전환 후 재실행)

DART 실적을 연간→분기로 전환(커밋 `7ecd44d`)함에 따라, 영향받는 두 조건만 재실행.

- **대상**: cond4 · cond4_no_reports (dart_fundamentals 사용 조건만 — cond1~3은 무영향)
- **모델**: 4종 전체 (20종목 × 2조건 = 40콜/모델 × 4 = 160콜)
- **결과**: 전 모델 20/20 성공 (cond1과 달리 실데이터가 있어 Claude 거절 없음)
- **검증**: `forward_verify.py` 재통과. 삼성전자 사례로 실시간 재조회 대조 → 저장된 JSON 값과 정확히 일치 확인 (revenue_yoy=69.16, operating_margin=42.75, debt_ratio=30.15, fiscal_period="2026 1분기")
- 커밋: `4ece002`

---

## 다음 실행 체크리스트 (매주 일요일)

```bash
python src/collect/crawl.py                    # ① 리포트 최신화 (필수 — 자동 안 됨)
python src/experiment/forward_run_all.py --model gemini-2.5-flash-lite
python src/experiment/forward_run_all.py --model gemma-4-31b-it
python src/experiment/forward_run_all.py --model gpt-5.4-mini
python src/experiment/forward_run_all.py --model claude-haiku-4-5
python src/experiment/forward_verify.py        # 입력 정보 신선도·정합성 점검
# (4주 뒤부터) python src/experiment/forward_eval.py
```

- Claude cond1 소수 실패는 예상된 패턴 — 재시도 없이 결측 처리
- 실패/이상 패턴 발견 시 이 문서에 날짜별로 추가

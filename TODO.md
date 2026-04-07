# TODO

## 완료

- [x] `crawl.py` — 증분 업데이트 (since_date 기반, nid 중복 제거)
- [x] `update.py` — Forward Test용 실시간 데이터 갱신 구현
- [x] `collect_financials.py` — 재무/기술지표 수집 (DART + FDR)
- [x] `collect_dart_fundamentals.py` — DART 연간 실적 수집 (cond4용)
- [x] `context_builders.py` — LLM 프롬프트 컨텍스트 빌더
- [x] `llm_experiment.py` — 체크포인트 재개 / cond1~cond4 통합 실험
- [x] `compare.py` — 조건별 성과 비교 + 섹터/종목 분석
- [x] `forward_test.py` — 오늘 날짜 기준 실시간 신호 생성
- [x] `app.py` — Streamlit 대시보드
- [x] 전체 코드 리뷰 및 버그 수정 (단위 오류, look-ahead bias, NaN 처리 등)
- [x] `EXPERIMENT_VARS.md` — 실험 변수 정리 문서

## 논문 제출 전

- [ ] `temperature` 결정 — 현재 0.3, 재현성 필요 시 0으로 변경
- [ ] `baseline_consensus.py` / `baseline_golden.py` — return_5d 추가 여부 결정
- [ ] 논문 본문에 실험 모델명(`gemini-2.5-flash-lite`) 및 temperature 명시

## 선택

- [ ] Streamlit Community Cloud 배포 (st.secrets로 API 키 전환 필요)
- [ ] 전체 종목 한눈에 보기 탭 추가 (app.py)
- [ ] `TICKERS` / `SECTORS` 단일 정의로 통합 (utils.py → compare.py 파생)
- [ ] `COND_LABELS` 단일 정의로 통합 (현재 compare.py / app.py 중복)

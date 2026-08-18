---
name: dashboard-verification
description: >-
  Procedure for verifying FastAPI backend APIs and Streamlit frontend dashboards
  using Antigravity Browser subagent or local verification tools.
  Use when dashboard views, charts, UI components, or dashboard API endpoints are modified.
---

# Dashboard & API Verification Guide

이 스킬은 FastAPI 백엔드와 Streamlit 대시보드 화면을 수정하거나 연동 기능을 변경했을 때, **Antigravity Browser Agent**를 활용하여 실제 브라우저 상에서 UI/UX와 API 응답이 정상적으로 동작하는지 종단간(End-to-End) 검증하는 절차입니다.

---

## 1. 사전 준비 및 서버 구동

1. **회귀 테스트 통과 확인**:
   - UI 검증 전 `regression-verification` 스킬에 따라 `python -m pytest`가 성공했는지 먼저 확인합니다.

2. **FastAPI 백엔드 구동**:
   - 백그라운드 프로세스로 FastAPI 서버를 실행합니다:
     ```powershell
     .\run_api.ps1
     # 또는
     .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
     ```
   - Swagger 문서 엔드포인트(`http://127.0.0.1:8000/docs`) 또는 헬스체크(`http://127.0.0.1:8000/health`) 정상 응답 확인.

3. **Streamlit 대시보드 구동**:
   - 백그라운드 프로세스로 대시보드를 실행합니다:
     ```powershell
     .\run_dashboard.ps1
     # 또는
     .\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.address 127.0.0.1 --server.port 8501
     ```

---

## 2. Browser Agent 검증 절차

`browser_subagent` 도구를 호출하여 실제 대시보드 화면을 렌더링하고 조작합니다.

### 주요 검증 항목
1. **대시보드 메인 접속**: `http://127.0.0.1:8501` 정상 로딩 확인
2. **페이지 전환 및 조작**:
   - `1_트렌드_대시보드`: 상위 트렌드 목록, 필터, 차트 렌더링
   - `2_AI_분석`: 기존 AI 분석 결과 카드 및 조회
   - `3_키워드_상세`: 키워드별 시계열 및 소스별 분포
   - `4_파이프라인_상태`: 수집/스케줄러/NER 통계
   - `5_여행_기회_V2`:
     - Funnel 메트릭 카드(Raw, Quality, Rule, Semantic, High Precision, Gemini Eligible, Final Accept) 정상 표시
     - LLM 감소율 및 연율화 추정치 표시
     - Accept 및 Review 결과 카드 레이아웃
     - High Precision 후보 테이블 렌더링

3. **UI 상태별 예외 처리 점검**:
   - **정상 데이터 상태**: 테이블, 카드, 메트릭이 깨짐 없이 표시되는지 확인
   - **데이터 없음 (Empty state)**: 안내 메시지(`st.info`)가 자연스럽게 표시되는지 확인
   - **API 오류 / 타임아웃**: `st.error`로 친절한 오류 안내가 노출되는지 확인
   - **필터 및 셀렉트박스**: 선택 변경 시 화면 갱신 여부

---

## 3. 핵심 안전 수칙

1. **Gemini 분석 버튼 자동 클릭 금지**:
   - 대시보드의 **"선택 후보 최종 AI 분석"** 또는 **"Gemini AI 분석 실행"** 버튼은 실제 외부 LLM 호출과 비용이 발생하므로, **사용자가 명시적으로 허용한 경우가 아니면 브라우저 상에서 자동 클릭하지 않습니다.**

2. **콘솔 및 네트워크 오류 점검**:
   - 브라우저 콘솔 오류, 500 내부 서버 오류, JavaScript 예외가 발생하는지 확인합니다.

3. **시각적 증거 확보**:
   - 코드만 보고 UI가 정상이라고 속단하지 않으며, 필요 시 스크린샷을 확인하여 레이아웃 무결성을 검증합니다.

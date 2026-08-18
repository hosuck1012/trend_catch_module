---
name: regression-verification
description: >-
  Execution and validation procedure for the full pytest regression test suite in Trend Catch Module.
  Use after any code modifications, refactorings, or bug fixes to verify test passes,
  detect regressions, and ensure safety before git commit or deployment.
---

# Regression Verification Guide

이 스킬은 코드 변경 후 **전체 회귀 테스트(Regression Test) 및 API 동작을 검증**하여 기존 V1/V2 기능이 깨지지 않았는지 확인하고 품질 기준을 유지하기 위한 표준 절차입니다.

---

## 1. 실행 절차

### 1단계: 가상환경 확인 및 활성화
Windows PowerShell 환경에서 프로젝트 루트의 가상환경이 활성화되어 있는지 확인합니다:
```powershell
Set-Location C:\Users\owner\Desktop\trend_catch_module
.\.venv\Scripts\Activate.ps1
```

### 2단계: Targeted Test (수정 영역 우선 실행)
수정한 영역과 직접 관련된 단위 테스트를 먼저 실행하여 빠른 피드백을 확인합니다:
```powershell
# 예시: AI 분석 수정 시
python -m pytest tests/test_ai_analysis.py

# 예시: 키워드 품질 수정 시
python -m pytest tests/test_keyword_quality_v2.py

# 예시: 여행 기회 V2 수정 시
python -m pytest tests/test_travel_opportunities_v2.py tests/test_travel_ranking_v2_step3.py tests/test_final_travel_opportunity_v2_step4.py
```

### 3단계: 전체 테스트 스위트 실행
수정된 테스트가 통과되면 반드시 전체 테스트 스위트를 실행합니다:
```powershell
python -m pytest
```

---

## 2. 검증 및 판정 규칙

1. **테스트 실패 시 조치**:
   - 단 1개의 테스트라도 실패할 경우 작업을 완료되었다고 보고하거나 종료하지 않습니다.
   - 실패 원인을 로그와 스택트레이스를 통해 정밀 분석하고 범위 내에서 코드를 수정한 후 다시 전체 테스트를 실행합니다.

2. **테스트 약화 금지**:
   - 코드가 통과하지 못한다고 해서 테스트의 assertion을 임의로 삭제하거나 허용 오차를 무단으로 늘리는 등 **테스트 자체를 약화시켜 통과시키는 행위는 엄격히 금지**합니다.

3. **테스트 격리 원칙 준수**:
   - 테스트는 실제 운영 DB(`trend_engine.sqlite3`)나 외부 API(YouTube, Newsis RSS, Wikipedia, Gemini)를 직접 호출하지 않고 Mock/Fake fixture(`conftest.py` 등)를 통해 격리되어 동작해야 합니다.

4. **최종 결과 보고**:
   - 테스트 완료 후 실제 실행된 테스트 결과(통과 수, 실패 수, 경고 수, 실행 시간)를 명시적으로 보고합니다.
   - *(참고 Baseline: 약 287 passed, 1 warning — 단, 수치는 현재 테스트 스위트의 실제 실행 결과를 기준으로 판단합니다.)*

5. **Commit / Push 차단**:
   - 테스트가 실패한 상태에서는 어떠한 경우에도 `git commit`이나 `git push`를 진행하지 않습니다.

---
name: codex-review
description: >-
  Secondary code review workflow utilizing the installed and authenticated OpenAI Codex CLI.
  Use for critical architecture changes, scoring formulas, DB migrations, or Gemini integration
  to perform independent quality, regression, security, and grounding reviews.
---

# Codex Review Workflow Guide

이 스킬은 Antigravity가 작성한 코드 및 설계 변경사항에 대해 시스템에 이미 로그인되어 있는 **OpenAI Codex CLI (`codex-cli`)**를 독립적인 2차 코드 리뷰어(Secondary Reviewer)로 활용하는 표준 절차를 정의합니다.

---

## 1. 역할 분담 원칙 (Roles & Separation)

1. **Antigravity Main Agent = 구현자 (Implementer)**:
   - 요구사항 분석, 아키텍처 설계, 코드 작성, 테스트 실행, 변경사항 적용 및 최종 검증을 주도합니다.

2. **Codex CLI = 독립 검토자 (Independent Reviewer)**:
   - 객관적인 관점에서 버그, 엣지 케이스, 환각 위험, 아키텍처 일관성을 검토하고 피드백을 제공합니다.
   - Codex에게 코드를 직접 수정하게 하거나 동시에 편집하게 하지 않고, **순수 검토(Review / Audit)** 목적으로만 프롬프트를 전달합니다.

---

## 2. 주요 검토 대상 (Review Checklist)

Codex 리뷰 시 집중적으로 점검할 항목:
1. **잠재적 버그 및 논리 오류**: 파이프라인 수식 계산, 누락된 None 체크, 인덱스 에러
2. **회귀(Regression) 위험**: 기존 V1/V2 API 계약 및 데이터 모델 호환성 훼손 여부
3. **DB 데이터 무결성**: 트랜잭션 처리, 중복 저장 방지, Unsafe Delete 방지
4. **비용 및 안전장치**: 불필요한 Gemini API 호출 루프, 페이로드 초과, 캐시 누락 여부
5. **Grounding & 환각 방지**: AI 응답이 원문 컨텍스트(`SourceDocument`)에 근거하는지 여부
6. **테스트 커버리지**: 새로운 로직에 대한 Mock 기반 단위/통합 테스트 누락 여부
7. **보안 및 키 유출**: 민감한 환경변수, API 키 하드코딩 여부
8. **중복 구현 여부**: 기존 리포지토리/서비스에 이미 존재하는 로직을 재창작했는지 여부

---

## 3. 실행 방법 및 명령어

Codex CLI의 리뷰 기능을 실행할 때는 **기존의 unrelated unstaged 변경사항이 포함되지 않도록 이번 작업의 특정 파일 또는 diff로 범위를 엄격히 한정**합니다:

```powershell
# 1. 특정 파일에 대한 Codex 리뷰 요청 예시
codex exec "Review the following file for potential bugs, regressions, and security issues: app/services/travel_ranking_service.py"

# 2. 특정 git diff 범위에 대한 리뷰 요청 예시
git diff origin/main...HEAD -- app/services/travel_ranking_service.py | codex exec "Review this diff for edge cases, performance issues, and grounding verification."
```

---

## 4. 피드백 반영 원칙

1. **비판적 검토**:
   - Codex의 피드백을 무조건적으로 맹신하여 적용하지 않습니다.
   - 제안된 내용이 프로젝트의 `AGENTS.md` 규칙, 기존 아키텍처, 성능 목표에 부합하는지 Antigravity Agent가 면밀히 타당성을 평가합니다.

2. **선택적 반영 후 재검증**:
   - 유의미한 지적 사항에 대해서만 필요한 최소한의 수정을 적용합니다.
   - 수정 후 반드시 `regression-verification` 스킬에 따라 전체 pytest를 다시 실행하여 모든 테스트가 통과하는지 확인합니다.

---
name: trend-v2-development
description: >-
  Development and maintenance workflow for Trend Catch V2 pipeline, including
  keyword quality filtering, keyword context extraction, rule/semantic prefiltering,
  high-precision ranking, and final Gemini travel opportunity analysis.
  Use when creating, modifying, or refactoring Trend Catch V2 features, models, services, or APIs.
---

# Trend Catch V2 Development Guide

이 스킬은 **Trend Catch Module**의 V2 파이프라인(Keyword Quality → Context/Rule Prefilter → Local Semantic Filter → High Precision Ranking → Final Gemini Analysis)을 구현, 개선, 리팩터링할 때 준수해야 하는 엔지니어링 가이드라인과 절차를 정의합니다.

---

## 1. 기본 원칙 및 제약사항

1. **프로젝트 루트 고정**:
   - 모든 작업은 항상 `C:\Users\owner\Desktop\trend_catch_module` 내에서 수행합니다.
   - `C:\Users\owner\Desktop\Travelen` 및 기타 외부 폴더는 절대 접근하거나 수정하지 않습니다.

2. **루트 가이드 우선 준수**:
   - 루트의 `AGENTS.md`에 명시된 규칙(SQLAlchemy 2.x 문법, UTC 시간 처리, 고정된 Entity Types 등)을 항상 최우선으로 준수합니다.

3. **기존 V1 기능 하위 호환성 유지**:
   - V2 기능을 추가하더라도 기존 수집(YouTube/RSS), 주간 트렌드 계산(`WeeklyTrend`), NER, 검색 관심도 검증, V1 대시보드 API 응답 호환성을 깨뜨리지 않습니다.

4. **아키텍처 재사용 및 중복 방지**:
   - 새로운 중복 service, model, repository를 임의로 만들지 말고 기존 계층(`app/services`, `app/repositories`, `app/models`, `app/schemas`)의 기존 함수와 패턴을 먼저 조사하고 재사용합니다.

5. **비용 및 외부 API 안전장치**:
   - Gemini LLM 호출은 사전 필터와 랭킹을 통과한 **최종 적격 후보(`gemini_eligible=True`)**에만 한정합니다.
   - 불필요한 외부 크롤러(`pytrends`, 무단 기사 크롤링 등)를 추가하지 않습니다.
   - `.env`, API 키, 비밀값은 코드, 테스트, 로그에 절대 하드코딩하지 않습니다.

6. **데이터 변경 작업 시 Dry-Run 기본 적용**:
   - 컨텍스트 빌드, 사전 필터링, 랭킹, AI 분석 등 배치/DB 변경 API는 `dry_run=True`를 기본값으로 지원하여 데이터 영향도를 사전에 확인할 수 있게 합니다.

---

## 2. Trend Catch V2 파이프라인 구조

V2 파이프라인은 5단계 Funnel 구조로 동작합니다:

```text
[Raw Keywords] (WeeklyTrend / KeywordOccurrence)
  ↓
[Step 1: Keyword Quality] (Kiwi 형태소/명사구 + 불용어 + 고유명사 보존 → KeywordCandidate)
  ↓
[Step 2: Context Extraction & Prefilter] (SourceDocument 문맥 추출 → KeywordContext + Rule/Semantic 평가 → TravelOpportunityCandidate)
  ↓
[Step 3: High Precision Ranking] (Trend/Context/Convertibility/Evidence 4종 합성 + Evidence Gate + 클러스터링 → gemini_eligible 선정)
  ↓
[Step 4: Final Gemini Analysis] (구조화 프롬프트 + 원문 근거 기반 목적지/콘텐츠 제안 + 캐싱 → FinalTravelOpportunity)
  ↓
[Streamlit V2 Dashboard] (5_여행_기회_V2.py)
```

### 단계별 핵심 참조 파일
- **Step 1 (Quality)**:
  - `app/services/keyword_extraction_v2_service.py`
  - `app/services/keyword_normalization_service.py`
  - `app/services/keyword_rebuild_service.py`
  - `app/models/keyword_candidate.py`
  - `app/api/keyword_quality.py`
- **Step 2 (Context & Prefilter & Semantic)**:
  - `app/context_v2/context_extractor.py`
  - `app/context_v2/travel_rules.py`
  - `app/context_v2/travel_taxonomy.py`
  - `app/services/keyword_context_service.py`
  - `app/services/travel_prefilter_service.py`
  - `app/models/keyword_context.py`
  - `app/models/travel_opportunity_candidate.py`
- **Step 3 (High Precision Ranking)**:
  - `app/services/travel_ranking_service.py`
  - `app/repositories/travel_ranking_repository.py`
- **Step 4 (Final AI Analysis)**:
  - `app/services/final_travel_opportunity_service.py`
  - `app/ai/travel_opportunity_prompt.py`
  - `app/ai/travel_opportunity_schemas.py`
  - `app/ai/travel_evidence_builder.py`
  - `app/models/final_travel_opportunity.py`
- **대시보드**:
  - `dashboard/pages/5_여행_기회_V2.py`
  - `dashboard/travel_opportunity_formatter.py`

---

## 3. 개발 워크플로우

1. **사전 조사 (Read Before Write)**:
   - 수정할 기능과 관련된 기존 모델, 리포지토리, 서비스, 테스트 코드를 먼저 읽어 구조와 시그니처를 파악합니다.
2. **최소 변경 및 단일 책임 구현**:
   - 변경 범위를 명확히 설정하고 필요한 파일만 수정합니다.
3. **타깃 테스트 작성 및 검증**:
   - 신규/수정 기능에 대한 단위 테스트를 `tests/`에 추가하거나 업데이트합니다.
   - 외부 API나 모델 추론은 반드시 Mock/Fake 객체로 격리합니다.
4. **회귀 검증 (Regression Verification)**:
   - 변경 후 `regression-verification` 스킬을 사용하여 전체 테스트 스위트를 실행하고 통과를 확인합니다.

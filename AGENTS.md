# 1. Project Overview

프로젝트 이름은 **Trend Catch Module**이다.

이 프로젝트의 목적은 다음과 같다.

- YouTube와 뉴스/RSS에서 문서를 수집한다.
- 수집 문서에서 키워드를 추출한다.
- 최근 7일과 이전 7일을 비교해 주간 트렌드를 계산한다.
- 하루짜리 급등 키워드는 `watchlist`로 분류한다.
- Google Trends와 네이버 데이터랩 검색 관심도 데이터를 검증 점수로 반영한다.
- 지역, 장소, 인물, 콘텐츠, 행사, 음식, 브랜드, 밈 객체를 추출한다.
- 향후 검증된 트렌드와 여행지를 연결해 Travelen에서 활용한다.

현재 `README.md`는 초기 골격 기준으로 오래되었으며 잘못된 프로젝트 경로를 포함한다. 현재 동작의 기준은 루트의 `app/`, `tests/`, `.env.example`, `requirements.txt`, 그리고 이 문서다.

# 2. Project Root

항상 다음 경로를 프로젝트 루트로 사용한다.

```text
C:\Users\owner\Desktop\trend_catch_module
```

다음 경로는 절대 수정하지 않는다.

```text
C:\Users\owner\Desktop\Travelen
```

별도의 `trend-engine` 또는 다른 중첩 프로젝트를 만들지 않는다. 루트에 이미 중첩 디렉터리가 있더라도 프로젝트 루트로 사용하거나 새 코드를 작성하지 않는다.

# 3. Technology Stack

현재 코드와 환경에서 확인된 기술은 다음과 같다.

- Python (현재 확인 환경 3.14.3)
- FastAPI와 Pydantic
- Uvicorn
- SQLAlchemy 2.x
- SQLite 기본 데이터베이스
- httpx
- pytest
- APScheduler 3.x의 `AsyncIOScheduler`
- feedparser
- python-multipart
- GLiNER 0.2.x (`urchade/gliner_multi-v2.1`)
- Google Gen AI SDK (`google-genai`)
- Streamlit, pandas, Plotly

GLiNER는 현재 설치 및 구현되어 있다. 모델은 NER 최초 요청 때 lazy loading하며 Hugging Face 캐시를 사용한다.

# 4. Current Architecture

- `app/api`: FastAPI 라우터와 HTTP 입력·응답 처리
- `app/models`: SQLAlchemy 테이블 모델과 제약조건
- `app/schemas`: Pydantic 요청·응답 스키마
- `app/services`: 수집, 점수 계산, 재점수화, NER 등 비즈니스 로직
- `app/repositories`: 데이터 조회, 집계, 중복 방지 및 저장 로직
- `app/collectors`: YouTube 공식 API와 허용된 뉴시스 RSS의 외부 요청·파싱
- `app/scheduler`: APScheduler 작업 등록, 실행 및 종료 관리
- `app/ner`: GLiNER adapter, 객체 label, 지역 사전, 규칙 및 겹침 해결
- `dashboard`: FastAPI REST API만 사용하는 독립형 Streamlit 대시보드와 화면 컴포넌트
- `tests`: 운영 DB와 분리된 테스트 DB를 사용하는 API·서비스 테스트
- `samples`: 합성 Google Trends 및 네이버 데이터랩 CSV 예시
- `data`: 한국 지역명 및 alias 사전

전체 파일을 한 API 모듈에 집중시키지 말고 현재 계층과 책임을 유지한다.

# 5. Existing Pipeline

현재 구현된 기본 흐름은 다음과 같다.

```text
Mock / YouTube / Newsis RSS 수집
-> SourceDocument 저장
-> 신규 문서 키워드 추출
-> KeywordOccurrence 저장
-> 최근 7일 및 이전 7일 WeeklyTrend 계산
-> weekly_trend / watchlist / stable / insufficient_data 분류
-> Google Trends 또는 네이버 데이터랩 CSV·수동 관측 입력
-> 검색 관심도 점수 및 WeeklyTrend final_score 재계산
-> NER 객체 및 Wikipedia/수동 맥락 연결
-> 설정된 경우 기존 근거 기반 Gemini 트렌드 설명 생성
-> Streamlit 대시보드에서 결과 조회 및 명시적인 단일 키워드 AI 분석 실행
```

APScheduler는 수집·키워드 추출과 주간 계산 작업을 지원하지만 기본 설정에서는 비활성화된다. NER 추출, 객체 요약, 트렌드-객체 연결도 구현되어 있으며 현재는 명시적 API 호출로 실행한다.

Gemini는 기존 DB 근거를 해석하는 선택 기능이며 기본 설정에서는 비활성화된다. Streamlit 대시보드는 DB에 직접 접근하지 않고 FastAPI 읽기 API를 사용한다. 실제 여행지 데이터 매핑은 아직 구현되지 않았다.

# 6. Entity Types

내부 객체 유형은 다음 8개로 고정한다.

- `LOCATION`
- `PLACE`
- `PERSON`
- `CONTENT_TITLE`
- `EVENT`
- `FOOD`
- `BRAND`
- `MEME`

현재 객체명 인식은 다음 조합을 사용하며 향후 확장에서도 이 구조를 우선 유지한다.

- 오픈소스 GLiNER multilingual
- 한국 지역명 및 alias 사전
- 규칙 기반 보정
- confidence, 사전·규칙 근거, 유형 구체성을 고려한 중복·겹침 해결

LLM이나 NER 모델은 통계 점수, `final_score`, 트렌드 순위 또는 status를 결정하지 않는다. 이 값은 검증 가능한 서비스 계산식으로만 산출한다.

# 7. Coding Rules

- 기존 기능과 API 응답 호환성을 유지한다.
- 새 기능은 기존 service와 repository 구조를 먼저 재사용한다.
- API 함수에 모든 비즈니스 로직을 직접 넣지 않는다.
- SQLAlchemy 2.x의 `select()` 및 typed model 문법을 사용한다.
- UTC, 로컬 시간대, 주간 경계를 명시적으로 처리한다.
- 테스트 DB와 운영 DB를 분리한다.
- 테스트의 외부 HTTP 요청과 모델 추론은 mock 또는 fake로 대체한다.
- 반복 수집·가져오기·계산은 unique constraint와 안전한 upsert로 중복 저장되지 않아야 한다.
- 환경변수와 API 키를 코드, 테스트, 로그에 하드코딩하거나 출력하지 않는다.
- `.env`, SQLite DB, 가상환경, 로그, Python cache, Hugging Face 모델 cache를 Git에 포함하지 않는다.
- 사용하지 않는 추상화, 불필요한 의존성, 요청 범위를 벗어난 대규모 리팩터링을 피한다.
- 사용자가 요구하지 않은 기능을 임의로 추가하지 않는다.
- 기존의 사용자 변경을 되돌리지 않으며 작업 전후 `git status`를 확인한다.

# 8. Model and External Data Rules

- YouTube 데이터는 공식 YouTube Data API를 우선 사용한다.
- 뉴스는 코드의 허용 목록에 있는 RSS 또는 공식 API만 사용한다.
- RSS 제목, 요약, 날짜, 원문 URL을 사용하며 기사 본문 페이지를 무단 크롤링하지 않는다.
- Google Trends 웹페이지를 자동 크롤링하지 않고 `pytrends`를 사용하지 않는다.
- 네이버 데이터랩 자동 연동 권한이 없으면 현재 CSV·수동 입력 구조를 유지한다.
- 나무위키는 향후 맥락 보조 자료로만 사용하고 단독 사실 근거로 확정하지 않는다.
- Gemini는 상위 키워드 설명과 여행 관련성 보조에만 사용하며 순위 계산, 원본 수치 변경 또는 외부 검색에는 사용하지 않는다.
- GLiNER는 lazy loading하고 프로세스 내 인스턴스를 재사용한다. 모델 로딩 실패가 FastAPI 서버 전체를 중단시키지 않게 한다.

# 9. Required Commands

Windows PowerShell 기준 명령은 다음과 같다.

프로젝트 이동:

```powershell
cd C:\Users\owner\Desktop\trend_catch_module
```

가상환경 활성화:

```powershell
.\.venv\Scripts\Activate.ps1
```

패키지 설치:

```powershell
python -m pip install -r requirements.txt
```

전체 테스트:

```powershell
python -m pytest
```

서버 실행:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

# 10. Testing Requirements

- 변경 전 관련 기존 테스트와 fixture를 확인한다.
- 변경 후 관련 테스트와 전체 `python -m pytest`를 실제 실행한다.
- 실행하지 않은 테스트를 성공했다고 말하지 않는다.
- 실패하면 원인을 확인하고 범위 내에서 수정한 뒤 다시 실행한다.
- 실제 외부 API·모델 실행 결과와 mock·fake 테스트 결과를 구분해 보고한다.
- 외부 API 테스트는 실제 YouTube 또는 RSS 서버를 호출하지 않는다.
- NER 자동 테스트에서는 실제 모델 다운로드와 추론을 금지하고 adapter를 mock 또는 fake로 교체한다.
- 실제 GLiNER smoke test가 명시적으로 필요하면 별도로 실행하고 문서 수를 최대 5개처럼 작게 제한한다.
- 운영 DB나 기존 사용자 데이터를 테스트 데이터로 사용하지 않는다.

# 11. Git Workflow

- Codex는 임의로 commit, push 또는 branch 생성을 하지 않는다.
- 구현과 테스트 완료 후 변경 파일과 권장 커밋 메시지를 사용자에게 보고한다.
- GitHub 기록과 실제 commit은 사용자가 수행한다.
- `.env`, API 키, SQLite DB, 로그, 가상환경, 모델 cache가 staged되지 않았는지 확인한다.
- 사용자 요청 없이 기존 commit history를 변경하거나 파일을 강제로 되돌리지 않는다.

# 12. Work Completion Report

각 작업 완료 시 다음 항목을 간단히 보고한다.

1. 생성·수정 파일
2. 구현 내용
3. 테스트 결과
4. 실제 외부 API 또는 모델 실행 여부
5. 남은 문제
6. 실행 명령
7. 권장 Git 커밋 메시지

# 13. Current and Planned Scope

## Current

- FastAPI 서버, lifespan 및 Swagger
- SQLite와 SQLAlchemy 모델
- Mock 문서 수집
- YouTube 공식 API 수집 구현 (실제 호출에는 API 키 필요)
- 허용 목록 기반 뉴시스 RSS 수집
- 신규 문서 키워드 추출과 `KeywordOccurrence` 중복 방지
- 최근 7일/이전 7일 주간 트렌드 계산과 watchlist 분류
- Google Trends·네이버 데이터랩 CSV 및 수동 검색 관심도 입력
- 검색 관심도 검증 점수와 `WeeklyTrend` 재점수화
- APScheduler 자동 수집·주간 계산, 수동 실행 및 실행 이력
- GLiNER lazy loading, 한국 지역명 사전, 규칙 보정 및 객체 병합
- `EntityMention` 저장, 객체 요약, `TrendEntityLink` 계산 및 조회
- 한국어 Wikipedia 공식 API 기반 객체 맥락 후보 검색·요약 저장
- 나무위키 URL 및 짧은 맥락의 사용자 수동 입력
- `EntityContext` 저장과 주간 트렌드 `TrendContextLink` 연결
- `google-genai` 구조화 출력 기반 트렌드 설명, 근거 참조 검증 및 여행 추천 보정
- FastAPI 대시보드 집계 API와 독립형 Streamlit 분석 대시보드
- 독립 테스트 DB와 외부 요청·모델 mock 테스트

## Planned

- 추가 Wikipedia provider 및 맥락 품질 검증 고도화
- Gemini 분석 품질 검증 및 운영 사용량 관측
- 객체를 실제 여행지 데이터와 매핑하는 검증 계층
- 배포·운영 관측 체계

# 14. Final Verification

작업을 끝내기 전에 다음을 확인한다.

- 작업 위치가 `C:\Users\owner\Desktop\trend_catch_module`인지 확인한다.
- 하위 프로젝트나 하위 `AGENTS.md`를 만들지 않았는지 확인한다.
- 요청하지 않은 기존 애플리케이션 코드를 변경하지 않았는지 확인한다.
- 문서가 실제 코드, requirements, 환경변수 예시 및 테스트 상태와 일치하는지 확인한다.
- 실제 비밀값, `.env` 내용, API 키 또는 민감한 로그가 변경 파일에 포함되지 않았는지 확인한다.

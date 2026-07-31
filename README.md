# Trend Catch Module

뉴스, YouTube, 검색 관심도, 객체 맥락과 AI 분석을 결합해 여행 트렌드를 확인하는 FastAPI 분석 서버와 독립형 Streamlit 대시보드입니다.

## 설치

Windows PowerShell에서 실행합니다.

```powershell
cd C:\Users\owner\Desktop\trend_catch_module
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.env.example`을 참고해 로컬 `.env`를 구성합니다. 비밀키가 필요한 기능은 명시적으로 활성화한 경우에만 외부 API를 호출합니다.

## Trend Catch 실행

터미널 1:

```powershell
.\run_api.ps1
```

터미널 2:

```powershell
.\run_dashboard.ps1
```

- FastAPI Swagger: http://127.0.0.1:8000/docs
- Trend Catch Dashboard: http://localhost:8501

## 환경변수

주요 설정은 다음과 같습니다. 실제 API 키와 연락처 값은 저장소 문서나 소스코드에 넣지 않습니다.

- `GEMINI_ENABLED`
- `GEMINI_API_KEY`
- `GEMINI_MODEL`
- `DASHBOARD_API_BASE_URL`
- `DASHBOARD_REQUEST_TIMEOUT_SECONDS`
- `DASHBOARD_PAGE_SIZE`
- `WIKIPEDIA_ENABLED`
- `WIKIPEDIA_LANGUAGE`
- `WIKIMEDIA_CONTACT_URL`
- `WIKIMEDIA_CONTACT_EMAIL`

## 테스트

```powershell
python -m pytest
```

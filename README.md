# Travelen Trend Engine

Travelen 트렌드 엔진의 첫 FastAPI 골격입니다. 현재 구현 범위는 `/health` API와 SQLite 연결 준비입니다.

## 설치

Windows PowerShell 기준:

```powershell
cd C:\Users\owner\Desktop\Travelen\trend-engine
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

## 실행

```powershell
cd C:\Users\owner\Desktop\Travelen\trend-engine
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## API 확인

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

예상 응답:

```json
{
  "status": "ok",
  "service": "trend-engine"
}
```

## 테스트

```powershell
cd C:\Users\owner\Desktop\Travelen\trend-engine
.\.venv\Scripts\Activate.ps1
pytest
```

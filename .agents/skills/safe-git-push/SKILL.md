---
name: safe-git-push
description: >-
  Safe and disciplined Git commit and push procedure for the Trend Catch Module repository.
  Use when staging specific changes, committing with conventional commit messages,
  preventing secret/cache leaks, avoiding accidental commits of unrelated files, and pushing to remote origin.
---

# Safe Git Commit & Push Guide

이 스킬은 **Trend Catch Module** 저장소에서 작업 완료 후 변경사항을 안전하고 격리된 방식으로 GitHub 원격 저장소에 반영하기 위한 표준 절차와 규칙입니다.

---

## 1. 사전 상태 점검 (Mandatory Pre-checks)

파일을 스테이징하거나 커밋하기 전에 항상 현재 Git 상태를 전체적으로 확인합니다:

```powershell
git status
git branch --show-current
git remote -v
git diff
```

### 주의: Unrelated Unstaged 변경 보호
- 저장소에는 사용자가 작업 중인 별도의 수정사항(Unrelated unstaged changes)이 존재할 수 있습니다.
- **`git add .` 또는 `git add -A` 명령은 절대 사용하지 않습니다.**
- 이번 작업에서 실제로 생성하거나 수정한 파일의 경로만 개별적으로 명시하여 스테이징합니다.

---

## 2. 절대 커밋 금지 항목 (Blacklist)

다음 항목은 어떤 경우에도 커밋이나 원격 저장소 푸시에 포함되어서는 안 됩니다:
- 환경 설정 및 비밀값: `.env`, `.env.local`, API Key, 토큰, 패스워드
- 데이터베이스 파일: `*.sqlite3`, `*.db`, `*.sqlite`
- 캐시 및 바이트코드: `__pycache__/`, `*.pyc`, `.pytest_cache/`, `.uv-cache/`, Hugging Face 모델 캐시
- 실행 산출물 및 임시 파일: `*.log`, `scratch/`, 임시 테스트 산출물

---

## 3. 안전한 Staging & Commit 절차

### 1단계: 대상 파일 명시적 Staging
```powershell
# 예시: 특정 파일만 명시적으로 스테이징
git add path/to/specific_file1.py path/to/specific_file2.py
```

### 2단계: Staged 파일 무결성 확인
커밋 직전 스테이징된 파일 목록을 확인하여 원치 않는 파일이나 비밀값이 포함되지 않았는지 점검합니다:
```powershell
git diff --cached --name-only
git status
```
*만약 원치 않는 파일이 스테이징되었다면 `git restore --staged <file>`로 즉시 제거합니다.*

### 3단계: 커밋 생성
Conventional Commits 규칙에 맞추어 명확한 커밋 메시지를 작성합니다:
```powershell
git commit -m "feat: <간결하고 명확한 작업 내용>"
# 또는
git commit -m "fix: <버그 수정 내용>"
# 또는
git commit -m "chore: <설정 및 스킬 추가>"
```

---

## 4. 원격 저장소 Push 절차

1. **현재 브랜치 푸시**:
   - 새 브랜치를 임의로 생성하지 않고 현재 브랜치를 기존 `origin`에 푸시합니다:
     ```powershell
     git push origin main
     ```

2. **금지 명령 (Strictly Forbidden)**:
   - `git push --force` 또는 `git push -f`
   - `git reset --hard`
   - `git clean -fd`

3. **Push 실패 시 조치**:
   - 네트워크 오류, 권한 문제 또는 원격 충돌로 인해 푸시가 거부될 경우, 강제 푸시를 시도하지 말고 원격 로그와 충돌 원인을 파악하여 사용자에게 보고합니다.

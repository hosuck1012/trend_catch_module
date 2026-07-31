import json
import os
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.scheduler_run import SchedulerRun


SCHEDULER_RUN_STATUSES = {
    "running",
    "success",
    "partial_success",
    "skipped",
    "failed",
}
SENSITIVE_ENV_NAMES = (
    "YOUTUBE_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def create_scheduler_run(
    session: Session,
    *,
    job_name: str,
    trigger_type: str,
) -> SchedulerRun:
    run = SchedulerRun(
        job_name=job_name,
        trigger_type=trigger_type,
        started_at=utc_now(),
        status="running",
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def finish_scheduler_run(
    session: Session,
    *,
    run_id: int,
    status: str,
    result_summary: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> SchedulerRun:
    if status not in SCHEDULER_RUN_STATUSES - {"running"}:
        raise ValueError(f"지원하지 않는 SchedulerRun status입니다: {status}")
    run = session.get(SchedulerRun, run_id)
    if run is None:
        raise RuntimeError("SchedulerRun 실행 이력을 찾을 수 없습니다.")
    run.finished_at = utc_now()
    run.status = status
    run.result_summary = (
        json.dumps(result_summary, ensure_ascii=False, sort_keys=True)
        if result_summary is not None
        else None
    )
    run.error_message = sanitize_error_message(error_message)
    session.commit()
    session.refresh(run)
    return run


def list_scheduler_runs(
    session: Session,
    *,
    job_name: str | None = None,
    status: str | None = None,
    limit: int = 20,
) -> list[SchedulerRun]:
    query = select(SchedulerRun)
    if job_name:
        query = query.where(SchedulerRun.job_name == job_name)
    if status:
        query = query.where(SchedulerRun.status == status)
    return list(
        session.scalars(
            query.order_by(SchedulerRun.started_at.desc(), SchedulerRun.id.desc()).limit(limit)
        ).all()
    )


def sanitize_error_message(message: str | None) -> str | None:
    if not message:
        return None
    sanitized = str(message)
    for name in SENSITIVE_ENV_NAMES:
        value = os.getenv(name, "").strip()
        if value:
            sanitized = sanitized.replace(value, "[REDACTED]")
    return sanitized[:2000]

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.scheduler.jobs import (
    JobAlreadyRunningError,
    run_calculate_weekly_trends,
    run_collect_sources,
)
from app.scheduler.scheduler_manager import scheduler_manager
from app.schemas.scheduler import (
    CollectionJobResponse,
    SchedulerRunsResponse,
    SchedulerStatusResponse,
    WeeklyJobResponse,
)
from app.services.scheduler_service import list_scheduler_runs


router = APIRouter(prefix="/api/scheduler", tags=["scheduler"])


@router.get("/status", response_model=SchedulerStatusResponse)
def scheduler_status(session: Session = Depends(get_db)) -> dict[str, object]:
    last_runs = list_scheduler_runs(session, limit=20)
    return {
        "enabled": scheduler_manager.enabled,
        "running": scheduler_manager.running,
        "timezone": scheduler_manager.timezone,
        "configuration_error": scheduler_manager.configuration_error,
        "jobs": [asdict(job) for job in scheduler_manager.get_jobs()],
        "last_runs": [
            {
                "job_name": run.job_name,
                "status": run.status,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
            }
            for run in last_runs
        ],
    }


@router.post("/run-collection", response_model=CollectionJobResponse)
async def run_collection_once() -> dict[str, object]:
    try:
        result = await run_collect_sources(trigger_type="manual")
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(result)


@router.post("/run-weekly", response_model=WeeklyJobResponse)
async def run_weekly_once() -> dict[str, object]:
    try:
        result = await run_calculate_weekly_trends(trigger_type="manual")
    except JobAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return asdict(result)


@router.get("/runs", response_model=SchedulerRunsResponse)
def scheduler_runs(
    job_name: str | None = Query(default=None),
    status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    runs = list_scheduler_runs(
        session,
        job_name=job_name,
        status=status,
        limit=limit,
    )
    return {
        "items": [
            {
                "id": run.id,
                "job_name": run.job_name,
                "trigger_type": run.trigger_type,
                "started_at": run.started_at,
                "finished_at": run.finished_at,
                "status": run.status,
                "result_summary": run.result_summary,
                "error_message": run.error_message,
            }
            for run in runs
        ]
    }

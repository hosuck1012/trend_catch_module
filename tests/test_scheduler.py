import asyncio
import json
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import get_settings
from app.database import SessionLocal
from app.main import app
from app.models.scheduler_run import SchedulerRun
from app.scheduler.jobs import (
    CALCULATE_WEEKLY_TRENDS_JOB_ID,
    COLLECT_SOURCES_JOB_ID,
    execution_registry,
    run_calculate_weekly_trends,
    run_collect_sources,
)
from app.scheduler.scheduler_manager import SchedulerManager, scheduler_manager
from app.services.newsis_rss_collection_service import (
    NewsisRssAllFeedsFailedError,
    NewsisRssFailedFeed,
)


def _set_scheduler_enabled(monkeypatch, enabled: bool = True) -> None:
    monkeypatch.setenv("SCHEDULER_ENABLED", "true" if enabled else "false")
    monkeypatch.setenv("RUN_COLLECTION_ON_STARTUP", "false")
    monkeypatch.setenv("COLLECTION_INTERVAL_HOURS", "6")
    monkeypatch.setenv("WEEKLY_CALCULATION_DAY", "mon")
    monkeypatch.setenv("WEEKLY_CALCULATION_HOUR", "8")
    monkeypatch.setenv("WEEKLY_CALCULATION_MINUTE", "0")
    monkeypatch.setenv("SCHEDULER_TIMEZONE", "Asia/Seoul")
    get_settings.cache_clear()


def _inspect_started_manager(monkeypatch):
    _set_scheduler_enabled(monkeypatch)

    async def inspect():
        manager = SchedulerManager()
        await manager.start()
        try:
            jobs = manager.scheduler.get_jobs()
            return manager, jobs
        finally:
            await manager.shutdown()

    return asyncio.run(inspect())


def _youtube_result(inserted=2, skipped=1):
    return SimpleNamespace(inserted_documents=inserted, skipped_documents=skipped)


def _newsis_result(inserted=3, skipped=2, failed_feeds=None):
    return SimpleNamespace(
        inserted_documents=inserted,
        skipped_documents=skipped,
        failed_feeds=failed_feeds or [],
    )


def _keyword_result(inserted=12):
    return SimpleNamespace(inserted_occurrences=inserted)


def _trend_result():
    return SimpleNamespace(
        week_start=date(2026, 7, 25),
        week_end=date(2026, 7, 31),
        weekly_trends=2,
        watchlist=1,
        stable=3,
        insufficient_data=20,
    )


def test_scheduler_disabled_status_response(client) -> None:
    response = client.get("/api/scheduler/status")

    assert response.status_code == 200
    assert response.json() == {
        "enabled": False,
        "running": False,
        "timezone": "Asia/Seoul",
        "configuration_error": None,
        "jobs": [],
        "last_runs": [],
    }


def test_scheduler_enabled_registers_jobs(monkeypatch) -> None:
    _, jobs = _inspect_started_manager(monkeypatch)

    assert {job.id for job in jobs} == {
        COLLECT_SOURCES_JOB_ID,
        CALCULATE_WEEKLY_TRENDS_JOB_ID,
    }


def test_collect_sources_is_registered_once(monkeypatch) -> None:
    _, jobs = _inspect_started_manager(monkeypatch)

    assert [job.id for job in jobs].count(COLLECT_SOURCES_JOB_ID) == 1


def test_calculate_weekly_trends_is_registered_once(monkeypatch) -> None:
    _, jobs = _inspect_started_manager(monkeypatch)

    assert [job.id for job in jobs].count(CALCULATE_WEEKLY_TRENDS_JOB_ID) == 1


def test_collection_uses_six_hour_interval(monkeypatch) -> None:
    _, jobs = _inspect_started_manager(monkeypatch)
    job = next(item for item in jobs if item.id == COLLECT_SOURCES_JOB_ID)

    assert job.trigger.interval.total_seconds() == 6 * 60 * 60


def test_weekly_uses_monday_at_eight_cron(monkeypatch) -> None:
    _, jobs = _inspect_started_manager(monkeypatch)
    job = next(item for item in jobs if item.id == CALCULATE_WEEKLY_TRENDS_JOB_ID)

    assert str(job.trigger.fields[4]) == "mon"
    assert str(job.trigger.fields[5]) == "8"
    assert str(job.trigger.fields[6]) == "0"


def test_scheduler_uses_asia_seoul_timezone(monkeypatch) -> None:
    _, jobs = _inspect_started_manager(monkeypatch)

    assert all(job.trigger.timezone == ZoneInfo("Asia/Seoul") for job in jobs)


def test_invalid_timezone_does_not_stop_application(monkeypatch) -> None:
    _set_scheduler_enabled(monkeypatch)
    monkeypatch.setenv("SCHEDULER_TIMEZONE", "Invalid/Timezone")
    get_settings.cache_clear()

    with TestClient(app) as test_client:
        body = test_client.get("/api/scheduler/status").json()

    assert body["enabled"] is True
    assert body["running"] is False
    assert "유효한 IANA timezone" in body["configuration_error"]


def test_lifespan_starts_scheduler(monkeypatch) -> None:
    start = AsyncMock(return_value=True)
    shutdown = AsyncMock(return_value=None)
    monkeypatch.setattr(scheduler_manager, "start", start)
    monkeypatch.setattr(scheduler_manager, "shutdown", shutdown)

    with TestClient(app):
        start.assert_awaited_once()


def test_lifespan_shuts_down_scheduler(monkeypatch) -> None:
    start = AsyncMock(return_value=True)
    shutdown = AsyncMock(return_value=None)
    monkeypatch.setattr(scheduler_manager, "start", start)
    monkeypatch.setattr(scheduler_manager, "shutdown", shutdown)

    with TestClient(app):
        pass

    shutdown.assert_awaited_once()


def test_manual_collection_execution(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_youtube_popular_videos",
        AsyncMock(return_value=_youtube_result()),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_newsis_rss_documents",
        AsyncMock(return_value=_newsis_result()),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_keywords_for_documents",
        lambda session: _keyword_result(),
    )

    response = client.post("/api/scheduler/run-collection")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "job_name": COLLECT_SOURCES_JOB_ID,
        "youtube_inserted": 2,
        "youtube_skipped": 1,
        "newsis_inserted": 3,
        "newsis_skipped": 2,
        "keyword_occurrences_inserted": 12,
        "failed_sources": [],
    }


def test_manual_weekly_execution(client, monkeypatch) -> None:
    monkeypatch.setattr(
        "app.scheduler.jobs.recalculate_weekly_trends",
        lambda session: _trend_result(),
    )

    response = client.post("/api/scheduler/run-weekly")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "job_name": CALCULATE_WEEKLY_TRENDS_JOB_ID,
        "weekly_trends": 2,
        "watchlist": 1,
        "stable": 3,
        "insufficient_data": 20,
    }


def test_rss_continues_when_youtube_key_is_missing(monkeypatch) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    get_settings.cache_clear()
    rss_collector = AsyncMock(return_value=_newsis_result(inserted=4))
    monkeypatch.setattr("app.scheduler.jobs.collect_newsis_rss_documents", rss_collector)
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_keywords_for_documents",
        lambda session: _keyword_result(9),
    )

    result = asyncio.run(run_collect_sources(trigger_type="manual"))

    rss_collector.assert_awaited_once()
    assert result.status == "success"
    assert result.youtube_inserted == 0
    assert result.source_statuses["youtube"] == "skipped"
    assert result.newsis_inserted == 4
    assert result.keyword_occurrences_inserted == 9


def test_source_failure_returns_partial_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_youtube_popular_videos",
        AsyncMock(side_effect=RuntimeError("youtube failed")),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_newsis_rss_documents",
        AsyncMock(return_value=_newsis_result()),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_keywords_for_documents",
        lambda session: _keyword_result(),
    )

    result = asyncio.run(run_collect_sources(trigger_type="scheduled"))

    assert result.status == "partial_success"
    assert result.failed_sources == ["youtube"]


def test_all_data_sources_failure_returns_failed(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_youtube_popular_videos",
        AsyncMock(side_effect=RuntimeError("youtube failed")),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_newsis_rss_documents",
        AsyncMock(
            side_effect=NewsisRssAllFeedsFailedError(
                [NewsisRssFailedFeed(feed="culture", error="failed")]
            )
        ),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_keywords_for_documents",
        lambda session: _keyword_result(0),
    )

    result = asyncio.run(run_collect_sources(trigger_type="scheduled"))

    assert result.status == "failed"
    assert result.failed_sources == ["youtube", "newsis_rss"]


def test_weekly_without_occurrences_is_skipped(db_session) -> None:
    result = asyncio.run(run_calculate_weekly_trends(trigger_type="manual"))
    run = db_session.scalar(
        select(SchedulerRun).where(
            SchedulerRun.job_name == CALCULATE_WEEKLY_TRENDS_JOB_ID
        )
    )

    assert result.status == "skipped"
    assert run is not None
    assert run.status == "skipped"


def test_scheduler_run_is_created_and_completed(db_session, monkeypatch) -> None:
    observed_statuses = []

    async def observe_running(session):
        observer = SessionLocal()
        try:
            observed_statuses.extend(
                observer.scalars(select(SchedulerRun.status)).all()
            )
        finally:
            observer.close()
        return _youtube_result()

    monkeypatch.setattr(
        "app.scheduler.jobs.collect_youtube_popular_videos",
        observe_running,
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.collect_newsis_rss_documents",
        AsyncMock(return_value=_newsis_result()),
    )
    monkeypatch.setattr(
        "app.scheduler.jobs.extract_keywords_for_documents",
        lambda session: _keyword_result(),
    )

    asyncio.run(run_collect_sources(trigger_type="manual"))
    run = db_session.scalar(select(SchedulerRun))

    assert run is not None
    assert run.status == "success"
    assert run.started_at is not None
    assert run.finished_at is not None
    assert observed_statuses == ["running"]
    assert json.loads(run.result_summary)["newsis_inserted"] == 3


def test_duplicate_manual_execution_returns_409(client) -> None:
    assert execution_registry.acquire(COLLECT_SOURCES_JOB_ID) is True
    try:
        response = client.post("/api/scheduler/run-collection")
    finally:
        execution_registry.release(COLLECT_SOURCES_JOB_ID)

    assert response.status_code == 409


def test_scheduler_status_includes_registered_jobs(monkeypatch) -> None:
    _set_scheduler_enabled(monkeypatch)

    with TestClient(app) as test_client:
        response = test_client.get("/api/scheduler/status")

    assert response.status_code == 200
    assert response.json()["running"] is True
    assert {job["id"] for job in response.json()["jobs"]} == {
        COLLECT_SOURCES_JOB_ID,
        CALCULATE_WEEKLY_TRENDS_JOB_ID,
    }


def test_scheduler_runs_api_orders_and_filters(client, db_session) -> None:
    older = SchedulerRun(
        job_name=COLLECT_SOURCES_JOB_ID,
        trigger_type="scheduled",
        started_at=datetime(2026, 7, 30, 8, 0),
        status="success",
    )
    newer = SchedulerRun(
        job_name=CALCULATE_WEEKLY_TRENDS_JOB_ID,
        trigger_type="manual",
        started_at=datetime(2026, 7, 31, 8, 0),
        status="skipped",
    )
    db_session.add_all([older, newer])
    db_session.commit()

    response = client.get("/api/scheduler/runs?status=skipped&limit=1")

    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert response.json()["items"][0]["job_name"] == CALCULATE_WEEKLY_TRENDS_JOB_ID
    assert client.get("/api/scheduler/runs?limit=0").status_code == 422

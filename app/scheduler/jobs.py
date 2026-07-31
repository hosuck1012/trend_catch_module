from dataclasses import asdict, dataclass
from threading import Lock

from app.database import SessionLocal
from app.repositories.keyword_repository import extract_keywords_for_documents
from app.services.newsis_rss_collection_service import (
    NewsisRssAllFeedsFailedError,
    collect_newsis_rss_documents,
)
from app.services.scheduler_service import create_scheduler_run, finish_scheduler_run
from app.services.trend_calculation_service import (
    NoKeywordOccurrencesError,
    recalculate_weekly_trends,
)
from app.services.youtube_collection_service import (
    YouTubeApiKeyMissingError,
    collect_youtube_popular_videos,
)


COLLECT_SOURCES_JOB_ID = "collect_sources"
CALCULATE_WEEKLY_TRENDS_JOB_ID = "calculate_weekly_trends"


class JobAlreadyRunningError(RuntimeError):
    pass


class _JobExecutionRegistry:
    def __init__(self) -> None:
        self._lock = Lock()
        self._running: set[str] = set()

    def acquire(self, job_name: str) -> bool:
        with self._lock:
            if job_name in self._running:
                return False
            self._running.add(job_name)
            return True

    def release(self, job_name: str) -> None:
        with self._lock:
            self._running.discard(job_name)

    def is_running(self, job_name: str) -> bool:
        with self._lock:
            return job_name in self._running


execution_registry = _JobExecutionRegistry()


@dataclass(frozen=True)
class CollectionJobResult:
    status: str
    job_name: str
    youtube_inserted: int
    youtube_skipped: int
    newsis_inserted: int
    newsis_skipped: int
    keyword_occurrences_inserted: int
    failed_sources: list[str]
    source_statuses: dict[str, str]


@dataclass(frozen=True)
class WeeklyJobResult:
    status: str
    job_name: str
    weekly_trends: int
    watchlist: int
    stable: int
    insufficient_data: int


async def run_collect_sources(*, trigger_type: str = "scheduled") -> CollectionJobResult:
    _acquire_or_raise(COLLECT_SOURCES_JOB_ID)
    session = SessionLocal()
    run_id: int | None = None
    try:
        run_id = create_scheduler_run(
            session,
            job_name=COLLECT_SOURCES_JOB_ID,
            trigger_type=trigger_type,
        ).id
        result = await _collect_sources(session)
        finish_scheduler_run(
            session,
            run_id=run_id,
            status=result.status,
            result_summary=asdict(result),
        )
        return result
    except Exception as exc:
        session.rollback()
        if run_id is not None:
            finish_scheduler_run(
                session,
                run_id=run_id,
                status="failed",
                error_message=str(exc),
            )
        raise
    finally:
        session.close()
        execution_registry.release(COLLECT_SOURCES_JOB_ID)


async def _collect_sources(session) -> CollectionJobResult:
    youtube_inserted = 0
    youtube_skipped = 0
    newsis_inserted = 0
    newsis_skipped = 0
    keyword_occurrences_inserted = 0
    failed_sources: list[str] = []
    youtube_status = "pending"
    newsis_status = "pending"
    keyword_status = "pending"

    try:
        youtube_result = await collect_youtube_popular_videos(session)
        youtube_inserted = youtube_result.inserted_documents
        youtube_skipped = youtube_result.skipped_documents
        youtube_status = "success"
    except YouTubeApiKeyMissingError:
        session.rollback()
        youtube_status = "skipped"
    except Exception:
        session.rollback()
        youtube_status = "failed"
        failed_sources.append("youtube")

    try:
        newsis_result = await collect_newsis_rss_documents(session)
        newsis_inserted = newsis_result.inserted_documents
        newsis_skipped = newsis_result.skipped_documents
        newsis_status = "success"
        failed_sources.extend(
            f"newsis_rss:{failed.feed}" for failed in newsis_result.failed_feeds
        )
    except NewsisRssAllFeedsFailedError:
        session.rollback()
        newsis_status = "failed"
        failed_sources.append("newsis_rss")
    except Exception:
        session.rollback()
        newsis_status = "failed"
        failed_sources.append("newsis_rss")

    try:
        keyword_result = extract_keywords_for_documents(session)
        keyword_occurrences_inserted = keyword_result.inserted_occurrences
        keyword_status = "success"
    except Exception:
        session.rollback()
        keyword_status = "failed"
        failed_sources.append("keyword_extraction")

    if newsis_status == "failed" and youtube_status != "success":
        status = "failed"
    elif failed_sources:
        status = "partial_success"
    else:
        status = "success"

    return CollectionJobResult(
        status=status,
        job_name=COLLECT_SOURCES_JOB_ID,
        youtube_inserted=youtube_inserted,
        youtube_skipped=youtube_skipped,
        newsis_inserted=newsis_inserted,
        newsis_skipped=newsis_skipped,
        keyword_occurrences_inserted=keyword_occurrences_inserted,
        failed_sources=failed_sources,
        source_statuses={
            "youtube": youtube_status,
            "newsis_rss": newsis_status,
            "keyword_extraction": keyword_status,
        },
    )


async def run_calculate_weekly_trends(*, trigger_type: str = "scheduled") -> WeeklyJobResult:
    _acquire_or_raise(CALCULATE_WEEKLY_TRENDS_JOB_ID)
    session = SessionLocal()
    run_id: int | None = None
    try:
        run_id = create_scheduler_run(
            session,
            job_name=CALCULATE_WEEKLY_TRENDS_JOB_ID,
            trigger_type=trigger_type,
        ).id
        try:
            calculation = recalculate_weekly_trends(session)
        except NoKeywordOccurrencesError:
            result = WeeklyJobResult(
                status="skipped",
                job_name=CALCULATE_WEEKLY_TRENDS_JOB_ID,
                weekly_trends=0,
                watchlist=0,
                stable=0,
                insufficient_data=0,
            )
        else:
            result = WeeklyJobResult(
                status="success",
                job_name=CALCULATE_WEEKLY_TRENDS_JOB_ID,
                weekly_trends=calculation.weekly_trends,
                watchlist=calculation.watchlist,
                stable=calculation.stable,
                insufficient_data=calculation.insufficient_data,
            )
        finish_scheduler_run(
            session,
            run_id=run_id,
            status=result.status,
            result_summary=asdict(result),
        )
        return result
    except Exception as exc:
        session.rollback()
        if run_id is not None:
            finish_scheduler_run(
                session,
                run_id=run_id,
                status="failed",
                error_message=str(exc),
            )
        raise
    finally:
        session.close()
        execution_registry.release(CALCULATE_WEEKLY_TRENDS_JOB_ID)


def _acquire_or_raise(job_name: str) -> None:
    if not execution_registry.acquire(job_name):
        raise JobAlreadyRunningError(f"{job_name} 작업이 이미 실행 중입니다.")

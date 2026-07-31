from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.config import Settings, get_settings
from app.scheduler.jobs import (
    CALCULATE_WEEKLY_TRENDS_JOB_ID,
    COLLECT_SOURCES_JOB_ID,
    run_calculate_weekly_trends,
    run_collect_sources,
)


WEEKDAY_LABELS = {
    "mon": "Monday",
    "tue": "Tuesday",
    "wed": "Wednesday",
    "thu": "Thursday",
    "fri": "Friday",
    "sat": "Saturday",
    "sun": "Sunday",
}


class SchedulerConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class SchedulerJobStatus:
    id: str
    next_run_time: object | None
    schedule: str


class SchedulerManager:
    def __init__(self) -> None:
        self.scheduler: AsyncIOScheduler | None = None
        self.enabled = False
        self.timezone = "Asia/Seoul"
        self.configuration_error: str | None = None
        self._collection_interval_hours = 6
        self._weekly_day = "mon"
        self._weekly_hour = 8
        self._weekly_minute = 0

    async def start(self) -> bool:
        settings = get_settings()
        await self.shutdown()
        self.enabled = settings.scheduler_enabled
        self.timezone = settings.scheduler_timezone
        self.configuration_error = None
        self._remember_schedule(settings)
        if not settings.scheduler_enabled:
            return False

        try:
            timezone = _validate_scheduler_settings(settings)
        except SchedulerConfigurationError as exc:
            self.configuration_error = str(exc)
            return False

        scheduler = AsyncIOScheduler(timezone=timezone)
        self._register_jobs(scheduler, settings, timezone)
        scheduler.start()
        self.scheduler = scheduler

        if settings.run_collection_on_startup:
            try:
                await run_collect_sources(trigger_type="startup")
            except Exception:
                # The failed run is persisted by the job; application startup stays available.
                pass
        return True

    async def shutdown(self) -> None:
        scheduler = self.scheduler
        self.scheduler = None
        if scheduler is not None and scheduler.running:
            scheduler.shutdown(wait=True)

    @property
    def running(self) -> bool:
        return bool(self.scheduler is not None and self.scheduler.running)

    def get_jobs(self) -> list[SchedulerJobStatus]:
        if self.scheduler is None:
            return []
        jobs_by_id = {job.id: job for job in self.scheduler.get_jobs()}
        statuses: list[SchedulerJobStatus] = []
        collect_job = jobs_by_id.get(COLLECT_SOURCES_JOB_ID)
        if collect_job is not None:
            statuses.append(
                SchedulerJobStatus(
                    id=collect_job.id,
                    next_run_time=collect_job.next_run_time,
                    schedule=f"every {self._collection_interval_hours} hours",
                )
            )
        weekly_job = jobs_by_id.get(CALCULATE_WEEKLY_TRENDS_JOB_ID)
        if weekly_job is not None:
            statuses.append(
                SchedulerJobStatus(
                    id=weekly_job.id,
                    next_run_time=weekly_job.next_run_time,
                    schedule=(
                        f"{WEEKDAY_LABELS[self._weekly_day]} "
                        f"{self._weekly_hour:02d}:{self._weekly_minute:02d}"
                    ),
                )
            )
        return statuses

    def _register_jobs(
        self,
        scheduler: AsyncIOScheduler,
        settings: Settings,
        timezone: ZoneInfo,
    ) -> None:
        common_options = {
            "replace_existing": True,
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": 3600,
        }
        scheduler.add_job(
            run_collect_sources,
            trigger=IntervalTrigger(
                hours=settings.collection_interval_hours,
                timezone=timezone,
            ),
            id=COLLECT_SOURCES_JOB_ID,
            kwargs={"trigger_type": "scheduled"},
            **common_options,
        )
        scheduler.add_job(
            run_calculate_weekly_trends,
            trigger=CronTrigger(
                day_of_week=settings.weekly_calculation_day,
                hour=settings.weekly_calculation_hour,
                minute=settings.weekly_calculation_minute,
                timezone=timezone,
            ),
            id=CALCULATE_WEEKLY_TRENDS_JOB_ID,
            kwargs={"trigger_type": "scheduled"},
            **common_options,
        )

    def _remember_schedule(self, settings: Settings) -> None:
        self._collection_interval_hours = settings.collection_interval_hours
        self._weekly_day = settings.weekly_calculation_day
        self._weekly_hour = settings.weekly_calculation_hour
        self._weekly_minute = settings.weekly_calculation_minute


def _validate_scheduler_settings(settings: Settings) -> ZoneInfo:
    if settings.collection_interval_hours < 1:
        raise SchedulerConfigurationError("COLLECTION_INTERVAL_HOURS는 1 이상이어야 합니다.")
    if settings.weekly_calculation_day not in WEEKDAY_LABELS:
        raise SchedulerConfigurationError(
            "WEEKLY_CALCULATION_DAY는 mon부터 sun까지의 영문 약어여야 합니다."
        )
    if not 0 <= settings.weekly_calculation_hour <= 23:
        raise SchedulerConfigurationError("WEEKLY_CALCULATION_HOUR는 0 이상 23 이하여야 합니다.")
    if not 0 <= settings.weekly_calculation_minute <= 59:
        raise SchedulerConfigurationError("WEEKLY_CALCULATION_MINUTE는 0 이상 59 이하여야 합니다.")
    try:
        return ZoneInfo(settings.scheduler_timezone)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise SchedulerConfigurationError(
            f"SCHEDULER_TIMEZONE이 유효한 IANA timezone이 아닙니다: {settings.scheduler_timezone}"
        ) from exc


scheduler_manager = SchedulerManager()

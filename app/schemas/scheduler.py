from datetime import datetime

from pydantic import BaseModel


class SchedulerJobResponse(BaseModel):
    id: str
    next_run_time: datetime | None
    schedule: str


class SchedulerRunBriefResponse(BaseModel):
    job_name: str
    status: str
    started_at: datetime
    finished_at: datetime | None


class SchedulerStatusResponse(BaseModel):
    enabled: bool
    running: bool
    timezone: str
    configuration_error: str | None = None
    jobs: list[SchedulerJobResponse]
    last_runs: list[SchedulerRunBriefResponse]


class CollectionJobResponse(BaseModel):
    status: str
    job_name: str
    youtube_inserted: int
    youtube_skipped: int
    newsis_inserted: int
    newsis_skipped: int
    keyword_occurrences_inserted: int
    failed_sources: list[str]


class WeeklyJobResponse(BaseModel):
    status: str
    job_name: str
    weekly_trends: int
    watchlist: int
    stable: int
    insufficient_data: int


class SchedulerRunResponse(BaseModel):
    id: int
    job_name: str
    trigger_type: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    result_summary: str | None
    error_message: str | None


class SchedulerRunsResponse(BaseModel):
    items: list[SchedulerRunResponse]

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field


class SearchInterestDateRange(BaseModel):
    start: date | None
    end: date | None


class SearchInterestImportErrorResponse(BaseModel):
    row: int
    reason: str


class SearchInterestImportResponse(BaseModel):
    status: str
    provider: str
    received_rows: int
    inserted_rows: int
    updated_rows: int
    skipped_rows: int
    keywords: list[str]
    date_range: SearchInterestDateRange
    errors: list[SearchInterestImportErrorResponse]


class ManualObservationRequest(BaseModel):
    date: date
    value: float = Field(ge=0, le=100)


class ManualSearchInterestRequest(BaseModel):
    provider: Literal["google_trends", "naver_datalab", "manual"]
    keyword: str = Field(min_length=1, max_length=255)
    geo: str = Field(default="KR", min_length=1, max_length=50)
    observations: list[ManualObservationRequest] = Field(min_length=1)


class SearchInterestRecalculateResponse(BaseModel):
    status: str
    week_start: date
    week_end: date
    recalculated_keywords: int
    validated_weekly_trends: int
    unvalidated_weekly_trends: int
    updated_weekly_trends: int


class SearchInterestProviderStatusResponse(BaseModel):
    observations: int
    keywords: int
    last_imported_at: datetime | None


class SearchInterestStatusResponse(BaseModel):
    google_trends: SearchInterestProviderStatusResponse
    naver_datalab: SearchInterestProviderStatusResponse
    validated_weekly_trends: int
    unvalidated_weekly_trends: int


class SearchValidationDetailResponse(BaseModel):
    keyword: str
    week_start: date
    week_end: date
    google_score: float | None
    naver_score: float | None
    combined_score: float | None
    provider_count: int
    coverage_score: float
    current_average: float
    previous_average: float
    growth_rate: float

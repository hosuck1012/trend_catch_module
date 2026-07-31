from pydantic import BaseModel


class TrendRecalculateResponse(BaseModel):
    status: str
    week_start: str
    week_end: str
    calculated_keywords: int
    weekly_trends: int
    watchlist: int
    stable: int
    insufficient_data: int


class TrendItemResponse(BaseModel):
    rank: int
    keyword: str
    weekly_mentions: int
    previous_weekly_mentions: int
    active_days: int
    source_count: int
    growth_rate: float
    peak_day_share: float
    final_score: float
    status: str
    search_interest_score: float = 50.0
    search_provider_count: int = 0
    search_coverage_score: float = 0.0
    google_trends_score: float | None = None
    naver_datalab_score: float | None = None
    primary_entity: str | None = None
    primary_entity_type: str | None = None
    travel_entity_count: int = 0


class TrendListResponse(BaseModel):
    week_start: str
    week_end: str
    total: int
    items: list[TrendItemResponse]


class TrendSummaryResponse(BaseModel):
    week_start: str
    week_end: str
    weekly_trend_count: int
    watchlist_count: int
    stable_count: int
    insufficient_data_count: int
    top_weekly_trend: str | None
    top_watchlist: str | None

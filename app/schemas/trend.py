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
    final_score: float | None
    trend_score: float | None = None
    keyword_quality_score: float | None = None
    status: str
    search_interest_score: float | None = None
    search_interest_available: bool = False
    search_provider_count: int = 0
    search_coverage_score: float = 0.0
    google_trends_score: float | None = None
    naver_datalab_score: float | None = None
    primary_entity: str | None = None
    primary_entity_type: str | None = None
    travel_entity_count: int = 0
    primary_context_title: str | None = None
    primary_context_provider: str | None = None
    context_available: bool = False
    ai_analysis_available: bool = False
    ai_trend_summary: str | None = None
    travel_relevance_score: float | None = None
    travel_relevance_level: str | None = None
    recommended_destination_count: int = 0
    pipeline_version: str = "legacy"
    suspicious: bool = False


class TrendListResponse(BaseModel):
    week_start: str
    week_end: str
    total: int
    items: list[TrendItemResponse]
    pipeline_version: str = "legacy"


class TrendSummaryResponse(BaseModel):
    week_start: str
    week_end: str
    weekly_trend_count: int
    watchlist_count: int
    stable_count: int
    insufficient_data_count: int
    top_weekly_trend: str | None
    top_watchlist: str | None

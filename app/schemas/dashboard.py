from datetime import date, datetime

from pydantic import BaseModel, Field


class DashboardMetricCard(BaseModel):
    key: str
    label: str
    value: int | float
    previous_value: int | float | None = None
    delta: int | float | None = None


class DashboardPipelineItem(BaseModel):
    key: str
    label: str
    status: str
    count: int | None = None
    details: dict[str, object] = Field(default_factory=dict)


class DashboardDistributionItem(BaseModel):
    name: str
    count: int


class DashboardTrendItem(BaseModel):
    rank: int
    keyword: str
    normalized_keyword: str
    week_start: date
    week_end: date
    final_score: float | None
    trend_score: float | None = None
    keyword_quality_score: float | None = None
    search_interest_score: float | None = None
    growth_rate: float | None = None
    acceleration: float | None = None
    source_count: int
    document_count: int
    sources: list[str] = Field(default_factory=list)
    primary_entity: str | None = None
    primary_context_title: str | None = None
    ai_status: str
    ai_analysis_available: bool
    ai_trend_summary: str | None = None
    travel_relevance_score: float | None = None
    travel_relevance_level: str | None = None
    watchlist: bool
    status: str
    suspicious: bool = False
    pipeline_version: str = "legacy"


class DashboardOverviewResponse(BaseModel):
    selected_week: date | None
    available_weeks: list[date]
    metric_cards: list[DashboardMetricCard]
    pipeline_status: list[DashboardPipelineItem]
    top_trends: list[DashboardTrendItem]
    source_distribution: list[DashboardDistributionItem]
    entity_distribution: list[DashboardDistributionItem]
    ai_distribution: list[DashboardDistributionItem]
    keyword_pipeline_version: str = "legacy"
    requested_week: date | None = None
    week_fallback_used: bool = False


class DashboardTrendListResponse(BaseModel):
    selected_week: date | None
    requested_week: date | None = None
    week_fallback_used: bool = False
    total: int
    limit: int
    offset: int
    items: list[DashboardTrendItem]


class DashboardEntityItem(BaseModel):
    entity_text: str
    normalized_entity: str
    entity_type: str
    relation_score: float
    mention_count: int
    source_count: int
    is_primary: bool


class DashboardContextItem(BaseModel):
    provider: str
    page_title: str
    page_url: str
    summary: str
    match_status: str
    context_score: float
    is_primary: bool


class DashboardDocumentItem(BaseModel):
    id: int
    source: str
    title: str
    published_at: datetime
    url: str | None
    snippet: str


class DashboardTrendDetailResponse(BaseModel):
    trend: DashboardTrendItem
    entities: list[DashboardEntityItem]
    contexts: list[DashboardContextItem]
    ai_analysis: dict[str, object] | None
    documents: list[DashboardDocumentItem]

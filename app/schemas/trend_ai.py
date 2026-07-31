from datetime import date, datetime

from pydantic import BaseModel, Field

from app.ai.gemini_schemas import ContentIdea, DestinationSuggestion


class AIAnalysisRunItemResponse(BaseModel):
    keyword: str
    week_start: date
    analysis_id: int | None
    status: str
    error_code: str | None = None
    error_message: str | None = None


class AIAnalysisGenerateResponse(BaseModel):
    status: str
    requested: int
    completed: int
    partial: int
    skipped: int
    errors: int
    model_name: str
    prompt_version: str
    results: list[AIAnalysisRunItemResponse]


class AISourceContextResponse(BaseModel):
    provider: str
    page_title: str
    page_url: str
    match_status: str


class AIAnalysisResponse(BaseModel):
    id: int
    keyword: str
    normalized_keyword: str
    week_start: date
    week_end: date
    analysis_status: str
    trend_summary: str | None
    rising_reason: str | None
    evidence_summary: list[str]
    travel_relevance_score: float | None
    travel_relevance_level: str | None
    travel_relevance_reason: str | None
    recommended_destinations: list[DestinationSuggestion]
    content_ideas: list[ContentIdea]
    cautions: list[str]
    evidence_refs: list[str]
    confidence_score: float | None
    model_name: str
    prompt_version: str
    generated_at: datetime
    error_code: str | None = None
    error_message: str | None = None
    source_contexts: list[AISourceContextResponse] = Field(default_factory=list)


class AIAnalysisListResponse(BaseModel):
    total: int
    items: list[AIAnalysisResponse]


class AIAnalysisStatusResponse(BaseModel):
    gemini_enabled: bool
    model_configured: bool
    api_key_configured: bool
    configured_model: str | None
    completed_count: int
    partial_count: int
    error_count: int
    cached_count: int
    last_generated_at: datetime | None
    prompt_version: str

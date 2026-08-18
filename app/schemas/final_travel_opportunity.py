from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.ai.travel_opportunity_schemas import (
    DestinationCandidate,
    TravelContentIdea,
)


class FinalizePreviewItemResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    candidate_id: int
    input_chars: int
    input_hash: str
    cache_hit: bool
    would_call: bool
    gemini_called: bool
    status: str
    final_decision: str | None = None
    final_travel_score: float | None = None
    travel_angle: str | None = None
    destination_candidates: list[DestinationCandidate] = Field(default_factory=list)
    content_ideas: list[TravelContentIdea] = Field(default_factory=list)
    evidence_refs: list[str] = Field(default_factory=list)
    needs_external_verification: bool | None = None
    error_code: str | None = None
    error_message: str | None = None


class FinalizeResponse(BaseModel):
    status: str
    dry_run: bool
    week_start: date | None = None
    eligible_candidates: int
    expected_gemini_calls: int
    gemini_calls: int
    cache_hits: int
    completed: int
    partial: int
    errors: int
    skipped_budget: int
    model: str | None = None
    prompt_version: str
    items: list[FinalizePreviewItemResponse] = Field(default_factory=list)


class FinalTravelOpportunityResponse(BaseModel):
    id: int
    keyword: str
    normalized_keyword: str
    week_start: date
    week_end: date
    final_decision: Literal["accept", "review", "reject"]
    final_travel_score: float
    trend_context_summary: str
    why_now: str
    travel_angle: str
    destination_candidates: list[DestinationCandidate]
    content_ideas: list[TravelContentIdea]
    evidence_refs: list[str]
    needs_external_verification: bool
    verification_queries: list[str]
    cautions: list[str]
    confidence_score: float
    analysis_status: Literal["completed", "partial", "cached", "error"]
    model: str
    prompt_version: str
    generated_at: datetime


class FinalTravelOpportunityListResponse(BaseModel):
    total: int
    items: list[FinalTravelOpportunityResponse]


class TravelOpportunityCostReportResponse(BaseModel):
    week_start: date | None = None
    raw_keyword_count: int
    quality_keyword_count: int
    rule_candidate_count: int
    semantic_candidate_count: int
    high_precision_candidate_count: int
    gemini_eligible_count: int
    final_accept_count: int
    gemini_calls_this_week: int
    gemini_cache_hits: int
    gemini_errors: int
    overall_llm_reduction_rate: float
    estimated_calls_per_year: float

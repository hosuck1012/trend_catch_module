from datetime import date, datetime

from pydantic import BaseModel, Field


class KeywordContextExampleResponse(BaseModel):
    document_id: int
    keyword: str
    previous_sentence: str | None = None
    matched_sentence: str
    next_sentence: str | None = None
    combined_context: str


class BuildContextsResponse(BaseModel):
    status: str
    dry_run: bool
    week_start: date | None = None
    week_end: date | None = None
    keywords_processed: int
    documents_processed: int
    contexts_found: int
    contexts_would_create: int
    duplicate_contexts: int
    context_examples: list[KeywordContextExampleResponse] = Field(default_factory=list)


class TravelCandidatePreviewResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    score: float
    status: str
    category: str
    reasoning_codes: list[str] = Field(default_factory=list)
    matched_positive_terms: list[str] = Field(default_factory=list)
    matched_negative_terms: list[str] = Field(default_factory=list)
    context: str


class PrefilterResponse(BaseModel):
    status: str
    dry_run: bool
    week_start: date | None = None
    week_end: date | None = None
    processed: int
    rejected: int
    weak: int
    review: int
    strong: int
    estimated_llm_candidates: int
    reduction_rate: float
    top_candidates: list[TravelCandidatePreviewResponse] = Field(default_factory=list)
    rejection_reason_counts: dict[str, int] = Field(default_factory=dict)
    raw_keyword_count: int
    quality_keyword_count: int
    context_candidate_count: int


class TravelOpportunityContextResponse(BaseModel):
    id: int
    document_id: int
    source: str
    published_at: datetime
    previous_sentence: str | None = None
    matched_sentence: str
    next_sentence: str | None = None
    combined_context: str


class TravelOpportunityItemResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    week_start: date
    week_end: date
    score: float
    status: str
    category: str
    primary_entity: str | None = None
    primary_entity_type: str | None = None
    matched_positive_terms: list[str] = Field(default_factory=list)
    matched_negative_terms: list[str] = Field(default_factory=list)
    reasoning_codes: list[str] = Field(default_factory=list)
    contexts: list[TravelOpportunityContextResponse] = Field(default_factory=list)


class TravelOpportunityListResponse(BaseModel):
    total: int
    items: list[TravelOpportunityItemResponse]


class TravelOpportunityEntityResponse(BaseModel):
    text: str
    entity_type: str


class TravelOpportunityDetailResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    score: float
    status: str
    category: str
    contexts: list[TravelOpportunityContextResponse]
    entities: list[TravelOpportunityEntityResponse]
    matched_positive_terms: list[str]
    matched_negative_terms: list[str]
    reasoning_codes: list[str]
    source_count: int
    document_count: int


class TravelOpportunitySummaryResponse(BaseModel):
    week_start: date | None = None
    raw_keyword_count: int
    quality_keyword_count: int
    context_candidate_count: int
    travel_prefilter_count: int
    strong_candidate_count: int
    estimated_gemini_calls: int
    llm_reduction_rate: float
    status_counts: dict[str, int]

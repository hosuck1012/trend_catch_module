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
    created: int = 0
    skipped: int = 0
    next_cursor: int | None = None
    has_more: bool = False
    batches: int = 0
    errors: int = 0
    existing_valid: int = 0
    stale_contexts: int = 0
    removed: int = 0
    unmatched_accepted_pairs: int = 0


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
    created: int = 0
    updated: int = 0
    would_create: int = 0
    would_update: int = 0
    skipped: int = 0
    next_cursor: int | None = None
    has_more: bool = False
    batches: int = 0
    errors: int = 0
    cache_hits: int = 0
    rule_version: str = "v2-rule-2"
    category_counts: dict[str, int] = Field(default_factory=dict)
    reasoning_code_counts: dict[str, int] = Field(default_factory=dict)
    primary_entity_count: int = 0
    other_percentage: float = 0.0


class SemanticCandidatePreviewResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    travel_category: str
    prefilter_status: str
    semantic_travel_score: float
    semantic_status: str
    semantic_positive_score: float
    semantic_positive_category: str
    semantic_negative_score: float
    semantic_negative_category: str
    semantic_margin: float
    semantic_confidence: float
    reasoning_codes: list[str] = Field(default_factory=list)


class SemanticFilterResponse(BaseModel):
    status: str
    dry_run: bool
    week_start: date | None = None
    processed: int
    semantic_rejected: int
    semantic_weak: int
    semantic_review: int
    semantic_strong: int
    estimated_gemini_candidates: int
    top_candidates: list[SemanticCandidatePreviewResponse] = Field(default_factory=list)
    model_name: str
    scoring_version: str
    cache_hits: int


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
    semantic_travel_score: float | None = None
    semantic_status: str | None = None
    embedding_model: str | None = None
    semantic_positive_score: float | None = None
    semantic_positive_category: str | None = None
    semantic_negative_score: float | None = None
    semantic_negative_category: str | None = None
    embedding_input_hash: str | None = None
    semantic_calculated_at: datetime | None = None
    trend_strength_score: float | None = None
    context_clarity_score: float | None = None
    travel_convertibility_score: float | None = None
    evidence_confidence_score: float | None = None
    high_precision_score: float | None = None
    evidence_gate: str | None = None
    evidence_codes: list[str] = Field(default_factory=list)
    evidence_document_count: int | None = None
    evidence_source_count: int | None = None
    ranking_status: str | None = None
    rank_in_week: int | None = None
    ranking_version: str | None = None
    calculated_at: datetime | None = None
    cluster_id: str | None = None
    cluster_representative: bool = False
    gemini_eligible: bool = False
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
    raw_keyword_occurrences: int = 0
    keyword_candidate_total: int = 0
    keyword_candidate_accepted_rows: int = 0
    distinct_accepted_keywords: int = 0
    weekly_trend_count: int = 0
    quality_keyword_count: int
    context_candidate_count: int
    travel_prefilter_count: int
    strong_candidate_count: int
    estimated_gemini_calls: int
    llm_reduction_rate: float
    status_counts: dict[str, int]


class RankingCandidateResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    week_start: date
    travel_category: str
    semantic_category: str | None = None
    semantic_status: str
    semantic_travel_score: float | None = None
    travel_pre_score: float
    trend_strength_score: float
    context_clarity_score: float
    travel_convertibility_score: float
    evidence_confidence_score: float
    high_precision_score: float
    evidence_gate: str
    evidence_codes: list[str] = Field(default_factory=list)
    evidence_document_count: int
    evidence_source_count: int
    ranking_status: str
    rank_in_week: int
    cluster_id: str
    cluster_representative: bool
    gemini_eligible: bool
    contexts: list[str] = Field(default_factory=list)


class FunnelResponse(BaseModel):
    raw_keyword: int
    keyword_quality_passed: int
    rule_candidate: int
    semantic_candidate: int
    high_precision_candidate: int
    gemini_eligible: int
    llm_reduction_rate: float


class RankingResponse(BaseModel):
    status: str
    dry_run: bool
    week_start: date | None = None
    processed: int
    rejected: int
    review: int
    gemini_candidates: int
    priority_candidates: int
    evidence_pass: int
    needs_evidence: int
    evidence_reject: int
    estimated_gemini_calls: int
    top_candidates: list[RankingCandidateResponse] = Field(default_factory=list)
    funnel: FunnelResponse
    annualized_candidate_estimate: float
    insufficient_history: bool


class CalibrationReportResponse(BaseModel):
    ranking_version: str
    week_start: date | None = None
    thresholds: dict[str, float]
    total_semantic_candidates: int
    rejected: int
    review: int
    gemini_candidate: int
    priority_candidate: int
    evidence_gate_counts: dict[str, int]
    score_distribution: dict[str, int]
    top_20_candidates: list[RankingCandidateResponse] = Field(default_factory=list)
    annualized_candidate_estimate: float
    insufficient_history: bool
    weekly_gemini_budget: int
    estimated_llm_calls: int
    overall_reduction_rate: float
    funnel: FunnelResponse

from datetime import date

from pydantic import BaseModel, Field


class AcceptedKeywordItem(BaseModel):
    keyword: str
    normalized_keyword: str
    quality_score: float
    document_count: int
    source_count: int
    extraction_reasons: list[str] = Field(default_factory=list)


class RejectedKeywordItem(BaseModel):
    candidate: str
    rejection_reason: str
    occurrence_count: int


class CandidateExample(BaseModel):
    document_id: int
    candidate: str
    extractor: str
    quality_score: float
    accepted: bool
    rejection_reason: str | None = None


class QualityPreviewResponse(BaseModel):
    pipeline_version: str
    processed_documents: int
    total_candidates: int
    accepted_candidates: int
    rejected_candidates: int
    top_accepted: list[AcceptedKeywordItem]
    top_rejected: list[RejectedKeywordItem]
    rejection_reason_counts: dict[str, int]
    extractor_counts: dict[str, int]
    examples: list[CandidateExample]


class KeywordRebuildResponse(BaseModel):
    status: str
    dry_run: bool
    week_start: date | None
    week_end: date | None
    processed_documents: int
    current_top_20: list[str]
    v2_top_20: list[str]
    removed_keywords: list[str]
    added_keywords: list[str]
    retained_keywords: list[str]
    search_interest_links_preserved: list[str]
    entity_links_to_recalculate: int
    ai_analyses_to_invalidate: int
    source_documents_preserved: int
    search_observations_preserved: int
    manual_contexts_preserved: int


class SuspiciousKeyword(BaseModel):
    keyword: str
    reasons: list[str]


class KeywordQualityReportResponse(BaseModel):
    pipeline_version: str
    active_keywords: int
    rejected_candidate_count: int
    stopword_rejection_count: int
    url_artifact_rejection_count: int
    generic_word_rejection_count: int
    average_quality_score: float | None
    lowest_accepted_quality_score: float | None
    keywords_without_search_validation: int
    suspicious_keywords: list[SuspiciousKeyword]

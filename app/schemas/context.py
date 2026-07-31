from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from app.ner.entity_labels import EntityType


class ContextEnrichmentResponse(BaseModel):
    status: str
    provider: str
    processed_entities: int
    skipped_entities: int
    matched: int
    ambiguous: int
    unmatched: int
    errors: int
    created_contexts: int
    updated_contexts: int


class ContextCandidateResponse(BaseModel):
    page_title: str
    page_url: str
    snippet: str
    match_score: float
    match_status: str


class ContextCandidatesResponse(BaseModel):
    status: str
    wikipedia_enabled: bool
    entity: str
    entity_type: str | None
    candidates: list[ContextCandidateResponse]


class ManualContextCreateRequest(BaseModel):
    provider: Literal["namuwiki_manual", "manual"]
    entity_text: str = Field(min_length=1, max_length=500)
    entity_type: EntityType
    page_title: str = Field(min_length=1, max_length=500)
    page_url: str = Field(min_length=1, max_length=2000)
    summary: str = Field(min_length=1, max_length=1000)
    keyword: str = Field(min_length=1, max_length=255)
    week_start: date


class ManualContextUpdateRequest(BaseModel):
    page_title: str | None = Field(default=None, min_length=1, max_length=500)
    page_url: str | None = Field(default=None, min_length=1, max_length=2000)
    summary: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_at_least_one_field(self):
        if self.page_title is None and self.page_url is None and self.summary is None:
            raise ValueError("수정할 필드가 하나 이상 필요합니다.")
        return self


class ContextRecordResponse(BaseModel):
    id: int
    entity_text: str
    normalized_entity: str
    entity_type: str
    provider: str
    page_title: str
    page_url: str
    summary: str
    match_score: float
    match_status: str
    source_language: str
    attribution: str | None
    retrieved_at: datetime
    updated_at: datetime


class ManualContextResponse(BaseModel):
    status: str
    context: ContextRecordResponse
    keyword: str | None = None
    week_start: date | None = None
    context_score: float | None = None
    guidance: str


class ContextDeleteResponse(BaseModel):
    status: str
    context_id: int


class TrendContextItemResponse(BaseModel):
    context_id: int
    entity: str
    entity_type: str
    provider: str
    page_title: str
    page_url: str
    summary: str
    match_status: str
    context_score: float
    is_primary: bool
    attribution: str | None


class TrendContextPrimaryResponse(BaseModel):
    entity: str
    entity_type: str
    provider: str
    page_title: str
    page_url: str
    summary: str
    context_score: float
    attribution: str | None


class TrendContextsResponse(BaseModel):
    keyword: str
    week_start: date
    week_end: date
    primary_context: TrendContextPrimaryResponse | None
    contexts: list[TrendContextItemResponse]


class ContextStatusResponse(BaseModel):
    wikipedia_enabled: bool
    wikipedia_contexts: int
    namuwiki_manual_contexts: int
    matched: int
    ambiguous: int
    unmatched: int
    keywords_with_context: int
    keywords_without_context: int
    last_retrieved_at: datetime | None


class ContextSummaryResponse(BaseModel):
    total: int
    items: list[ContextRecordResponse]

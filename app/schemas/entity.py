from datetime import date

from pydantic import BaseModel, Field


class EntityExtractionResponse(BaseModel):
    status: str
    model_status: str
    processed_documents: int
    skipped_documents: int
    inserted_entities: int
    entity_counts: dict[str, int]
    extractor_counts: dict[str, int]
    errors: list[str] = Field(default_factory=list)
    model_error: str | None = None


class EntityModelStatusResponse(BaseModel):
    enabled: bool
    status: str
    model_name: str
    device: str
    threshold: float
    last_error: str | None


class EntitySummaryItemResponse(BaseModel):
    text: str
    canonical_text: str
    entity_type: str
    mentions: int
    document_count: int
    source_count: int
    average_confidence: float


class EntitySummaryResponse(BaseModel):
    total_entities: int
    unique_entities: int
    items: list[EntitySummaryItemResponse]


class TrendEntityLinkResponse(BaseModel):
    status: str
    week_start: str | None
    week_end: str | None
    processed_keywords: int
    linked_keywords: int
    inserted_links: int
    primary_links: int


class KeywordEntityItemResponse(BaseModel):
    text: str
    entity_type: str
    mention_count: int
    document_count: int
    source_count: int
    average_confidence: float
    relation_score: float
    is_primary: bool


class PrimaryEntityResponse(BaseModel):
    text: str
    entity_type: str
    relation_score: float


class KeywordEntityResponse(BaseModel):
    keyword: str
    week_start: date
    week_end: date
    primary_entity: PrimaryEntityResponse | None
    entities: list[KeywordEntityItemResponse]
    message: str | None = None

from datetime import datetime

from pydantic import BaseModel


class KeywordExtractionResponse(BaseModel):
    status: str
    processed_documents: int
    skipped_documents: int
    inserted_occurrences: int


class TopKeywordItemResponse(BaseModel):
    keyword: str
    normalized_keyword: str
    mentions: int
    active_days: int
    source_count: int
    sources: list[str]


class TopKeywordsResponse(BaseModel):
    total_occurrences: int
    unique_keywords: int
    items: list[TopKeywordItemResponse]


class KeywordDetailResponse(BaseModel):
    keyword: str
    mentions: int
    active_days: int
    source_count: int
    sources: list[str]
    first_seen: datetime
    last_seen: datetime

from pydantic import BaseModel


class YouTubeCollectionResponse(BaseModel):
    status: str
    source: str
    region_code: str
    requested_results: int
    received_results: int
    inserted_documents: int
    skipped_documents: int
    collected_at: str


class RecentDocumentItem(BaseModel):
    id: int
    source: str
    source_id: str
    title: str
    published_at: str
    views: int | None
    likes: int | None
    comments: int | None
    url: str | None


class RecentDocumentsResponse(BaseModel):
    total: int
    items: list[RecentDocumentItem]

from pydantic import BaseModel


class NewsisRssFailedFeedResponse(BaseModel):
    feed: str
    error: str


class NewsisRssCollectionResponse(BaseModel):
    status: str
    source: str
    feeds: list[str]
    requested_feeds: int
    received_items: int
    inserted_documents: int
    skipped_documents: int
    failed_feeds: list[NewsisRssFailedFeedResponse]
    collected_at: str

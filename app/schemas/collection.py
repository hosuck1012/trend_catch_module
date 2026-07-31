from pydantic import BaseModel


class MockDateRange(BaseModel):
    start: str
    end: str


class MockCollectionResponse(BaseModel):
    status: str
    inserted_documents: int
    skipped_documents: int
    date_range: MockDateRange


class DocumentCountResponse(BaseModel):
    total_documents: int
    youtube: int
    naver_news: int

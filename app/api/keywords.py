from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.keyword_repository import (
    extract_keywords_for_documents,
    get_keyword_detail,
    get_top_keywords,
)
from app.schemas.keyword import (
    KeywordDetailResponse,
    KeywordExtractionResponse,
    TopKeywordsResponse,
)


router = APIRouter(prefix="/api/keywords", tags=["keywords"])


@router.post("/extract", response_model=KeywordExtractionResponse)
def extract_keywords(session: Session = Depends(get_db)) -> dict[str, int | str]:
    result = extract_keywords_for_documents(session)
    return {
        "status": "ok",
        "processed_documents": result.processed_documents,
        "skipped_documents": result.skipped_documents,
        "inserted_occurrences": result.inserted_occurrences,
    }


@router.get("/top", response_model=TopKeywordsResponse)
def top_keywords(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    result = get_top_keywords(session, limit)
    return {
        "total_occurrences": result.total_occurrences,
        "unique_keywords": result.unique_keywords,
        "items": [item.__dict__ for item in result.items],
    }


@router.get("/{normalized_keyword}", response_model=KeywordDetailResponse)
def keyword_detail(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    detail = get_keyword_detail(session, normalized_keyword)
    if detail is None:
        raise HTTPException(status_code=404, detail="Keyword not found")
    return detail.__dict__

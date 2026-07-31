from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.newsis_rss import NewsisRssCollectionResponse
from app.services.newsis_rss_collection_service import (
    NewsisRssAllFeedsFailedError,
    NewsisRssCollectionInputError,
    collect_newsis_rss_documents,
)


router = APIRouter(prefix="/api", tags=["newsis-rss"])


@router.post("/collect/newsis-rss", response_model=NewsisRssCollectionResponse)
async def collect_newsis_rss(
    feeds: str | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        result = await collect_newsis_rss_documents(session, feeds=feeds)
    except NewsisRssCollectionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except NewsisRssAllFeedsFailedError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": str(exc),
                "failed_feeds": [failed.__dict__ for failed in exc.failed_feeds],
            },
        ) from exc

    return {
        "status": "ok",
        "source": result.source,
        "feeds": result.feeds,
        "requested_feeds": result.requested_feeds,
        "received_items": result.received_items,
        "inserted_documents": result.inserted_documents,
        "skipped_documents": result.skipped_documents,
        "failed_feeds": [failed.__dict__ for failed in result.failed_feeds],
        "collected_at": result.collected_at.isoformat(),
    }

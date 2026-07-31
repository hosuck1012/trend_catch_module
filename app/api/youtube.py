from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.collectors.youtube_collector import (
    YouTubeApiAuthError,
    YouTubeApiError,
    YouTubeNetworkError,
)
from app.database import get_db
from app.models.source_document import SourceDocument
from app.schemas.youtube import RecentDocumentsResponse, YouTubeCollectionResponse
from app.services.youtube_collection_service import (
    YouTubeApiKeyMissingError,
    YouTubeCollectionInputError,
    collect_youtube_popular_videos,
)


router = APIRouter(prefix="/api", tags=["youtube"])


@router.post("/collect/youtube", response_model=YouTubeCollectionResponse)
async def collect_youtube(
    region_code: str | None = Query(default=None),
    max_results: int | None = Query(default=None, ge=1, le=50),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        result = await collect_youtube_popular_videos(
            session,
            region_code=region_code,
            max_results=max_results,
        )
    except YouTubeCollectionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except YouTubeApiKeyMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YouTubeApiAuthError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except YouTubeNetworkError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except YouTubeApiError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "status": "ok",
        "source": result.source,
        "region_code": result.region_code,
        "requested_results": result.requested_results,
        "received_results": result.received_results,
        "inserted_documents": result.inserted_documents,
        "skipped_documents": result.skipped_documents,
        "collected_at": result.collected_at.isoformat(),
    }


@router.get("/documents/recent", response_model=RecentDocumentsResponse)
def recent_documents(
    source: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    query = select(SourceDocument)
    count_query = select(func.count(SourceDocument.id))
    if source:
        query = query.where(SourceDocument.source == source)
        count_query = count_query.where(SourceDocument.source == source)

    documents = session.scalars(
        query.order_by(SourceDocument.published_at.desc(), SourceDocument.id.desc()).limit(limit)
    ).all()
    total = session.scalar(count_query) or 0
    return {
        "total": total,
        "items": [
            {
                "id": document.id,
                "source": document.source,
                "source_id": document.source_id,
                "title": document.title,
                "published_at": document.published_at.isoformat(),
                "views": document.views,
                "likes": document.likes,
                "comments": document.comments,
                "url": document.url,
            }
            for document in documents
        ],
    }

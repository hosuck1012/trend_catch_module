from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.ner.entity_labels import EntityType
from app.ner.gliner_adapter import gliner_adapter
from app.repositories.entity_repository import (
    get_entity_summary,
    get_latest_links_for_keyword,
)
from app.repositories.context_repository import get_primary_context_metadata_for_keyword
from app.schemas.entity import (
    EntityExtractionResponse,
    EntityModelStatusResponse,
    EntitySummaryResponse,
    KeywordEntityResponse,
    TrendEntityLinkResponse,
)
from app.services.entity_extraction_service import extract_entities
from app.services.keyword_normalization_service import normalize_keyword
from app.services.trend_entity_link_service import link_trends_to_entities


router = APIRouter(prefix="/api/entities", tags=["entities"])


@router.post("/extract", response_model=EntityExtractionResponse)
async def extract(
    limit: int = Query(default=100, ge=1, le=500),
    force: bool = Query(default=False),
    source: str | None = Query(default=None),
    since_days: int = Query(default=14, ge=1, le=3650),
) -> dict[str, object]:
    result = await extract_entities(
        limit=limit,
        force=force,
        source=source,
        since_days=since_days,
    )
    return result.__dict__


@router.get("/model-status", response_model=EntityModelStatusResponse)
def model_status() -> dict[str, object]:
    return gliner_adapter.get_status().__dict__


@router.get("/summary", response_model=EntitySummaryResponse)
def summary(
    entity_type: EntityType | None = Query(default=None),
    source: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    total, unique, rows = get_entity_summary(
        session,
        entity_type=entity_type.value if entity_type else None,
        source=source,
        limit=limit,
    )
    return {
        "total_entities": total,
        "unique_entities": unique,
        "items": [row.__dict__ for row in rows],
    }


@router.post("/link-trends", response_model=TrendEntityLinkResponse)
def link_trends(session: Session = Depends(get_db)) -> dict[str, object]:
    return link_trends_to_entities(session).__dict__


@router.get("/by-keyword/{normalized_keyword}", response_model=KeywordEntityResponse)
def by_keyword(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    keyword = normalize_keyword(normalized_keyword)
    if not keyword:
        raise HTTPException(status_code=404, detail="키워드를 찾을 수 없습니다.")
    trend, links = get_latest_links_for_keyword(session, keyword)
    if trend is None:
        raise HTTPException(status_code=404, detail="키워드 트렌드를 찾을 수 없습니다.")
    primary = next((link for link in links if link.is_primary), None)
    primary_context = get_primary_context_metadata_for_keyword(
        session,
        keyword=trend.keyword,
        week_start=trend.week_start,
    )
    return {
        "keyword": trend.keyword,
        "week_start": trend.week_start,
        "week_end": trend.week_end,
        "primary_entity": (
            {
                "text": primary.entity_text,
                "entity_type": primary.entity_type,
                "relation_score": primary.relation_score,
            }
            if primary
            else None
        ),
        "entities": [
            {
                "text": link.entity_text,
                "entity_type": link.entity_type,
                "mention_count": link.mention_count,
                "document_count": link.document_count,
                "source_count": link.source_count,
                "average_confidence": link.average_confidence,
                "relation_score": link.relation_score,
                "is_primary": link.is_primary,
            }
            for link in links
        ],
        "message": (
            None
            if links
            else "아직 객체 연결 결과가 없습니다. POST /api/entities/link-trends를 실행하세요."
        ),
        "context_available": primary_context is not None,
        "context_provider": primary_context.provider if primary_context else None,
        "context_page_title": primary_context.page_title if primary_context else None,
    }

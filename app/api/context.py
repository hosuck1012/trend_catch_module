from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.config import get_settings
from app.context.context_normalizer import normalize_context_text
from app.context.provider_registry import (
    ALLOWED_MATCH_STATUSES,
    ALLOWED_CONTEXT_PROVIDERS,
    WIKIPEDIA_PROVIDER,
)
from app.context.wikipedia_client import (
    WikimediaConfigurationError,
    WikipediaClient,
    WikipediaClientError,
)
from app.database import get_db
from app.ner.entity_dictionary import canonical_location_for
from app.ner.entity_labels import EntityType
from app.repositories.context_repository import (
    get_context_status_counts,
    get_context_summary,
    get_latest_context_links_for_keyword,
)
from app.schemas.context import (
    ContextCandidatesResponse,
    ContextDeleteResponse,
    ContextEnrichmentResponse,
    ContextStatusResponse,
    ContextSummaryResponse,
    ManualContextCreateRequest,
    ManualContextResponse,
    ManualContextUpdateRequest,
    TrendContextsResponse,
)
from app.services.context_enrichment_service import (
    ContextInputError,
    ContextNotFoundError,
    ContextPermissionError,
    create_manual_context,
    enrich_contexts,
    get_candidates_for_entity,
    remove_manual_context,
    update_manual_context,
)
from app.services.keyword_normalization_service import normalize_keyword


router = APIRouter(prefix="/api/context", tags=["context"])
MANUAL_GUIDANCE = "원문 전체가 아닌 사용자가 직접 확인한 URL과 1000자 이하의 짧은 맥락만 저장하세요."


async def get_wikipedia_client() -> AsyncGenerator[WikipediaClient, None]:
    client = WikipediaClient()
    try:
        yield client
    finally:
        await client.close()


@router.post("/enrich", response_model=ContextEnrichmentResponse)
async def enrich(
    limit: int = Query(default=20, ge=1, le=100),
    force: bool = Query(default=False),
    entity_type: EntityType | None = Query(default=None),
    provider: str = Query(default=WIKIPEDIA_PROVIDER),
    wikipedia_client: WikipediaClient = Depends(get_wikipedia_client),
) -> dict[str, object]:
    try:
        result = await enrich_contexts(
            limit=limit,
            force=force,
            entity_type=entity_type.value if entity_type else None,
            provider=provider,
            wikipedia_client=wikipedia_client,
        )
    except ContextInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except WikimediaConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return result.__dict__


@router.get(
    "/candidates/{normalized_entity}",
    response_model=ContextCandidatesResponse,
)
async def candidates(
    normalized_entity: str,
    wikipedia_client: WikipediaClient = Depends(get_wikipedia_client),
) -> dict[str, object]:
    settings = get_settings()
    canonical = canonical_location_for(normalized_entity) or normalized_entity
    normalized = normalize_context_text(canonical)
    if not normalized:
        raise HTTPException(status_code=404, detail="객체를 찾을 수 없습니다.")
    if not settings.wikipedia_enabled:
        return {
            "status": "disabled",
            "wikipedia_enabled": False,
            "entity": normalized_entity,
            "entity_type": None,
            "candidates": [],
        }
    try:
        target, rows = await get_candidates_for_entity(
            normalized_entity=canonical,
            client=wikipedia_client,
        )
    except WikimediaConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except WikipediaClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    if target is None:
        raise HTTPException(status_code=404, detail="TrendEntityLink 객체를 찾을 수 없습니다.")
    return {
        "status": "ok",
        "wikipedia_enabled": True,
        "entity": target.entity_text,
        "entity_type": target.entity_type,
        "candidates": [
            {
                "page_title": row.page_title,
                "page_url": row.page_url,
                "snippet": row.snippet,
                "match_score": row.match_score,
                "match_status": row.match_status,
            }
            for row in rows
        ],
    }


@router.post("/manual", response_model=ManualContextResponse)
def create_manual(
    request: ManualContextCreateRequest,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        result = create_manual_context(
            session,
            provider=request.provider,
            entity_text=request.entity_text,
            entity_type=request.entity_type.value,
            page_title=request.page_title,
            page_url=request.page_url,
            summary=request.summary,
            keyword=request.keyword,
            week_start=request.week_start,
        )
    except ContextInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ContextNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "status": "ok",
        "context": _context_response(result.context),
        "keyword": result.keyword,
        "week_start": result.week_start,
        "context_score": result.context_score,
        "guidance": MANUAL_GUIDANCE,
    }


@router.patch("/manual/{context_id}", response_model=ManualContextResponse)
def update_manual(
    context_id: int,
    request: ManualContextUpdateRequest,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        context = update_manual_context(
            session,
            context_id=context_id,
            page_title=request.page_title,
            page_url=request.page_url,
            summary=request.summary,
        )
    except ContextInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ContextNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContextPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {
        "status": "ok",
        "context": _context_response(context),
        "guidance": MANUAL_GUIDANCE,
    }


@router.delete("/manual/{context_id}", response_model=ContextDeleteResponse)
def delete_manual(
    context_id: int,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        remove_manual_context(session, context_id=context_id)
    except ContextNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ContextPermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"status": "deleted", "context_id": context_id}


@router.get(
    "/by-keyword/{normalized_keyword}",
    response_model=TrendContextsResponse,
)
def by_keyword(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    keyword = normalize_keyword(normalized_keyword)
    if keyword is None:
        raise HTTPException(status_code=404, detail="키워드를 찾을 수 없습니다.")
    trend, links = get_latest_context_links_for_keyword(session, keyword)
    if trend is None:
        raise HTTPException(status_code=404, detail="키워드 트렌드를 찾을 수 없습니다.")
    primary = next((link for link in links if link.is_primary), None)
    return {
        "keyword": trend.keyword,
        "week_start": trend.week_start,
        "week_end": trend.week_end,
        "primary_context": (
            _primary_context_response(primary) if primary else None
        ),
        "contexts": [_trend_context_response(link) for link in links],
    }


@router.get("/status", response_model=ContextStatusResponse)
def status(session: Session = Depends(get_db)) -> dict[str, object]:
    counts = get_context_status_counts(session)
    return {
        "wikipedia_enabled": get_settings().wikipedia_enabled,
        **counts.__dict__,
    }


@router.get("/summary", response_model=ContextSummaryResponse)
def summary(
    provider: str | None = Query(default=None),
    entity_type: EntityType | None = Query(default=None),
    match_status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    if provider and provider not in ALLOWED_CONTEXT_PROVIDERS:
        raise HTTPException(status_code=422, detail="지원하지 않는 provider입니다.")
    if match_status and match_status not in ALLOWED_MATCH_STATUSES:
        raise HTTPException(status_code=422, detail="지원하지 않는 match_status입니다.")
    rows = get_context_summary(
        session,
        provider=provider,
        entity_type=entity_type.value if entity_type else None,
        match_status=match_status,
        limit=limit,
    )
    return {"total": len(rows), "items": [_context_response(row) for row in rows]}


def _context_response(context) -> dict[str, object]:
    return {
        "id": context.id,
        "entity_text": context.entity_text,
        "normalized_entity": context.normalized_entity,
        "entity_type": context.entity_type,
        "provider": context.provider,
        "page_title": context.page_title,
        "page_url": context.page_url,
        "summary": context.summary,
        "match_score": context.match_score,
        "match_status": context.match_status,
        "source_language": context.source_language,
        "attribution": context.attribution_text,
        "retrieved_at": context.retrieved_at,
        "updated_at": context.updated_at,
    }


def _trend_context_response(link) -> dict[str, object]:
    context = link.entity_context
    return {
        "context_id": context.id,
        "entity": context.normalized_entity,
        "entity_type": link.entity_type,
        "provider": context.provider,
        "page_title": context.page_title,
        "page_url": context.page_url,
        "summary": context.summary,
        "match_status": context.match_status,
        "context_score": link.context_score,
        "is_primary": link.is_primary,
        "attribution": context.attribution_text,
    }


def _primary_context_response(link) -> dict[str, object]:
    context = link.entity_context
    return {
        "entity": context.normalized_entity,
        "entity_type": link.entity_type,
        "provider": context.provider,
        "page_title": context.page_title,
        "page_url": context.page_url,
        "summary": context.summary,
        "context_score": link.context_score,
        "attribution": context.attribution_text,
    }

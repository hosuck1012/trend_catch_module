from collections.abc import AsyncGenerator
from dataclasses import asdict
from datetime import date
from functools import lru_cache
from threading import Lock
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.ai.gemini_adapter import GeminiAdapter
from app.config import get_settings
from app.context_v2.embedding_adapter import get_embedding_adapter
from app.context_v2.semantic_scorer import (
    SemanticAnchors,
    SemanticScorer,
    load_semantic_anchors,
)
from app.repositories import final_travel_opportunity_repository as final_repo
from app.repositories import travel_opportunity_repository as repo
from app.schemas.travel_opportunity import (
    BuildContextsResponse,
    CalibrationReportResponse,
    PrefilterResponse,
    RelatedDestinationExpansionResponse,
    RankingResponse,
    SemanticFilterResponse,
    TravelOpportunityDetailResponse,
    TravelOpportunityListResponse,
    TravelOpportunitySummaryResponse,
)
from app.schemas.final_travel_opportunity import (
    FinalizeResponse,
    FinalTravelOpportunityListResponse,
    FinalTravelOpportunityResponse,
    TravelOpportunityCostReportResponse,
)
from app.services.final_travel_opportunity_service import (
    cost_report,
    finalize_travel_opportunities,
    serialize_final,
    serialize_finalize_result,
)
from app.services.keyword_normalization_service import normalize_keyword
from app.services.keyword_context_service import build_keyword_contexts, serialize_build_result
from app.services.related_destination_expansion_service import (
    expand_related_destinations,
    serialize_expansion_result,
)
from app.services.travel_prefilter_service import (
    detail_for_keyword,
    prefilter_travel_opportunities,
    serialize_candidate,
    serialize_prefilter_result,
)
from app.services.travel_ranking_service import (
    calibration_report,
    rank_travel_opportunities,
    serialize_ranking_result,
)
from app.services.travel_semantic_filter_service import (
    semantic_filter_travel_opportunities,
    serialize_semantic_filter_result,
)


router = APIRouter(prefix="/api/travel-opportunities", tags=["travel-opportunities-v2"])
_SCORER_CACHE_LOCK = Lock()


async def get_final_gemini_adapter() -> AsyncGenerator[GeminiAdapter, None]:
    adapter = GeminiAdapter()
    try:
        yield adapter
    finally:
        await adapter.close()


@lru_cache(maxsize=8)
def _cached_semantic_scorer(
    model_name: str,
    device: str,
    batch_size: int,
    enabled: bool,
    anchor_version: str,
    scoring_version: str,
    reject_threshold: float,
    review_threshold: float,
    strong_threshold: float,
) -> SemanticScorer:
    adapter = get_embedding_adapter(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        enabled=enabled,
    )
    anchors = (
        load_semantic_anchors(expected_version=anchor_version)
        if enabled
        else SemanticAnchors(
            version=anchor_version,
            positive={},
            negative={},
            content_hash="disabled",
        )
    )
    return SemanticScorer(
        adapter=adapter,
        anchors=anchors,
        reject_threshold=reject_threshold,
        review_threshold=review_threshold,
        strong_threshold=strong_threshold,
        scoring_version=scoring_version,
    )


def get_travel_semantic_scorer() -> SemanticScorer:
    settings = get_settings()
    try:
        with _SCORER_CACHE_LOCK:
            return _cached_semantic_scorer(
                settings.travel_embedding_model,
                settings.travel_embedding_device,
                settings.travel_embedding_batch_size,
                settings.travel_embedding_enabled,
                settings.travel_semantic_anchor_version,
                settings.travel_semantic_scoring_version,
                settings.travel_semantic_reject_threshold,
                settings.travel_semantic_review_threshold,
                settings.travel_semantic_strong_threshold,
            )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local semantic configuration unavailable: {type(exc).__name__}",
        ) from exc


@router.post("/build-contexts", response_model=BuildContextsResponse)
def build_contexts(
    week_start: date | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
    after_id: int | None = Query(default=None, ge=0),
    process_all: bool = Query(default=False),
    force: bool = Query(default=False),
    dry_run: bool = Query(default=True),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return serialize_build_result(
        build_keyword_contexts(
            session,
            week_start=week_start,
            limit=limit,
            force=force,
            dry_run=dry_run,
            after_id=after_id,
            process_all=process_all,
        )
    )


@router.post(
    "/expand-destinations",
    response_model=RelatedDestinationExpansionResponse,
)
def expand_destinations(
    week_start: date | None = Query(default=None),
    dry_run: bool = Query(default=True),
    force: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return serialize_expansion_result(
        expand_related_destinations(
            session,
            week_start=week_start,
            dry_run=dry_run,
            force=force,
            limit=limit,
        )
    )


@router.post("/prefilter", response_model=PrefilterResponse)
def prefilter(
    week_start: date | None = Query(default=None),
    dry_run: bool = Query(default=True),
    force: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
    after_id: int | None = Query(default=None, ge=0),
    process_all: bool = Query(default=False),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return serialize_prefilter_result(
        prefilter_travel_opportunities(
            session,
            week_start=week_start,
            dry_run=dry_run,
            force=force,
            limit=limit,
            after_id=after_id,
            process_all=process_all,
        )
    )


@router.post("/semantic-filter", response_model=SemanticFilterResponse)
def semantic_filter(
    week_start: date | None = Query(default=None),
    dry_run: bool = Query(default=True),
    force: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=5000),
    session: Session = Depends(get_db),
    scorer: SemanticScorer = Depends(get_travel_semantic_scorer),
) -> dict[str, object]:
    try:
        result = semantic_filter_travel_opportunities(
            session,
            scorer=scorer,
            week_start=week_start,
            dry_run=dry_run,
            force=force,
            limit=limit,
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Local semantic model unavailable: {type(exc).__name__}",
        ) from exc
    return serialize_semantic_filter_result(result)


@router.get("/summary", response_model=TravelOpportunitySummaryResponse)
def summary(
    week_start: date | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return repo.summarize_v2(session, week_start=week_start)


@router.post("/rank", response_model=RankingResponse)
def rank(
    week_start: date | None = Query(default=None),
    dry_run: bool = Query(default=True),
    force: bool = Query(default=False),
    limit: int = Query(default=100, ge=1, le=5000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return serialize_ranking_result(
        rank_travel_opportunities(
            session,
            week_start=week_start,
            dry_run=dry_run,
            force=force,
            limit=limit,
        )
    )


@router.get("/calibration-report", response_model=CalibrationReportResponse)
def get_calibration_report(
    week_start: date | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return calibration_report(session, week_start=week_start)


@router.post("/finalize", response_model=FinalizeResponse)
async def finalize(
    week_start: date | None = Query(default=None),
    keyword: str | None = Query(default=None, min_length=2, max_length=255),
    limit: int = Query(default=3, ge=1, le=20),
    force: bool = Query(default=False),
    dry_run: bool = Query(default=False),
    session: Session = Depends(get_db),
    adapter: GeminiAdapter = Depends(get_final_gemini_adapter),
) -> dict[str, object]:
    result = await finalize_travel_opportunities(
        session,
        week_start=week_start,
        keyword=keyword,
        limit=limit,
        force=force,
        dry_run=dry_run,
        adapter=adapter,
    )
    return serialize_finalize_result(result)


@router.get("/cost-report", response_model=TravelOpportunityCostReportResponse)
def get_cost_report(
    week_start: date | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return cost_report(session, week_start=week_start)


@router.get("/final", response_model=FinalTravelOpportunityListResponse)
def list_final(
    week_start: date | None = Query(default=None),
    decision: Literal["accept", "review", "reject"] | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100),
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    rows = final_repo.list_final_opportunities(
        session,
        week_start=week_start,
        decision=decision,
        min_score=min_score,
        limit=limit,
    )
    return {"total": len(rows), "items": [serialize_final(row) for row in rows]}


@router.get(
    "/final/{normalized_keyword}",
    response_model=FinalTravelOpportunityResponse,
)
def final_detail(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    normalized = normalize_keyword(normalized_keyword)
    row = final_repo.get_latest_final(
        session,
        normalized_keyword=normalized or normalized_keyword,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Final travel opportunity not found")
    return serialize_final(row)


@router.get("", response_model=TravelOpportunityListResponse)
def list_opportunities(
    week_start: date | None = Query(default=None),
    status: str | None = Query(default=None),
    min_score: float | None = Query(default=None, ge=0, le=100),
    travel_category: str | None = Query(default=None),
    semantic_status: str | None = Query(default=None),
    ranking_status: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    rows = repo.get_candidates(
        session,
        week_start=week_start,
        status=status,
        min_score=min_score,
        travel_category=travel_category,
        semantic_status=semantic_status,
        ranking_status=ranking_status,
        limit=limit,
    )
    related_by_key: dict[tuple[date, str], list[object]] = {}
    for row_week_start in {row.week_start for row in rows}:
        week_rows = [row for row in rows if row.week_start == row_week_start]
        destinations = repo.get_related_destination_contexts(
            session,
            normalized_keywords=sorted(
                {row.normalized_keyword for row in week_rows}
            ),
            week_start=row_week_start,
        )
        for normalized_keyword, contexts in destinations.items():
            related_by_key[(row_week_start, normalized_keyword)] = contexts
    return {
        "total": len(rows),
        "items": [
            serialize_candidate(
                row,
                related_destinations=related_by_key.get(
                    (row.week_start, row.normalized_keyword), []
                ),
            )
            for row in rows
        ],
    }


@router.get("/{normalized_keyword}", response_model=TravelOpportunityDetailResponse)
def opportunity_detail(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    result = detail_for_keyword(session, normalized_keyword)
    if result is None:
        raise HTTPException(status_code=404, detail="Travel opportunity candidate not found")
    return result

from dataclasses import asdict
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories import travel_opportunity_repository as repo
from app.schemas.travel_opportunity import (
    BuildContextsResponse,
    CalibrationReportResponse,
    PrefilterResponse,
    RankingResponse,
    TravelOpportunityDetailResponse,
    TravelOpportunityListResponse,
    TravelOpportunitySummaryResponse,
)
from app.services.keyword_context_service import build_keyword_contexts, serialize_build_result
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


router = APIRouter(prefix="/api/travel-opportunities", tags=["travel-opportunities-v2"])


@router.post("/build-contexts", response_model=BuildContextsResponse)
def build_contexts(
    week_start: date | None = Query(default=None),
    limit: int = Query(default=500, ge=1, le=5000),
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
        )
    )


@router.post("/prefilter", response_model=PrefilterResponse)
def prefilter(
    week_start: date | None = Query(default=None),
    dry_run: bool = Query(default=True),
    force: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return serialize_prefilter_result(
        prefilter_travel_opportunities(
            session,
            week_start=week_start,
            dry_run=dry_run,
            force=force,
            limit=limit,
        )
    )


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
    return {"total": len(rows), "items": [serialize_candidate(row) for row in rows]}


@router.get("/{normalized_keyword}", response_model=TravelOpportunityDetailResponse)
def opportunity_detail(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    result = detail_for_keyword(session, normalized_keyword)
    if result is None:
        raise HTTPException(status_code=404, detail="Travel opportunity candidate not found")
    return result

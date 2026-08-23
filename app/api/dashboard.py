from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.dashboard import (
    DashboardOverviewResponse,
    DashboardTrendDetailResponse,
    DashboardTrendListResponse,
)
from app.services.dashboard_service import (
    DashboardTrendNotFoundError,
    DashboardWeekNotFoundError,
    get_dashboard_overview,
    get_dashboard_trend_detail,
    get_dashboard_trends,
)


router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/overview", response_model=DashboardOverviewResponse)
def overview(
    week_start: date | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        return get_dashboard_overview(session, week_start=week_start)
    except DashboardWeekNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/trends", response_model=DashboardTrendListResponse)
def trends(
    week_start: date | None = Query(default=None),
    query: str | None = Query(default=None, max_length=255),
    source: str | None = Query(default=None, max_length=50),
    watchlist_only: bool = Query(default=False),
    include_low_quality: bool = Query(default=True),
    min_final_score: float | None = Query(default=None, ge=0, le=100),
    min_travel_score: float | None = Query(default=None, ge=0, le=100),
    travel_level: str | None = Query(default=None),
    ai_status: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        return get_dashboard_trends(
            session,
            week_start=week_start,
            query=query,
            source=source,
            watchlist_only=watchlist_only,
            include_low_quality=include_low_quality,
            min_final_score=min_final_score,
            min_travel_score=min_travel_score,
            travel_level=travel_level,
            ai_status=ai_status,
            limit=limit,
            offset=offset,
        )
    except DashboardWeekNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/trends/{normalized_keyword}", response_model=DashboardTrendDetailResponse)
def trend_detail(
    normalized_keyword: str,
    week_start: date | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        return get_dashboard_trend_detail(
            session,
            normalized_keyword=normalized_keyword,
            week_start=week_start,
        )
    except DashboardTrendNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

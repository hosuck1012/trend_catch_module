from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.trend_repository import (
    get_latest_week_range,
    get_status_counts,
    get_trends_by_status,
)
from app.repositories.search_interest_repository import get_validations_for_week
from app.repositories.entity_repository import get_link_metadata_for_week
from app.schemas.trend import (
    TrendListResponse,
    TrendRecalculateResponse,
    TrendSummaryResponse,
)
from app.services.trend_calculation_service import (
    NoKeywordOccurrencesError,
    recalculate_weekly_trends,
)


router = APIRouter(prefix="/api/trends", tags=["trends"])


@router.post("/recalculate", response_model=TrendRecalculateResponse)
def recalculate_trends(session: Session = Depends(get_db)) -> dict[str, object]:
    try:
        result = recalculate_weekly_trends(session)
    except NoKeywordOccurrencesError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "week_start": result.week_start.isoformat(),
        "week_end": result.week_end.isoformat(),
        "calculated_keywords": result.calculated_keywords,
        "weekly_trends": result.weekly_trends,
        "watchlist": result.watchlist,
        "stable": result.stable,
        "insufficient_data": result.insufficient_data,
    }


@router.get("/weekly", response_model=TrendListResponse)
def weekly_trends(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return _trend_list_response(session, status="weekly_trend", limit=limit)


@router.get("/watchlist", response_model=TrendListResponse)
def watchlist(
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return _trend_list_response(session, status="watchlist", limit=limit)


@router.get("/summary", response_model=TrendSummaryResponse)
def summary(session: Session = Depends(get_db)) -> dict[str, object]:
    week_range = get_latest_week_range(session)
    if week_range is None:
        raise HTTPException(
            status_code=404,
            detail="계산된 트렌드 데이터가 없습니다. POST /api/trends/recalculate를 먼저 실행하세요.",
        )
    week_start, week_end = week_range
    status_counts = get_status_counts(session)
    top_weekly = get_trends_by_status(session, status="weekly_trend", limit=1)
    top_watchlist = get_trends_by_status(session, status="watchlist", limit=1)
    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "weekly_trend_count": status_counts.get("weekly_trend", 0),
        "watchlist_count": status_counts.get("watchlist", 0),
        "stable_count": status_counts.get("stable", 0),
        "insufficient_data_count": status_counts.get("insufficient_data", 0),
        "top_weekly_trend": top_weekly[0].keyword if top_weekly else None,
        "top_watchlist": top_watchlist[0].keyword if top_watchlist else None,
    }


def _trend_list_response(
    session: Session,
    *,
    status: str,
    limit: int,
) -> dict[str, object]:
    week_range = get_latest_week_range(session)
    if week_range is None:
        raise HTTPException(
            status_code=404,
            detail="계산된 트렌드 데이터가 없습니다. POST /api/trends/recalculate를 먼저 실행하세요.",
        )
    week_start, week_end = week_range
    trends = get_trends_by_status(session, status=status, limit=limit)
    validations = get_validations_for_week(
        session,
        week_start=week_start,
        keywords=[trend.keyword for trend in trends],
    )
    entity_metadata = get_link_metadata_for_week(
        session,
        week_start=week_start,
        keywords=[trend.keyword for trend in trends],
    )
    return {
        "week_start": week_start.isoformat(),
        "week_end": week_end.isoformat(),
        "total": len(trends),
        "items": [
            _trend_item_response(
                index,
                trend,
                validations.get(trend.keyword),
                entity_metadata.get(trend.keyword),
            )
            for index, trend in enumerate(trends, start=1)
        ],
    }


def _trend_item_response(index, trend, validation, entity_metadata=None) -> dict[str, object]:
    primary_entity, primary_entity_type, travel_count = entity_metadata or (
        None,
        None,
        0,
    )
    return {
        "rank": index,
        "keyword": trend.keyword,
        "weekly_mentions": trend.weekly_mentions,
        "previous_weekly_mentions": trend.previous_weekly_mentions,
        "active_days": trend.active_days,
        "source_count": trend.source_count,
        "growth_rate": trend.growth_rate,
        "peak_day_share": trend.peak_day_share,
        "final_score": trend.final_score,
        "status": trend.status,
        "search_interest_score": trend.search_interest_score,
        "search_provider_count": validation.provider_count if validation else 0,
        "search_coverage_score": validation.coverage_score if validation else 0.0,
        "google_trends_score": validation.google_score if validation else None,
        "naver_datalab_score": validation.naver_score if validation else None,
        "primary_entity": primary_entity,
        "primary_entity_type": primary_entity_type,
        "travel_entity_count": travel_count,
    }

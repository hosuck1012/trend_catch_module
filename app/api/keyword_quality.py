from dataclasses import asdict
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.keyword_quality import (
    KeywordQualityReportResponse,
    KeywordRebuildResponse,
    QualityPreviewResponse,
)
from app.services.keyword_rebuild_service import (
    preview_quality,
    quality_report,
    rebuild_keywords,
)


router = APIRouter(prefix="/api/keywords", tags=["keyword-quality"])


@router.post("/quality-preview", response_model=QualityPreviewResponse)
def quality_preview(
    limit: int = Query(default=100, ge=1, le=500),
    source: str | None = Query(default=None, max_length=50),
    since_days: int = Query(default=14, ge=1, le=365),
    include_rejected: bool = Query(default=True),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return preview_quality(
        session,
        limit=limit,
        source=source,
        since_days=since_days,
        include_rejected=include_rejected,
    )


@router.post("/rebuild", response_model=KeywordRebuildResponse)
def rebuild(
    week_start: date | None = Query(default=None),
    since_days: int = Query(default=14, ge=1, le=365),
    dry_run: bool = Query(default=True),
    force: bool = Query(default=False),
    limit: int = Query(default=500, ge=1, le=5000),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        return asdict(
            rebuild_keywords(
                session,
                week_start=week_start,
                since_days=since_days,
                dry_run=dry_run,
                force=force,
                limit=limit,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/quality-report", response_model=KeywordQualityReportResponse)
def report(session: Session = Depends(get_db)) -> dict[str, object]:
    return quality_report(session)

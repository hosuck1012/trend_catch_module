from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.repositories.search_interest_repository import (
    get_latest_validation,
    get_provider_import_status,
    get_validation_coverage_counts,
)
from app.repositories.trend_repository import get_latest_week_range
from app.schemas.search_interest import (
    ManualSearchInterestRequest,
    SearchInterestImportResponse,
    SearchInterestRecalculateResponse,
    SearchInterestStatusResponse,
    SearchValidationDetailResponse,
)
from app.services.keyword_normalization_service import normalize_keyword
from app.services.search_interest_import_service import (
    MAX_CSV_BYTES,
    ManualObservationValue,
    SearchInterestImportError,
    SearchInterestImportResult,
    import_manual_observations,
    import_search_interest_csv,
)
from app.services.search_interest_scoring_service import (
    NoWeeklyTrendsError,
    recalculate_search_interest,
)


router = APIRouter(prefix="/api/search-interest", tags=["search-interest"])


@router.post("/import/google-trends", response_model=SearchInterestImportResponse)
async def import_google_trends(
    file: UploadFile = File(...),
    geo: str = Form(default="KR"),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return await _import_csv(file, geo, "google_trends", session)


@router.post("/import/naver-datalab", response_model=SearchInterestImportResponse)
async def import_naver_datalab(
    file: UploadFile = File(...),
    geo: str = Form(default="KR"),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    return await _import_csv(file, geo, "naver_datalab", session)


@router.post("/manual", response_model=SearchInterestImportResponse)
def import_manual(
    request: ManualSearchInterestRequest,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    try:
        result = import_manual_observations(
            session,
            provider=request.provider,
            keyword=request.keyword,
            geo=request.geo,
            observations=[
                ManualObservationValue(
                    observed_date=observation.date,
                    interest_value=observation.value,
                )
                for observation in request.observations
            ],
        )
    except SearchInterestImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _import_result_response(result)


@router.post("/recalculate", response_model=SearchInterestRecalculateResponse)
def recalculate(session: Session = Depends(get_db)) -> dict[str, object]:
    try:
        result = recalculate_search_interest(session)
    except NoWeeklyTrendsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "ok",
        "week_start": result.week_start,
        "week_end": result.week_end,
        "recalculated_keywords": result.recalculated_keywords,
        "validated_weekly_trends": result.validated_weekly_trends,
        "unvalidated_weekly_trends": result.unvalidated_weekly_trends,
        "updated_weekly_trends": result.updated_weekly_trends,
    }


@router.get("/status", response_model=SearchInterestStatusResponse)
def status(session: Session = Depends(get_db)) -> dict[str, object]:
    google = get_provider_import_status(session, "google_trends")
    naver = get_provider_import_status(session, "naver_datalab")
    week_range = get_latest_week_range(session)
    validated, unvalidated = get_validation_coverage_counts(
        session,
        week_start=week_range[0] if week_range else None,
    )
    return {
        "google_trends": google.__dict__,
        "naver_datalab": naver.__dict__,
        "validated_weekly_trends": validated,
        "unvalidated_weekly_trends": unvalidated,
    }


@router.get("/{normalized_keyword}", response_model=SearchValidationDetailResponse)
def validation_detail(
    normalized_keyword: str,
    session: Session = Depends(get_db),
) -> dict[str, object]:
    normalized = normalize_keyword(normalized_keyword)
    result = get_latest_validation(session, normalized) if normalized else None
    if result is None:
        raise HTTPException(status_code=404, detail="검색 관심도 검증 결과가 없습니다.")
    return {
        "keyword": result.keyword,
        "week_start": result.week_start,
        "week_end": result.week_end,
        "google_score": result.google_score,
        "naver_score": result.naver_score,
        "combined_score": result.combined_score,
        "provider_count": result.provider_count,
        "coverage_score": result.coverage_score,
        "current_average": result.current_average,
        "previous_average": result.previous_average,
        "growth_rate": result.growth_rate,
    }


async def _import_csv(
    file: UploadFile,
    geo: str,
    provider: str,
    session: Session,
) -> dict[str, object]:
    content = await file.read(MAX_CSV_BYTES + 1)
    try:
        result = import_search_interest_csv(
            session,
            provider=provider,
            content=content,
            default_geo=geo,
        )
    except SearchInterestImportError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _import_result_response(result)


def _import_result_response(result: SearchInterestImportResult) -> dict[str, object]:
    return {
        "status": "ok",
        "provider": result.provider,
        "received_rows": result.received_rows,
        "inserted_rows": result.inserted_rows,
        "updated_rows": result.updated_rows,
        "skipped_rows": result.skipped_rows,
        "keywords": result.keywords,
        "date_range": {"start": result.start_date, "end": result.end_date},
        "errors": [error.__dict__ for error in result.errors],
    }

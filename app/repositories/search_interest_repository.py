from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.models.search_interest_observation import SearchInterestObservation
from app.models.search_validation_result import SearchValidationResult
from app.models.weekly_trend import WeeklyTrend


@dataclass(frozen=True)
class ObservationValues:
    provider: str
    keyword: str
    normalized_keyword: str
    observed_date: date
    interest_value: float
    geo: str
    source_type: str
    imported_at: datetime


@dataclass(frozen=True)
class ObservationPoint:
    provider: str
    normalized_keyword: str
    observed_date: date
    interest_value: float


@dataclass(frozen=True)
class ValidationValues:
    keyword: str
    week_start: date
    week_end: date
    google_score: float | None
    naver_score: float | None
    combined_score: float | None
    provider_count: int
    coverage_score: float
    current_average: float
    previous_average: float
    growth_rate: float
    calculated_at: datetime


@dataclass(frozen=True)
class ProviderImportStatus:
    observations: int
    keywords: int
    last_imported_at: datetime | None


def upsert_observation(session: Session, values: ObservationValues) -> str:
    observation = session.scalar(
        select(SearchInterestObservation).where(
            SearchInterestObservation.provider == values.provider,
            SearchInterestObservation.normalized_keyword == values.normalized_keyword,
            SearchInterestObservation.observed_date == values.observed_date,
            SearchInterestObservation.geo == values.geo,
        )
    )
    if observation is None:
        session.add(SearchInterestObservation(**values.__dict__))
        session.flush()
        return "inserted"
    if (
        observation.interest_value == values.interest_value
        and observation.keyword == values.keyword
        and observation.source_type == values.source_type
    ):
        return "skipped"
    observation.keyword = values.keyword
    observation.interest_value = values.interest_value
    observation.source_type = values.source_type
    observation.imported_at = values.imported_at
    session.flush()
    return "updated"


def get_observation_points(
    session: Session,
    *,
    start_date: date,
    end_date: date,
) -> list[ObservationPoint]:
    rows = session.execute(
        select(
            SearchInterestObservation.provider,
            SearchInterestObservation.normalized_keyword,
            SearchInterestObservation.observed_date,
            func.avg(SearchInterestObservation.interest_value),
        )
        .where(
            SearchInterestObservation.provider.in_(("google_trends", "naver_datalab")),
            SearchInterestObservation.observed_date >= start_date,
            SearchInterestObservation.observed_date <= end_date,
        )
        .group_by(
            SearchInterestObservation.provider,
            SearchInterestObservation.normalized_keyword,
            SearchInterestObservation.observed_date,
        )
    ).all()
    return [
        ObservationPoint(
            provider=row[0],
            normalized_keyword=row[1],
            observed_date=row[2],
            interest_value=float(row[3]),
        )
        for row in rows
    ]


def get_weekly_trends_for_week(session: Session, week_start: date) -> list[WeeklyTrend]:
    return list(
        session.scalars(
            select(WeeklyTrend)
            .where(WeeklyTrend.week_start == week_start)
            .order_by(WeeklyTrend.keyword)
        ).all()
    )


def upsert_validation_result(session: Session, values: ValidationValues) -> SearchValidationResult:
    result = session.scalar(
        select(SearchValidationResult).where(
            SearchValidationResult.keyword == values.keyword,
            SearchValidationResult.week_start == values.week_start,
        )
    )
    if result is None:
        result = SearchValidationResult(keyword=values.keyword, week_start=values.week_start)
        session.add(result)
    for name, value in values.__dict__.items():
        setattr(result, name, value)
    return result


def get_validations_for_week(
    session: Session,
    *,
    week_start: date,
    keywords: list[str] | None = None,
) -> dict[str, SearchValidationResult]:
    if keywords is not None and not keywords:
        return {}
    query = select(SearchValidationResult).where(
        SearchValidationResult.week_start == week_start
    )
    if keywords is not None:
        query = query.where(SearchValidationResult.keyword.in_(keywords))
    return {
        result.keyword: result
        for result in session.scalars(query).all()
    }


def get_latest_validation(
    session: Session,
    normalized_keyword: str,
) -> SearchValidationResult | None:
    return session.scalar(
        select(SearchValidationResult)
        .where(SearchValidationResult.keyword == normalized_keyword)
        .order_by(SearchValidationResult.week_start.desc())
        .limit(1)
    )


def get_provider_import_status(session: Session, provider: str) -> ProviderImportStatus:
    row = session.execute(
        select(
            func.count(SearchInterestObservation.id),
            func.count(distinct(SearchInterestObservation.normalized_keyword)),
            func.max(SearchInterestObservation.imported_at),
        ).where(SearchInterestObservation.provider == provider)
    ).one()
    return ProviderImportStatus(
        observations=row[0] or 0,
        keywords=row[1] or 0,
        last_imported_at=row[2],
    )


def get_validation_coverage_counts(
    session: Session,
    *,
    week_start: date | None,
) -> tuple[int, int]:
    if week_start is None:
        return 0, 0
    total = session.scalar(
        select(func.count(WeeklyTrend.id)).where(WeeklyTrend.week_start == week_start)
    ) or 0
    validated = session.scalar(
        select(func.count(SearchValidationResult.id)).where(
            SearchValidationResult.week_start == week_start,
            SearchValidationResult.provider_count > 0,
        )
    ) or 0
    return validated, max(total - validated, 0)

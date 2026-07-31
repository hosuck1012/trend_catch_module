from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.repositories.search_interest_repository import (
    ObservationPoint,
    ValidationValues,
    get_observation_points,
    get_weekly_trends_for_week,
    upsert_validation_result,
)
from app.repositories.trend_repository import get_latest_week_range
from app.services.trend_rescoring_service import rescore_weekly_trend


SEARCH_PROVIDERS = ("google_trends", "naver_datalab")


class NoWeeklyTrendsError(ValueError):
    pass


@dataclass(frozen=True)
class ProviderScore:
    current_average: float
    previous_average: float
    growth_rate: float
    growth_score: float
    persistence_score: float
    provider_score: float


@dataclass(frozen=True)
class CombinedSearchScore:
    google_score: float | None
    naver_score: float | None
    combined_score: float | None
    provider_count: int
    coverage_score: float
    current_average: float
    previous_average: float
    growth_rate: float


@dataclass(frozen=True)
class SearchInterestRecalculationResult:
    week_start: date
    week_end: date
    recalculated_keywords: int
    validated_weekly_trends: int
    unvalidated_weekly_trends: int
    updated_weekly_trends: int


def recalculate_search_interest(session: Session) -> SearchInterestRecalculationResult:
    week_range = get_latest_week_range(session)
    if week_range is None:
        raise NoWeeklyTrendsError(
            "계산된 WeeklyTrend가 없습니다. POST /api/trends/recalculate를 먼저 실행하세요."
        )
    week_start, week_end = week_range
    previous_week_start = week_start - timedelta(days=7)
    trends = get_weekly_trends_for_week(session, week_start)
    points = get_observation_points(
        session,
        start_date=previous_week_start,
        end_date=week_end,
    )
    points_by_keyword_provider: dict[tuple[str, str], list[ObservationPoint]] = defaultdict(list)
    for point in points:
        points_by_keyword_provider[(point.normalized_keyword, point.provider)].append(point)

    calculated_at = datetime.now(timezone.utc).replace(tzinfo=None)
    validated = 0
    updated = 0
    for trend in trends:
        provider_scores = {
            provider: calculate_provider_score(
                points_by_keyword_provider.get((trend.keyword, provider), []),
                week_start=week_start,
                week_end=week_end,
            )
            for provider in SEARCH_PROVIDERS
        }
        available_scores = {
            provider: score
            for provider, score in provider_scores.items()
            if points_by_keyword_provider.get((trend.keyword, provider))
        }
        combined = combine_provider_scores(available_scores)
        upsert_validation_result(
            session,
            ValidationValues(
                keyword=trend.keyword,
                week_start=week_start,
                week_end=week_end,
                google_score=combined.google_score,
                naver_score=combined.naver_score,
                combined_score=combined.combined_score,
                provider_count=combined.provider_count,
                coverage_score=combined.coverage_score,
                current_average=combined.current_average,
                previous_average=combined.previous_average,
                growth_rate=combined.growth_rate,
                calculated_at=calculated_at,
            ),
        )
        if combined.provider_count > 0:
            validated += 1
        rescore_weekly_trend(
            trend, combined.combined_score, provider_count=combined.provider_count
        )
        updated += 1

    session.commit()
    return SearchInterestRecalculationResult(
        week_start=week_start,
        week_end=week_end,
        recalculated_keywords=len(trends),
        validated_weekly_trends=validated,
        unvalidated_weekly_trends=max(len(trends) - validated, 0),
        updated_weekly_trends=updated,
    )


def calculate_provider_score(
    points: list[ObservationPoint],
    *,
    week_start: date,
    week_end: date,
) -> ProviderScore:
    previous_week_start = week_start - timedelta(days=7)
    previous_week_end = week_start - timedelta(days=1)
    current_points = [
        point for point in points if week_start <= point.observed_date <= week_end
    ]
    previous_points = [
        point
        for point in points
        if previous_week_start <= point.observed_date <= previous_week_end
    ]
    current_average = _average(point.interest_value for point in current_points)
    previous_average = _average(point.interest_value for point in previous_points)
    growth_rate = calculate_search_growth_rate(current_average, previous_average)
    growth_score = growth_rate_to_score(growth_rate)
    observed_days = len({point.observed_date for point in current_points})
    persistence_score = _clamp(observed_days / 7 * 100)
    provider_score = _clamp(
        current_average * 0.40
        + growth_score * 0.40
        + persistence_score * 0.20
    )
    return ProviderScore(
        current_average=round(current_average, 2),
        previous_average=round(previous_average, 2),
        growth_rate=round(growth_rate, 4),
        growth_score=round(growth_score, 2),
        persistence_score=round(persistence_score, 2),
        provider_score=round(provider_score, 2),
    )


def combine_provider_scores(scores: dict[str, ProviderScore]) -> CombinedSearchScore:
    google = scores.get("google_trends")
    naver = scores.get("naver_datalab")
    present = [score for score in (google, naver) if score is not None]
    provider_count = len(present)
    if provider_count == 2:
        combined_score = google.provider_score * 0.5 + naver.provider_score * 0.5
    elif provider_count == 1:
        combined_score = present[0].provider_score
    else:
        combined_score = None
    current_average = _average(score.current_average for score in present)
    previous_average = _average(score.previous_average for score in present)
    growth_rate = calculate_search_growth_rate(current_average, previous_average)
    return CombinedSearchScore(
        google_score=google.provider_score if google else None,
        naver_score=naver.provider_score if naver else None,
        combined_score=round(_clamp(combined_score), 2) if combined_score is not None else None,
        provider_count=provider_count,
        coverage_score=float(provider_count * 50),
        current_average=round(current_average, 2),
        previous_average=round(previous_average, 2),
        growth_rate=round(growth_rate, 4),
    )


def calculate_search_growth_rate(current_average: float, previous_average: float) -> float:
    if previous_average == 0 and current_average > 0:
        return 1.0
    if previous_average == 0:
        return 0.0
    return (current_average - previous_average) / previous_average


def growth_rate_to_score(growth_rate: float) -> float:
    if growth_rate <= 0:
        return 0.0
    if growth_rate >= 1:
        return 100.0
    return growth_rate * 100


def _average(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))

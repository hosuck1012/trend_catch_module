from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

from sqlalchemy.orm import Session

from app.repositories.trend_repository import (
    OccurrenceRow,
    get_latest_occurrence_date,
    get_occurrences_between,
    upsert_weekly_trends,
)
from app.services.trend_scoring_service import KeywordMetricsInput, KeywordTrendScore, score_keyword


@dataclass(frozen=True)
class TrendCalculationResult:
    week_start: date
    week_end: date
    calculated_keywords: int
    weekly_trends: int
    watchlist: int
    stable: int
    insufficient_data: int
    scores: list[KeywordTrendScore]


class NoKeywordOccurrencesError(ValueError):
    pass


def recalculate_weekly_trends(session: Session) -> TrendCalculationResult:
    week_end = get_latest_occurrence_date(session)
    if week_end is None:
        raise NoKeywordOccurrencesError(
            "키워드 발생 데이터가 없습니다. 먼저 POST /api/collect/mock 및 POST /api/keywords/extract를 실행하세요."
        )

    week_start = week_end - timedelta(days=6)
    previous_week_start = week_start - timedelta(days=7)
    previous_week_end = week_start - timedelta(days=1)
    rows = get_occurrences_between(session, previous_week_start, week_end)
    scores = calculate_scores(
        rows=rows,
        week_start=week_start,
        week_end=week_end,
        previous_week_start=previous_week_start,
        previous_week_end=previous_week_end,
    )
    upsert_weekly_trends(
        session,
        week_start=week_start,
        week_end=week_end,
        scores=scores,
    )
    status_counts = Counter(score.status for score in scores)
    return TrendCalculationResult(
        week_start=week_start,
        week_end=week_end,
        calculated_keywords=len(scores),
        weekly_trends=status_counts.get("weekly_trend", 0),
        watchlist=status_counts.get("watchlist", 0),
        stable=status_counts.get("stable", 0),
        insufficient_data=status_counts.get("insufficient_data", 0),
        scores=scores,
    )


def calculate_scores(
    *,
    rows: list[OccurrenceRow],
    week_start: date,
    week_end: date,
    previous_week_start: date,
    previous_week_end: date,
) -> list[KeywordTrendScore]:
    metrics_by_keyword = _build_metrics(
        rows=rows,
        week_start=week_start,
        week_end=week_end,
        previous_week_start=previous_week_start,
        previous_week_end=previous_week_end,
    )
    max_weekly_mentions = max(
        (metrics.weekly_mentions for metrics in metrics_by_keyword.values()),
        default=0,
    )
    return [
        score_keyword(metrics, max_weekly_mentions)
        for metrics in sorted(metrics_by_keyword.values(), key=lambda item: item.keyword)
    ]


def _build_metrics(
    *,
    rows: list[OccurrenceRow],
    week_start: date,
    week_end: date,
    previous_week_start: date,
    previous_week_end: date,
) -> dict[str, KeywordMetricsInput]:
    weekly_counts: Counter[str] = Counter()
    previous_counts: Counter[str] = Counter()
    daily_counts: dict[str, Counter[date]] = defaultdict(Counter)
    sources: dict[str, set[str]] = defaultdict(set)
    quality_scores: dict[str, list[float]] = defaultdict(list)
    keywords = {row.keyword for row in rows}
    recent_start = week_end - timedelta(days=1)

    for row in rows:
        if week_start <= row.occurred_date <= week_end:
            weekly_counts[row.keyword] += 1
            daily_counts[row.keyword][row.occurred_date] += 1
            sources[row.keyword].add(row.source)
            if row.keyword_quality_score is not None:
                quality_scores[row.keyword].append(row.keyword_quality_score)
        elif previous_week_start <= row.occurred_date <= previous_week_end:
            previous_counts[row.keyword] += 1

    return {
        keyword: KeywordMetricsInput(
            keyword=keyword,
            weekly_mentions=weekly_counts[keyword],
            previous_weekly_mentions=previous_counts[keyword],
            active_days=len(daily_counts[keyword]),
            source_count=len(sources[keyword]),
            peak_day_mentions=max(daily_counts[keyword].values(), default=0),
            recent_mentions=sum(
                count
                for occurred_date, count in daily_counts[keyword].items()
                if recent_start <= occurred_date <= week_end
            ),
            keyword_quality_score=(
                sum(quality_scores[keyword]) / len(quality_scores[keyword])
                if quality_scores[keyword]
                else 0.0
            ),
        )
        for keyword in keywords
    }

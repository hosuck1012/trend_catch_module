from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.keyword_occurrence import KeywordOccurrence
from app.models.weekly_trend import WeeklyTrend
from app.services.trend_scoring_service import KeywordTrendScore


@dataclass(frozen=True)
class OccurrenceRow:
    keyword: str
    source: str
    occurred_date: date
    keyword_quality_score: float | None = None


def get_latest_occurrence_date(session: Session) -> date | None:
    latest = session.scalar(select(func.max(KeywordOccurrence.occurred_at)))
    return latest.date() if latest is not None else None


def get_occurrences_between(
    session: Session,
    start_date: date,
    end_date: date,
) -> list[OccurrenceRow]:
    rows = session.execute(
        select(
            KeywordOccurrence.normalized_keyword,
            KeywordOccurrence.source,
            func.date(KeywordOccurrence.occurred_at),
            KeywordOccurrence.keyword_quality_score,
        )
        .where(
            func.date(KeywordOccurrence.occurred_at) >= start_date.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= end_date.isoformat(),
        )
        .order_by(KeywordOccurrence.normalized_keyword, KeywordOccurrence.occurred_at)
    ).all()
    return [
        OccurrenceRow(
            keyword=row.normalized_keyword,
            source=row.source,
            occurred_date=date.fromisoformat(row[2]),
            keyword_quality_score=row[3],
        )
        for row in rows
    ]


def upsert_weekly_trends(
    session: Session,
    *,
    week_start: date,
    week_end: date,
    scores: list[KeywordTrendScore],
    commit: bool = True,
) -> None:
    calculated_at = datetime.now()
    existing_by_keyword = {
        trend.keyword: trend
        for trend in session.scalars(
            select(WeeklyTrend).where(WeeklyTrend.week_start == week_start)
        ).all()
    }

    for score in scores:
        trend = existing_by_keyword.get(score.keyword)
        if trend is None:
            trend = WeeklyTrend(keyword=score.keyword, week_start=week_start)
            session.add(trend)
        trend.week_end = week_end
        trend.weekly_mentions = score.weekly_mentions
        trend.previous_weekly_mentions = score.previous_weekly_mentions
        trend.active_days = score.active_days
        trend.source_count = score.source_count
        trend.growth_rate = score.growth_rate
        trend.peak_day_share = score.peak_day_share
        trend.persistence_score = score.persistence_score
        trend.diversity_score = score.diversity_score
        trend.freshness_score = score.freshness_score
        trend.volume_score = score.volume_score
        trend.growth_score = score.growth_score
        trend.trend_score = score.trend_score
        trend.keyword_quality_score = score.keyword_quality_score
        trend.search_interest_score = score.search_interest_score
        trend.search_interest_available = score.search_interest_available
        trend.search_provider_count = score.search_provider_count
        trend.one_day_spike_penalty = score.one_day_spike_penalty
        trend.spam_penalty = score.spam_penalty
        trend.final_score = score.final_score
        trend.status = score.status
        trend.calculated_at = calculated_at
        trend.pipeline_version = "v2" if score.keyword_quality_score > 0 else "legacy"

    if commit:
        session.commit()


def get_latest_week_range(session: Session) -> tuple[date, date] | None:
    row = session.execute(
        select(WeeklyTrend.week_start, WeeklyTrend.week_end)
        .order_by(WeeklyTrend.week_start.desc())
        .limit(1)
    ).first()
    if row is None:
        return None
    return row.week_start, row.week_end


def get_trends_by_status(
    session: Session,
    *,
    status: str,
    limit: int,
) -> list[WeeklyTrend]:
    week_range = get_latest_week_range(session)
    if week_range is None:
        return []
    week_start, _ = week_range
    order_by = (
        (
            WeeklyTrend.final_score.desc(),
            WeeklyTrend.weekly_mentions.desc(),
            WeeklyTrend.keyword.asc(),
        )
        if status == "weekly_trend"
        else (
            WeeklyTrend.weekly_mentions.desc(),
            WeeklyTrend.peak_day_share.desc(),
            WeeklyTrend.final_score.desc(),
        )
    )
    return list(
        session.scalars(
            select(WeeklyTrend)
            .where(WeeklyTrend.week_start == week_start, WeeklyTrend.status == status)
            .order_by(*order_by)
            .limit(limit)
        ).all()
    )


def get_status_counts(session: Session) -> dict[str, int]:
    week_range = get_latest_week_range(session)
    if week_range is None:
        return {}
    week_start, _ = week_range
    rows = session.execute(
        select(WeeklyTrend.status, func.count(WeeklyTrend.id))
        .where(WeeklyTrend.week_start == week_start)
        .group_by(WeeklyTrend.status)
    ).all()
    return {row.status: row[1] for row in rows}


def count_weekly_trends_for_week(session: Session, week_start: date) -> int:
    return session.scalar(
        select(func.count(WeeklyTrend.id)).where(WeeklyTrend.week_start == week_start)
    ) or 0

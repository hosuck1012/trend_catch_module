from datetime import date, timedelta

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.models.entity_context import EntityContext
from app.models.entity_mention import EntityMention
from app.models.keyword_candidate import KeywordCandidate
from app.models.keyword_context import KeywordContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend


SEMANTIC_RANKING_STATUSES = ("semantic_review", "semantic_strong")
SEMANTIC_ADMIN_STATUSES = (*SEMANTIC_RANKING_STATUSES, "semantic_weak")


def get_semantic_candidates(
    session: Session,
    *,
    week_start: date | None,
    limit: int,
    include_weak: bool = False,
) -> list[TravelOpportunityCandidate]:
    statuses = SEMANTIC_ADMIN_STATUSES if include_weak else SEMANTIC_RANKING_STATUSES
    query = (
        select(TravelOpportunityCandidate)
        .options(joinedload(TravelOpportunityCandidate.keyword_context))
        .where(TravelOpportunityCandidate.semantic_status.in_(statuses))
        .order_by(
            TravelOpportunityCandidate.semantic_travel_score.desc(),
            TravelOpportunityCandidate.travel_pre_score.desc(),
            TravelOpportunityCandidate.normalized_keyword.asc(),
        )
        .limit(limit)
    )
    if week_start is not None:
        query = query.where(TravelOpportunityCandidate.week_start == week_start)
    return list(session.scalars(query).unique().all())


def get_keyword_contexts(
    session: Session,
    *,
    keywords: list[str],
    week_start: date,
) -> list[KeywordContext]:
    if not keywords:
        return []
    week_end = week_start + timedelta(days=6)
    return list(
        session.scalars(
            select(KeywordContext)
            .options(joinedload(KeywordContext.document))
            .where(
                KeywordContext.normalized_keyword.in_(keywords),
                func.date(KeywordContext.published_at) >= week_start.isoformat(),
                func.date(KeywordContext.published_at) <= week_end.isoformat(),
            )
            .order_by(KeywordContext.published_at.desc(), KeywordContext.id.desc())
        ).unique().all()
    )


def get_weekly_trends(
    session: Session,
    *,
    keywords: list[str],
    week_start: date,
) -> dict[str, WeeklyTrend]:
    if not keywords:
        return {}
    rows = session.scalars(
        select(WeeklyTrend).where(
            WeeklyTrend.keyword.in_(keywords),
            WeeklyTrend.week_start == week_start,
        )
    ).all()
    return {row.keyword: row for row in rows}


def get_document_ids_by_keyword(
    session: Session,
    *,
    keywords: list[str],
    week_start: date,
) -> dict[str, set[int]]:
    if not keywords:
        return {}
    week_end = week_start + timedelta(days=6)
    rows = session.execute(
        select(KeywordOccurrence.normalized_keyword, KeywordOccurrence.document_id)
        .where(
            KeywordOccurrence.normalized_keyword.in_(keywords),
            func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
        )
        .distinct()
    ).all()
    result: dict[str, set[int]] = {}
    for keyword, document_id in rows:
        result.setdefault(keyword, set()).add(document_id)
    return result


def get_documents(session: Session, document_ids: set[int]) -> dict[int, SourceDocument]:
    if not document_ids:
        return {}
    rows = session.scalars(
        select(SourceDocument).where(SourceDocument.id.in_(document_ids))
    ).all()
    return {row.id: row for row in rows}


def get_entity_mentions(
    session: Session, document_ids: set[int]
) -> list[EntityMention]:
    if not document_ids:
        return []
    return list(
        session.scalars(
            select(EntityMention).where(EntityMention.document_id.in_(document_ids))
        ).all()
    )


def get_trend_entity_links(
    session: Session,
    *,
    keywords: list[str],
    week_start: date,
) -> list[TrendEntityLink]:
    if not keywords:
        return []
    return list(
        session.scalars(
            select(TrendEntityLink).where(
                TrendEntityLink.keyword.in_(keywords),
                TrendEntityLink.week_start == week_start,
            )
        ).all()
    )


def get_entity_contexts(
    session: Session,
    *,
    keywords: list[str],
    week_start: date,
) -> dict[str, list[EntityContext]]:
    if not keywords:
        return {}
    rows = session.execute(
        select(TrendContextLink.keyword, EntityContext)
        .join(EntityContext, EntityContext.id == TrendContextLink.entity_context_id)
        .where(
            TrendContextLink.keyword.in_(keywords),
            TrendContextLink.week_start == week_start,
        )
    ).all()
    result: dict[str, list[EntityContext]] = {}
    for keyword, context in rows:
        result.setdefault(keyword, []).append(context)
    return result


def save_rankings(
    session: Session,
    *,
    values_by_keyword: dict[str, dict[str, object]],
    week_start: date,
) -> None:
    rows = session.scalars(
        select(TravelOpportunityCandidate).where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.normalized_keyword.in_(values_by_keyword),
        )
    ).all()
    for row in rows:
        for name, value in values_by_keyword[row.normalized_keyword].items():
            setattr(row, name, value)
    session.commit()


def get_ranked_candidates(
    session: Session,
    *,
    week_start: date | None,
    limit: int | None = None,
) -> list[TravelOpportunityCandidate]:
    query = select(TravelOpportunityCandidate).where(
        TravelOpportunityCandidate.ranking_version.is_not(None)
    )
    if week_start is not None:
        query = query.where(TravelOpportunityCandidate.week_start == week_start)
    query = query.order_by(
        TravelOpportunityCandidate.week_start.desc(),
        TravelOpportunityCandidate.rank_in_week.asc(),
        TravelOpportunityCandidate.high_precision_score.desc(),
    )
    if limit is not None:
        query = query.limit(limit)
    return list(session.scalars(query).all())


def funnel_counts(session: Session, *, week_start: date | None) -> dict[str, int]:
    if week_start is None:
        return {
            "raw": 0,
            "quality": 0,
            "rule": 0,
            "semantic": 0,
            "high_precision": 0,
            "gemini_eligible": 0,
        }
    settings = get_settings()
    raw = session.scalar(
        select(func.count(WeeklyTrend.id)).where(WeeklyTrend.week_start == week_start)
    ) or 0
    quality = session.scalar(
        select(func.count(WeeklyTrend.id)).where(
            WeeklyTrend.week_start == week_start,
            WeeklyTrend.keyword_quality_score >= settings.keyword_min_quality_score,
        )
    ) or 0
    if raw == 0:
        week_end = week_start + timedelta(days=6)
        raw = session.scalar(
            select(func.count(distinct(KeywordOccurrence.normalized_keyword))).where(
                func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
            )
        ) or 0
    if quality == 0:
        quality = session.scalar(
            select(func.count(distinct(KeywordCandidate.normalized_candidate)))
            .join(SourceDocument, SourceDocument.id == KeywordCandidate.document_id)
            .where(
                KeywordCandidate.accepted.is_(True),
                func.date(SourceDocument.published_at) >= week_start.isoformat(),
                func.date(SourceDocument.published_at) <= (week_start + timedelta(days=6)).isoformat(),
            )
        ) or 0
    rule = session.scalar(
        select(func.count(distinct(TravelOpportunityCandidate.normalized_keyword))).where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.prefilter_status.in_(("review", "strong")),
        )
    ) or 0
    semantic = session.scalar(
        select(func.count(distinct(TravelOpportunityCandidate.normalized_keyword))).where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.semantic_status.in_(SEMANTIC_RANKING_STATUSES),
        )
    ) or 0
    high_precision = session.scalar(
        select(func.count(distinct(TravelOpportunityCandidate.normalized_keyword))).where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.ranking_status.in_(
                ("review", "gemini_candidate", "priority_candidate")
            ),
        )
    ) or 0
    gemini_eligible = session.scalar(
        select(func.count(distinct(TravelOpportunityCandidate.normalized_keyword))).where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.gemini_eligible.is_(True),
        )
    ) or 0
    return {
        "raw": raw,
        "quality": quality,
        "rule": rule,
        "semantic": semantic,
        "high_precision": high_precision,
        "gemini_eligible": gemini_eligible,
    }


def ranked_history(session: Session) -> list[tuple[date, str]]:
    return list(
        session.execute(
            select(
                TravelOpportunityCandidate.week_start,
                TravelOpportunityCandidate.ranking_status,
            )
            .where(TravelOpportunityCandidate.ranking_version.is_not(None))
            .group_by(
                TravelOpportunityCandidate.week_start,
                TravelOpportunityCandidate.normalized_keyword,
                TravelOpportunityCandidate.ranking_status,
            )
        ).all()
    )

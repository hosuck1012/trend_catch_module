from dataclasses import dataclass
from datetime import date, datetime, timedelta

from sqlalchemy import case, distinct, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.context.provider_registry import (
    CONTEXT_TRAVEL_PRIORITY,
    ENTITY_SEARCH_PRIORITY,
    WIKIPEDIA_PROVIDER,
)
from app.models.entity_context import EntityContext
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend


@dataclass(frozen=True)
class ContextTarget:
    keyword: str
    week_start: date
    week_end: date
    entity_text: str
    normalized_entity: str
    entity_type: str
    relation_score: float
    related_locations: tuple[str, ...]


@dataclass(frozen=True)
class EntityContextValues:
    normalized_entity: str
    entity_text: str
    entity_type: str
    provider: str
    page_id: str | None
    page_title: str
    page_url: str
    summary: str
    description: str | None
    match_score: float
    match_status: str
    source_language: str
    license_name: str | None
    attribution_text: str | None
    revision_id: str | None
    retrieved_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ContextStatusCounts:
    wikipedia_contexts: int
    namuwiki_manual_contexts: int
    matched: int
    ambiguous: int
    unmatched: int
    keywords_with_context: int
    keywords_without_context: int
    last_retrieved_at: datetime | None


def get_context_targets(
    session: Session,
    *,
    entity_type: str | None,
    limit: int,
) -> list[ContextTarget]:
    latest_week = session.scalar(select(func.max(WeeklyTrend.week_start)))
    if latest_week is None:
        return []
    query = (
        select(TrendEntityLink)
        .join(
            WeeklyTrend,
            (WeeklyTrend.keyword == TrendEntityLink.keyword)
            & (WeeklyTrend.week_start == TrendEntityLink.week_start),
        )
        .where(
            TrendEntityLink.week_start == latest_week,
            WeeklyTrend.status.in_(("weekly_trend", "watchlist")),
        )
        .order_by(
            case(
                ENTITY_SEARCH_PRIORITY,
                value=TrendEntityLink.entity_type,
                else_=99,
            ),
            TrendEntityLink.is_primary.desc(),
            TrendEntityLink.relation_score.desc(),
        )
    )
    if entity_type:
        query = query.where(TrendEntityLink.entity_type == entity_type)
    links = list(session.scalars(query).all())
    locations_by_keyword: dict[str, list[str]] = {}
    keywords = {link.keyword for link in links}
    if keywords:
        location_rows = session.execute(
            select(TrendEntityLink.keyword, TrendEntityLink.entity_text).where(
                TrendEntityLink.week_start == latest_week,
                TrendEntityLink.keyword.in_(keywords),
                TrendEntityLink.entity_type == "LOCATION",
            )
        ).all()
        for keyword, entity_text in location_rows:
            locations_by_keyword.setdefault(keyword, []).append(entity_text)
    return [
        ContextTarget(
            keyword=link.keyword,
            week_start=link.week_start,
            week_end=link.week_end,
            entity_text=link.entity_text,
            normalized_entity=link.normalized_entity,
            entity_type=link.entity_type,
            relation_score=link.relation_score,
            related_locations=tuple(locations_by_keyword.get(link.keyword, [])),
        )
        for link in links[:limit]
    ]


def get_context_target_by_entity(
    session: Session, normalized_entity: str
) -> ContextTarget | None:
    link = session.scalar(
        select(TrendEntityLink)
        .where(TrendEntityLink.normalized_entity == normalized_entity)
        .order_by(
            TrendEntityLink.week_start.desc(),
            TrendEntityLink.is_primary.desc(),
            TrendEntityLink.relation_score.desc(),
        )
        .limit(1)
    )
    if link is None:
        return None
    locations = session.scalars(
        select(TrendEntityLink.entity_text).where(
            TrendEntityLink.keyword == link.keyword,
            TrendEntityLink.week_start == link.week_start,
            TrendEntityLink.entity_type == "LOCATION",
        )
    ).all()
    return ContextTarget(
        keyword=link.keyword,
        week_start=link.week_start,
        week_end=link.week_end,
        entity_text=link.entity_text,
        normalized_entity=link.normalized_entity,
        entity_type=link.entity_type,
        relation_score=link.relation_score,
        related_locations=tuple(locations),
    )


def get_cached_context(
    session: Session,
    *,
    normalized_entity: str,
    entity_type: str,
    provider: str,
    now: datetime,
) -> EntityContext | None:
    cutoff = now - timedelta(hours=24)
    return session.scalar(
        select(EntityContext)
        .where(
            EntityContext.normalized_entity == normalized_entity,
            EntityContext.entity_type == entity_type,
            EntityContext.provider == provider,
            or_(
                EntityContext.match_status == "matched",
                EntityContext.retrieved_at >= cutoff,
            ),
        )
        .order_by(
            case((EntityContext.match_status == "matched", 0), else_=1),
            EntityContext.retrieved_at.desc(),
        )
        .limit(1)
    )


def upsert_entity_context(
    session: Session, values: EntityContextValues
) -> tuple[EntityContext, str]:
    context = session.scalar(
        select(EntityContext).where(
            EntityContext.normalized_entity == values.normalized_entity,
            EntityContext.entity_type == values.entity_type,
            EntityContext.provider == values.provider,
            EntityContext.page_url == values.page_url,
        )
    )
    action = "updated"
    if context is None:
        context = EntityContext(
            normalized_entity=values.normalized_entity,
            entity_type=values.entity_type,
            provider=values.provider,
            page_url=values.page_url,
        )
        session.add(context)
        action = "created"
    for name, value in values.__dict__.items():
        setattr(context, name, value)
    session.flush()
    return context, action


def upsert_trend_context_link(
    session: Session,
    *,
    target: ContextTarget,
    context: EntityContext,
    context_score: float,
    now: datetime,
) -> TrendContextLink:
    link = session.scalar(
        select(TrendContextLink).where(
            TrendContextLink.keyword == target.keyword,
            TrendContextLink.week_start == target.week_start,
            TrendContextLink.entity_context_id == context.id,
        )
    )
    if link is None:
        link = TrendContextLink(
            keyword=target.keyword,
            week_start=target.week_start,
            entity_context_id=context.id,
            created_at=now,
        )
        session.add(link)
    link.week_end = target.week_end
    link.normalized_entity = target.normalized_entity
    link.entity_type = target.entity_type
    link.context_score = round(min(max(context_score, 0.0), 100.0), 2)
    link.is_primary = False
    link.updated_at = now
    session.flush()
    return link


def recalculate_primary_context(
    session: Session, *, keyword: str, week_start: date
) -> None:
    links = list(
        session.scalars(
            select(TrendContextLink)
            .options(joinedload(TrendContextLink.entity_context))
            .where(
                TrendContextLink.keyword == keyword,
                TrendContextLink.week_start == week_start,
            )
        ).all()
    )
    for link in links:
        link.is_primary = False
    eligible = [
        link
        for link in links
        if link.entity_context.match_status == "manual"
        or (
            link.entity_context.provider == WIKIPEDIA_PROVIDER
            and link.entity_context.match_status == "matched"
        )
    ]
    if eligible:
        primary = max(
            eligible,
            key=lambda link: (
                link.context_score,
                -CONTEXT_TRAVEL_PRIORITY.get(link.entity_type, 99),
            ),
        )
        primary.is_primary = True
    session.flush()


def get_weekly_trend(
    session: Session, *, keyword: str, week_start: date
) -> WeeklyTrend | None:
    return session.scalar(
        select(WeeklyTrend).where(
            WeeklyTrend.keyword == keyword,
            WeeklyTrend.week_start == week_start,
        )
    )


def get_relation_score(
    session: Session,
    *,
    keyword: str,
    week_start: date,
    normalized_entity: str,
    entity_type: str,
) -> float:
    return session.scalar(
        select(TrendEntityLink.relation_score).where(
            TrendEntityLink.keyword == keyword,
            TrendEntityLink.week_start == week_start,
            TrendEntityLink.normalized_entity == normalized_entity,
            TrendEntityLink.entity_type == entity_type,
        )
    ) or 0.0


def get_context_by_id(session: Session, context_id: int) -> EntityContext | None:
    return session.get(EntityContext, context_id)


def get_links_for_context(session: Session, context_id: int) -> list[tuple[str, date]]:
    return list(
        session.execute(
            select(TrendContextLink.keyword, TrendContextLink.week_start).where(
                TrendContextLink.entity_context_id == context_id
            )
        ).all()
    )


def delete_manual_context(session: Session, context: EntityContext) -> None:
    session.delete(context)
    session.flush()


def get_latest_context_links_for_keyword(
    session: Session, keyword: str
) -> tuple[WeeklyTrend | None, list[TrendContextLink]]:
    trend = session.scalar(
        select(WeeklyTrend)
        .where(WeeklyTrend.keyword == keyword)
        .order_by(WeeklyTrend.week_start.desc())
        .limit(1)
    )
    if trend is None:
        return None, []
    type_order = case(
        CONTEXT_TRAVEL_PRIORITY,
        value=TrendContextLink.entity_type,
        else_=99,
    )
    links = list(
        session.scalars(
            select(TrendContextLink)
            .options(joinedload(TrendContextLink.entity_context))
            .where(
                TrendContextLink.keyword == keyword,
                TrendContextLink.week_start == trend.week_start,
            )
            .order_by(
                TrendContextLink.is_primary.desc(),
                TrendContextLink.context_score.desc(),
                type_order,
            )
        ).all()
    )
    return trend, links


def get_context_status_counts(session: Session) -> ContextStatusCounts:
    provider_rows = session.execute(
        select(EntityContext.provider, func.count(EntityContext.id)).group_by(
            EntityContext.provider
        )
    ).all()
    status_rows = session.execute(
        select(EntityContext.match_status, func.count(EntityContext.id)).group_by(
            EntityContext.match_status
        )
    ).all()
    providers = dict(provider_rows)
    statuses = dict(status_rows)
    latest_week = session.scalar(select(func.max(WeeklyTrend.week_start)))
    if latest_week is None:
        total_keywords = 0
        with_context = 0
    else:
        total_keywords = session.scalar(
            select(func.count(WeeklyTrend.id)).where(
                WeeklyTrend.week_start == latest_week,
                WeeklyTrend.status.in_(("weekly_trend", "watchlist")),
            )
        ) or 0
        with_context = session.scalar(
            select(func.count(distinct(TrendContextLink.keyword))).where(
                TrendContextLink.week_start == latest_week
            )
        ) or 0
    return ContextStatusCounts(
        wikipedia_contexts=providers.get(WIKIPEDIA_PROVIDER, 0),
        namuwiki_manual_contexts=providers.get("namuwiki_manual", 0),
        matched=statuses.get("matched", 0),
        ambiguous=statuses.get("ambiguous", 0),
        unmatched=statuses.get("unmatched", 0),
        keywords_with_context=with_context,
        keywords_without_context=max(total_keywords - with_context, 0),
        last_retrieved_at=session.scalar(select(func.max(EntityContext.retrieved_at))),
    )


def get_context_summary(
    session: Session,
    *,
    provider: str | None,
    entity_type: str | None,
    match_status: str | None,
    limit: int,
) -> list[EntityContext]:
    query = select(EntityContext)
    if provider:
        query = query.where(EntityContext.provider == provider)
    if entity_type:
        query = query.where(EntityContext.entity_type == entity_type)
    if match_status:
        query = query.where(EntityContext.match_status == match_status)
    return list(
        session.scalars(
            query.order_by(EntityContext.updated_at.desc(), EntityContext.id.desc()).limit(limit)
        ).all()
    )


def get_primary_context_metadata_for_keyword(
    session: Session, *, keyword: str, week_start: date
) -> EntityContext | None:
    return session.scalar(
        select(EntityContext)
        .join(TrendContextLink, TrendContextLink.entity_context_id == EntityContext.id)
        .where(
            TrendContextLink.keyword == keyword,
            TrendContextLink.week_start == week_start,
            TrendContextLink.is_primary.is_(True),
        )
        .limit(1)
    )


def get_context_metadata_for_week(
    session: Session, *, week_start: date, keywords: list[str]
) -> dict[str, tuple[str | None, str | None, bool]]:
    if not keywords:
        return {}
    rows = session.execute(
        select(
            TrendContextLink.keyword,
            EntityContext.page_title,
            EntityContext.provider,
        )
        .join(EntityContext, EntityContext.id == TrendContextLink.entity_context_id)
        .where(
            TrendContextLink.week_start == week_start,
            TrendContextLink.keyword.in_(keywords),
            TrendContextLink.is_primary.is_(True),
        )
    ).all()
    result = {row.keyword: (row.page_title, row.provider, True) for row in rows}
    for keyword in keywords:
        result.setdefault(keyword, (None, None, False))
    return result

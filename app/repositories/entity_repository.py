from dataclasses import dataclass
from datetime import date, datetime

from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import Session

from app.models.entity_mention import EntityMention
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.ner.entity_labels import EntityCandidate


@dataclass(frozen=True)
class DocumentSnapshot:
    id: int
    title: str
    text: str
    source: str
    occurred_at: datetime
    has_entities: bool


@dataclass(frozen=True)
class EntitySummaryRow:
    text: str
    canonical_text: str
    entity_type: str
    mentions: int
    document_count: int
    source_count: int
    average_confidence: float


def get_recent_document_snapshots(
    session: Session,
    *,
    since: datetime,
    source: str | None,
    limit: int,
) -> list[DocumentSnapshot]:
    mention_exists = (
        select(EntityMention.id)
        .where(EntityMention.document_id == SourceDocument.id)
        .exists()
    )
    statement = (
        select(SourceDocument, mention_exists.label("has_entities"))
        .where(SourceDocument.published_at >= since)
        .order_by(SourceDocument.published_at.desc(), SourceDocument.id.desc())
        .limit(limit)
    )
    if source:
        statement = statement.where(SourceDocument.source == source)
    rows = session.execute(statement).all()
    return [
        DocumentSnapshot(
            id=document.id,
            title=document.title or "",
            text=document.text or "",
            source=document.source,
            occurred_at=document.published_at,
            has_entities=bool(has_entities),
        )
        for document, has_entities in rows
    ]


def replace_document_entities(
    session: Session,
    *,
    document: DocumentSnapshot,
    candidates: list[EntityCandidate],
    force: bool,
) -> int:
    if force:
        session.execute(
            delete(EntityMention).where(EntityMention.document_id == document.id)
        )
    else:
        existing_keys = {
            (
                row.normalized_text,
                row.entity_type,
                row.start_char,
                row.end_char,
            )
            for row in session.scalars(
                select(EntityMention).where(EntityMention.document_id == document.id)
            ).all()
        }
        if existing_keys:
            return 0

    created_at = datetime.now()
    inserted = 0
    seen: set[tuple[str, str, int | None, int | None]] = set()
    for candidate in candidates:
        if candidate.normalized_text is None:
            continue
        key = (
            candidate.normalized_text,
            candidate.entity_type.value,
            candidate.start_char,
            candidate.end_char,
        )
        if key in seen:
            continue
        seen.add(key)
        session.add(
            EntityMention(
                document_id=document.id,
                text=candidate.text,
                normalized_text=candidate.normalized_text,
                entity_type=candidate.entity_type.value,
                confidence=round(candidate.confidence, 4),
                extractor=candidate.extractor,
                start_char=candidate.start_char,
                end_char=candidate.end_char,
                source=document.source,
                occurred_at=document.occurred_at,
                created_at=created_at,
            )
        )
        inserted += 1
    session.commit()
    return inserted


def get_entity_summary(
    session: Session,
    *,
    entity_type: str | None,
    source: str | None,
    limit: int,
) -> tuple[int, int, list[EntitySummaryRow]]:
    filters = []
    if entity_type:
        filters.append(EntityMention.entity_type == entity_type)
    if source:
        filters.append(EntityMention.source == source)

    total = session.scalar(
        select(func.count(EntityMention.id)).where(*filters)
    ) or 0
    unique_entities = session.scalar(
        select(
            func.count(
                distinct(EntityMention.normalized_text + "|" + EntityMention.entity_type)
            )
        ).where(*filters)
    ) or 0
    rows = session.execute(
        select(
            func.min(EntityMention.text).label("text"),
            EntityMention.normalized_text,
            EntityMention.entity_type,
            func.count(EntityMention.id).label("mentions"),
            func.count(distinct(EntityMention.document_id)).label("document_count"),
            func.count(distinct(EntityMention.source)).label("source_count"),
            func.avg(EntityMention.confidence).label("average_confidence"),
        )
        .where(*filters)
        .group_by(EntityMention.normalized_text, EntityMention.entity_type)
        .order_by(
            func.count(EntityMention.id).desc(),
            func.count(distinct(EntityMention.source)).desc(),
            func.avg(EntityMention.confidence).desc(),
        )
        .limit(limit)
    ).all()
    return (
        total,
        unique_entities,
        [
            EntitySummaryRow(
                text=row.text,
                canonical_text=row.normalized_text,
                entity_type=row.entity_type,
                mentions=row.mentions,
                document_count=row.document_count,
                source_count=row.source_count,
                average_confidence=round(float(row.average_confidence), 4),
            )
            for row in rows
        ],
    )


def get_latest_linkable_trends(session: Session) -> list[WeeklyTrend]:
    latest_week = session.scalar(select(func.max(WeeklyTrend.week_start)))
    if latest_week is None:
        return []
    return list(
        session.scalars(
            select(WeeklyTrend).where(
                WeeklyTrend.week_start == latest_week,
                WeeklyTrend.status.in_(("weekly_trend", "watchlist")),
            )
        ).all()
    )


def get_keyword_document_ids(
    session: Session,
    *,
    keyword: str,
    week_start: date,
    week_end: date,
) -> list[int]:
    return list(
        session.scalars(
            select(distinct(KeywordOccurrence.document_id)).where(
                KeywordOccurrence.normalized_keyword == keyword,
                func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
            )
        ).all()
    )


def get_mentions_for_documents(
    session: Session, document_ids: list[int]
) -> list[EntityMention]:
    if not document_ids:
        return []
    return list(
        session.scalars(
            select(EntityMention).where(EntityMention.document_id.in_(document_ids))
        ).all()
    )


def replace_trend_entity_links(
    session: Session,
    *,
    keyword: str,
    week_start: date,
    links: list[dict[str, object]],
) -> None:
    session.execute(
        delete(TrendEntityLink).where(
            TrendEntityLink.keyword == keyword,
            TrendEntityLink.week_start == week_start,
        )
    )
    for values in links:
        session.add(TrendEntityLink(**values))
    session.commit()


def get_latest_links_for_keyword(
    session: Session, keyword: str
) -> tuple[WeeklyTrend | None, list[TrendEntityLink]]:
    trend = session.scalar(
        select(WeeklyTrend)
        .where(WeeklyTrend.keyword == keyword)
        .order_by(WeeklyTrend.week_start.desc())
        .limit(1)
    )
    if trend is None:
        return None, []
    links = list(
        session.scalars(
            select(TrendEntityLink)
            .where(
                TrendEntityLink.keyword == keyword,
                TrendEntityLink.week_start == trend.week_start,
            )
            .order_by(
                TrendEntityLink.is_primary.desc(),
                TrendEntityLink.relation_score.desc(),
            )
        ).all()
    )
    return trend, links


def get_link_metadata_for_week(
    session: Session, *, week_start: date, keywords: list[str]
) -> dict[str, tuple[str | None, str | None, int]]:
    if not keywords:
        return {}
    links = session.scalars(
        select(TrendEntityLink).where(
            TrendEntityLink.week_start == week_start,
            TrendEntityLink.keyword.in_(keywords),
        )
    ).all()
    grouped: dict[str, list[TrendEntityLink]] = {}
    for link in links:
        grouped.setdefault(link.keyword, []).append(link)
    result: dict[str, tuple[str | None, str | None, int]] = {}
    for keyword, items in grouped.items():
        primary = next((item for item in items if item.is_primary), None)
        travel_count = sum(
            item.entity_type in {"LOCATION", "PLACE"} for item in items
        )
        result[keyword] = (
            primary.entity_text if primary else None,
            primary.entity_type if primary else None,
            travel_count,
        )
    return result

from datetime import date, datetime, timedelta, timezone
import json

from sqlalchemy import and_, delete, distinct, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.models.entity_mention import EntityMention
from app.models.entity_context import EntityContext
from app.models.keyword_candidate import KeywordCandidate
from app.models.keyword_context import KeywordContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_entity_link import TrendEntityLink
from app.models.trend_context_link import TrendContextLink
from app.models.weekly_trend import WeeklyTrend


SEMANTIC_PREFILTER_STATUSES = ("weak", "review", "strong")
RELATED_DESTINATION_PAGE_ID_PREFIX = "travel-destination:"


def get_related_destination_contexts(
    session: Session,
    *,
    normalized_keywords: list[str],
    week_start: date,
) -> dict[str, list[EntityContext]]:
    if not normalized_keywords:
        return {}
    rows = session.execute(
        select(TrendContextLink.keyword, EntityContext)
        .join(EntityContext, EntityContext.id == TrendContextLink.entity_context_id)
        .where(
            TrendContextLink.week_start == week_start,
            TrendContextLink.keyword.in_(normalized_keywords),
            EntityContext.page_id.like(f"{RELATED_DESTINATION_PAGE_ID_PREFIX}%"),
            EntityContext.match_status.in_(("matched", "manual")),
        )
        .order_by(
            TrendContextLink.context_score.desc(),
            EntityContext.entity_text.asc(),
        )
    ).all()
    result: dict[str, list[EntityContext]] = {}
    for keyword, context in rows:
        result.setdefault(keyword, []).append(context)
    return result


def resolve_week_range(session: Session, week_start: date | None) -> tuple[date | None, date | None]:
    if week_start:
        return week_start, week_start + timedelta(days=6)
    row = session.execute(
        select(WeeklyTrend.week_start, WeeklyTrend.week_end)
        .order_by(WeeklyTrend.week_start.desc())
        .limit(1)
    ).first()
    if row:
        return row.week_start, row.week_end
    latest = session.scalar(select(func.max(SourceDocument.published_at)))
    if latest is None:
        return None, None
    end = latest.date()
    return end - timedelta(days=6), end


def count_raw_keywords(session: Session, *, week_start: date | None) -> int:
    if week_start:
        trend_count = session.scalar(
            select(func.count(WeeklyTrend.id)).where(WeeklyTrend.week_start == week_start)
        ) or 0
        if trend_count:
            return trend_count
    return session.scalar(select(func.count(distinct(KeywordOccurrence.normalized_keyword)))) or 0


def get_quality_keywords(
    session: Session,
    *,
    week_start: date | None,
    week_end: date | None,
    limit: int | None,
) -> list[str]:
    settings = get_settings()
    if week_start:
        statement = (
            select(WeeklyTrend.keyword)
            .where(
                WeeklyTrend.week_start == week_start,
                WeeklyTrend.keyword_quality_score >= settings.keyword_min_quality_score,
            )
            .order_by(
                WeeklyTrend.final_score.desc(),
                WeeklyTrend.keyword_quality_score.desc(),
                WeeklyTrend.source_count.desc(),
                WeeklyTrend.keyword.asc(),
            )
        )
        if limit is not None:
            statement = statement.limit(limit)
        rows = session.scalars(statement).all()
        if rows:
            return list(rows)
    filters = [KeywordCandidate.accepted.is_(True)]
    if week_start and week_end:
        filters.extend(
            [
                func.date(SourceDocument.published_at) >= week_start.isoformat(),
                func.date(SourceDocument.published_at) <= week_end.isoformat(),
            ]
        )
    statement = (
        select(KeywordCandidate.normalized_candidate)
        .join(SourceDocument, SourceDocument.id == KeywordCandidate.document_id)
        .where(*filters)
        .distinct()
        .order_by(KeywordCandidate.quality_score.desc(), KeywordCandidate.normalized_candidate.asc())
    )
    if limit is not None:
        statement = statement.limit(limit)
    rows = session.scalars(statement).all()
    if rows:
        return list(rows)
    occurrence_filters = []
    if week_start and week_end:
        occurrence_filters.extend(
            [
                func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
            ]
        )
    statement = (
        select(KeywordOccurrence.normalized_keyword)
        .where(*occurrence_filters)
        .distinct()
        .order_by(KeywordOccurrence.normalized_keyword.asc())
    )
    if limit is not None:
        statement = statement.limit(limit)
    return list(session.scalars(statement).all())


def get_keyword_occurrence_documents(
    session: Session,
    *,
    normalized_keywords: list[str],
    week_start: date | None,
    week_end: date | None,
    limit: int,
) -> list[tuple[KeywordOccurrence, SourceDocument]]:
    if not normalized_keywords:
        return []
    conditions = [KeywordOccurrence.normalized_keyword.in_(normalized_keywords)]
    if week_start and week_end:
        conditions.extend(
            [
                func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
            ]
        )
    return list(
        session.execute(
            select(KeywordOccurrence, SourceDocument)
            .join(SourceDocument, SourceDocument.id == KeywordOccurrence.document_id)
            .where(*conditions)
            .order_by(KeywordOccurrence.occurred_at.desc(), KeywordOccurrence.id.desc())
            .limit(limit)
        ).all()
    )


def get_keyword_occurrence_documents_page(
    session: Session,
    *,
    normalized_keywords: list[str],
    week_start: date | None,
    week_end: date | None,
    after_id: int | None,
    limit: int,
) -> tuple[list[tuple[KeywordOccurrence, SourceDocument]], int | None, bool]:
    if not normalized_keywords:
        return [], None, False
    conditions = [KeywordOccurrence.normalized_keyword.in_(normalized_keywords)]
    if week_start and week_end:
        conditions.extend(
            [
                func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
            ]
        )
    if after_id is not None:
        conditions.append(KeywordOccurrence.id > after_id)
    rows = list(
        session.execute(
            select(KeywordOccurrence, SourceDocument)
            .join(SourceDocument, SourceDocument.id == KeywordOccurrence.document_id)
            .where(*conditions)
            .order_by(KeywordOccurrence.id.asc())
            .limit(limit + 1)
        ).all()
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = page[-1][0].id if has_more and page else None
    return page, next_cursor, has_more


def existing_context_keys(
    session: Session,
    *,
    triples: list[tuple[int, str, str]],
) -> set[tuple[int, str, str]]:
    if not triples:
        return set()
    requested = set(triples)
    document_ids = sorted({document_id for document_id, _keyword, _hash in requested})
    keywords = sorted({keyword for _document_id, keyword, _hash in requested})
    hashes = sorted({context_hash for _document_id, _keyword, context_hash in requested})
    rows = session.execute(
        select(
            KeywordContext.document_id,
            KeywordContext.normalized_keyword,
            KeywordContext.context_hash,
        ).where(
            KeywordContext.document_id.in_(document_ids),
            KeywordContext.normalized_keyword.in_(keywords),
            KeywordContext.context_hash.in_(hashes),
        )
    ).all()
    return {
        (row.document_id, row.normalized_keyword, row.context_hash)
        for row in rows
        if (row.document_id, row.normalized_keyword, row.context_hash) in requested
    }

def add_keyword_contexts(session: Session, rows: list[dict[str, object]]) -> None:
    session.add_all(KeywordContext(**row) for row in rows)
    session.flush()


def get_keyword_contexts_in_range(
    session: Session,
    *,
    week_start: date,
    week_end: date,
) -> list[KeywordContext]:
    return list(
        session.scalars(
            select(KeywordContext).where(
                func.date(KeywordContext.published_at) >= week_start.isoformat(),
                func.date(KeywordContext.published_at) <= week_end.isoformat(),
            )
        ).all()
    )


def delete_keyword_contexts(session: Session, *, context_ids: list[int]) -> int:
    if not context_ids:
        return 0
    result = session.execute(
        delete(KeywordContext).where(KeywordContext.id.in_(context_ids))
    )
    return max(result.rowcount or 0, 0)


def get_keyword_contexts_for_week(
    session: Session,
    *,
    week_start: date | None,
    week_end: date | None,
    limit: int,
) -> list[KeywordContext]:
    conditions = []
    if week_start and week_end:
        conditions.extend(
            [
                func.date(KeywordContext.published_at) >= week_start.isoformat(),
                func.date(KeywordContext.published_at) <= week_end.isoformat(),
            ]
        )
    return list(
        session.scalars(
            select(KeywordContext)
            .where(*conditions)
            .order_by(KeywordContext.published_at.desc(), KeywordContext.id.desc())
            .limit(limit)
        ).all()
    )


def get_keyword_contexts_page(
    session: Session,
    *,
    week_start: date | None,
    week_end: date | None,
    candidate_week_start: date | None,
    after_id: int | None,
    limit: int,
    force: bool,
) -> tuple[list[KeywordContext], int | None, bool]:
    conditions = []
    if week_start and week_end:
        conditions.extend(
            [
                func.date(KeywordContext.published_at) >= week_start.isoformat(),
                func.date(KeywordContext.published_at) <= week_end.isoformat(),
            ]
        )
    if after_id is not None:
        conditions.append(KeywordContext.id > after_id)
    rows = list(
        session.scalars(
            select(KeywordContext)
            .where(*conditions)
            .order_by(KeywordContext.id.asc())
            .limit(limit + 1)
        ).all()
    )
    has_more = len(rows) > limit
    page = rows[:limit]
    next_cursor = page[-1].id if has_more and page else None
    return page, next_cursor, has_more


def count_rule_materialization_coverage(
    session: Session,
    *,
    week_start: date | None,
    week_end: date | None,
    candidate_week_start: date | None,
) -> tuple[int, int, int]:
    conditions = []
    if week_start and week_end:
        conditions.extend(
            [
                func.date(KeywordContext.published_at) >= week_start.isoformat(),
                func.date(KeywordContext.published_at) <= week_end.isoformat(),
            ]
        )
    eligible = session.scalar(select(func.count(KeywordContext.id)).where(*conditions)) or 0
    if candidate_week_start is None:
        return eligible, 0, eligible
    materialized = session.scalar(
        select(func.count(distinct(TravelOpportunityCandidate.keyword_context_id)))
        .join(KeywordContext, KeywordContext.id == TravelOpportunityCandidate.keyword_context_id)
        .where(
            TravelOpportunityCandidate.week_start == candidate_week_start,
            *conditions,
        )
    ) or 0
    return eligible, materialized, max(eligible - materialized, 0)


def get_existing_candidate_context_ids(
    session: Session,
    *,
    week_start: date,
    context_ids: list[int],
) -> set[int]:
    if not context_ids:
        return set()
    return set(
        session.scalars(
            select(TravelOpportunityCandidate.keyword_context_id).where(
                TravelOpportunityCandidate.week_start == week_start,
                TravelOpportunityCandidate.keyword_context_id.in_(context_ids),
            )
        ).all()
    )


def get_existing_candidates_by_context(
    session: Session,
    *,
    week_start: date,
    context_ids: list[int],
) -> dict[int, TravelOpportunityCandidate]:
    if not context_ids:
        return {}
    rows = session.scalars(
        select(TravelOpportunityCandidate).where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.keyword_context_id.in_(context_ids),
        )
    ).all()
    return {row.keyword_context_id: row for row in rows}


def get_trend_by_keyword(session: Session, *, keyword: str, week_start: date | None) -> WeeklyTrend | None:
    statement = select(WeeklyTrend).where(WeeklyTrend.keyword == keyword)
    if week_start:
        statement = statement.where(WeeklyTrend.week_start == week_start)
    return session.scalar(statement.order_by(WeeklyTrend.week_start.desc()).limit(1))


def get_entities_for_context(
    session: Session,
    *,
    context: KeywordContext,
    week_start: date | None,
) -> tuple[list[TrendEntityLink], list[EntityMention]]:
    trend_links = []
    if week_start:
        trend_links = list(
            session.scalars(
                select(TrendEntityLink)
                .where(
                    TrendEntityLink.keyword == context.normalized_keyword,
                    TrendEntityLink.week_start == week_start,
                )
                .order_by(TrendEntityLink.is_primary.desc(), TrendEntityLink.relation_score.desc())
            ).all()
        )
    mentions = list(
        session.scalars(
            select(EntityMention)
            .where(EntityMention.document_id == context.document_id)
            .order_by(EntityMention.confidence.desc())
        ).all()
    )
    return trend_links, mentions


def upsert_travel_candidates(
    session: Session,
    *,
    week_start: date,
    rows: list[dict[str, object]],
    force: bool,
) -> tuple[int, int]:
    context_ids = [int(row["keyword_context_id"]) for row in rows]
    existing = {
        (row.normalized_keyword, row.week_start, row.keyword_context_id): row
        for row in session.scalars(
            select(TravelOpportunityCandidate).where(
                TravelOpportunityCandidate.week_start == week_start,
                TravelOpportunityCandidate.keyword_context_id.in_(context_ids),
            )
        ).all()
    } if context_ids else {}
    created = 0
    updated = 0
    for values in rows:
        key = (
            values["normalized_keyword"],
            values["week_start"],
            values["keyword_context_id"],
        )
        existing_row = existing.get(key)
        if existing_row is None:
            session.add(TravelOpportunityCandidate(**values))
            created += 1
            continue
        updated += 1
        if existing_row.semantic_status is not None:
            existing_row.semantic_status = "stale"
        if existing_row.ranking_status is not None:
            existing_row.ranking_status = "stale"
        if existing_row.gemini_eligible:
            existing_row.gemini_eligible = False
        for name, value in values.items():
            if name not in {
                "keyword_context_id",
                "week_start",
                "normalized_keyword",
                "created_at",
            }:
                setattr(existing_row, name, value)
    session.commit()
    return created, updated


def get_candidates(
    session: Session,
    *,
    week_start: date | None,
    status: str | None,
    min_score: float | None,
    travel_category: str | None,
    limit: int,
    semantic_status: str | None = None,
    ranking_status: str | None = None,
) -> list[TravelOpportunityCandidate]:
    conditions = []
    if week_start:
        conditions.append(TravelOpportunityCandidate.week_start == week_start)
    if status:
        conditions.append(TravelOpportunityCandidate.prefilter_status == status)
    if min_score is not None:
        conditions.append(TravelOpportunityCandidate.travel_pre_score >= min_score)
    if travel_category:
        conditions.append(TravelOpportunityCandidate.travel_category == travel_category)
    if semantic_status:
        conditions.append(TravelOpportunityCandidate.semantic_status == semantic_status)
    if ranking_status:
        conditions.append(TravelOpportunityCandidate.ranking_status == ranking_status)
    return list(
        session.scalars(
            select(TravelOpportunityCandidate)
            .options(joinedload(TravelOpportunityCandidate.keyword_context))
            .where(*conditions)
            .order_by(
                TravelOpportunityCandidate.high_precision_score.desc(),
                TravelOpportunityCandidate.travel_pre_score.desc(),
                TravelOpportunityCandidate.normalized_keyword.asc(),
            )
            .limit(limit)
        ).all()
    )


def get_candidates_for_keyword(session: Session, normalized_keyword: str) -> list[TravelOpportunityCandidate]:
    return list(
        session.scalars(
            select(TravelOpportunityCandidate)
            .options(joinedload(TravelOpportunityCandidate.keyword_context))
            .where(TravelOpportunityCandidate.normalized_keyword == normalized_keyword)
            .order_by(
                TravelOpportunityCandidate.week_start.desc(),
                TravelOpportunityCandidate.travel_pre_score.desc(),
            )
        ).all()
    )


def get_semantic_filter_candidates(
    session: Session,
    *,
    week_start: date,
    limit: int,
) -> list[TravelOpportunityCandidate]:
    return list(
        session.scalars(
            select(TravelOpportunityCandidate)
            .options(joinedload(TravelOpportunityCandidate.keyword_context))
            .where(
                TravelOpportunityCandidate.week_start == week_start,
                TravelOpportunityCandidate.prefilter_status.in_(SEMANTIC_PREFILTER_STATUSES),
            )
            .order_by(
                TravelOpportunityCandidate.travel_pre_score.desc(),
                TravelOpportunityCandidate.normalized_keyword.asc(),
                TravelOpportunityCandidate.id.asc(),
            )
            .limit(limit)
        ).unique().all()
    )


def get_semantic_keyword_quality_signals(
    session: Session,
    *,
    candidates: list[TravelOpportunityCandidate],
) -> dict[int, KeywordCandidate]:
    if not candidates:
        return {}
    document_ids = {
        candidate.keyword_context.document_id for candidate in candidates
    }
    normalized_keywords = {candidate.normalized_keyword for candidate in candidates}
    quality_rows = session.scalars(
        select(KeywordCandidate)
        .where(
            KeywordCandidate.document_id.in_(document_ids),
            KeywordCandidate.normalized_candidate.in_(normalized_keywords),
            KeywordCandidate.accepted.is_(True),
            KeywordCandidate.pipeline_version == get_settings().keyword_pipeline_version,
        )
        .order_by(KeywordCandidate.quality_score.desc(), KeywordCandidate.id.desc())
    ).all()
    best_by_key: dict[tuple[int, str], KeywordCandidate] = {}
    for row in quality_rows:
        best_by_key.setdefault(
            (row.document_id, row.normalized_candidate),
            row,
        )
    return {
        candidate.id: signal
        for candidate in candidates
        if (
            signal := best_by_key.get(
                (
                    candidate.keyword_context.document_id,
                    candidate.normalized_keyword,
                )
            )
        )
        is not None
    }


def get_semantic_context_entities(
    session: Session,
    *,
    candidates: list[TravelOpportunityCandidate],
) -> dict[int, list[EntityMention]]:
    if not candidates:
        return {}
    document_ids = {
        candidate.keyword_context.document_id for candidate in candidates
    }
    mentions = session.scalars(
        select(EntityMention)
        .where(EntityMention.document_id.in_(document_ids))
        .order_by(EntityMention.confidence.desc(), EntityMention.id.asc())
    ).all()
    by_document: dict[int, list[EntityMention]] = {
        document_id: [] for document_id in document_ids
    }
    for mention in mentions:
        by_document[mention.document_id].append(mention)
    return {
        candidate.id: by_document[candidate.keyword_context.document_id]
        for candidate in candidates
    }


def save_semantic_results(
    session: Session,
    *,
    values_by_id: dict[int, dict[str, object]],
) -> None:
    if not values_by_id:
        return
    semantic_rows = session.scalars(
        select(TravelOpportunityCandidate).where(
            TravelOpportunityCandidate.id.in_(values_by_id)
        )
    ).all()
    affected_keys = {
        (row.week_start, row.normalized_keyword) for row in semantic_rows
    }
    affected_rows = session.scalars(
        select(TravelOpportunityCandidate).where(
            or_(
                *(
                    and_(
                        TravelOpportunityCandidate.week_start == affected_week_start,
                        TravelOpportunityCandidate.normalized_keyword == normalized_keyword,
                    )
                    for affected_week_start, normalized_keyword in affected_keys
                )
            )
        )
    ).all()
    downstream_fields = (
        "trend_strength_score",
        "context_clarity_score",
        "travel_convertibility_score",
        "evidence_confidence_score",
        "high_precision_score",
        "evidence_gate",
        "evidence_codes_json",
        "evidence_document_count",
        "evidence_source_count",
        "ranking_status",
        "rank_in_week",
        "ranking_version",
        "calculated_at",
        "cluster_id",
        "cluster_representative",
        "gemini_eligible",
    )
    for row in affected_rows:
        for field_name in downstream_fields:
            setattr(row, field_name, None)
    for row in semantic_rows:
        for name, value in values_by_id[row.id].items():
            setattr(row, name, value)
    session.commit()


def count_keyword_candidate_funnel(
    session: Session,
    *,
    week_start: date | None,
    week_end: date | None,
) -> tuple[int, int, int]:
    week_filters = []
    if week_start and week_end:
        document_ids = select(SourceDocument.id).where(
            SourceDocument.published_at
            >= datetime.combine(week_start, datetime.min.time()),
            SourceDocument.published_at
            < datetime.combine(week_end + timedelta(days=1), datetime.min.time()),
        )
        week_filters.append(KeywordCandidate.document_id.in_(document_ids))

    total = session.scalar(
        select(func.count(KeywordCandidate.id)).where(*week_filters)
    ) or 0
    accepted_rows = session.scalar(
        select(func.count(KeywordCandidate.id)).where(
            KeywordCandidate.accepted.is_(True),
            *week_filters,
        )
    ) or 0
    distinct_accepted = session.scalar(
        select(func.count(distinct(KeywordCandidate.normalized_candidate))).where(
            KeywordCandidate.accepted.is_(True),
            *week_filters,
        )
    ) or 0
    return total, accepted_rows, distinct_accepted


def summarize_v2(session: Session, *, week_start: date | None) -> dict[str, object]:
    resolved_start, resolved_end = resolve_week_range(session, week_start)

    raw = count_raw_keywords(session, week_start=resolved_start)
    raw_occurrences = (
        session.scalar(
            select(func.count(KeywordOccurrence.id)).where(
                KeywordOccurrence.occurred_at
                >= datetime.combine(resolved_start, datetime.min.time()),
                KeywordOccurrence.occurred_at
                < datetime.combine(resolved_end + timedelta(days=1), datetime.min.time()),
            )
        )
        if resolved_start and resolved_end
        else session.scalar(select(func.count(KeywordOccurrence.id))) or 0
    )

    candidate_total, candidate_accepted_rows, distinct_accepted_keywords = (
        count_keyword_candidate_funnel(
            session,
            week_start=resolved_start,
            week_end=resolved_end,
        )
    )

    weekly_trend_count = session.scalar(
        select(func.count(WeeklyTrend.id)).where(
            WeeklyTrend.week_start == resolved_start
        )
    ) if resolved_start else session.scalar(
        select(func.count(WeeklyTrend.id))
    ) or 0

    context_count = 0
    if resolved_start:
        context_count = session.scalar(
            select(func.count(KeywordContext.id)).where(
                func.date(KeywordContext.published_at) >= resolved_start.isoformat(),
                func.date(KeywordContext.published_at) <= (resolved_start + timedelta(days=6)).isoformat(),
            )
        ) or 0
    status_rows = session.execute(
        select(TravelOpportunityCandidate.prefilter_status, func.count(TravelOpportunityCandidate.id))
        .where(TravelOpportunityCandidate.week_start == resolved_start if resolved_start else True)
        .group_by(TravelOpportunityCandidate.prefilter_status)
    ).all()
    counts = {row[0]: row[1] for row in status_rows}
    review = counts.get("review", 0)
    strong = counts.get("strong", 0)
    estimated_calls = strong
    reduction = _reduction_rate(raw, estimated_calls)
    return {
        "week_start": resolved_start,
        "raw_keyword_count": raw,
        "raw_keyword_occurrences": raw_occurrences,
        "keyword_candidate_total": candidate_total,
        "keyword_candidate_accepted_rows": candidate_accepted_rows,
        "distinct_accepted_keywords": distinct_accepted_keywords,
        "weekly_trend_count": weekly_trend_count,
        "quality_keyword_count": distinct_accepted_keywords,
        "context_candidate_count": context_count,
        "travel_prefilter_count": review + strong,
        "strong_candidate_count": strong,
        "estimated_gemini_calls": estimated_calls,
        "llm_reduction_rate": reduction,
        "status_counts": {
            "rejected": counts.get("rejected", 0),
            "weak": counts.get("weak", 0),
            "review": review,
            "strong": strong,
        },
    }


def encode_json(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _reduction_rate(raw: int, estimated_calls: int) -> float:
    if raw <= 0:
        return 0.0
    return round((1 - estimated_calls / raw) * 100, 2)

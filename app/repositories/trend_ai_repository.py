from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json

from sqlalchemy import distinct, exists, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.models.entity_context import EntityContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.keyword_candidate import KeywordCandidate
from app.models.source_document import SourceDocument
from app.models.trend_ai_analysis import TrendAIAnalysis
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.config import get_settings
from app.keywords.stopword_filter import load_word_set
from app.keywords.keyword_quality import KEYWORD_HIGH_CONFIDENCE_SCORE


COMPLETED_STATUSES = ("completed", "partial")


@dataclass(frozen=True)
class AIStatusCounts:
    completed_count: int
    partial_count: int
    error_count: int
    cached_count: int
    last_generated_at: datetime | None


def get_analysis_targets(
    session: Session,
    *,
    normalized_keyword: str | None,
    week_start: date | None,
    limit: int,
) -> list[WeeklyTrend]:
    target_week = week_start or session.scalar(select(func.max(WeeklyTrend.week_start)))
    if target_week is None:
        return []
    settings = get_settings()
    blocked = set(load_word_set("stopwords_ko.txt"))
    blocked.update(load_word_set("stopwords_en.txt"))
    blocked.update(load_word_set("url_artifacts.txt"))
    blocked.update(load_word_set("generic_travel_words.txt"))
    document_count = (
        select(func.count(distinct(KeywordOccurrence.document_id)))
        .where(
            KeywordOccurrence.normalized_keyword == WeeklyTrend.keyword,
            func.date(KeywordOccurrence.occurred_at) >= WeeklyTrend.week_start,
            func.date(KeywordOccurrence.occurred_at) <= WeeklyTrend.week_end,
        )
        .correlate(WeeklyTrend)
        .scalar_subquery()
    )
    query = select(WeeklyTrend).where(
        WeeklyTrend.week_start == target_week,
        WeeklyTrend.status.in_(("weekly_trend", "watchlist")),
        WeeklyTrend.keyword_quality_score >= settings.keyword_min_quality_score,
        or_(
            WeeklyTrend.keyword_quality_score >= KEYWORD_HIGH_CONFIDENCE_SCORE,
            WeeklyTrend.search_interest_available.is_(True),
            exists(
                select(1).where(
                    KeywordCandidate.normalized_candidate == WeeklyTrend.keyword,
                    KeywordCandidate.accepted.is_(True),
                    KeywordCandidate.extractor.in_(
                        ("protected_phrase", "ner", "title_phrase", "hashtag")
                    ),
                )
            ),
        ),
        WeeklyTrend.trend_score.is_not(None),
        WeeklyTrend.final_score.is_not(None),
        ~WeeklyTrend.keyword.in_(blocked),
        func.length(WeeklyTrend.keyword) > 1,
        ~WeeklyTrend.keyword.op("GLOB")("[0-9]*"),
        exists(
            select(1).where(
                KeywordOccurrence.normalized_keyword == WeeklyTrend.keyword,
                func.date(KeywordOccurrence.occurred_at) >= WeeklyTrend.week_start,
                func.date(KeywordOccurrence.occurred_at) <= WeeklyTrend.week_end,
            )
        ),
    )
    if normalized_keyword:
        query = query.where(WeeklyTrend.keyword == normalized_keyword)
    return list(
        session.scalars(
            query.order_by(
                WeeklyTrend.final_score.desc(),
                WeeklyTrend.keyword_quality_score.desc(),
                WeeklyTrend.source_count.desc(),
                document_count.desc(),
                WeeklyTrend.keyword.asc(),
            ).limit(limit)
        ).all()
    )


def get_candidate_documents(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date,
    week_end: date,
    limit: int,
) -> list[SourceDocument]:
    return list(
        session.scalars(
            select(SourceDocument)
            .join(
                KeywordOccurrence,
                KeywordOccurrence.document_id == SourceDocument.id,
            )
            .where(
                KeywordOccurrence.normalized_keyword == normalized_keyword,
                func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
            )
            .distinct()
            .order_by(SourceDocument.published_at.desc(), SourceDocument.id.desc())
            .limit(limit)
        ).all()
    )


def count_keyword_documents(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date,
    week_end: date,
) -> int:
    return session.scalar(
        select(func.count(distinct(KeywordOccurrence.document_id))).where(
            KeywordOccurrence.normalized_keyword == normalized_keyword,
            func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
        )
    ) or 0


def get_entity_links(
    session: Session,
    *,
    keyword: str,
    week_start: date,
) -> list[TrendEntityLink]:
    return list(
        session.scalars(
            select(TrendEntityLink)
            .where(
                TrendEntityLink.keyword == keyword,
                TrendEntityLink.week_start == week_start,
            )
            .order_by(
                TrendEntityLink.is_primary.desc(),
                TrendEntityLink.relation_score.desc(),
            )
        ).all()
    )


def get_eligible_context_links(
    session: Session,
    *,
    keyword: str,
    week_start: date,
) -> list[TrendContextLink]:
    return list(
        session.scalars(
            select(TrendContextLink)
            .options(joinedload(TrendContextLink.entity_context))
            .join(EntityContext, EntityContext.id == TrendContextLink.entity_context_id)
            .where(
                TrendContextLink.keyword == keyword,
                TrendContextLink.week_start == week_start,
                EntityContext.match_status.in_(("matched", "manual")),
            )
            .order_by(
                TrendContextLink.is_primary.desc(),
                TrendContextLink.context_score.desc(),
            )
        ).unique().all()
    )


def get_analysis(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date,
    model_name: str,
    prompt_version: str,
) -> TrendAIAnalysis | None:
    return session.scalar(
        select(TrendAIAnalysis).where(
            TrendAIAnalysis.normalized_keyword == normalized_keyword,
            TrendAIAnalysis.week_start == week_start,
            TrendAIAnalysis.model_name == model_name,
            TrendAIAnalysis.prompt_version == prompt_version,
        )
    )


def get_cached_analysis(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date,
    model_name: str,
    prompt_version: str,
    input_hash: str,
    now: datetime,
    cache_hours: int,
) -> TrendAIAnalysis | None:
    cutoff = now - timedelta(hours=max(cache_hours, 0))
    return session.scalar(
        select(TrendAIAnalysis).where(
            TrendAIAnalysis.normalized_keyword == normalized_keyword,
            TrendAIAnalysis.week_start == week_start,
            TrendAIAnalysis.model_name == model_name,
            TrendAIAnalysis.prompt_version == prompt_version,
            TrendAIAnalysis.input_hash == input_hash,
            TrendAIAnalysis.analysis_status.in_(COMPLETED_STATUSES),
            TrendAIAnalysis.generated_at >= cutoff,
        )
    )


def upsert_pending_analysis(
    session: Session,
    *,
    trend: WeeklyTrend,
    normalized_keyword: str,
    model_name: str,
    prompt_version: str,
    input_hash: str,
    now: datetime,
) -> TrendAIAnalysis:
    row = get_analysis(
        session,
        normalized_keyword=normalized_keyword,
        week_start=trend.week_start,
        model_name=model_name,
        prompt_version=prompt_version,
    )
    if row is None:
        row = TrendAIAnalysis(
            keyword=trend.keyword,
            normalized_keyword=normalized_keyword,
            week_start=trend.week_start,
            model_name=model_name,
            prompt_version=prompt_version,
            input_hash=input_hash,
            generated_at=now,
            updated_at=now,
            analysis_status="pending",
        )
        session.add(row)
    row.keyword = trend.keyword
    row.week_end = trend.week_end
    row.input_hash = input_hash
    row.analysis_status = "pending"
    row.trend_summary = None
    row.rising_reason = None
    row.evidence_summary = None
    row.travel_relevance_score = None
    row.travel_relevance_level = None
    row.travel_relevance_reason = None
    row.recommended_destinations_json = None
    row.content_ideas_json = None
    row.cautions_json = None
    row.evidence_refs_json = None
    row.confidence_score = None
    row.raw_response_json = None
    row.error_code = None
    row.error_message = None
    row.generated_at = now
    row.updated_at = now
    session.flush()
    return row


def complete_analysis(
    row: TrendAIAnalysis,
    *,
    analysis_status: str,
    explanation,
    raw_response: dict[str, object],
    now: datetime,
) -> None:
    row.analysis_status = analysis_status
    row.trend_summary = explanation.trend_summary
    row.rising_reason = explanation.rising_reason
    row.evidence_summary = _json_dump(explanation.evidence_summary)
    row.travel_relevance_score = explanation.travel_relevance_score
    row.travel_relevance_level = explanation.travel_relevance_level
    row.travel_relevance_reason = explanation.travel_relevance_reason
    row.recommended_destinations_json = _json_dump(
        [item.model_dump(mode="json") for item in explanation.recommended_destinations]
    )
    row.content_ideas_json = _json_dump(
        [item.model_dump(mode="json") for item in explanation.content_ideas]
    )
    row.cautions_json = _json_dump(explanation.cautions)
    row.evidence_refs_json = _json_dump(explanation.evidence_refs)
    row.confidence_score = explanation.confidence_score
    row.raw_response_json = _json_dump(raw_response)
    row.error_code = None
    row.error_message = None
    row.generated_at = now
    row.updated_at = now


def fail_analysis(
    row: TrendAIAnalysis,
    *,
    error_code: str,
    error_message: str,
    now: datetime,
) -> None:
    row.analysis_status = "error"
    row.error_code = error_code[:100]
    row.error_message = error_message[:1000]
    row.raw_response_json = None
    row.updated_at = now


def get_latest_analysis_for_keyword(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date | None,
) -> TrendAIAnalysis | None:
    query = select(TrendAIAnalysis).where(
        TrendAIAnalysis.normalized_keyword == normalized_keyword,
    )
    if week_start:
        query = query.where(TrendAIAnalysis.week_start == week_start)
    return session.scalar(
        query.order_by(
            TrendAIAnalysis.week_start.desc(),
            TrendAIAnalysis.updated_at.desc(),
        ).limit(1)
    )


def list_analyses(
    session: Session,
    *,
    week_start: date | None,
    status: str | None,
    travel_level: str | None,
    min_travel_score: float | None,
    limit: int,
) -> list[TrendAIAnalysis]:
    query = select(TrendAIAnalysis)
    if week_start:
        query = query.where(TrendAIAnalysis.week_start == week_start)
    if status:
        query = query.where(TrendAIAnalysis.analysis_status == status)
    if travel_level:
        query = query.where(TrendAIAnalysis.travel_relevance_level == travel_level)
    if min_travel_score is not None:
        query = query.where(TrendAIAnalysis.travel_relevance_score >= min_travel_score)
    return list(
        session.scalars(
            query.order_by(
                TrendAIAnalysis.week_start.desc(),
                TrendAIAnalysis.travel_relevance_score.desc(),
                TrendAIAnalysis.updated_at.desc(),
            ).limit(limit)
        ).all()
    )


def get_ai_status_counts(
    session: Session,
    *,
    now: datetime,
    cache_hours: int,
) -> AIStatusCounts:
    rows = dict(
        session.execute(
            select(TrendAIAnalysis.analysis_status, func.count(TrendAIAnalysis.id))
            .group_by(TrendAIAnalysis.analysis_status)
        ).all()
    )
    cutoff = now - timedelta(hours=max(cache_hours, 0))
    cached_count = session.scalar(
        select(func.count(TrendAIAnalysis.id)).where(
            TrendAIAnalysis.analysis_status.in_(COMPLETED_STATUSES),
            TrendAIAnalysis.generated_at >= cutoff,
        )
    ) or 0
    return AIStatusCounts(
        completed_count=rows.get("completed", 0),
        partial_count=rows.get("partial", 0),
        error_count=rows.get("error", 0),
        cached_count=cached_count,
        last_generated_at=session.scalar(
            select(func.max(TrendAIAnalysis.generated_at)).where(
                TrendAIAnalysis.analysis_status.in_(COMPLETED_STATUSES)
            )
        ),
    )


def get_ai_metadata_for_week(
    session: Session,
    *,
    week_start: date,
    keywords: list[str],
) -> dict[str, tuple[bool, str | None, float | None, str | None, int]]:
    if not keywords:
        return {}
    rows = session.scalars(
        select(TrendAIAnalysis)
        .where(
            TrendAIAnalysis.week_start == week_start,
            TrendAIAnalysis.keyword.in_(keywords),
            TrendAIAnalysis.analysis_status.in_(COMPLETED_STATUSES),
        )
        .order_by(TrendAIAnalysis.updated_at.desc())
    ).all()
    result = {}
    for row in rows:
        destination_count = len(_json_list(row.recommended_destinations_json))
        result.setdefault(row.keyword, (
            True,
            row.trend_summary,
            row.travel_relevance_score,
            row.travel_relevance_level,
            destination_count,
        ))
    return result


def get_source_contexts_for_analysis(
    session: Session,
    analysis: TrendAIAnalysis,
) -> list[EntityContext]:
    return list(
        session.scalars(
            select(EntityContext)
            .join(TrendContextLink, TrendContextLink.entity_context_id == EntityContext.id)
            .where(
                TrendContextLink.keyword == analysis.keyword,
                TrendContextLink.week_start == analysis.week_start,
                EntityContext.match_status.in_(("matched", "manual")),
            )
            .order_by(TrendContextLink.is_primary.desc(), TrendContextLink.context_score.desc())
        ).all()
    )


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

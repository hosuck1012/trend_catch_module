from datetime import date, datetime, timedelta, timezone
import json

from sqlalchemy import and_, distinct, exists, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.config import get_settings
from app.context.context_normalizer import clean_plain_text
from app.models.entity_context import EntityContext
from app.models.entity_mention import EntityMention
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.keyword_candidate import KeywordCandidate
from app.models.search_interest_observation import SearchInterestObservation
from app.models.source_document import SourceDocument
from app.models.trend_ai_analysis import TrendAIAnalysis
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.repositories.trend_ai_repository import get_ai_status_counts
from app.services.keyword_normalization_service import normalize_keyword
from app.keywords.stopword_filter import load_word_set, rejection_reason
from app.keywords.keyword_quality import KEYWORD_HIGH_CONFIDENCE_SCORE, is_numeric_artifact
from app.keywords.keyword_normalizer import canonical_display


ENTITY_TYPES = (
    "LOCATION",
    "PLACE",
    "EVENT",
    "FOOD",
    "PERSON",
    "CONTENT_TITLE",
    "BRAND",
    "MEME",
)
AI_COMPLETE_STATUSES = ("completed", "partial")


class DashboardWeekNotFoundError(LookupError):
    pass


class DashboardTrendNotFoundError(LookupError):
    pass


def get_dashboard_overview(
    session: Session,
    *,
    week_start: date | None,
) -> dict[str, object]:
    weeks = get_available_weeks(session)
    selected_week = _resolve_week(weeks, week_start)
    if selected_week is None:
        return {
            "selected_week": None,
            "available_weeks": [],
            "metric_cards": [],
            "pipeline_status": _pipeline_status(session, None),
            "top_trends": [],
            "source_distribution": [],
            "entity_distribution": [],
            "ai_distribution": [],
            "keyword_pipeline_version": get_settings().keyword_pipeline_version,
            "requested_week": week_start,
            "week_fallback_used": False,
        }
    trend_result = get_dashboard_trends(
        session,
        week_start=selected_week,
        query=None,
        source=None,
        watchlist_only=False,
        include_low_quality=False,
        min_final_score=None,
        min_travel_score=None,
        travel_level=None,
        ai_status=None,
        limit=10,
        offset=0,
    )
    return {
        "selected_week": selected_week,
        "available_weeks": weeks,
        "metric_cards": _metric_cards(session, selected_week),
        "pipeline_status": _pipeline_status(session, selected_week),
        "top_trends": trend_result["items"],
        "source_distribution": _source_distribution(session, selected_week),
        "entity_distribution": _entity_distribution(session, selected_week),
        "ai_distribution": _ai_distribution(session, selected_week),
        "keyword_pipeline_version": get_settings().keyword_pipeline_version,
        "requested_week": week_start,
        "week_fallback_used": week_start is not None and week_start != selected_week,
    }


def get_dashboard_trends(
    session: Session,
    *,
    week_start: date | None,
    query: str | None,
    source: str | None,
    watchlist_only: bool,
    include_low_quality: bool,
    min_final_score: float | None,
    min_travel_score: float | None,
    travel_level: str | None,
    ai_status: str | None,
    limit: int,
    offset: int,
) -> dict[str, object]:
    weeks = get_available_weeks(session)
    selected_week = _resolve_week(weeks, week_start)
    if selected_week is None:
        return {
            "selected_week": None,
            "requested_week": week_start,
            "week_fallback_used": False,
            "total": 0,
            "limit": limit,
            "offset": offset,
            "items": [],
        }

    statement = select(WeeklyTrend).where(
        WeeklyTrend.week_start == selected_week,
        WeeklyTrend.trend_score.is_not(None),
        WeeklyTrend.final_score.is_not(None),
    )
    if not include_low_quality:
        settings = get_settings()
        blocked = set(load_word_set("stopwords_ko.txt"))
        blocked.update(load_word_set("stopwords_en.txt"))
        blocked.update(load_word_set("url_artifacts.txt"))
        blocked.update(load_word_set("generic_travel_words.txt"))
        statement = statement.where(
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
            WeeklyTrend.status.in_(("weekly_trend", "watchlist", "stable")),
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
    if query and query.strip():
        raw_query = query.strip()
        normalized_query = normalize_keyword(raw_query)
        conditions = [WeeklyTrend.keyword.contains(raw_query)]
        if normalized_query and normalized_query != raw_query:
            conditions.append(WeeklyTrend.keyword.contains(normalized_query))
        statement = statement.where(or_(*conditions))
    if source:
        source_exists = exists(
            select(1)
            .select_from(KeywordOccurrence)
            .where(
                KeywordOccurrence.normalized_keyword == WeeklyTrend.keyword,
                KeywordOccurrence.source == source,
                func.date(KeywordOccurrence.occurred_at) >= WeeklyTrend.week_start,
                func.date(KeywordOccurrence.occurred_at) <= WeeklyTrend.week_end,
            )
        )
        statement = statement.where(source_exists)
    if watchlist_only:
        statement = statement.where(WeeklyTrend.status == "watchlist")
    if min_final_score is not None:
        statement = statement.where(WeeklyTrend.final_score >= min_final_score)
    if min_travel_score is not None or travel_level or ai_status:
        ai_conditions = [
            TrendAIAnalysis.normalized_keyword == WeeklyTrend.keyword,
            TrendAIAnalysis.week_start == WeeklyTrend.week_start,
        ]
        if min_travel_score is not None:
            ai_conditions.append(TrendAIAnalysis.travel_relevance_score >= min_travel_score)
        if travel_level:
            ai_conditions.append(TrendAIAnalysis.travel_relevance_level == travel_level)
        if ai_status == "not_analyzed":
            statement = statement.where(~exists(select(1).where(and_(*ai_conditions[:2]))))
        else:
            if ai_status:
                ai_conditions.append(TrendAIAnalysis.analysis_status == ai_status)
            statement = statement.where(exists(select(1).where(and_(*ai_conditions))))

    total = session.scalar(select(func.count()).select_from(statement.subquery())) or 0
    trends = list(
        session.scalars(
            statement.order_by(
                WeeklyTrend.final_score.desc(),
                WeeklyTrend.keyword_quality_score.desc(),
                WeeklyTrend.source_count.desc(),
                WeeklyTrend.weekly_mentions.desc(),
                WeeklyTrend.keyword.asc(),
            )
            .offset(offset)
            .limit(limit)
        ).all()
    )
    items = _trend_items(session, trends, selected_week, offset=offset)
    return {
        "selected_week": selected_week,
        "requested_week": week_start,
        "week_fallback_used": week_start is not None and week_start != selected_week,
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }


def get_dashboard_trend_detail(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date | None,
) -> dict[str, object]:
    normalized = normalize_keyword(normalized_keyword)
    if not normalized:
        raise DashboardTrendNotFoundError("트렌드 키워드를 찾을 수 없습니다.")
    statement = select(WeeklyTrend).where(WeeklyTrend.keyword == normalized)
    if week_start:
        statement = statement.where(WeeklyTrend.week_start == week_start)
    trend = session.scalar(statement.order_by(WeeklyTrend.week_start.desc()).limit(1))
    if trend is None:
        raise DashboardTrendNotFoundError("트렌드 키워드를 찾을 수 없습니다.")
    trend_item = _trend_items(session, [trend], trend.week_start, offset=0)[0]
    entities = list(
        session.scalars(
            select(TrendEntityLink)
            .where(
                TrendEntityLink.keyword == trend.keyword,
                TrendEntityLink.week_start == trend.week_start,
            )
            .order_by(TrendEntityLink.is_primary.desc(), TrendEntityLink.relation_score.desc())
        ).all()
    )
    contexts = list(
        session.scalars(
            select(TrendContextLink)
            .options(joinedload(TrendContextLink.entity_context))
            .where(
                TrendContextLink.keyword == trend.keyword,
                TrendContextLink.week_start == trend.week_start,
            )
            .order_by(TrendContextLink.is_primary.desc(), TrendContextLink.context_score.desc())
        ).unique().all()
    )
    ai_row = _latest_ai_rows(session, trend.week_start, [trend.keyword]).get(trend.keyword)
    documents = list(
        session.scalars(
            select(SourceDocument)
            .join(KeywordOccurrence, KeywordOccurrence.document_id == SourceDocument.id)
            .where(
                KeywordOccurrence.normalized_keyword == trend.keyword,
                func.date(KeywordOccurrence.occurred_at) >= trend.week_start.isoformat(),
                func.date(KeywordOccurrence.occurred_at) <= trend.week_end.isoformat(),
            )
            .distinct()
            .order_by(SourceDocument.published_at.desc())
            .limit(8)
        ).all()
    )
    return {
        "trend": trend_item,
        "entities": [
            {
                "entity_text": entity.entity_text,
                "normalized_entity": entity.normalized_entity,
                "entity_type": entity.entity_type,
                "relation_score": entity.relation_score,
                "mention_count": entity.mention_count,
                "source_count": entity.source_count,
                "is_primary": entity.is_primary,
            }
            for entity in entities
        ],
        "contexts": [
            {
                "provider": link.entity_context.provider,
                "page_title": link.entity_context.page_title,
                "page_url": link.entity_context.page_url,
                "summary": clean_plain_text(link.entity_context.summary, max_chars=1000),
                "match_status": link.entity_context.match_status,
                "context_score": link.context_score,
                "is_primary": link.is_primary,
            }
            for link in contexts
        ],
        "ai_analysis": _serialize_ai(ai_row) if ai_row else None,
        "documents": [
            {
                "id": document.id,
                "source": document.source,
                "title": document.title,
                "published_at": document.published_at,
                "url": document.url,
                "snippet": clean_plain_text(document.text, max_chars=320),
            }
            for document in documents
        ],
    }


def get_available_weeks(session: Session) -> list[date]:
    return list(
        session.scalars(
            select(distinct(WeeklyTrend.week_start)).order_by(WeeklyTrend.week_start.desc())
        ).all()
    )


def _resolve_week(weeks: list[date], requested: date | None) -> date | None:
    if not weeks:
        return None
    if requested is None:
        return weeks[0]
    if requested not in weeks:
        return weeks[0]
    return requested


def _trend_items(
    session: Session,
    trends: list[WeeklyTrend],
    week_start: date,
    *,
    offset: int,
) -> list[dict[str, object]]:
    if not trends:
        return []
    keywords = [trend.keyword for trend in trends]
    ai_rows = _latest_ai_rows(session, week_start, keywords)
    entity_rows = session.execute(
        select(TrendEntityLink.keyword, TrendEntityLink.entity_text)
        .where(
            TrendEntityLink.week_start == week_start,
            TrendEntityLink.keyword.in_(keywords),
            TrendEntityLink.is_primary.is_(True),
        )
    ).all()
    primary_entities = {row.keyword: row.entity_text for row in entity_rows}
    context_rows = session.execute(
        select(TrendContextLink.keyword, EntityContext.page_title)
        .join(EntityContext, EntityContext.id == TrendContextLink.entity_context_id)
        .where(
            TrendContextLink.week_start == week_start,
            TrendContextLink.keyword.in_(keywords),
            TrendContextLink.is_primary.is_(True),
        )
    ).all()
    primary_contexts = {row.keyword: row.page_title for row in context_rows}
    document_rows = session.execute(
        select(
            KeywordOccurrence.normalized_keyword,
            func.count(distinct(KeywordOccurrence.document_id)),
        )
        .where(
            KeywordOccurrence.normalized_keyword.in_(keywords),
            func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= trends[0].week_end.isoformat(),
        )
        .group_by(KeywordOccurrence.normalized_keyword)
    ).all()
    document_counts = {row[0]: row[1] for row in document_rows}
    display_rows = session.execute(
        select(
            KeywordOccurrence.normalized_keyword,
            KeywordOccurrence.keyword,
            func.count(KeywordOccurrence.id),
        )
        .where(
            KeywordOccurrence.normalized_keyword.in_(keywords),
            func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= trends[0].week_end.isoformat(),
        )
        .group_by(KeywordOccurrence.normalized_keyword, KeywordOccurrence.keyword)
    ).all()
    display_values: dict[str, list[str]] = {}
    for normalized, display, count in display_rows:
        display_values.setdefault(normalized, []).extend([display] * int(count))
    source_rows = session.execute(
        select(KeywordOccurrence.normalized_keyword, KeywordOccurrence.source)
        .where(
            KeywordOccurrence.normalized_keyword.in_(keywords),
            func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= trends[0].week_end.isoformat(),
        )
        .distinct()
        .order_by(KeywordOccurrence.source)
    ).all()
    sources: dict[str, list[str]] = {}
    for keyword, source in source_rows:
        sources.setdefault(keyword, []).append(source)
    return [
        _trend_item(
            trend,
            rank=offset + index,
            display_keyword=canonical_display(display_values.get(trend.keyword, [trend.keyword])),
            document_count=document_counts.get(trend.keyword, 0),
            sources=sources.get(trend.keyword, []),
            primary_entity=primary_entities.get(trend.keyword),
            primary_context=primary_contexts.get(trend.keyword),
            ai_row=ai_rows.get(trend.keyword),
        )
        for index, trend in enumerate(trends, start=1)
    ]


def _trend_item(
    trend: WeeklyTrend,
    *,
    rank: int,
    display_keyword: str,
    document_count: int,
    sources: list[str],
    primary_entity: str | None,
    primary_context: str | None,
    ai_row: TrendAIAnalysis | None,
) -> dict[str, object]:
    return {
        "rank": rank,
        "keyword": display_keyword,
        "normalized_keyword": trend.keyword,
        "week_start": trend.week_start,
        "week_end": trend.week_end,
        "final_score": trend.final_score if trend.trend_score is not None else None,
        "trend_score": trend.trend_score,
        "keyword_quality_score": trend.keyword_quality_score,
        "search_interest_score": (
            trend.search_interest_score if trend.search_interest_available else None
        ),
        "growth_rate": trend.growth_rate,
        "acceleration": None,
        "source_count": trend.source_count,
        "document_count": document_count,
        "sources": sources,
        "primary_entity": primary_entity,
        "primary_context_title": primary_context,
        "ai_status": ai_row.analysis_status if ai_row else "not_analyzed",
        "ai_analysis_available": bool(ai_row and ai_row.analysis_status in AI_COMPLETE_STATUSES),
        "ai_trend_summary": ai_row.trend_summary if ai_row else None,
        "travel_relevance_score": ai_row.travel_relevance_score if ai_row else None,
        "travel_relevance_level": ai_row.travel_relevance_level if ai_row else None,
        "watchlist": trend.status == "watchlist",
        "status": trend.status,
        "suspicious": bool(
            rejection_reason(trend.keyword, trend.keyword.lower().replace(" ", ""))
            or len(trend.keyword.replace(" ", "")) <= 1
            or is_numeric_artifact(trend.keyword)
            or (
                trend.keyword_quality_score is not None
                and trend.keyword_quality_score < get_settings().keyword_min_quality_score
            )
            or document_count == 0
        ),
        "pipeline_version": trend.pipeline_version,
    }


def _latest_ai_rows(
    session: Session,
    week_start: date,
    keywords: list[str],
) -> dict[str, TrendAIAnalysis]:
    if not keywords:
        return {}
    rows = session.scalars(
        select(TrendAIAnalysis)
        .where(
            TrendAIAnalysis.week_start == week_start,
            TrendAIAnalysis.normalized_keyword.in_(keywords),
        )
        .order_by(TrendAIAnalysis.updated_at.desc())
    ).all()
    result: dict[str, TrendAIAnalysis] = {}
    for row in rows:
        result.setdefault(row.normalized_keyword, row)
    return result


def _metric_cards(session: Session, week_start: date) -> list[dict[str, object]]:
    values = _metric_values(session, week_start)
    previous_week = week_start - timedelta(days=7)
    previous_exists = session.scalar(
        select(func.count(WeeklyTrend.id)).where(WeeklyTrend.week_start == previous_week)
    ) or 0
    previous = _metric_values(session, previous_week) if previous_exists else {}
    labels = {
        "trend_count": "이번 주 트렌드 수",
        "watchlist_count": "Watchlist 키워드 수",
        "average_final_score": "평균 final_score",
        "ai_completed_count": "AI 분석 완료 수",
        "travel_high_count": "여행 연관성 High 키워드 수",
        "document_count": "수집 문서 수",
    }
    return [
        {
            "key": key,
            "label": label,
            "value": values[key],
            "previous_value": previous.get(key),
            "delta": round(values[key] - previous[key], 2) if key in previous else None,
        }
        for key, label in labels.items()
    ]


def _metric_values(session: Session, week_start: date) -> dict[str, int | float]:
    trends = list(session.scalars(select(WeeklyTrend).where(WeeklyTrend.week_start == week_start)).all())
    ai_rows = list(
        session.scalars(
            select(TrendAIAnalysis).where(TrendAIAnalysis.week_start == week_start)
        ).all()
    )
    week_end = week_start + timedelta(days=6)
    document_count = session.scalar(
        select(func.count(SourceDocument.id)).where(
            func.date(SourceDocument.published_at) >= week_start.isoformat(),
            func.date(SourceDocument.published_at) <= week_end.isoformat(),
        )
    ) or 0
    return {
        "trend_count": sum(trend.status == "weekly_trend" for trend in trends),
        "watchlist_count": sum(trend.status == "watchlist" for trend in trends),
        "average_final_score": round(
            sum(trend.final_score for trend in trends) / len(trends), 2
        ) if trends else 0.0,
        "ai_completed_count": sum(row.analysis_status == "completed" for row in ai_rows),
        "travel_high_count": sum(
            row.analysis_status in AI_COMPLETE_STATUSES
            and row.travel_relevance_level == "high"
            for row in ai_rows
        ),
        "document_count": document_count,
    }


def _source_distribution(session: Session, week_start: date) -> list[dict[str, object]]:
    week_end = week_start + timedelta(days=6)
    rows = session.execute(
        select(SourceDocument.source, func.count(SourceDocument.id))
        .where(
            func.date(SourceDocument.published_at) >= week_start.isoformat(),
            func.date(SourceDocument.published_at) <= week_end.isoformat(),
        )
        .group_by(SourceDocument.source)
        .order_by(func.count(SourceDocument.id).desc(), SourceDocument.source.asc())
    ).all()
    return [{"name": source, "count": count} for source, count in rows]


def _entity_distribution(session: Session, week_start: date) -> list[dict[str, object]]:
    rows = dict(
        session.execute(
            select(TrendEntityLink.entity_type, func.count(TrendEntityLink.id))
            .where(TrendEntityLink.week_start == week_start)
            .group_by(TrendEntityLink.entity_type)
        ).all()
    )
    return [
        {"name": entity_type, "count": rows.get(entity_type, 0)}
        for entity_type in ENTITY_TYPES
        if rows.get(entity_type, 0) > 0
    ]


def _ai_distribution(session: Session, week_start: date) -> list[dict[str, object]]:
    rows = session.execute(
        select(TrendAIAnalysis.analysis_status, func.count(TrendAIAnalysis.id))
        .where(TrendAIAnalysis.week_start == week_start)
        .group_by(TrendAIAnalysis.analysis_status)
    ).all()
    return [{"name": status, "count": count} for status, count in rows]


def _pipeline_status(session: Session, week_start: date | None) -> list[dict[str, object]]:
    settings = get_settings()
    document_count = session.scalar(select(func.count(SourceDocument.id))) or 0
    keyword_count = session.scalar(select(func.count(KeywordOccurrence.id))) or 0
    trend_count = (
        session.scalar(
            select(func.count(WeeklyTrend.id)).where(WeeklyTrend.week_start == week_start)
        ) or 0
        if week_start
        else 0
    )
    provider_counts = dict(
        session.execute(
            select(SearchInterestObservation.provider, func.count(SearchInterestObservation.id))
            .group_by(SearchInterestObservation.provider)
        ).all()
    )
    entity_count = session.scalar(select(func.count(EntityMention.id))) or 0
    context_counts = dict(
        session.execute(
            select(EntityContext.match_status, func.count(EntityContext.id))
            .group_by(EntityContext.match_status)
        ).all()
    )
    matched_contexts = context_counts.get("matched", 0)
    manual_contexts = context_counts.get("manual", 0)
    ambiguous_contexts = context_counts.get("ambiguous", 0)
    unmatched_contexts = context_counts.get("unmatched", 0)
    context_errors = context_counts.get("error", 0)
    ai_counts = get_ai_status_counts(
        session,
        now=datetime.now(timezone.utc).replace(tzinfo=None),
        cache_hours=settings.gemini_analysis_cache_hours,
    )
    return [
        _pipeline_item("documents", "Source Document 수집", "healthy" if document_count else "no_data", document_count),
        _pipeline_item("keywords", "키워드 추출", "healthy" if keyword_count else "no_data", keyword_count),
        _pipeline_item("weekly_trends", "Weekly Trend 계산", "healthy" if trend_count else "no_data", trend_count),
        _pipeline_item("google_trends", "Google Trends CSV", "healthy" if provider_counts.get("google_trends", 0) else "no_data", provider_counts.get("google_trends", 0)),
        _pipeline_item("naver_datalab", "Naver DataLab CSV", "healthy" if provider_counts.get("naver_datalab", 0) else "no_data", provider_counts.get("naver_datalab", 0)),
        _pipeline_item(
            "ner",
            "GLiNER NER",
            "disabled" if not settings.ner_enabled else "healthy" if entity_count else "no_data",
            entity_count,
            {"enabled": settings.ner_enabled},
        ),
        _pipeline_item(
            "wikipedia",
            "Wikipedia Context",
            _context_pipeline_status(
                settings.wikipedia_enabled,
                matched_contexts + manual_contexts,
                context_errors,
            ),
            matched_contexts + manual_contexts,
            {
                "matched": matched_contexts,
                "manual": manual_contexts,
                "ambiguous": ambiguous_contexts,
                "unmatched": unmatched_contexts,
                "error": context_errors,
                "errors": context_errors,
            },
        ),
        _pipeline_item(
            "gemini",
            "Gemini AI 분석",
            _gemini_pipeline_status(settings, ai_counts),
            ai_counts.completed_count + ai_counts.partial_count,
            {
                "enabled": settings.gemini_enabled,
                "model_configured": bool(settings.gemini_model),
                "api_key_configured": bool(settings.gemini_api_key),
                "configured_model": settings.gemini_model or None,
                "completed_count": ai_counts.completed_count,
                "partial_count": ai_counts.partial_count,
                "cached_count": ai_counts.cached_count,
                "error_count": ai_counts.error_count,
                "last_generated_at": ai_counts.last_generated_at,
            },
        ),
        _keyword_quality_pipeline_item(session, settings, week_start),
    ]


def _keyword_quality_pipeline_item(session, settings, week_start):
    if week_start is None:
        active = 0
        average = None
    else:
        active = session.scalar(
            select(func.count(WeeklyTrend.id)).where(
                WeeklyTrend.week_start == week_start,
                WeeklyTrend.keyword_quality_score >= settings.keyword_min_quality_score,
            )
        ) or 0
        average = session.scalar(
            select(func.avg(WeeklyTrend.keyword_quality_score)).where(
                WeeklyTrend.week_start == week_start,
                WeeklyTrend.keyword_quality_score >= settings.keyword_min_quality_score,
            )
        )
    rejected = session.scalar(
        select(func.count(KeywordCandidate.id)).where(KeywordCandidate.accepted.is_(False))
    ) or 0
    suspicious = 0
    if week_start is not None:
        trends = list(
            session.scalars(
                select(WeeklyTrend).where(WeeklyTrend.week_start == week_start)
            ).all()
        )
        week_end = trends[0].week_end if trends else week_start
        linked = set(
            session.scalars(
                select(distinct(KeywordOccurrence.normalized_keyword)).where(
                    func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                    func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
                )
            ).all()
        )
        suspicious = sum(
            bool(
                rejection_reason(trend.keyword, trend.keyword.lower().replace(" ", ""))
                or len(trend.keyword.replace(" ", "")) <= 1
                or is_numeric_artifact(trend.keyword)
                or trend.keyword_quality_score is None
                or trend.keyword_quality_score < settings.keyword_min_quality_score
                or trend.keyword not in linked
            )
            for trend in trends
        )
    status = "no_data" if not active else "partial" if suspicious else "healthy"
    return _pipeline_item(
        "keyword_quality",
        "키워드 품질",
        status,
        active,
        {
            "pipeline_version": settings.keyword_pipeline_version,
            "active_keywords": active,
            "rejected_candidates": rejected,
            "suspicious_keywords": suspicious,
            "average_quality_score": round(float(average), 2) if average is not None else None,
        },
    )


def _pipeline_item(key, label, status, count, details=None):
    return {
        "key": key,
        "label": label,
        "status": status,
        "count": count,
        "details": details or {},
    }


def _context_pipeline_status(enabled: bool, matched: int, errors: int) -> str:
    if not enabled:
        return "disabled"
    if errors and matched:
        return "partial"
    if errors:
        return "error"
    if matched:
        return "healthy"
    return "no_data"


def _gemini_pipeline_status(settings, counts) -> str:
    if not settings.gemini_enabled:
        return "disabled"
    if not settings.gemini_api_key or not settings.gemini_model:
        return "configuration_required"
    if counts.error_count and (counts.completed_count or counts.partial_count):
        return "partial"
    if counts.error_count:
        return "error"
    if counts.completed_count or counts.partial_count:
        return "healthy"
    return "no_data"


def _serialize_ai(row: TrendAIAnalysis) -> dict[str, object]:
    return {
        "id": row.id,
        "keyword": row.keyword,
        "analysis_status": row.analysis_status,
        "trend_summary": row.trend_summary,
        "rising_reason": row.rising_reason,
        "evidence_summary": _json_list(row.evidence_summary),
        "travel_relevance_score": row.travel_relevance_score,
        "travel_relevance_level": row.travel_relevance_level,
        "travel_relevance_reason": row.travel_relevance_reason,
        "recommended_destinations": _json_list(row.recommended_destinations_json),
        "content_ideas": _json_list(row.content_ideas_json),
        "cautions": _json_list(row.cautions_json),
        "evidence_refs": _json_list(row.evidence_refs_json),
        "confidence_score": row.confidence_score,
        "model_name": row.model_name,
        "generated_at": row.generated_at,
        "error_code": row.error_code,
        "error_message": row.error_message,
    }


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []

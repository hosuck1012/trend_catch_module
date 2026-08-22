from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib

from sqlalchemy import delete, distinct, func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.keywords.stopword_filter import rejection_reason
from app.keywords.keyword_quality import is_numeric_artifact
from app.models.entity_context import EntityContext
from app.models.keyword_candidate import KeywordCandidate
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.search_interest_observation import SearchInterestObservation
from app.models.search_validation_result import SearchValidationResult
from app.models.source_document import SourceDocument
from app.models.trend_ai_analysis import TrendAIAnalysis
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.repositories.entity_repository import get_mentions_for_documents
from app.repositories.keyword_quality_repository import (
    delete_occurrences_for_documents,
    get_documents_for_quality,
    replace_candidate_audit,
)
from app.repositories.search_interest_repository import (
    ValidationValues,
    get_observation_points,
    upsert_validation_result,
)
from app.repositories.trend_repository import get_occurrences_between, upsert_weekly_trends
from app.services.keyword_extraction_v2_service import QualityAnalysis, analyze_documents
from app.services.search_interest_scoring_service import (
    SEARCH_PROVIDERS,
    calculate_provider_score,
    combine_provider_scores,
)
from app.services.trend_calculation_service import calculate_scores
from app.services.trend_entity_link_service import _calculate_links, _group_mentions
from app.services.trend_rescoring_service import rescore_weekly_trend


@dataclass(frozen=True)
class RebuildResult:
    status: str
    dry_run: bool
    week_start: date | None
    week_end: date | None
    processed_documents: int
    current_top_20: list[str]
    v2_top_20: list[str]
    removed_keywords: list[str]
    added_keywords: list[str]
    retained_keywords: list[str]
    search_interest_links_preserved: list[str]
    entity_links_to_recalculate: int
    ai_analyses_to_invalidate: int
    source_documents_preserved: int
    search_observations_preserved: int
    manual_contexts_preserved: int


def preview_quality(
    session: Session,
    *,
    limit: int,
    source: str | None,
    since_days: int,
    include_rejected: bool,
) -> dict[str, object]:
    documents = get_documents_for_quality(
        session, since_days=since_days, limit=limit, source=source
    )
    analysis = analyze_documents(documents, now=_analysis_now(documents))
    rejected_grouped: Counter[tuple[str, str]] = Counter(
        (row.candidate_text, row.rejection_reason or "unknown")
        for row in analysis.candidates
        if not row.accepted
    )
    examples = [
        {
            "document_id": row.document_id,
            "candidate": row.candidate_text,
            "extractor": row.extractor,
            "quality_score": row.quality_score,
            "accepted": row.accepted,
            "rejection_reason": row.rejection_reason,
        }
        for row in analysis.candidates
        if include_rejected or row.accepted
    ][:100]
    return {
        "pipeline_version": get_settings().keyword_pipeline_version,
        "processed_documents": analysis.processed_documents,
        "total_candidates": len(analysis.candidates),
        "accepted_candidates": sum(row.accepted for row in analysis.candidates),
        "rejected_candidates": sum(not row.accepted for row in analysis.candidates),
        "top_accepted": [asdict(row) for row in analysis.accepted[:20]],
        "top_rejected": [
            {"candidate": candidate, "rejection_reason": reason, "occurrence_count": count}
            for (candidate, reason), count in rejected_grouped.most_common(20)
        ],
        "rejection_reason_counts": dict(analysis.rejected_counts),
        "extractor_counts": dict(analysis.extractor_counts),
        "examples": examples,
    }


def rebuild_keywords(
    session: Session,
    *,
    week_start: date | None,
    since_days: int,
    dry_run: bool,
    force: bool,
    limit: int,
) -> RebuildResult:
    documents = get_documents_for_quality(
        session, since_days=since_days, limit=limit, week_start=week_start
    )
    analysis = analyze_documents(documents, now=_analysis_now(documents))
    resolved_week_start, week_end = _resolve_week(documents, week_start)
    result = _comparison(
        session, documents, analysis, resolved_week_start, week_end, dry_run
    )
    if dry_run:
        return result
    if not force:
        raise ValueError("실제 재생성에는 force=true가 필요합니다. 먼저 dry-run 결과를 확인하세요.")
    if not documents or resolved_week_start is None or week_end is None:
        return result
    try:
        _apply_rebuild(session, documents, analysis, resolved_week_start, week_end)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return RebuildResult(**{**asdict(result), "status": "ok", "dry_run": False})


def quality_report(session: Session) -> dict[str, object]:
    settings = get_settings()
    latest_week = session.scalar(select(func.max(WeeklyTrend.week_start)))
    trends = (
        list(
            session.scalars(
                select(WeeklyTrend).where(WeeklyTrend.week_start == latest_week)
            ).all()
        )
        if latest_week
        else []
    )
    rejected_rows = session.execute(
        select(KeywordCandidate.rejection_reason, func.count(KeywordCandidate.id))
        .where(KeywordCandidate.accepted.is_(False))
        .group_by(KeywordCandidate.rejection_reason)
    ).all()
    rejected = {reason or "unknown": count for reason, count in rejected_rows}
    accepted_scores = [
        trend.keyword_quality_score
        for trend in trends
        if trend.keyword_quality_score is not None
        and trend.keyword_quality_score >= settings.keyword_min_quality_score
    ]
    linked_keywords = set()
    if latest_week and trends:
        week_end = trends[0].week_end
        linked_keywords = set(
            session.scalars(
                select(distinct(KeywordOccurrence.normalized_keyword)).where(
                    func.date(KeywordOccurrence.occurred_at) >= latest_week.isoformat(),
                    func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
                )
            ).all()
        )
    suspicious = []
    for trend in trends:
        reasons = _suspicious_reasons_from_linked(trend, linked_keywords)
        if reasons:
            suspicious.append({"keyword": trend.keyword, "reasons": reasons})
        if len(suspicious) >= 500:
            break
    return {
        "pipeline_version": settings.keyword_pipeline_version,
        "active_keywords": len(trends),
        "rejected_candidate_count": sum(rejected.values()),
        "stopword_rejection_count": rejected.get("korean_stopword", 0)
        + rejected.get("english_stopword", 0),
        "url_artifact_rejection_count": rejected.get("url_artifact", 0),
        "generic_word_rejection_count": rejected.get("generic_word", 0),
        "average_quality_score": round(sum(accepted_scores) / len(accepted_scores), 2)
        if accepted_scores
        else None,
        "lowest_accepted_quality_score": min(accepted_scores) if accepted_scores else None,
        "keywords_without_search_validation": sum(
            not trend.search_interest_available for trend in trends
        ),
        "suspicious_keywords": suspicious,
    }


def suspicious_reasons(session: Session, trend: WeeklyTrend) -> list[str]:
    linked = session.scalar(
        select(func.count(KeywordOccurrence.id)).where(
            KeywordOccurrence.normalized_keyword == trend.keyword,
            func.date(KeywordOccurrence.occurred_at) >= trend.week_start.isoformat(),
            func.date(KeywordOccurrence.occurred_at) <= trend.week_end.isoformat(),
        )
    ) or 0
    return _suspicious_reasons_from_linked(
        trend, {trend.keyword} if linked else set()
    )


def _suspicious_reasons_from_linked(
    trend: WeeklyTrend, linked_keywords: set[str]
) -> list[str]:
    settings = get_settings()
    keyword = trend.keyword
    reasons: list[str] = []
    lexical = rejection_reason(keyword, keyword.lower().replace(" ", ""))
    if lexical:
        reasons.append(lexical)
    if len(keyword.replace(" ", "")) == 1:
        reasons.append("too_short")
    if is_numeric_artifact(keyword):
        reasons.append("numeric_only")
    if (
        trend.keyword_quality_score is not None
        and trend.keyword_quality_score < settings.keyword_min_quality_score
    ):
        reasons.append("low_quality")
    if trend.keyword not in linked_keywords:
        reasons.append("no_document_link")
    return sorted(set(reasons))


def _comparison(session, documents, analysis, week_start, week_end, dry_run) -> RebuildResult:
    ids = [document.id for document in documents]
    current_rows = session.execute(
        select(KeywordOccurrence.normalized_keyword, func.count(KeywordOccurrence.id))
        .where(KeywordOccurrence.document_id.in_(ids) if ids else False)
        .group_by(KeywordOccurrence.normalized_keyword)
        .order_by(func.count(KeywordOccurrence.id).desc(), KeywordOccurrence.normalized_keyword)
    ).all()
    current = [row[0] for row in current_rows]
    proposed = [row.normalized_keyword for row in analysis.accepted]
    current_set, proposed_set = set(current), set(proposed)
    observation_keywords = set(
        session.scalars(
            select(distinct(SearchInterestObservation.normalized_keyword)).where(
                SearchInterestObservation.normalized_keyword.in_(proposed_set)
                if proposed_set
                else False
            )
        ).all()
    )
    entity_count = (
        session.scalar(
            select(func.count(TrendEntityLink.id)).where(
                TrendEntityLink.week_start == week_start
            )
        ) or 0
        if week_start
        else 0
    )
    ai_count = (
        session.scalar(
            select(func.count(TrendAIAnalysis.id)).where(
                TrendAIAnalysis.week_start == week_start
            )
        ) or 0
        if week_start
        else 0
    )
    return RebuildResult(
        status="dry_run" if dry_run else "pending",
        dry_run=dry_run,
        week_start=week_start,
        week_end=week_end,
        processed_documents=len(documents),
        current_top_20=current[:20],
        v2_top_20=proposed[:20],
        removed_keywords=sorted(current_set - proposed_set),
        added_keywords=sorted(proposed_set - current_set),
        retained_keywords=sorted(current_set & proposed_set),
        search_interest_links_preserved=sorted(proposed_set & observation_keywords),
        entity_links_to_recalculate=entity_count,
        ai_analyses_to_invalidate=ai_count,
        source_documents_preserved=session.scalar(select(func.count(SourceDocument.id))) or 0,
        search_observations_preserved=session.scalar(
            select(func.count(SearchInterestObservation.id))
        ) or 0,
        manual_contexts_preserved=session.scalar(
            select(func.count(EntityContext.id)).where(EntityContext.match_status == "manual")
        ) or 0,
    )


def _apply_rebuild(session, documents, analysis, week_start, week_end) -> None:
    settings = get_settings()
    document_ids = [document.id for document in documents]
    documents_by_id = {document.id: document for document in documents}
    delete_occurrences_for_documents(session, document_ids)
    accepted = [row for row in analysis.candidates if row.accepted]
    session.add_all(
        KeywordOccurrence(
            document_id=row.document_id,
            keyword=row.candidate_text,
            normalized_keyword=row.normalized_candidate,
            source=documents_by_id[row.document_id].source,
            occurred_at=documents_by_id[row.document_id].published_at,
            keyword_quality_score=row.quality_score,
            pipeline_version=settings.keyword_pipeline_version,
        )
        for row in accepted
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    replace_candidate_audit(
        session,
        document_ids=document_ids,
        rows=[
            {
                "document_id": row.document_id,
                "candidate_text": row.candidate_text,
                "normalized_candidate": row.normalized_candidate,
                "candidate_type": row.candidate_type,
                "extractor": row.extractor,
                "quality_score": row.quality_score,
                "accepted": row.accepted,
                "rejection_reason": row.rejection_reason,
                "title_occurrence": row.title_occurrence,
                "body_occurrence": row.body_occurrence,
                "entity_type": row.entity_type,
                "entity_confidence": row.entity_confidence,
                "created_at": now,
                "pipeline_version": settings.keyword_pipeline_version,
            }
            for row in analysis.candidates
        ],
    )
    session.flush()

    context_snapshots = list(
        session.execute(
            select(
                TrendContextLink.keyword,
                TrendContextLink.entity_context_id,
                TrendContextLink.normalized_entity,
                TrendContextLink.entity_type,
                TrendContextLink.context_score,
                TrendContextLink.is_primary,
                TrendContextLink.created_at,
            ).where(TrendContextLink.week_start == week_start)
        ).all()
    )
    session.execute(delete(TrendContextLink).where(TrendContextLink.week_start == week_start))
    session.execute(delete(TrendEntityLink).where(TrendEntityLink.week_start == week_start))
    session.execute(
        delete(SearchValidationResult).where(SearchValidationResult.week_start == week_start)
    )
    session.execute(delete(WeeklyTrend).where(WeeklyTrend.week_start == week_start))
    session.flush()

    previous_start = week_start - timedelta(days=7)
    rows = get_occurrences_between(session, previous_start, week_end)
    scores = calculate_scores(
        rows=rows,
        week_start=week_start,
        week_end=week_end,
        previous_week_start=previous_start,
        previous_week_end=week_start - timedelta(days=1),
    )
    scores = [
        score
        for score in scores
        if score.keyword_quality_score >= settings.keyword_min_quality_score
    ]
    upsert_weekly_trends(
        session,
        week_start=week_start,
        week_end=week_end,
        scores=scores,
        commit=False,
    )
    session.flush()
    _reconnect_search(session, week_start, week_end)
    _rebuild_entity_links(session, week_start, week_end)
    _restore_context_links(session, context_snapshots, week_start, week_end)
    for row in session.scalars(
        select(TrendAIAnalysis).where(TrendAIAnalysis.week_start == week_start)
    ).all():
        row.input_hash = hashlib.sha256(
            f"invalidated:{row.input_hash}:{settings.keyword_pipeline_version}".encode()
        ).hexdigest()


def _reconnect_search(session, week_start, week_end) -> None:
    trends = list(
        session.scalars(select(WeeklyTrend).where(WeeklyTrend.week_start == week_start)).all()
    )
    points = get_observation_points(
        session, start_date=week_start - timedelta(days=7), end_date=week_end
    )
    grouped = {
        (trend.keyword, provider): []
        for trend in trends
        for provider in SEARCH_PROVIDERS
    }
    for point in points:
        if (point.normalized_keyword, point.provider) in grouped:
            grouped[(point.normalized_keyword, point.provider)].append(point)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for trend in trends:
        scores = {
            provider: calculate_provider_score(
                grouped[(trend.keyword, provider)],
                week_start=week_start,
                week_end=week_end,
            )
            for provider in SEARCH_PROVIDERS
            if grouped[(trend.keyword, provider)]
        }
        combined = combine_provider_scores(scores)
        if combined.provider_count > 0 and combined.combined_score is not None:
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
                    calculated_at=now,
                ),
            )
        rescore_weekly_trend(
            trend, combined.combined_score, provider_count=combined.provider_count
        )


def _rebuild_entity_links(session, week_start, week_end) -> None:
    trends = list(
        session.scalars(
            select(WeeklyTrend).where(
                WeeklyTrend.week_start == week_start,
                WeeklyTrend.status.in_(("weekly_trend", "watchlist")),
            )
        ).all()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for trend in trends:
        document_ids = list(
            session.scalars(
                select(distinct(KeywordOccurrence.document_id)).where(
                    KeywordOccurrence.normalized_keyword == trend.keyword,
                    func.date(KeywordOccurrence.occurred_at) >= week_start.isoformat(),
                    func.date(KeywordOccurrence.occurred_at) <= week_end.isoformat(),
                )
            ).all()
        )
        mentions = get_mentions_for_documents(session, document_ids)
        links = _calculate_links(
            keyword=trend.keyword,
            week_start=week_start,
            week_end=week_end,
            document_ids=document_ids,
            total_source_count=max(trend.source_count, 1),
            groups=_group_mentions(mentions, trend.keyword),
            calculated_at=now,
        )
        session.add_all(TrendEntityLink(**link) for link in links)
    session.flush()


def _restore_context_links(session, snapshots, week_start, week_end) -> None:
    valid = set(
        session.execute(
            select(
                TrendEntityLink.keyword,
                TrendEntityLink.normalized_entity,
                TrendEntityLink.entity_type,
            ).where(TrendEntityLink.week_start == week_start)
        ).all()
    )
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for row in snapshots:
        if (row.keyword, row.normalized_entity, row.entity_type) not in valid:
            continue
        context = session.get(EntityContext, row.entity_context_id)
        if context is None or context.match_status not in {"matched", "manual"}:
            continue
        session.add(
            TrendContextLink(
                keyword=row.keyword,
                week_start=week_start,
                week_end=week_end,
                entity_context_id=row.entity_context_id,
                normalized_entity=row.normalized_entity,
                entity_type=row.entity_type,
                context_score=row.context_score,
                is_primary=row.is_primary,
                created_at=row.created_at,
                updated_at=now,
            )
        )


def _resolve_week(documents, requested):
    if requested:
        return requested, requested + timedelta(days=6)
    if not documents:
        return None, None
    latest = max(document.published_at.date() for document in documents)
    return latest - timedelta(days=6), latest


def _analysis_now(documents):
    return max(
        (document.published_at for document in documents),
        default=datetime.now(timezone.utc).replace(tzinfo=None),
    )

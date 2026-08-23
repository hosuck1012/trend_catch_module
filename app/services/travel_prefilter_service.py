from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime
import hashlib
import json

from sqlalchemy.orm import Session

from app.config import get_settings
from app.context_v2.context_extractor import extract_keyword_contexts
from app.context_v2.travel_rules import (
    EntitySignal,
    TrendSignal,
    evaluate_travel_rules,
    load_terms,
)
from app.models.entity_mention import EntityMention
from app.models.keyword_context import KeywordContext
from app.models.entity_context import EntityContext
from app.models.trend_entity_link import TrendEntityLink
from app.repositories import travel_opportunity_repository as repo
from app.services.keyword_normalization_service import normalize_keyword
from app.services.related_destination_expansion_service import (
    related_destination_metadata,
    serialize_related_destination,
)


TRAVEL_PREFILTER_VERSION = "v2-rule-3-related-destination"


@dataclass(frozen=True)
class TransientKeywordContext:
    id: int
    document_id: int
    keyword: str
    normalized_keyword: str
    previous_sentence: str | None
    matched_sentence: str
    next_sentence: str | None
    combined_context: str
    source: str
    published_at: datetime


@dataclass(frozen=True)
class CandidatePreview:
    keyword: str
    normalized_keyword: str
    score: float
    status: str
    category: str
    reasoning_codes: list[str]
    matched_positive_terms: list[str]
    matched_negative_terms: list[str]
    context: str


@dataclass(frozen=True)
class PrefilterResult:
    status: str
    dry_run: bool
    week_start: date | None
    week_end: date | None
    processed: int
    rejected: int
    weak: int
    review: int
    strong: int
    estimated_llm_candidates: int
    reduction_rate: float
    top_candidates: list[CandidatePreview]
    rejection_reason_counts: dict[str, int]
    raw_keyword_count: int
    quality_keyword_count: int
    context_candidate_count: int
    created: int = 0
    updated: int = 0
    would_create: int = 0
    would_update: int = 0
    skipped: int = 0
    next_cursor: int | None = None
    has_more: bool = False
    batches: int = 0
    errors: int = 0
    cache_hits: int = 0
    rule_version: str = TRAVEL_PREFILTER_VERSION
    category_counts: dict[str, int] = field(default_factory=dict)
    reasoning_code_counts: dict[str, int] = field(default_factory=dict)
    primary_entity_count: int = 0
    other_percentage: float = 0.0


def prefilter_travel_opportunities(
    session: Session,
    *,
    week_start: date | None,
    dry_run: bool,
    force: bool,
    limit: int,
    after_id: int | None = None,
    process_all: bool = False,
) -> PrefilterResult:
    settings = get_settings()
    resolved_start, resolved_end = repo.resolve_week_range(session, week_start)
    if not settings.travel_opportunity_v2_enabled or not settings.travel_prefilter_enabled:
        return PrefilterResult(
            "disabled", dry_run, resolved_start, resolved_end, 0, 0, 0, 0, 0, 0,
            0.0, [], {}, 0, 0, 0
        )
    now = repo.utc_now()
    positive_terms = load_terms("travel_positive_terms.json")
    negative_terms = load_terms("travel_negative_terms.json")
    rows: list[dict[str, object]] = []
    previews: list[CandidatePreview] = []
    status_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    reasoning_code_counts: Counter[str] = Counter()
    primary_entity_count = 0
    eligible_count, materialized_before, _remaining_before = repo.count_rule_materialization_coverage(
        session,
        week_start=resolved_start,
        week_end=resolved_end,
        candidate_week_start=resolved_start,
    )
    processed = 0
    created = 0
    updated = 0
    would_create = 0
    would_update = 0
    batches = 0
    cache_hits = 0
    cursor = after_id
    next_cursor: int | None = None
    has_more = False
    while True:
        contexts, page_next_cursor, page_has_more = repo.get_keyword_contexts_page(
            session,
            week_start=resolved_start,
            week_end=resolved_end,
            candidate_week_start=resolved_start,
            after_id=cursor,
            limit=limit,
            force=force,
        )
        if dry_run and not contexts and eligible_count == 0 and after_id is None:
            contexts = _build_transient_contexts(
                session,
                week_start=resolved_start,
                week_end=resolved_end,
                limit=limit,
            )
        if not contexts:
            next_cursor = None
            has_more = False
            break
        batches += 1
        existing_by_context = repo.get_existing_candidates_by_context(
            session,
            week_start=resolved_start,
            context_ids=[context.id for context in contexts if context.id > 0],
        ) if resolved_start else {}
        related_destinations_by_keyword = repo.get_related_destination_contexts(
            session,
            normalized_keywords=sorted(
                {context.normalized_keyword for context in contexts}
            ),
            week_start=resolved_start,
        ) if resolved_start else {}
        rows = []
        evaluated_in_page = 0
        for context in contexts:
            trend = repo.get_trend_by_keyword(
                session,
                keyword=context.normalized_keyword,
                week_start=resolved_start,
            )
            trend_links, mentions = repo.get_entities_for_context(
                session,
                context=context,
                week_start=resolved_start,
            )
            entities = _entity_signals(context, trend_links, mentions)
            related_destinations = related_destinations_by_keyword.get(
                context.normalized_keyword, []
            )
            trend_signal = TrendSignal(
                weekly_mentions=trend.weekly_mentions if trend else 0,
                final_score=trend.final_score if trend else None,
                source_count=trend.source_count if trend else max(1, _source_count_from_mentions(mentions)),
                document_count=1,
            )
            input_hash = rule_input_hash(
                context=context,
                entities=entities,
                trend=trend_signal,
                positive_terms=positive_terms,
                negative_terms=negative_terms,
                related_destinations=related_destinations,
            )
            existing = existing_by_context.get(context.id)
            if (
                not force
                and existing is not None
                and existing.rule_input_hash == input_hash
                and existing.rule_version == TRAVEL_PREFILTER_VERSION
            ):
                cache_hits += 1
                status_counts[existing.prefilter_status] += 1
                category_counts[existing.travel_category] += 1
                primary_entity_count += int(existing.primary_entity is not None)
                reasoning_code_counts.update(_json_list(existing.reasoning_codes_json))
                if existing.prefilter_status == "rejected":
                    for code in _json_list(existing.reasoning_codes_json) or ["NO_TRAVEL_SIGNAL"]:
                        rejection_reasons[code] += 1
                continue
            evaluated_in_page += 1
            if existing is None:
                would_create += 1
            else:
                would_update += 1
            rule_result = evaluate_travel_rules(
                keyword=context.keyword,
                context=context.combined_context,
                entities=entities,
                trend=trend_signal,
                positive_terms=positive_terms,
                negative_terms=negative_terms,
            )
            rule_result = _apply_related_destination_evidence(
                rule_result,
                keyword=context.keyword,
                related_destinations=related_destinations,
                settings=settings,
            )
            status_counts[rule_result.prefilter_status] += 1
            category_counts[rule_result.travel_category] += 1
            primary_entity_count += int(rule_result.primary_entity is not None)
            reasoning_code_counts.update(rule_result.reasoning_codes)
            if rule_result.prefilter_status == "rejected":
                for code in rule_result.reasoning_codes or ["NO_TRAVEL_SIGNAL"]:
                    rejection_reasons[code] += 1
            rows.append(
                {
                    "keyword": context.keyword,
                    "normalized_keyword": context.normalized_keyword,
                    "week_start": resolved_start,
                    "week_end": resolved_end,
                    "keyword_context_id": context.id,
                    "primary_entity": rule_result.primary_entity,
                    "primary_entity_type": rule_result.primary_entity_type,
                    "travel_category": rule_result.travel_category,
                    "entity_prior_score": rule_result.entity_prior_score,
                    "positive_context_score": rule_result.positive_context_score,
                    "negative_context_penalty": rule_result.negative_context_penalty,
                    "trend_evidence_score": rule_result.trend_evidence_score,
                    "source_diversity_score": rule_result.source_diversity_score,
                    "travel_pre_score": rule_result.travel_pre_score,
                    "prefilter_status": rule_result.prefilter_status,
                    "matched_positive_terms_json": repo.encode_json(rule_result.matched_positive_terms),
                    "matched_negative_terms_json": repo.encode_json(rule_result.matched_negative_terms),
                    "reasoning_codes_json": repo.encode_json(rule_result.reasoning_codes),
                    "rule_input_hash": input_hash,
                    "rule_version": TRAVEL_PREFILTER_VERSION,
                    "rule_calculated_at": now,
                    "created_at": now,
                    "updated_at": now,
                }
            )
            if rule_result.prefilter_status in {"review", "strong"}:
                previews.append(
                    CandidatePreview(
                        keyword=context.keyword,
                        normalized_keyword=context.normalized_keyword,
                        score=rule_result.travel_pre_score,
                        status=rule_result.prefilter_status,
                        category=rule_result.travel_category,
                        reasoning_codes=rule_result.reasoning_codes,
                        matched_positive_terms=rule_result.matched_positive_terms,
                        matched_negative_terms=rule_result.matched_negative_terms,
                        context=context.combined_context,
                    )
                )
        processed += evaluated_in_page
        if not dry_run and rows and resolved_start and resolved_end:
            page_created, page_updated = repo.upsert_travel_candidates(
                session,
                week_start=resolved_start,
                rows=rows,
                force=force,
            )
            created += page_created
            updated += page_updated
        has_more = page_has_more
        next_cursor = page_next_cursor
        if not process_all or not page_has_more:
            break
        if page_next_cursor is None or page_next_cursor <= (cursor or 0):
            raise RuntimeError("Rule pagination cursor did not advance")
        if batches >= 100_000:
            raise RuntimeError("Rule pagination exceeded safety limit")
        cursor = page_next_cursor
    raw = repo.count_raw_keywords(session, week_start=resolved_start)
    _candidate_total, _accepted_rows, quality = repo.count_keyword_candidate_funnel(
        session,
        week_start=resolved_start,
        week_end=resolved_end,
    )
    estimated = status_counts["strong"]
    reduction = round((1 - estimated / raw) * 100, 2) if raw else 0.0
    top = sorted(
        previews,
        key=lambda item: (-item.score, item.normalized_keyword),
    )[:20]
    return PrefilterResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        week_end=resolved_end,
        processed=processed,
        rejected=status_counts["rejected"],
        weak=status_counts["weak"],
        review=status_counts["review"],
        strong=status_counts["strong"],
        estimated_llm_candidates=estimated,
        reduction_rate=reduction,
        top_candidates=top,
        rejection_reason_counts=dict(rejection_reasons),
        raw_keyword_count=raw,
        quality_keyword_count=quality,
        context_candidate_count=processed,
        created=created,
        updated=updated,
        would_create=would_create,
        would_update=would_update,
        skipped=cache_hits,
        next_cursor=next_cursor,
        has_more=has_more,
        batches=batches,
        cache_hits=cache_hits,
        rule_version=TRAVEL_PREFILTER_VERSION,
        category_counts=dict(category_counts),
        reasoning_code_counts=dict(reasoning_code_counts),
        primary_entity_count=primary_entity_count,
        other_percentage=round(category_counts["OTHER"] / sum(category_counts.values()) * 100, 2)
        if category_counts
        else 0.0,
    )


def serialize_prefilter_result(result: PrefilterResult) -> dict[str, object]:
    payload = asdict(result)
    payload["top_candidates"] = [asdict(item) for item in result.top_candidates]
    return payload


def serialize_candidate(
    row,
    *,
    related_destinations: list[EntityContext] | None = None,
) -> dict[str, object]:
    context = row.keyword_context
    return {
        "keyword": row.keyword,
        "normalized_keyword": row.normalized_keyword,
        "week_start": row.week_start,
        "week_end": row.week_end,
        "score": row.travel_pre_score,
        "status": row.prefilter_status,
        "category": row.travel_category,
        "primary_entity": row.primary_entity,
        "primary_entity_type": row.primary_entity_type,
        "matched_positive_terms": _json_list(row.matched_positive_terms_json),
        "matched_negative_terms": _json_list(row.matched_negative_terms_json),
        "reasoning_codes": _json_list(row.reasoning_codes_json),
        "semantic_travel_score": row.semantic_travel_score,
        "semantic_status": row.semantic_status,
        "embedding_model": row.embedding_model,
        "semantic_positive_score": row.semantic_positive_score,
        "semantic_positive_category": row.semantic_positive_category,
        "semantic_negative_score": row.semantic_negative_score,
        "semantic_negative_category": row.semantic_negative_category,
        "embedding_input_hash": row.embedding_input_hash,
        "semantic_calculated_at": row.semantic_calculated_at,
        "trend_strength_score": row.trend_strength_score,
        "context_clarity_score": row.context_clarity_score,
        "travel_convertibility_score": row.travel_convertibility_score,
        "evidence_confidence_score": row.evidence_confidence_score,
        "high_precision_score": row.high_precision_score,
        "evidence_gate": row.evidence_gate,
        "evidence_codes": _json_list(row.evidence_codes_json),
        "evidence_document_count": row.evidence_document_count,
        "evidence_source_count": row.evidence_source_count,
        "ranking_status": row.ranking_status,
        "rank_in_week": row.rank_in_week,
        "ranking_version": row.ranking_version,
        "calculated_at": row.calculated_at,
        "cluster_id": row.cluster_id,
        "cluster_representative": bool(row.cluster_representative),
        "gemini_eligible": bool(row.gemini_eligible),
        "contexts": [_serialize_context(context)] if context else [],
        "related_destinations": [
            serialize_related_destination(item)
            for item in (related_destinations or [])
        ],
    }


def detail_for_keyword(session: Session, normalized_keyword: str) -> dict[str, object] | None:
    normalized = normalize_keyword(normalized_keyword) or normalized_keyword
    rows = repo.get_candidates_for_keyword(session, normalized)
    if not rows:
        return None
    best = rows[0]
    contexts = []
    entities = []
    sources = set()
    documents = set()
    positives = set()
    negatives = set()
    codes = set()
    for row in rows:
        context = row.keyword_context
        positives.update(_json_list(row.matched_positive_terms_json))
        negatives.update(_json_list(row.matched_negative_terms_json))
        codes.update(_json_list(row.reasoning_codes_json))
        if context:
            sources.add(context.source)
            documents.add(context.document_id)
            contexts.append(_serialize_context(context))
        if row.primary_entity and row.primary_entity_type:
            entities.append({"text": row.primary_entity, "entity_type": row.primary_entity_type})
    related_destinations = repo.get_related_destination_contexts(
        session,
        normalized_keywords=[best.normalized_keyword],
        week_start=best.week_start,
    ).get(best.normalized_keyword, [])
    return {
        "keyword": best.keyword,
        "normalized_keyword": best.normalized_keyword,
        "score": best.travel_pre_score,
        "status": best.prefilter_status,
        "category": best.travel_category,
        "contexts": contexts,
        "entities": _dedupe_entities(entities),
        "matched_positive_terms": sorted(positives),
        "matched_negative_terms": sorted(negatives),
        "reasoning_codes": sorted(codes),
        "source_count": len(sources),
        "document_count": len(documents),
        "related_destinations": [
            serialize_related_destination(item) for item in related_destinations
        ],
    }


def _build_transient_contexts(
    session: Session,
    *,
    week_start: date | None,
    week_end: date | None,
    limit: int,
) -> list[TransientKeywordContext]:
    settings = get_settings()
    keywords = repo.get_quality_keywords(
        session,
        week_start=week_start,
        week_end=week_end,
        limit=limit,
    )
    rows = repo.get_keyword_occurrence_documents(
        session,
        normalized_keywords=keywords,
        week_start=week_start,
        week_end=week_end,
        limit=limit,
    )
    contexts: list[TransientKeywordContext] = []
    for occurrence, document in rows:
        extracted = extract_keyword_contexts(
            text=f"{document.title}\n{document.text}".strip(),
            keyword=occurrence.keyword,
            normalized_keyword=occurrence.normalized_keyword,
            sentences_before=settings.context_sentences_before,
            sentences_after=settings.context_sentences_after,
            max_chars=settings.context_max_chars,
        )
        for item in extracted:
            contexts.append(
                TransientKeywordContext(
                    id=0,
                    document_id=document.id,
                    keyword=item.keyword,
                    normalized_keyword=item.normalized_keyword,
                    previous_sentence=item.previous_sentence,
                    matched_sentence=item.matched_sentence,
                    next_sentence=item.next_sentence,
                    combined_context=item.combined_context,
                    source=document.source,
                    published_at=document.published_at,
                )
            )
            if len(contexts) >= limit:
                return contexts
    return contexts


def _entity_signals(
    context: KeywordContext | TransientKeywordContext,
    trend_links: list[TrendEntityLink],
    mentions: list[EntityMention],
) -> list[EntitySignal]:
    signals = [
        EntitySignal(
            text=link.entity_text,
            normalized_text=link.normalized_entity,
            entity_type=link.entity_type,
            relation_score=link.relation_score,
            is_primary=link.is_primary,
        )
        for link in trend_links
    ]
    seen = {(signal.normalized_text, signal.entity_type) for signal in signals}
    context_text = context.combined_context.lower()
    keyword_norm = context.normalized_keyword.lower().replace(" ", "")
    for mention in mentions:
        key = (mention.normalized_text, mention.entity_type)
        if key in seen:
            continue
        mention_norm = mention.normalized_text.lower().replace(" ", "")
        in_context = mention.text.lower() in context_text or mention.normalized_text.lower() in context_text
        keyword_match = mention_norm == keyword_norm
        if in_context or keyword_match:
            signals.append(
                EntitySignal(
                    text=mention.text,
                    normalized_text=mention.normalized_text,
                    entity_type=mention.entity_type,
                    relation_score=mention.confidence * 100,
                    is_primary=keyword_match,
                )
            )
            seen.add(key)
    return signals


def rule_input_hash(
    *,
    context: KeywordContext | TransientKeywordContext,
    entities: list[EntitySignal],
    trend: TrendSignal,
    positive_terms: list[str],
    negative_terms: list[str],
    related_destinations: list[EntityContext] | None = None,
) -> str:
    settings = get_settings()
    payload = {
        "version": TRAVEL_PREFILTER_VERSION,
        "keyword": context.keyword,
        "normalized_keyword": context.normalized_keyword,
        "context": context.combined_context,
        "context_hash": getattr(context, "context_hash", None),
        "entities": sorted(
            (
                entity.normalized_text,
                entity.entity_type,
                round(entity.relation_score, 4),
                entity.is_primary,
            )
            for entity in entities
        ),
        "trend": {
            "weekly_mentions": trend.weekly_mentions,
            "final_score": trend.final_score,
            "source_count": trend.source_count,
            "document_count": trend.document_count,
        },
        "thresholds": {
            "minimum": settings.travel_prefilter_min_score,
            "review": settings.travel_prefilter_review_score,
            "strong": settings.travel_prefilter_strong_score,
        },
        "positive_terms": positive_terms,
        "negative_terms": negative_terms,
        "related_destinations": sorted(
            (
                destination.page_id,
                destination.page_url,
                destination.summary,
                round(destination.match_score, 4),
                destination.revision_id,
            )
            for destination in (related_destinations or [])
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _apply_related_destination_evidence(
    result,
    *,
    keyword: str,
    related_destinations: list[EntityContext],
    settings,
):
    if not related_destinations:
        return result
    metadata_rows = [
        related_destination_metadata(destination)
        for destination in related_destinations
    ]
    metadata_rows = [metadata for metadata in metadata_rows if metadata]
    if not metadata_rows:
        return result
    travel_categories = {
        str(metadata.get("travel_category", "")).strip()
        for metadata in metadata_rows
        if metadata.get("travel_category")
    }
    keyword_entity_types = {
        str(metadata.get("keyword_entity_type", "")).strip()
        for metadata in metadata_rows
        if metadata.get("keyword_entity_type")
    }
    theme_prior = 20.0 if "CONTENT_TITLE" in keyword_entity_types else 15.0
    destination_score = min(30.0, 15.0 + len(related_destinations) * 5.0)
    entity_prior = max(result.entity_prior_score, theme_prior)
    positive_score = max(result.positive_context_score, destination_score)
    travel_score = round(
        min(
            100.0,
            max(
                0.0,
                entity_prior
                + positive_score
                + result.trend_evidence_score
                + result.source_diversity_score
                - result.negative_context_penalty,
            ),
        ),
        2,
    )
    if travel_score >= settings.travel_prefilter_strong_score:
        status = "strong"
    elif travel_score >= settings.travel_prefilter_review_score:
        status = "review"
    elif travel_score >= settings.travel_prefilter_min_score:
        status = "weak"
    else:
        status = "rejected"
    matched_positive_terms = sorted(
        set(result.matched_positive_terms) | {"여행", "방문", "체험"}
    )
    reasoning_codes = sorted(
        (
            set(result.reasoning_codes)
            - {"NO_TRAVEL_SIGNAL"}
        )
        | {
            "CONTENT_THEME_DESTINATION",
            "OFFICIAL_DESTINATION_SOURCE",
            "CURATED_DESTINATION_SUGGESTION",
            "RELATED_DESTINATION_VERIFIED",
        }
    )
    return replace(
        result,
        primary_entity=result.primary_entity or keyword,
        primary_entity_type=result.primary_entity_type
        or next(iter(keyword_entity_types), "CONTENT_TITLE"),
        travel_category=next(iter(travel_categories), "LOCAL_CULTURE"),
        entity_prior_score=entity_prior,
        positive_context_score=positive_score,
        travel_pre_score=travel_score,
        prefilter_status=status,
        matched_positive_terms=matched_positive_terms,
        reasoning_codes=reasoning_codes,
    )


def _serialize_context(context) -> dict[str, object]:
    return {
        "id": context.id,
        "document_id": context.document_id,
        "source": context.source,
        "published_at": context.published_at,
        "previous_sentence": context.previous_sentence,
        "matched_sentence": context.matched_sentence,
        "next_sentence": context.next_sentence,
        "combined_context": context.combined_context,
    }


def _source_count_from_mentions(mentions: list[EntityMention]) -> int:
    return len({mention.source for mention in mentions})


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def _dedupe_entities(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    result = []
    for row in rows:
        key = (row["text"], row["entity_type"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result

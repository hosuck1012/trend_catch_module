from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime
import json

from sqlalchemy.orm import Session

from app.config import get_settings
from app.context_v2.context_extractor import extract_keyword_contexts
from app.context_v2.travel_rules import EntitySignal, TrendSignal, evaluate_travel_rules
from app.models.entity_mention import EntityMention
from app.models.keyword_context import KeywordContext
from app.models.trend_entity_link import TrendEntityLink
from app.repositories import travel_opportunity_repository as repo
from app.services.keyword_normalization_service import normalize_keyword


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


def prefilter_travel_opportunities(
    session: Session,
    *,
    week_start: date | None,
    dry_run: bool,
    force: bool,
    limit: int,
) -> PrefilterResult:
    settings = get_settings()
    resolved_start, resolved_end = repo.resolve_week_range(session, week_start)
    if not settings.travel_opportunity_v2_enabled or not settings.travel_prefilter_enabled:
        return PrefilterResult(
            "disabled", dry_run, resolved_start, resolved_end, 0, 0, 0, 0, 0, 0,
            0.0, [], {}, 0, 0, 0
        )
    contexts: list[KeywordContext | TransientKeywordContext] = repo.get_keyword_contexts_for_week(
        session,
        week_start=resolved_start,
        week_end=resolved_end,
        limit=limit,
    )
    if dry_run and not contexts:
        contexts = _build_transient_contexts(
            session,
            week_start=resolved_start,
            week_end=resolved_end,
            limit=limit,
        )
    now = repo.utc_now()
    rows: list[dict[str, object]] = []
    previews: list[CandidatePreview] = []
    status_counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
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
        trend_signal = TrendSignal(
            weekly_mentions=trend.weekly_mentions if trend else 0,
            final_score=trend.final_score if trend else None,
            source_count=trend.source_count if trend else max(1, _source_count_from_mentions(mentions)),
            document_count=1,
        )
        rule_result = evaluate_travel_rules(
            keyword=context.keyword,
            context=context.combined_context,
            entities=entities,
            trend=trend_signal,
        )
        status_counts[rule_result.prefilter_status] += 1
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
                "created_at": now,
                "updated_at": now,
            }
        )
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
    if not dry_run and rows and resolved_start and resolved_end:
        repo.upsert_travel_candidates(session, week_start=resolved_start, rows=rows, force=force)
    raw = repo.count_raw_keywords(session, week_start=resolved_start)
    quality = len(
        repo.get_quality_keywords(
            session,
            week_start=resolved_start,
            week_end=resolved_end,
            limit=100000,
        )
    )
    estimated = status_counts["strong"]
    reduction = round((1 - estimated / raw) * 100, 2) if raw else 0.0
    top = sorted(
        [item for item in previews if item.status in {"review", "strong"}],
        key=lambda item: (-item.score, item.normalized_keyword),
    )[:20]
    return PrefilterResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        week_end=resolved_end,
        processed=len(contexts),
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
        context_candidate_count=len(contexts),
    )


def serialize_prefilter_result(result: PrefilterResult) -> dict[str, object]:
    payload = asdict(result)
    payload["top_candidates"] = [asdict(item) for item in result.top_candidates]
    return payload


def serialize_candidate(row) -> dict[str, object]:
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
        "contexts": [_serialize_context(context)] if context else [],
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

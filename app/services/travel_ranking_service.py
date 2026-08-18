from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import json
import re
from typing import Iterable, Sequence

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models.entity_context import EntityContext
from app.models.entity_mention import EntityMention
from app.models.keyword_context import KeywordContext
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.repositories import travel_opportunity_repository as opportunity_repo
from app.repositories import travel_ranking_repository as repo


RANKING_VERSION = "v2-step3-local-1"
RANKING_THRESHOLDS = {
    "rejected_max": 69.99,
    "review_min": 70.0,
    "gemini_candidate_min": 85.0,
    "priority_candidate_min": 90.0,
}
NEGATIVE_SEMANTIC_CATEGORIES = {
    "FINANCE",
    "LEGAL",
    "POLITICS",
    "ACCIDENT",
    "CRIME",
}
LOCATION_REQUIRED_CATEGORIES = {
    "FILM_LOCATION",
    "DRAMA_LOCATION",
    "SHOW_LOCATION",
    "FESTIVAL",
    "CONCERT",
    "EXHIBITION",
    "POPUP",
    "SPORTS_EVENT",
    "FOOD",
}
CONCRETE_TRAVEL_CATEGORIES = {
    *LOCATION_REQUIRED_CATEGORIES,
    "LOCAL_CULTURE",
    "LANDMARK",
    "NATURE",
    "REGIONAL_MEME",
}
CLUSTER_SUFFIXES = (
    "촬영지",
    "로케이션",
    "여행지",
    "개최지",
    "여행",
    "방문",
    "명소",
)


@dataclass(frozen=True)
class EvidenceAssessment:
    score: float
    gate: str
    codes: list[str]
    document_count: int
    source_count: int


@dataclass
class RankedCandidate:
    keyword: str
    normalized_keyword: str
    week_start: date
    travel_category: str
    semantic_category: str | None
    semantic_status: str
    semantic_travel_score: float | None
    travel_pre_score: float
    trend_strength_score: float
    context_clarity_score: float
    travel_convertibility_score: float
    evidence_confidence_score: float
    high_precision_score: float
    evidence_gate: str
    evidence_codes: list[str]
    evidence_document_count: int
    evidence_source_count: int
    ranking_status: str
    rank_in_week: int = 0
    cluster_id: str = ""
    cluster_representative: bool = False
    gemini_eligible: bool = False
    contexts: list[str] = field(default_factory=list)
    _document_ids: set[int] = field(default_factory=set, repr=False)
    _entity_keys: set[tuple[str, str]] = field(default_factory=set, repr=False)


@dataclass(frozen=True)
class RankingResult:
    status: str
    dry_run: bool
    week_start: date | None
    processed: int
    rejected: int
    review: int
    gemini_candidates: int
    priority_candidates: int
    evidence_pass: int
    needs_evidence: int
    evidence_reject: int
    estimated_gemini_calls: int
    top_candidates: list[dict[str, object]]
    funnel: dict[str, object]
    annualized_candidate_estimate: float
    insufficient_history: bool


def rank_travel_opportunities(
    session: Session,
    *,
    week_start: date | None,
    dry_run: bool,
    force: bool,
    limit: int,
) -> RankingResult:
    resolved_start, _ = opportunity_repo.resolve_week_range(session, week_start)
    if resolved_start is None:
        return _empty_result(dry_run=dry_run, week_start=None)

    candidate_rows = repo.get_semantic_candidates(
        session,
        week_start=resolved_start,
        limit=max(limit * 20, limit),
    )
    grouped_rows: dict[str, list[TravelOpportunityCandidate]] = defaultdict(list)
    for row in candidate_rows:
        if row.normalized_keyword not in grouped_rows and len(grouped_rows) >= limit:
            continue
        grouped_rows[row.normalized_keyword].append(row)
    keywords = list(grouped_rows)
    contexts = repo.get_keyword_contexts(
        session, keywords=keywords, week_start=resolved_start
    )
    contexts_by_keyword: dict[str, list[KeywordContext]] = defaultdict(list)
    for context in contexts:
        contexts_by_keyword[context.normalized_keyword].append(context)
    for keyword, rows in grouped_rows.items():
        known_ids = {context.id for context in contexts_by_keyword[keyword]}
        for row in rows:
            if row.keyword_context and row.keyword_context.id not in known_ids:
                contexts_by_keyword[keyword].append(row.keyword_context)
                known_ids.add(row.keyword_context.id)

    trends = repo.get_weekly_trends(
        session, keywords=keywords, week_start=resolved_start
    )
    document_ids_by_keyword = repo.get_document_ids_by_keyword(
        session, keywords=keywords, week_start=resolved_start
    )
    for keyword, keyword_contexts in contexts_by_keyword.items():
        document_ids_by_keyword.setdefault(keyword, set()).update(
            context.document_id for context in keyword_contexts
        )
    all_document_ids = set().union(*document_ids_by_keyword.values()) if keywords else set()
    documents = repo.get_documents(session, all_document_ids)
    mentions = repo.get_entity_mentions(session, all_document_ids)
    mentions_by_document: dict[int, list[EntityMention]] = defaultdict(list)
    for mention in mentions:
        mentions_by_document[mention.document_id].append(mention)
    links_by_keyword: dict[str, list[TrendEntityLink]] = defaultdict(list)
    for link in repo.get_trend_entity_links(
        session, keywords=keywords, week_start=resolved_start
    ):
        links_by_keyword[link.keyword].append(link)
    entity_contexts = repo.get_entity_contexts(
        session, keywords=keywords, week_start=resolved_start
    )

    ranked: list[RankedCandidate] = []
    for keyword, rows in grouped_rows.items():
        best = max(
            rows,
            key=lambda row: (
                row.semantic_travel_score if row.semantic_travel_score is not None else -1,
                row.travel_pre_score,
            ),
        )
        keyword_contexts = contexts_by_keyword[keyword]
        document_ids = document_ids_by_keyword.get(keyword, set())
        document_mentions = [
            mention
            for document_id in document_ids
            for mention in mentions_by_document.get(document_id, [])
        ]
        keyword_mentions = _relevant_mentions(
            keyword=keyword,
            contexts=keyword_contexts,
            mentions=document_mentions,
        )
        keyword_links = links_by_keyword.get(keyword, [])
        entity_types = {
            mention.entity_type for mention in keyword_mentions
        } | {link.entity_type for link in keyword_links}
        entity_keys = {
            (mention.normalized_text, mention.entity_type) for mention in keyword_mentions
        } | {(link.normalized_entity, link.entity_type) for link in keyword_links}
        source_count = len(
            {
                documents[document_id].source
                for document_id in document_ids
                if document_id in documents
            }
            | {context.source for context in keyword_contexts}
        )
        trend_score = score_trend_strength(
            trends.get(keyword), document_count=len(document_ids)
        )
        clarity_score = score_context_clarity(
            keyword=keyword,
            contexts=keyword_contexts,
            mentions=keyword_mentions,
        )
        convertibility_score = score_travel_convertibility(
            rows=rows,
            entity_types=entity_types,
        )
        evidence = assess_evidence(
            rows=rows,
            contexts=keyword_contexts,
            document_count=len(document_ids),
            source_count=source_count,
            mentions=keyword_mentions,
            trend_links=keyword_links,
            entity_contexts=entity_contexts.get(keyword, []),
        )
        high_precision = calculate_high_precision_score(
            trend_strength=trend_score,
            context_clarity=clarity_score,
            travel_convertibility=convertibility_score,
            evidence_confidence=evidence.score,
        )
        ranking_status = classify_ranking(high_precision, evidence.gate)
        ranked.append(
            RankedCandidate(
                keyword=best.keyword,
                normalized_keyword=keyword,
                week_start=resolved_start,
                travel_category=best.travel_category,
                semantic_category=best.semantic_positive_category,
                semantic_status=best.semantic_status or "",
                semantic_travel_score=best.semantic_travel_score,
                travel_pre_score=round(best.travel_pre_score, 2),
                trend_strength_score=trend_score,
                context_clarity_score=clarity_score,
                travel_convertibility_score=convertibility_score,
                evidence_confidence_score=evidence.score,
                high_precision_score=high_precision,
                evidence_gate=evidence.gate,
                evidence_codes=evidence.codes,
                evidence_document_count=evidence.document_count,
                evidence_source_count=evidence.source_count,
                ranking_status=ranking_status,
                contexts=[context.combined_context for context in keyword_contexts[:5]],
                _document_ids=set(document_ids),
                _entity_keys=entity_keys,
            )
        )

    ranked.sort(key=lambda item: (-item.high_precision_score, item.normalized_keyword))
    for index, item in enumerate(ranked, start=1):
        item.rank_in_week = index
    assign_clusters(ranked)
    assign_gemini_budget(
        ranked,
        max_candidates=get_settings().travel_gemini_max_candidates_per_week,
    )

    if not dry_run and ranked:
        values_by_keyword = {
            item.normalized_keyword: _persistence_values(item)
            for item in ranked
        }
        if force or any(
            row.ranking_version != RANKING_VERSION for row in candidate_rows
        ):
            repo.save_rankings(
                session,
                values_by_keyword=values_by_keyword,
                week_start=resolved_start,
            )

    status_counts = Counter(item.ranking_status for item in ranked)
    gate_counts = Counter(item.evidence_gate for item in ranked)
    funnel_counts = repo.funnel_counts(session, week_start=resolved_start)
    funnel_counts["high_precision"] = sum(
        item.ranking_status != "rejected" for item in ranked
    )
    funnel_counts["gemini_eligible"] = sum(item.gemini_eligible for item in ranked)
    funnel = _funnel_payload(funnel_counts)
    estimate, insufficient = annualized_estimate(
        session,
        current_week=resolved_start,
        current_candidates=ranked,
    )
    return RankingResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        processed=len(ranked),
        rejected=status_counts["rejected"],
        review=status_counts["review"],
        gemini_candidates=status_counts["gemini_candidate"],
        priority_candidates=status_counts["priority_candidate"],
        evidence_pass=gate_counts["PASS"],
        needs_evidence=gate_counts["NEEDS_EVIDENCE"],
        evidence_reject=gate_counts["REJECT"],
        estimated_gemini_calls=sum(item.gemini_eligible for item in ranked),
        top_candidates=[serialize_ranked_candidate(item) for item in ranked[:20]],
        funnel=funnel,
        annualized_candidate_estimate=estimate,
        insufficient_history=insufficient,
    )


def score_trend_strength(
    trend: WeeklyTrend | None,
    *,
    document_count: int,
) -> float:
    if trend is None:
        return 0.0
    components: list[tuple[float, float]] = []
    _append_score(components, trend.final_score, 0.22)
    _append_score(components, trend.trend_score, 0.18)
    _append_score(components, trend.growth_score, 0.14)
    acceleration = getattr(trend, "acceleration", None)
    _append_score(components, _rate_score(acceleration), 0.10)
    _append_score(components, trend.persistence_score, 0.14)
    _append_score(components, min(trend.source_count / 3 * 100, 100), 0.10)
    _append_score(components, min(document_count / 3 * 100, 100), 0.07)
    if trend.search_interest_available and trend.search_interest_score is not None:
        _append_score(components, trend.search_interest_score, 0.10)
    status_score = 45.0 if trend.status == "watchlist" else 100.0
    _append_score(components, status_score, 0.05)
    if not components:
        return 0.0
    weight = sum(item_weight for _, item_weight in components)
    return _clamp_round(sum(score * item_weight for score, item_weight in components) / weight)


def score_context_clarity(
    *,
    keyword: str,
    contexts: Sequence[KeywordContext],
    mentions: Sequence[EntityMention],
) -> float:
    if not contexts:
        return 0.0
    normalized_keyword = _normalize_text(keyword)
    keyword_presence = _average(
        1.0 if normalized_keyword in _normalize_text(context.matched_sentence) else 0.0
        for context in contexts
    )
    length_quality = _average(
        1.0
        if 20 <= len(context.matched_sentence.strip()) <= 500
        and 30 <= len(context.combined_context.strip()) <= 1500
        else 0.35
        if len(context.matched_sentence.strip()) >= 10
        else 0.0
        for context in contexts
    )
    surrounding_scores = []
    for context in contexts:
        neighbors = [context.previous_sentence, context.next_sentence]
        similarities = [
            _text_similarity(context.matched_sentence, neighbor)
            for neighbor in neighbors
            if neighbor
        ]
        surrounding_scores.append(max(similarities, default=0.0))
    cross_document = _cross_document_similarity(contexts)
    entity_stability = _entity_stability(mentions)
    score = (
        keyword_presence * 30
        + length_quality * 20
        + _average(surrounding_scores) * 15
        + cross_document * 20
        + entity_stability * 15
    )
    score -= _duplicate_context_penalty(contexts)
    score -= _list_or_tag_penalty(contexts)
    if len({context.source for context in contexts}) >= 2 and cross_document < 0.12:
        score -= 10
    return _clamp_round(score)


def score_travel_convertibility(
    *,
    rows: Sequence[TravelOpportunityCandidate],
    entity_types: set[str],
) -> float:
    if not rows:
        return 0.0
    best = max(
        rows,
        key=lambda row: (
            row.semantic_travel_score if row.semantic_travel_score is not None else -1,
            row.travel_pre_score,
        ),
    )
    category_scores = {
        "FILM_LOCATION": 92,
        "DRAMA_LOCATION": 92,
        "SHOW_LOCATION": 85,
        "FESTIVAL": 100,
        "CONCERT": 95,
        "EXHIBITION": 90,
        "POPUP": 95,
        "SPORTS_EVENT": 88,
        "FOOD": 90,
        "LOCAL_CULTURE": 85,
        "LANDMARK": 95,
        "NATURE": 95,
        "REGIONAL_MEME": 72,
        "OTHER": 20,
    }
    components = [
        (_clamp(best.travel_pre_score), 0.30),
        (float(category_scores.get(best.travel_category, 35)), 0.25),
    ]
    if best.semantic_travel_score is not None:
        components.append((_clamp(best.semantic_travel_score), 0.30))
    entity_score = 100.0 if entity_types & {"LOCATION", "PLACE", "EVENT", "FOOD"} else 70.0 if "CONTENT_TITLE" in entity_types else 20.0
    components.append((entity_score, 0.10))
    positive_count = max(
        (len(_json_list(row.matched_positive_terms_json)) for row in rows),
        default=0,
    )
    components.append((min(positive_count / 4 * 100, 100), 0.05))
    weight = sum(item_weight for _, item_weight in components)
    score = sum(value * item_weight for value, item_weight in components) / weight
    negative_penalty = max((row.negative_context_penalty for row in rows), default=0.0)
    negative_categories = {
        (row.semantic_negative_category or "").upper() for row in rows
    }
    score -= min(negative_penalty, 40) * 0.75
    if negative_categories & NEGATIVE_SEMANTIC_CATEGORIES:
        score -= 45
    if best.travel_category not in CONCRETE_TRAVEL_CATEGORIES:
        score = min(score, 55)
    return _clamp_round(score)


def assess_evidence(
    *,
    rows: Sequence[TravelOpportunityCandidate],
    contexts: Sequence[KeywordContext],
    document_count: int,
    source_count: int,
    mentions: Sequence[EntityMention],
    trend_links: Sequence[TrendEntityLink],
    entity_contexts: Sequence[EntityContext],
) -> EvidenceAssessment:
    codes: set[str] = set()
    if document_count >= 2:
        codes.add("MULTI_DOCUMENT_CONFIRMATION")
    else:
        codes.add("SINGLE_DOCUMENT_ONLY")
    if source_count >= 2:
        codes.add("MULTI_SOURCE_CONFIRMATION")
    else:
        codes.add("SINGLE_SOURCE_ONLY")

    entity_types = {mention.entity_type for mention in mentions} | {
        link.entity_type for link in trend_links
    }
    confidence_values = [mention.confidence for mention in mentions] + [
        min(link.average_confidence, 1.0) for link in trend_links
    ]
    if "LOCATION" in entity_types:
        codes.add("LOCATION_EVIDENCE")
    if "PLACE" in entity_types:
        codes.add("PLACE_EVIDENCE")
    if "CONTENT_TITLE" in entity_types:
        codes.add("CONTENT_TITLE_CONTEXT")
    if "EVENT" in entity_types and entity_types & {"LOCATION", "PLACE"}:
        codes.add("EVENT_LOCATION_PAIR")
    if "FOOD" in entity_types and entity_types & {"LOCATION", "PLACE"}:
        codes.add("FOOD_LOCATION_PAIR")

    categories = {row.travel_category for row in rows}
    if categories & {"FILM_LOCATION", "DRAMA_LOCATION", "SHOW_LOCATION"}:
        codes.add("FILM_CONTEXT")
    eligible_contexts = []
    for context in entity_contexts:
        if context.match_status == "manual" or context.provider in {"manual", "namuwiki_manual"} and context.match_status == "manual":
            codes.add("MANUAL_CONTEXT")
            eligible_contexts.append(context)
        elif context.provider == "wikipedia_ko" and context.match_status == "matched":
            codes.add("MATCHED_CONTEXT")
            eligible_contexts.append(context)

    location_evidence = bool(entity_types & {"LOCATION", "PLACE"})
    if not location_evidence:
        codes.add("NO_LOCATION_EVIDENCE")
    repeat_score = _cross_document_similarity(contexts)
    if len(contexts) >= 2 and repeat_score < 0.12:
        codes.add("AMBIGUOUS_CONTEXT")

    score = min(document_count / 3 * 25, 25)
    score += min(source_count / 2 * 20, 20)
    score += repeat_score * 15
    if "EVENT_LOCATION_PAIR" in codes or "FOOD_LOCATION_PAIR" in codes:
        score += 20
    elif location_evidence:
        score += 15
    elif "CONTENT_TITLE_CONTEXT" in codes:
        score += 8
    if confidence_values:
        score += min(_average(confidence_values), 1.0) * 5
    if "MANUAL_CONTEXT" in codes:
        score += 20
    elif "MATCHED_CONTEXT" in codes:
        score += 15

    negative_categories = {
        (row.semantic_negative_category or "").upper() for row in rows
    }
    negative_dominant = bool(negative_categories & NEGATIVE_SEMANTIC_CATEGORIES)
    if negative_dominant:
        codes.add("NEGATIVE_SEMANTIC_DOMINANT")
    best = max(rows, key=lambda row: row.travel_pre_score)
    semantic_scores = [
        row.semantic_travel_score
        for row in rows
        if row.semantic_travel_score is not None
    ]
    semantic_score = max(semantic_scores, default=0.0)
    category_unclear = best.travel_category == "OTHER"
    no_linked_evidence = document_count == 0 or not contexts
    if negative_dominant or no_linked_evidence or category_unclear or (
        semantic_score < 45 and best.travel_pre_score < 65
    ):
        gate = "REJECT"
    elif best.travel_category in LOCATION_REQUIRED_CATEGORIES and not location_evidence:
        gate = "NEEDS_EVIDENCE"
    elif score >= 65 and (location_evidence or bool(eligible_contexts)):
        gate = "PASS"
    else:
        gate = "NEEDS_EVIDENCE"
    if gate != "PASS":
        codes.add("INSUFFICIENT_EVIDENCE")
    return EvidenceAssessment(
        score=_clamp_round(score),
        gate=gate,
        codes=sorted(codes),
        document_count=document_count,
        source_count=source_count,
    )


def calculate_high_precision_score(
    *,
    trend_strength: float,
    context_clarity: float,
    travel_convertibility: float,
    evidence_confidence: float,
) -> float:
    return _clamp_round(
        trend_strength * 0.30
        + context_clarity * 0.20
        + travel_convertibility * 0.30
        + evidence_confidence * 0.20
    )


def classify_ranking(score: float, evidence_gate: str) -> str:
    if evidence_gate == "REJECT" or score < 70:
        return "rejected"
    if score < 85:
        return "review"
    if score < 90:
        return "gemini_candidate"
    if evidence_gate == "PASS":
        return "priority_candidate"
    return "gemini_candidate"


def assign_clusters(candidates: list[RankedCandidate]) -> None:
    parents = list(range(len(candidates)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    canonical = [_cluster_key(item) for item in candidates]
    for left in range(len(candidates)):
        for right in range(left + 1, len(candidates)):
            shared_documents = candidates[left]._document_ids & candidates[right]._document_ids
            shared_entities = candidates[left]._entity_keys & candidates[right]._entity_keys
            similar_keyword = _text_similarity(canonical[left], canonical[right]) >= 0.72
            if canonical[left] == canonical[right] or shared_documents or (
                shared_entities and similar_keyword
            ):
                union(left, right)

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(candidates)):
        groups[find(index)].append(index)
    for indices in groups.values():
        representative_index = max(
            indices,
            key=lambda index: (
                candidates[index].high_precision_score,
                candidates[index].travel_convertibility_score,
                candidates[index].normalized_keyword,
            ),
        )
        cluster_seed = min(canonical[index] for index in indices)
        cluster_id = "cluster-" + hashlib.sha256(cluster_seed.encode("utf-8")).hexdigest()[:12]
        for index in indices:
            candidates[index].cluster_id = cluster_id
            candidates[index].cluster_representative = index == representative_index


def assign_gemini_budget(
    candidates: list[RankedCandidate], *, max_candidates: int
) -> None:
    for item in candidates:
        item.gemini_eligible = False
    eligible = [
        item
        for item in candidates
        if item.cluster_representative
        and item.ranking_status in {"priority_candidate", "gemini_candidate"}
        and item.evidence_gate != "REJECT"
    ]
    eligible.sort(
        key=lambda item: (
            0 if item.ranking_status == "priority_candidate" else 1,
            -item.high_precision_score,
            item.normalized_keyword,
        )
    )
    for item in eligible[: max(0, max_candidates)]:
        item.gemini_eligible = True


def annualized_estimate(
    session: Session,
    *,
    current_week: date | None = None,
    current_candidates: Sequence[RankedCandidate] = (),
) -> tuple[float, bool]:
    history = repo.ranked_history(session)
    weeks = {row[0] for row in history}
    candidate_count = sum(
        row[1] in {"gemini_candidate", "priority_candidate"} for row in history
    )
    if current_week is not None:
        weeks.add(current_week)
        candidate_count += sum(
            item.ranking_status in {"gemini_candidate", "priority_candidate"}
            for item in current_candidates
        )
    if not weeks:
        return 0.0, True
    observed_weeks = max(((max(weeks) - min(weeks)).days // 7) + 1, len(weeks))
    estimate = round(candidate_count / observed_weeks * 52, 2)
    return estimate, observed_weeks < 4


def calibration_report(
    session: Session, *, week_start: date | None
) -> dict[str, object]:
    resolved_start, _ = opportunity_repo.resolve_week_range(session, week_start)
    rows = repo.get_ranked_candidates(session, week_start=resolved_start)
    distinct: dict[str, TravelOpportunityCandidate] = {}
    for row in rows:
        distinct.setdefault(row.normalized_keyword, row)
    candidates = list(distinct.values())
    ranking_counts = Counter(row.ranking_status for row in candidates)
    gate_counts = Counter(row.evidence_gate for row in candidates)
    distribution = {"0_69": 0, "70_84": 0, "85_89": 0, "90_100": 0}
    for row in candidates:
        score = row.high_precision_score or 0
        if score < 70:
            distribution["0_69"] += 1
        elif score < 85:
            distribution["70_84"] += 1
        elif score < 90:
            distribution["85_89"] += 1
        else:
            distribution["90_100"] += 1
    funnel_counts = repo.funnel_counts(session, week_start=resolved_start)
    estimate, insufficient = annualized_estimate(session)
    top = sorted(
        candidates,
        key=lambda row: (-(row.high_precision_score or 0), row.normalized_keyword),
    )[:20]
    return {
        "ranking_version": RANKING_VERSION,
        "week_start": resolved_start,
        "thresholds": RANKING_THRESHOLDS,
        "total_semantic_candidates": funnel_counts["semantic"],
        "rejected": ranking_counts["rejected"],
        "review": ranking_counts["review"],
        "gemini_candidate": ranking_counts["gemini_candidate"],
        "priority_candidate": ranking_counts["priority_candidate"],
        "evidence_gate_counts": {
            "PASS": gate_counts["PASS"],
            "NEEDS_EVIDENCE": gate_counts["NEEDS_EVIDENCE"],
            "REJECT": gate_counts["REJECT"],
        },
        "score_distribution": distribution,
        "top_20_candidates": [_serialize_persisted(row) for row in top],
        "annualized_candidate_estimate": estimate,
        "insufficient_history": insufficient,
        "weekly_gemini_budget": get_settings().travel_gemini_max_candidates_per_week,
        "estimated_llm_calls": funnel_counts["gemini_eligible"],
        "overall_reduction_rate": _reduction_rate(
            funnel_counts["raw"], funnel_counts["gemini_eligible"]
        ),
        "funnel": _funnel_payload(funnel_counts),
    }


def serialize_ranking_result(result: RankingResult) -> dict[str, object]:
    return asdict(result)


def serialize_ranked_candidate(item: RankedCandidate) -> dict[str, object]:
    return {
        key: value
        for key, value in asdict(item).items()
        if not key.startswith("_")
    }


def _serialize_persisted(row: TravelOpportunityCandidate) -> dict[str, object]:
    context = row.keyword_context
    return {
        "keyword": row.keyword,
        "normalized_keyword": row.normalized_keyword,
        "week_start": row.week_start,
        "travel_category": row.travel_category,
        "semantic_category": row.semantic_positive_category,
        "semantic_status": row.semantic_status,
        "semantic_travel_score": row.semantic_travel_score,
        "travel_pre_score": row.travel_pre_score,
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
        "cluster_id": row.cluster_id,
        "cluster_representative": bool(row.cluster_representative),
        "gemini_eligible": bool(row.gemini_eligible),
        "contexts": [context.combined_context] if context else [],
    }


def _persistence_values(item: RankedCandidate) -> dict[str, object]:
    from app.repositories.travel_opportunity_repository import utc_now

    return {
        "trend_strength_score": item.trend_strength_score,
        "context_clarity_score": item.context_clarity_score,
        "travel_convertibility_score": item.travel_convertibility_score,
        "evidence_confidence_score": item.evidence_confidence_score,
        "high_precision_score": item.high_precision_score,
        "evidence_gate": item.evidence_gate,
        "evidence_codes_json": json.dumps(item.evidence_codes, ensure_ascii=False),
        "evidence_document_count": item.evidence_document_count,
        "evidence_source_count": item.evidence_source_count,
        "ranking_status": item.ranking_status,
        "rank_in_week": item.rank_in_week,
        "ranking_version": RANKING_VERSION,
        "calculated_at": utc_now(),
        "cluster_id": item.cluster_id,
        "cluster_representative": item.cluster_representative,
        "gemini_eligible": item.gemini_eligible,
    }


def _cluster_key(item: RankedCandidate) -> str:
    key = _normalize_text(item.normalized_keyword)
    location_entities = sorted(
        normalized
        for normalized, entity_type in item._entity_keys
        if entity_type in {"LOCATION", "PLACE"}
    )
    for entity in location_entities:
        normalized_entity = _normalize_text(entity)
        if key.startswith(normalized_entity) and len(key) > len(normalized_entity) + 2:
            key = key[len(normalized_entity):]
    changed = True
    while changed:
        changed = False
        for suffix in CLUSTER_SUFFIXES:
            normalized_suffix = _normalize_text(suffix)
            if key.endswith(normalized_suffix) and len(key) > len(normalized_suffix):
                key = key[: -len(normalized_suffix)]
                changed = True
    return key or _normalize_text(item.normalized_keyword)


def _cross_document_similarity(contexts: Sequence[KeywordContext]) -> float:
    pairs = []
    for left, left_context in enumerate(contexts):
        for right_context in contexts[left + 1:]:
            if left_context.document_id == right_context.document_id:
                continue
            pairs.append(
                _text_similarity(
                    left_context.combined_context,
                    right_context.combined_context,
                )
            )
    return max(pairs, default=0.0)


def _duplicate_context_penalty(contexts: Sequence[KeywordContext]) -> float:
    keys = [
        (context.document_id, _normalize_text(context.combined_context))
        for context in contexts
    ]
    if not keys:
        return 0.0
    duplicate_ratio = 1 - len(set(keys)) / len(keys)
    return duplicate_ratio * 20


def _list_or_tag_penalty(contexts: Sequence[KeywordContext]) -> float:
    penalties = []
    for context in contexts:
        sentence = context.matched_sentence.strip()
        separator_count = sum(sentence.count(mark) for mark in (",", "#", "|", "/"))
        penalties.append(12.0 if separator_count >= 4 else 6.0 if sentence.startswith("#") else 0.0)
    return _average(penalties)


def _entity_stability(mentions: Sequence[EntityMention]) -> float:
    if not mentions:
        return 0.0
    counts = Counter(mention.entity_type for mention in mentions)
    return max(counts.values()) / sum(counts.values())


def _relevant_mentions(
    *,
    keyword: str,
    contexts: Sequence[KeywordContext],
    mentions: Sequence[EntityMention],
) -> list[EntityMention]:
    context_by_document: dict[int, str] = defaultdict(str)
    for context in contexts:
        context_by_document[context.document_id] += " " + _normalize_text(
            context.combined_context
        )
    normalized_keyword = _normalize_text(keyword)
    relevant = []
    for mention in mentions:
        normalized_text = _normalize_text(mention.text)
        normalized_entity = _normalize_text(mention.normalized_text)
        document_context = context_by_document.get(mention.document_id, "")
        if (
            normalized_entity == normalized_keyword
            or normalized_text and normalized_text in document_context
            or normalized_entity and normalized_entity in document_context
        ):
            relevant.append(mention)
    return relevant


def _text_similarity(left: str | None, right: str | None) -> float:
    if not left or not right:
        return 0.0
    left_normalized = _normalize_text(left)
    right_normalized = _normalize_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    left_grams = _ngrams(left_normalized)
    right_grams = _ngrams(right_normalized)
    union = left_grams | right_grams
    return len(left_grams & right_grams) / len(union) if union else 0.0


def _ngrams(value: str, size: int = 2) -> set[str]:
    if len(value) <= size:
        return {value}
    return {value[index:index + size] for index in range(len(value) - size + 1)}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]", "", value).lower()


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def _append_score(
    components: list[tuple[float, float]], value: float | None, weight: float
) -> None:
    if value is not None:
        components.append((_clamp(float(value)), weight))


def _rate_score(value: float | None) -> float | None:
    if value is None:
        return None
    return _clamp(value * 100 if -1 <= value <= 1 else value)


def _average(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(100.0, value))


def _clamp_round(value: float) -> float:
    return round(_clamp(value), 2)


def _reduction_rate(raw: int, calls: int) -> float:
    return round((1 - calls / raw) * 100, 2) if raw else 0.0


def _funnel_payload(counts: dict[str, int]) -> dict[str, object]:
    return {
        "raw_keyword": counts["raw"],
        "keyword_quality_passed": counts["quality"],
        "rule_candidate": counts["rule"],
        "semantic_candidate": counts["semantic"],
        "high_precision_candidate": counts["high_precision"],
        "gemini_eligible": counts["gemini_eligible"],
        "llm_reduction_rate": _reduction_rate(
            counts["raw"], counts["gemini_eligible"]
        ),
    }


def _empty_result(*, dry_run: bool, week_start: date | None) -> RankingResult:
    return RankingResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=week_start,
        processed=0,
        rejected=0,
        review=0,
        gemini_candidates=0,
        priority_candidates=0,
        evidence_pass=0,
        needs_evidence=0,
        evidence_reject=0,
        estimated_gemini_calls=0,
        top_candidates=[],
        funnel=_funnel_payload(
            {
                "raw": 0,
                "quality": 0,
                "rule": 0,
                "semantic": 0,
                "high_precision": 0,
                "gemini_eligible": 0,
            }
        ),
        annualized_candidate_estimate=0.0,
        insufficient_history=True,
    )

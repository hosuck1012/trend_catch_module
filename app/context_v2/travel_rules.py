from dataclasses import dataclass
import json
from pathlib import Path

from app.config import get_settings
from app.context_v2.travel_taxonomy import (
    ACCIDENT_TERMS,
    CONCERT_TERMS,
    DRAMA_TERMS,
    ENTITY_PRIORS,
    EXHIBITION_TERMS,
    FESTIVAL_TERMS,
    FILM_TERMS,
    FINANCE_TERMS,
    FOOD_TERMS,
    LANDMARK_TERMS,
    LEGAL_TERMS,
    LOCAL_CULTURE_TERMS,
    NATURE_TERMS,
    POPUP_TERMS,
    PREFILTER_REJECTED,
    PREFILTER_REVIEW,
    PREFILTER_STRONG,
    PREFILTER_WEAK,
    SHOW_TERMS,
    SPORTS_TERMS,
    TravelCategory,
)


@dataclass(frozen=True)
class EntitySignal:
    text: str
    normalized_text: str
    entity_type: str
    relation_score: float = 0.0
    is_primary: bool = False


@dataclass(frozen=True)
class TrendSignal:
    weekly_mentions: int = 0
    final_score: float | None = None
    source_count: int = 0
    document_count: int = 0


@dataclass(frozen=True)
class TravelRuleResult:
    primary_entity: str | None
    primary_entity_type: str | None
    travel_category: str
    entity_prior_score: float
    positive_context_score: float
    negative_context_penalty: float
    trend_evidence_score: float
    source_diversity_score: float
    travel_pre_score: float
    prefilter_status: str
    matched_positive_terms: list[str]
    matched_negative_terms: list[str]
    reasoning_codes: list[str]


def load_terms(filename: str) -> list[str]:
    path = Path(__file__).resolve().parents[2] / "data" / filename
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    return [str(term) for term in payload.get("terms", []) if str(term).strip()]


def evaluate_travel_rules(
    *,
    keyword: str,
    context: str,
    entities: list[EntitySignal],
    trend: TrendSignal,
    positive_terms: list[str] | None = None,
    negative_terms: list[str] | None = None,
) -> TravelRuleResult:
    settings = get_settings()
    positive_terms = positive_terms if positive_terms is not None else load_terms("travel_positive_terms.json")
    negative_terms = negative_terms if negative_terms is not None else load_terms("travel_negative_terms.json")
    text = f"{keyword} {context}".lower()
    matched_positive = _matched_terms(text, positive_terms)
    matched_negative = _matched_terms(text, negative_terms)
    primary = _primary_entity(entities, keyword)
    primary_matches_keyword = _matches_keyword(primary, keyword)
    entity_type = primary.entity_type if primary else None
    entity_prior = float(ENTITY_PRIORS.get(entity_type or "", 0)) if primary_matches_keyword else 0.0
    positive_score = _positive_score(matched_positive, entity_type if primary_matches_keyword else None)
    negative_penalty = _negative_penalty(matched_negative, entity_type)
    trend_score = _trend_evidence_score(trend)
    diversity_score = min(15.0, max(float(trend.source_count), 0.0) * 5.0)
    category, category_codes = _category_and_codes(
        entity_type,
        matched_positive,
        entities,
        primary_matches_keyword,
    )
    reasoning_codes = list(category_codes)
    if entity_type == "BRAND" and set(matched_negative) & FINANCE_TERMS:
        reasoning_codes.append("FINANCE_CONTEXT")
    if entity_type == "PERSON" and set(matched_negative) & (LEGAL_TERMS | ACCIDENT_TERMS):
        reasoning_codes.append("LEGAL_CONTEXT")
    if (not matched_positive or not primary_matches_keyword) and entity_prior < 25:
        reasoning_codes.append("NO_TRAVEL_SIGNAL")
    if trend.source_count < 2:
        reasoning_codes.append("LOW_SOURCE_DIVERSITY")
    raw_score = entity_prior + positive_score + trend_score + diversity_score - negative_penalty
    travel_score = round(min(100.0, max(0.0, raw_score)), 2)
    status = _status(travel_score, settings)
    if negative_penalty >= 40 and entity_type in {"BRAND", "PERSON"}:
        travel_score = min(travel_score, settings.travel_prefilter_min_score - 1)
        status = PREFILTER_REJECTED
    return TravelRuleResult(
        primary_entity=primary.text if primary else None,
        primary_entity_type=entity_type,
        travel_category=category.value,
        entity_prior_score=entity_prior,
        positive_context_score=positive_score,
        negative_context_penalty=negative_penalty,
        trend_evidence_score=trend_score,
        source_diversity_score=diversity_score,
        travel_pre_score=travel_score,
        prefilter_status=status,
        matched_positive_terms=matched_positive,
        matched_negative_terms=matched_negative,
        reasoning_codes=sorted(set(reasoning_codes)),
    )


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    seen: list[str] = []
    for term in terms:
        normalized = term.lower().strip()
        if len(normalized) == 1:
            continue
        if normalized and normalized in text and term not in seen:
            seen.append(term)
    return seen


def _primary_entity(entities: list[EntitySignal], keyword: str) -> EntitySignal | None:
    if not entities:
        return None
    normalized_keyword = keyword.lower().replace(" ", "")
    return max(
        entities,
        key=lambda entity: (
            entity.normalized_text.lower().replace(" ", "") == normalized_keyword,
            entity.is_primary,
            ENTITY_PRIORS.get(entity.entity_type, 0),
            entity.relation_score,
        ),
    )


def _matches_keyword(entity: EntitySignal | None, keyword: str) -> bool:
    if entity is None:
        return False
    normalized_keyword = keyword.lower().replace(" ", "")
    return entity.normalized_text.lower().replace(" ", "") == normalized_keyword


def _positive_score(matched_positive: list[str], entity_type: str | None) -> float:
    score = min(30.0, len(matched_positive) * 5.0)
    if entity_type in {"PLACE", "LOCATION", "EVENT"} and matched_positive:
        score = min(30.0, score + 5.0)
    return score


def _negative_penalty(matched_negative: list[str], entity_type: str | None) -> float:
    penalty = min(30.0, len(matched_negative) * 10.0)
    negatives = set(matched_negative)
    if entity_type == "BRAND" and negatives & FINANCE_TERMS:
        penalty = 40.0
    if entity_type == "PERSON" and negatives & (LEGAL_TERMS | ACCIDENT_TERMS):
        penalty = 40.0
    return penalty


def _trend_evidence_score(trend: TrendSignal) -> float:
    if trend.final_score is not None:
        return round(min(20.0, max(0.0, trend.final_score) * 0.2), 2)
    mention_score = min(12.0, trend.weekly_mentions * 3.0)
    document_score = min(8.0, trend.document_count * 2.0)
    return round(min(20.0, mention_score + document_score), 2)


def _category_and_codes(
    entity_type: str | None,
    matched_positive: list[str],
    entities: list[EntitySignal],
    primary_matches_keyword: bool,
) -> tuple[TravelCategory, list[str]]:
    if not primary_matches_keyword:
        return TravelCategory.OTHER, []
    terms = set(matched_positive)
    has_location = any(entity.entity_type in {"LOCATION", "PLACE"} for entity in entities)
    if entity_type == "CONTENT_TITLE":
        if terms & DRAMA_TERMS:
            return TravelCategory.DRAMA_LOCATION, ["CONTENT_TITLE_WITH_DRAMA_CONTEXT"]
        if terms & SHOW_TERMS:
            return TravelCategory.SHOW_LOCATION, ["CONTENT_TITLE_WITH_SHOW_CONTEXT"]
        if terms & FILM_TERMS:
            return TravelCategory.FILM_LOCATION, ["CONTENT_TITLE_WITH_FILM_CONTEXT"]
    if entity_type == "EVENT":
        if terms & FESTIVAL_TERMS:
            return TravelCategory.FESTIVAL, ["EVENT_WITH_LOCATION"]
        if terms & CONCERT_TERMS:
            return TravelCategory.CONCERT, ["EVENT_WITH_LOCATION"]
        if terms & EXHIBITION_TERMS:
            return TravelCategory.EXHIBITION, ["EVENT_WITH_LOCATION"]
        if terms & POPUP_TERMS:
            return TravelCategory.POPUP, ["EVENT_WITH_LOCATION"]
        if terms & SPORTS_TERMS:
            return TravelCategory.SPORTS_EVENT, ["EVENT_WITH_LOCATION"]
        return TravelCategory.LOCAL_CULTURE, ["EVENT_WITH_LOCATION"] if has_location else []
    if entity_type == "FOOD":
        return TravelCategory.FOOD, ["FOOD_TREND"]
    if entity_type in {"PLACE", "LOCATION"}:
        if terms & NATURE_TERMS:
            return TravelCategory.NATURE, ["PLACE_TREND"]
        if terms & LANDMARK_TERMS:
            return TravelCategory.LANDMARK, ["PLACE_TREND"]
        if terms & LOCAL_CULTURE_TERMS:
            return TravelCategory.LOCAL_CULTURE, ["PLACE_TREND"]
        return TravelCategory.LANDMARK, ["PLACE_TREND"]
    if entity_type == "MEME" and has_location:
        return TravelCategory.REGIONAL_MEME, ["REGIONAL_MEME"]
    return TravelCategory.OTHER, []


def _status(score: float, settings) -> str:
    if score >= settings.travel_prefilter_strong_score:
        return PREFILTER_STRONG
    if score >= settings.travel_prefilter_review_score:
        return PREFILTER_REVIEW
    if score >= settings.travel_prefilter_min_score:
        return PREFILTER_WEAK
    return PREFILTER_REJECTED

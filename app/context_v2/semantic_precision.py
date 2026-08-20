from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

from app.context_v2.semantic_scorer import (
    classify_semantic_score,
    semantic_confidence_from_margin,
    semantic_score_from_similarities,
)
from app.context_v2.travel_taxonomy import TravelCategory
from app.keywords.keyword_normalizer import normalize_keyword
from app.keywords.tokenizer import (
    KiwiTokenizerAdapter,
    ResilientTokenizer,
    Token,
    Tokenizer,
)


HIGH_VALUE_ENTITY_TYPES = {
    "LOCATION",
    "PLACE",
    "EVENT",
    "FOOD",
    "CONTENT_TITLE",
    "MEME",
}
WEAK_ENTITY_TYPES = {"BRAND", "PERSON"}
SPECIFIC_TOPIC_TERMS = {
    "촬영지",
    "불꽃축제",
    "축제",
    "페스티벌",
    "콘서트",
    "공연",
    "전시",
    "팝업",
    "초콜릿",
    "카페",
    "맛집",
    "오름",
    "해수욕장",
    "마라톤",
}
LOW_MARGIN_THRESHOLD = 0.005
COHERENT_TOPIC_BONUS = 0.08
SCORE_EPSILON = 1e-4
DEFAULT_GENERIC_TOPIC_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "travel_generic_topic_terms.json"
)


class QualitySignal(Protocol):
    candidate_type: str
    extractor: str
    quality_score: float
    title_occurrence: int
    entity_type: str | None
    entity_confidence: float | None
    accepted: bool
    rejection_reason: str | None
    pipeline_version: str


class EntitySignal(Protocol):
    text: str
    normalized_text: str
    entity_type: str
    confidence: float


@dataclass(frozen=True)
class GenericTopicTerms:
    version: str
    terms: frozenset[str]
    content_hash: str


@dataclass(frozen=True)
class SemanticPrecisionEvidence:
    topic_specificity_pass: bool
    generic_topic: bool
    brand_person_only: bool
    entity_types: frozenset[str]
    context: str
    travel_category: str
    topic_codes: tuple[str, ...]
    cache_signature: str


@dataclass(frozen=True)
class CalibratedSemanticScore:
    semantic_travel_score: float
    semantic_status: str
    semantic_margin: float
    semantic_confidence: float
    category_coherence_pass: bool
    reasoning_codes: tuple[str, ...]


@lru_cache(maxsize=4)
def load_generic_topic_terms(
    path: Path = DEFAULT_GENERIC_TOPIC_PATH,
) -> GenericTopicTerms:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("version", "")).strip()
    raw_terms = payload.get("terms")
    if not version or not isinstance(raw_terms, list):
        raise ValueError("Generic topic terms require a version and terms list")
    terms = frozenset(
        normalized
        for value in raw_terms
        if isinstance(value, str)
        and (normalized := normalize_keyword(value)) is not None
    )
    if not terms:
        raise ValueError("Generic topic terms must not be empty")
    canonical = json.dumps(
        {"version": version, "terms": sorted(terms)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return GenericTopicTerms(
        version=version,
        terms=terms,
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@lru_cache(maxsize=1)
def get_topic_tokenizer() -> Tokenizer:
    return ResilientTokenizer(KiwiTokenizerAdapter())


def build_semantic_precision_evidence(
    candidate: object,
    *,
    quality_signal: QualitySignal | None,
    context_entities: Sequence[EntitySignal],
    generic_terms: GenericTopicTerms | None = None,
    tokenizer: Tokenizer | None = None,
) -> SemanticPrecisionEvidence:
    generic_terms = generic_terms or load_generic_topic_terms()
    tokenizer = tokenizer or get_topic_tokenizer()
    normalized = normalize_keyword(candidate.normalized_keyword or candidate.keyword) or ""
    context = str(candidate.keyword_context.combined_context or "")
    normalized_context = normalize_keyword(context) or ""
    scoped_entities = tuple(
        entity
        for entity in context_entities
        if _entity_is_in_context(entity, context, normalized_context)
    )
    tokens = tokenizer.tokenize(candidate.keyword)
    meaningful_tokens = tuple(token for token in tokens if _is_meaningful_token(token))
    token_count = len(meaningful_tokens)
    primary_matches_keyword = _normalized_matches_keyword(
        candidate.primary_entity,
        normalized,
    )
    primary_type = (
        str(candidate.primary_entity_type or "") if primary_matches_keyword else ""
    )
    quality_type = str(quality_signal.entity_type or "") if quality_signal else ""
    matching_context_types = {
        str(entity.entity_type or "")
        for entity in scoped_entities
        if _normalized_matches_keyword(entity.normalized_text or entity.text, normalized)
    }
    entity_types = frozenset(
        value
        for value in (
            primary_type,
            quality_type,
            *(str(entity.entity_type or "") for entity in scoped_entities),
        )
        if value
    )
    high_value_entity = bool(
        {primary_type, quality_type, *matching_context_types}
        & HIGH_VALUE_ENTITY_TYPES
    )
    proper_topic = _has_proper_topic_signal(
        tokens=meaningful_tokens,
        quality_signal=quality_signal,
    )
    generic_components = {
        normalize_keyword(token.text)
        for token in meaningful_tokens
        if normalize_keyword(token.text) in generic_terms.terms
    }
    is_generic_term = normalized in generic_terms.terms
    multi_token_topic = bool(
        token_count >= 2
        and not generic_components
        and (
            high_value_entity
            or proper_topic
            or any(token.tag == "NNP" for token in meaningful_tokens)
            or _contains_any(normalized, SPECIFIC_TOPIC_TERMS)
        )
    )
    generic_single_token = token_count <= 1 and not high_value_entity and not proper_topic
    generic_topic = is_generic_term or generic_single_token or bool(
        generic_components and not (high_value_entity or proper_topic)
    )

    codes: set[str] = set()
    if is_generic_term or generic_single_token or generic_components:
        codes.add("GENERIC_TOPIC")
    if high_value_entity:
        codes.add("HIGH_VALUE_ENTITY")
    if multi_token_topic:
        codes.add("MULTI_TOKEN_TOPIC")
    weak_types = {value for value in (primary_type, quality_type) if value in WEAK_ENTITY_TYPES}
    brand_person_only = bool(weak_types) and not (
        entity_types & HIGH_VALUE_ENTITY_TYPES
    )
    if "BRAND" in weak_types:
        codes.add("BRAND_ONLY")
    if "PERSON" in weak_types:
        codes.add("PERSON_ONLY")

    topic_specificity_pass = not generic_topic and (
        high_value_entity or multi_token_topic or proper_topic
    )
    codes.add(
        "TOPIC_SPECIFICITY_PASS"
        if topic_specificity_pass
        else "TOPIC_SPECIFICITY_FAIL"
    )
    payload = {
        "generic_terms": generic_terms.content_hash,
        "keyword": candidate.keyword,
        "normalized_keyword": normalized,
        "tokens": [(token.text, token.tag) for token in meaningful_tokens],
        "primary_entity": candidate.primary_entity,
        "primary_entity_type": candidate.primary_entity_type,
        "primary_entity_matches_keyword": primary_matches_keyword,
        "travel_category": candidate.travel_category,
        "matched_positive_terms": _json_terms(candidate.matched_positive_terms_json),
        "quality": _quality_payload(quality_signal),
        "entities": sorted(
            (
                entity.normalized_text,
                entity.entity_type,
                round(float(entity.confidence), 6),
            )
            for entity in scoped_entities
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SemanticPrecisionEvidence(
        topic_specificity_pass=topic_specificity_pass,
        generic_topic=generic_topic,
        brand_person_only=brand_person_only,
        entity_types=entity_types,
        context=context,
        travel_category=str(candidate.travel_category),
        topic_codes=tuple(sorted(codes)),
        cache_signature=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def calibrate_semantic_score(
    *,
    positive_similarity: float,
    positive_category: str,
    negative_similarity: float,
    negative_category: str,
    evidence: SemanticPrecisionEvidence,
    reject_threshold: float,
    review_threshold: float,
    strong_threshold: float,
) -> CalibratedSemanticScore:
    margin = positive_similarity - negative_similarity
    confidence = semantic_confidence_from_margin(margin)
    score = semantic_score_from_similarities(
        positive_similarity,
        negative_similarity,
        negative_category=negative_category,
    )
    category_pass, category_codes = _category_coherence(
        positive_category=positive_category,
        evidence=evidence,
    )
    reasoning_codes = set(evidence.topic_codes) | category_codes
    status_ceiling: str | None = None
    if margin <= 0:
        reasoning_codes.add("NEGATIVE_SEMANTIC_DOMINANT")
        score = 0.0
    elif margin < LOW_MARGIN_THRESHOLD:
        reasoning_codes.add("LOW_SEMANTIC_MARGIN")
        score = min(score, max(0.0, review_threshold - SCORE_EPSILON))
        status_ceiling = "semantic_weak"
    else:
        reasoning_codes.add("POSITIVE_SEMANTIC_MARGIN")

    if margin > 0 and evidence.topic_specificity_pass and category_pass:
        score = min(1.0, score + COHERENT_TOPIC_BONUS)
    if not evidence.topic_specificity_pass:
        score = 0.0
        status_ceiling = "semantic_rejected"
    if not category_pass:
        score = min(score, max(0.0, review_threshold - SCORE_EPSILON))
        if status_ceiling != "semantic_rejected":
            status_ceiling = "semantic_weak"
    if evidence.brand_person_only and not category_pass:
        score = min(score, max(0.0, review_threshold - SCORE_EPSILON))
        if status_ceiling != "semantic_rejected":
            status_ceiling = "semantic_weak"

    status = classify_semantic_score(
        score,
        reject_threshold=reject_threshold,
        review_threshold=review_threshold,
        strong_threshold=strong_threshold,
    )
    if status_ceiling == "semantic_rejected":
        status = "semantic_rejected"
    elif status_ceiling == "semantic_weak" and status in {
        "semantic_review",
        "semantic_strong",
    }:
        status = "semantic_weak"
    return CalibratedSemanticScore(
        semantic_travel_score=round(score * 100, 2),
        semantic_status=status,
        semantic_margin=round(margin, 6),
        semantic_confidence=round(confidence, 6),
        category_coherence_pass=category_pass,
        reasoning_codes=tuple(sorted(reasoning_codes)),
    )


def _category_coherence(
    *,
    positive_category: str,
    evidence: SemanticPrecisionEvidence,
) -> tuple[bool, set[str]]:
    context = evidence.context.lower()
    entity_types = evidence.entity_types
    codes: set[str] = set()
    entity_match = False
    context_match = False

    if positive_category in {
        TravelCategory.FILM_LOCATION,
        TravelCategory.DRAMA_LOCATION,
        TravelCategory.SHOW_LOCATION,
    }:
        entity_match = "CONTENT_TITLE" in entity_types
        terms = {
            TravelCategory.FILM_LOCATION: ("영화", "촬영지", "로케이션", "개봉"),
            TravelCategory.DRAMA_LOCATION: ("드라마", "촬영지", "로케이션"),
            TravelCategory.SHOW_LOCATION: ("예능", "방송", "촬영지"),
        }[TravelCategory(positive_category)]
        context_match = _contains_any(context, terms)
    elif positive_category == TravelCategory.FESTIVAL:
        entity_match = "EVENT" in entity_types
        context_match = _contains_any(context, ("축제", "페스티벌"))
    elif positive_category == TravelCategory.CONCERT:
        entity_match = "EVENT" in entity_types
        context_match = _contains_any(context, ("콘서트", "공연", "라이브"))
    elif positive_category == TravelCategory.EXHIBITION:
        entity_match = "EVENT" in entity_types
        context_match = _contains_any(context, ("전시", "전시회", "미술관", "박물관"))
    elif positive_category == TravelCategory.POPUP:
        entity_match = "EVENT" in entity_types
        context_match = _contains_any(context, ("팝업", "팝업스토어"))
    elif positive_category == TravelCategory.FOOD:
        entity_match = "FOOD" in entity_types
        context_match = _contains_any(context, ("음식", "맛집", "디저트", "카페", "요리"))
    elif positive_category == TravelCategory.SPORTS_EVENT:
        explicit_sport = _contains_any(
            context,
            ("마라톤", "야구", "축구", "농구", "배구", "테니스", "골프"),
        )
        visit_or_venue = _contains_any(
            context,
            ("개최", "열리", "관람", "직관", "방문", "경기장", "야구장", "구장", "스타디움"),
        )
        entity_match = "EVENT" in entity_types
        context_match = (explicit_sport or entity_match) and visit_or_venue
    elif positive_category == TravelCategory.NATURE:
        entity_match = bool(entity_types & {"LOCATION", "PLACE"})
        context_match = _contains_any(context, ("해변", "해수욕장", "섬", "공원", "숲", "계곡", "오름"))
    elif positive_category == TravelCategory.LANDMARK:
        entity_match = bool(entity_types & {"LOCATION", "PLACE"})
    elif positive_category == TravelCategory.LOCAL_CULTURE:
        entity_match = bool(entity_types & {"LOCATION", "PLACE", "EVENT"})
        context_match = _contains_any(context, ("지역", "문화", "마을", "체험", "관광"))
    elif positive_category == TravelCategory.REGIONAL_MEME:
        entity_match = "MEME" in entity_types and bool(
            entity_types & {"LOCATION", "PLACE"}
        )
        context_match = entity_match and _contains_any(context, ("지역", "밈", "명소", "관광"))
    elif positive_category == TravelCategory.OTHER:
        entity_match = bool(entity_types & HIGH_VALUE_ENTITY_TYPES)

    if entity_match:
        codes.add("CATEGORY_ENTITY_MATCH")
    if context_match:
        codes.add("CATEGORY_CONTEXT_MATCH")
    category_pass = entity_match or context_match
    if not category_pass:
        codes.add("SEMANTIC_CATEGORY_UNSUPPORTED")
    return category_pass, codes


def _has_proper_topic_signal(
    *,
    tokens: Sequence[Token],
    quality_signal: QualitySignal | None,
) -> bool:
    if quality_signal is None:
        return False
    if quality_signal.candidate_type in {"protected_phrase", "protected_entity"}:
        return True
    if quality_signal.entity_type in HIGH_VALUE_ENTITY_TYPES:
        return True
    if (
        quality_signal.extractor == "ner"
        and quality_signal.entity_type in WEAK_ENTITY_TYPES
        and (quality_signal.entity_confidence or 0) >= 0.8
        and quality_signal.title_occurrence > 0
    ):
        return True
    return bool(
        quality_signal.quality_score >= 60
        and quality_signal.title_occurrence > 0
        and any(token.tag == "NNP" for token in tokens)
    )


def _is_meaningful_token(token: Token) -> bool:
    return (
        token.tag.startswith("NN")
        or token.tag.startswith("VV")
        or token.tag.startswith("VA")
        or token.tag in {"SL", "SN"}
    )


def _quality_payload(signal: QualitySignal | None) -> dict[str, object] | None:
    if signal is None:
        return None
    return {
        "candidate_type": signal.candidate_type,
        "extractor": signal.extractor,
        "quality_score": round(float(signal.quality_score), 6),
        "title_occurrence": int(signal.title_occurrence),
        "entity_type": signal.entity_type,
        "entity_confidence": (
            round(float(signal.entity_confidence), 6)
            if signal.entity_confidence is not None
            else None
        ),
        "accepted": bool(signal.accepted),
        "rejection_reason": signal.rejection_reason,
        "pipeline_version": signal.pipeline_version,
    }


def _json_terms(raw: str | None) -> list[str]:
    try:
        values = json.loads(raw or "[]")
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values if str(value).strip())


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    return any(term in text for term in terms)


def _entity_is_in_context(
    entity: EntitySignal,
    context: str,
    normalized_context: str,
) -> bool:
    if entity.text in context:
        return True
    normalized_entity = normalize_keyword(entity.normalized_text or entity.text)
    return bool(normalized_entity and normalized_entity in normalized_context)


def _normalized_matches_keyword(value: str | None, normalized_keyword: str) -> bool:
    normalized_value = normalize_keyword(value or "")
    return bool(normalized_keyword and normalized_value == normalized_keyword)

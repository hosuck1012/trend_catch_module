from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
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
PRECISION_EVIDENCE_VERSION = "v2-step4e5-coherence-1"
MATCHED_SENTENCE = "MATCHED_SENTENCE"
ADJACENT_SENTENCE = "ADJACENT_SENTENCE"
DOCUMENT_ONLY = "DOCUMENT_ONLY"
NO_LOCALITY = "NONE"
EXACT_ALIGNMENT = "EXACT"
PARTIAL_ALIGNMENT = "PARTIAL"
UNALIGNED = "UNALIGNED"
NO_ALIGNMENT = "NONE"
KOREAN_PARTICLE_SUFFIXES = (
    "으로부터",
    "에서는",
    "으로는",
    "에게서",
    "까지는",
    "부터는",
    "에서",
    "에게",
    "께서",
    "으로",
    "부터",
    "까지",
    "처럼",
    "보다",
    "로는",
    "에는",
    "와는",
    "과는",
    "이라",
    "라고",
    "이며",
    "이고",
    "이나",
    "라도",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "와",
    "과",
    "로",
    "도",
    "만",
    "엔",
    "이다",
    "였다",
    "라고는",
)
INCOMPLETE_FRAGMENT_TERMS = {
    "서",
    "로",
    "에",
    "의",
    "와",
    "과",
    "을",
    "를",
    "은",
    "는",
    "이",
    "가",
    "도",
    "만",
}
EVENT_TOPIC_TERMS = ("축제", "페스티벌", "페스트", "박람회", "엑스포", "행사")
EVENT_ACTION_TERMS = (
    "개최",
    "개막",
    "열리",
    "열린",
    "연다",
    "진행",
    "행사",
    "축제",
    "페스티벌",
    "페스트",
    "공연",
    "전시",
)
VISITOR_TRAVEL_TERMS = (
    "여행",
    "관광",
    "여행객",
    "관광객",
    "방문객",
    "방문",
    "관람",
    "직관",
    "찾는",
    "찾아",
    "명소",
    "체험",
)
EXPLICIT_TRAVEL_TERMS = (
    "여행",
    "관광",
    "여행객",
    "관광객",
    "방문객",
    "명소",
)
NEWS_DATELINE_PATTERN = re.compile(
    r"\[[^\]\r\n=]{1,30}=\s*뉴시스\]",
    flags=re.IGNORECASE,
)
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
    matched_context: str
    adjacent_context: str
    normalized_keyword: str
    travel_category: str
    aligned_entity_types_matched: frozenset[str]
    aligned_entity_types_adjacent: frozenset[str]
    aligned_entity_types_document: frozenset[str]
    entity_alignment: str
    entity_locality: str
    keyword_locality: str
    single_token_topic: bool
    malformed_topic: bool
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
    context_object = candidate.keyword_context
    context = str(getattr(context_object, "combined_context", "") or "")
    matched_context = str(getattr(context_object, "matched_sentence", "") or "")
    previous_context = str(getattr(context_object, "previous_sentence", "") or "")
    next_context = str(getattr(context_object, "next_sentence", "") or "")
    if not matched_context:
        matched_context = context
    adjacent_context = " ".join(
        value for value in (previous_context, next_context) if value
    )
    tokens = tokenizer.tokenize(candidate.keyword)
    meaningful_tokens = tuple(token for token in tokens if _is_meaningful_token(token))
    token_count = len(meaningful_tokens)
    single_token_topic = token_count <= 1
    keyword_locality = _text_locality(
        str(candidate.keyword),
        matched_context=matched_context,
        adjacent_context=adjacent_context,
        document_context=context,
    )
    malformed_topic = _is_malformed_topic(
        str(candidate.keyword),
        matched_context=matched_context,
        adjacent_context=adjacent_context,
        tokenizer=tokenizer,
    )

    entity_rows = tuple(
        (
            entity,
            _entity_alignment(entity.normalized_text or entity.text, candidate.keyword),
            _text_locality(
                str(entity.text),
                matched_context=matched_context,
                adjacent_context=adjacent_context,
                document_context=context,
            ),
        )
        for entity in context_entities
    )
    aligned_types: dict[str, set[str]] = {
        MATCHED_SENTENCE: set(),
        ADJACENT_SENTENCE: set(),
        DOCUMENT_ONLY: set(),
    }
    alignment_values: set[str] = set()
    for entity, alignment, locality in entity_rows:
        if alignment not in {EXACT_ALIGNMENT, PARTIAL_ALIGNMENT}:
            continue
        alignment_values.add(alignment)
        aligned_types[locality].add(str(entity.entity_type or ""))

    primary_alignment = _entity_alignment(candidate.primary_entity, candidate.keyword)
    primary_type = str(candidate.primary_entity_type or "")
    primary_locality = _signal_locality(
        value=candidate.primary_entity,
        entity_type=primary_type,
        entity_rows=entity_rows,
        fallback_locality=keyword_locality,
    )
    if primary_type and primary_alignment in {EXACT_ALIGNMENT, PARTIAL_ALIGNMENT}:
        alignment_values.add(primary_alignment)
        aligned_types[primary_locality].add(primary_type)

    quality_type = str(quality_signal.entity_type or "") if quality_signal else ""
    quality_alignment = (
        _entity_alignment(candidate.keyword, candidate.keyword)
        if quality_type
        else NO_ALIGNMENT
    )
    quality_locality = _signal_locality(
        value=candidate.keyword if quality_type else None,
        entity_type=quality_type,
        entity_rows=entity_rows,
        fallback_locality=keyword_locality,
    )
    if quality_type and quality_alignment in {EXACT_ALIGNMENT, PARTIAL_ALIGNMENT}:
        alignment_values.add(quality_alignment)
        aligned_types[quality_locality].add(quality_type)

    local_aligned_types = aligned_types[MATCHED_SENTENCE] | aligned_types[ADJACENT_SENTENCE]
    entity_types = frozenset(local_aligned_types)
    high_value_entity = bool(local_aligned_types & HIGH_VALUE_ENTITY_TYPES)
    entity_alignment = (
        EXACT_ALIGNMENT
        if EXACT_ALIGNMENT in alignment_values
        else PARTIAL_ALIGNMENT
        if PARTIAL_ALIGNMENT in alignment_values
        else UNALIGNED
        if context_entities or primary_type or quality_type
        else NO_ALIGNMENT
    )
    entity_locality = (
        MATCHED_SENTENCE
        if aligned_types[MATCHED_SENTENCE]
        else ADJACENT_SENTENCE
        if aligned_types[ADJACENT_SENTENCE]
        else DOCUMENT_ONLY
        if aligned_types[DOCUMENT_ONLY]
        else NO_LOCALITY
    )
    proper_topic = _has_proper_topic_signal(
        tokens=meaningful_tokens,
        quality_signal=quality_signal,
        aligned_high_value_entity=high_value_entity,
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
    generic_single_token = single_token_topic and not high_value_entity and not proper_topic
    generic_topic = is_generic_term or generic_single_token or bool(
        generic_components and not (high_value_entity or proper_topic)
    )

    codes: set[str] = set()
    if is_generic_term or generic_single_token or generic_components:
        codes.add("GENERIC_TOPIC")
    if high_value_entity:
        codes.add("HIGH_VALUE_ENTITY")
    if entity_alignment == EXACT_ALIGNMENT:
        codes.add("ENTITY_KEYWORD_EXACT_MATCH")
    elif entity_alignment == PARTIAL_ALIGNMENT:
        codes.add("ENTITY_KEYWORD_PARTIAL_MATCH")
    elif entity_alignment == UNALIGNED:
        codes.add("ENTITY_KEYWORD_UNALIGNED")
    if entity_locality == MATCHED_SENTENCE:
        codes.add("ENTITY_IN_MATCHED_SENTENCE")
    elif entity_locality == ADJACENT_SENTENCE:
        codes.add("ENTITY_IN_ADJACENT_CONTEXT")
    elif entity_locality == DOCUMENT_ONLY:
        codes.update({"ENTITY_DOCUMENT_ONLY", "DOCUMENT_EVIDENCE_ONLY"})
    if single_token_topic:
        codes.add("SINGLE_TOKEN_TOPIC")
    if malformed_topic:
        codes.add("MALFORMED_TOPIC")
    if multi_token_topic:
        codes.add("MULTI_TOKEN_TOPIC")
    weak_types = {
        value
        for value in local_aligned_types
        if value in WEAK_ENTITY_TYPES
    }
    brand_person_only = bool(weak_types) and not (
        entity_types & HIGH_VALUE_ENTITY_TYPES
    )
    if "BRAND" in weak_types:
        codes.add("BRAND_ONLY")
    if "PERSON" in weak_types:
        codes.add("PERSON_ONLY")

    topic_specificity_pass = not generic_topic and not malformed_topic and (
        high_value_entity or multi_token_topic or proper_topic
    )
    codes.add(
        "TOPIC_SPECIFICITY_PASS"
        if topic_specificity_pass
        else "TOPIC_SPECIFICITY_FAIL"
    )
    payload = {
        "generic_terms": generic_terms.content_hash,
        "precision_version": PRECISION_EVIDENCE_VERSION,
        "keyword": candidate.keyword,
        "normalized_keyword": normalized,
        "tokens": [(token.text, token.tag) for token in meaningful_tokens],
        "primary_entity": candidate.primary_entity,
        "primary_entity_type": candidate.primary_entity_type,
        "primary_entity_alignment": primary_alignment,
        "primary_entity_locality": primary_locality,
        "keyword_locality": keyword_locality,
        "malformed_topic": malformed_topic,
        "matched_context_hash": _text_hash(matched_context),
        "adjacent_context_hash": _text_hash(adjacent_context),
        "travel_category": candidate.travel_category,
        "matched_positive_terms": _json_terms(candidate.matched_positive_terms_json),
        "quality": _quality_payload(quality_signal),
        "entities": sorted(
            (
                entity.normalized_text,
                entity.entity_type,
                alignment,
                locality,
            )
            for entity, alignment, locality in entity_rows
            if alignment in {EXACT_ALIGNMENT, PARTIAL_ALIGNMENT}
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
        matched_context=matched_context,
        adjacent_context=adjacent_context,
        normalized_keyword=normalized,
        travel_category=str(candidate.travel_category),
        aligned_entity_types_matched=frozenset(aligned_types[MATCHED_SENTENCE]),
        aligned_entity_types_adjacent=frozenset(aligned_types[ADJACENT_SENTENCE]),
        aligned_entity_types_document=frozenset(aligned_types[DOCUMENT_ONLY]),
        entity_alignment=entity_alignment,
        entity_locality=entity_locality,
        keyword_locality=keyword_locality,
        single_token_topic=single_token_topic,
        malformed_topic=malformed_topic,
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
    if evidence.malformed_topic:
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
    matched_context = evidence.matched_context.lower()
    adjacent_context = evidence.adjacent_context.lower()
    local_context = " ".join(value for value in (matched_context, adjacent_context) if value)
    topic_context = (
        adjacent_context
        if evidence.keyword_locality == ADJACENT_SENTENCE
        else matched_context
    )
    category_context = (
        local_context if not evidence.single_token_topic else topic_context
    )
    matched_types = evidence.aligned_entity_types_matched
    adjacent_types = evidence.aligned_entity_types_adjacent
    entity_types = matched_types | adjacent_types
    codes: set[str] = set()
    entity_match = False
    context_match = False
    keyword_is_local = evidence.keyword_locality in {MATCHED_SENTENCE, ADJACENT_SENTENCE}
    visitor_travel = _contains_any(category_context, VISITOR_TRAVEL_TERMS)
    category_travel_evidence = False

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
        media_context = _contains_any(category_context, terms)
        context_match = entity_match and media_context
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.FESTIVAL:
        entity_match = "EVENT" in entity_types
        keyword_signal = _contains_any(evidence.normalized_keyword, EVENT_TOPIC_TERMS)
        event_context = _contains_any(category_context, EVENT_ACTION_TERMS)
        explicit_event_topic = _contains_any(category_context, EVENT_TOPIC_TERMS)
        event_evidence = entity_match and event_context and (
            keyword_signal or explicit_event_topic
        )
        location_travel = (
            bool(entity_types & {"LOCATION", "PLACE"})
            and visitor_travel
            and explicit_event_topic
        )
        context_match = keyword_is_local and (event_evidence or location_travel)
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.CONCERT:
        entity_match = "EVENT" in entity_types
        concert_context = _contains_any(category_context, ("콘서트", "공연", "라이브", "무대"))
        context_match = keyword_is_local and entity_match and concert_context
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.EXHIBITION:
        entity_match = bool(entity_types & {"EVENT", "CONTENT_TITLE"})
        exhibition_context = _contains_any(
            category_context,
            ("전시", "전시회", "박람회", "엑스포", "미술관", "박물관", "관람"),
        )
        context_match = keyword_is_local and entity_match and exhibition_context
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.POPUP:
        entity_match = "EVENT" in entity_types
        context_match = (keyword_is_local or not evidence.single_token_topic) and _contains_any(
            category_context,
            ("팝업", "팝업스토어"),
        ) and (entity_match or _contains_any(evidence.normalized_keyword, ("팝업", "팝업스토어")))
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.FOOD:
        entity_match = "FOOD" in entity_types
        food_travel = _contains_any(
            category_context,
            (
                "맛집",
                "디저트",
                "카페",
                "요리",
                "먹거리",
                "먹방",
                "맛보",
                "먹으",
                "관광객",
                "여행객",
                "방문",
                "명물",
                "향토",
            ),
        )
        context_match = keyword_is_local and entity_match and food_travel
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.SPORTS_EVENT:
        explicit_sport = _contains_any(
            category_context,
            ("마라톤", "야구", "축구", "농구", "배구", "테니스", "골프"),
        )
        visitor_signal = _contains_any(
            category_context,
            ("관람", "직관", "방문", "여행", "관광"),
        )
        hosting_signal = _contains_any(
            category_context,
            ("개최", "개막", "열리", "열린", "경기장", "야구장", "구장", "스타디움"),
        )
        entity_match = "EVENT" in entity_types
        context_match = keyword_is_local and explicit_sport and (
            visitor_signal or (entity_match and hosting_signal)
        )
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.NATURE:
        entity_match = bool(entity_types & {"LOCATION", "PLACE"})
        nature_context = _contains_any(
            category_context,
            (
                "해변",
                "해수욕장",
                "섬",
                "산",
                "강",
                "호수",
                "공원",
                "숲",
                "계곡",
                "오름",
                "산책",
                "트레킹",
            ),
        )
        context_match = keyword_is_local and entity_match and nature_context and (
            visitor_travel or not evidence.single_token_topic
        )
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.LANDMARK:
        entity_match = bool(entity_types & {"LOCATION", "PLACE"})
        landmark_context = _contains_any(
            category_context,
            ("명소", "랜드마크", "성지", "순례", "관광", "여행", "방문", "관람", "찾는"),
        )
        context_match = keyword_is_local and entity_match and landmark_context
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.LOCAL_CULTURE:
        entity_match = bool(entity_types & {"LOCATION", "PLACE", "EVENT"})
        cultural_context = _contains_any(
            category_context,
            ("문화", "마을", "체험", "관광", "여행", "불교", "전통", "국악", "박람회"),
        )
        specific_entity = bool(entity_types & {"PLACE", "EVENT"})
        context_match = keyword_is_local and cultural_context and (
            specific_entity or visitor_travel
        )
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.REGIONAL_MEME:
        entity_match = "MEME" in entity_types
        viral_context = _contains_any(category_context, ("밈", "유행", "바이럴", "화제"))
        regional_context = _contains_any(
            category_context,
            ("지역", "도시", "마을", "명소", "관광", "여행", "방문"),
        )
        context_match = keyword_is_local and entity_match and viral_context and regional_context
        category_travel_evidence = context_match
    elif positive_category == TravelCategory.OTHER:
        entity_match = bool(entity_types & HIGH_VALUE_ENTITY_TYPES)
        context_match = keyword_is_local and entity_match and _contains_any(
            category_context,
            EXPLICIT_TRAVEL_TERMS,
        )
        category_travel_evidence = context_match

    if entity_match:
        codes.add("CATEGORY_ENTITY_MATCH")
    if context_match:
        codes.add("CATEGORY_CONTEXT_MATCH")
    category_pass = context_match
    if category_pass:
        codes.add("CATEGORY_LOCAL_EVIDENCE_PASS")
    else:
        codes.add("CATEGORY_LOCAL_EVIDENCE_FAIL")
    if evidence.single_token_topic:
        codes.add("SINGLE_TOKEN_TOPIC")
        if category_travel_evidence:
            codes.add("SINGLE_TOKEN_WITH_TRAVEL_EVIDENCE")
        else:
            codes.add("SINGLE_TOKEN_INSUFFICIENT_EVIDENCE")
            category_pass = False
    if evidence.entity_locality == DOCUMENT_ONLY and not context_match:
        codes.add("DOCUMENT_EVIDENCE_ONLY")
    if not category_pass:
        codes.add("SEMANTIC_CATEGORY_UNSUPPORTED")
    return category_pass, codes


def _has_proper_topic_signal(
    *,
    tokens: Sequence[Token],
    quality_signal: QualitySignal | None,
    aligned_high_value_entity: bool,
) -> bool:
    if quality_signal is None:
        return False
    if quality_signal.candidate_type in {"protected_phrase", "protected_entity"}:
        return True
    if quality_signal.entity_type in HIGH_VALUE_ENTITY_TYPES and aligned_high_value_entity:
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


def _entity_alignment(value: str | None, keyword: str) -> str:
    normalized_value = normalize_keyword(value or "") or ""
    normalized_keyword = normalize_keyword(keyword) or ""
    if not normalized_value or not normalized_keyword:
        return NO_ALIGNMENT
    if normalized_value == normalized_keyword:
        return EXACT_ALIGNMENT
    if _standalone_occurs(str(value or ""), keyword) or _standalone_occurs(keyword, str(value or "")):
        return PARTIAL_ALIGNMENT
    relationship_suffixes = (
        "촬영지",
        "축제",
        "페스티벌",
        "박람회",
        "엑스포",
        "공연",
        "콘서트",
        "전시",
        "팝업",
        "공원",
        "경기장",
    )
    if normalized_keyword.startswith(normalized_value):
        remainder = normalized_keyword[len(normalized_value) :]
        if _contains_any(remainder, relationship_suffixes):
            return PARTIAL_ALIGNMENT
    if normalized_value.startswith(normalized_keyword):
        remainder = normalized_value[len(normalized_keyword) :]
        if _contains_any(remainder, relationship_suffixes):
            return PARTIAL_ALIGNMENT
    return UNALIGNED


def _signal_locality(
    *,
    value: str | None,
    entity_type: str,
    entity_rows: Sequence[tuple[EntitySignal, str, str]],
    fallback_locality: str,
) -> str:
    if not value or not entity_type:
        return NO_LOCALITY
    normalized_value = normalize_keyword(value) or ""
    localities = {
        locality
        for entity, alignment, locality in entity_rows
        if str(entity.entity_type or "") == entity_type
        and alignment in {EXACT_ALIGNMENT, PARTIAL_ALIGNMENT}
        and (
            (normalize_keyword(entity.normalized_text or entity.text) or "") == normalized_value
            or _entity_alignment(entity.text, value) in {EXACT_ALIGNMENT, PARTIAL_ALIGNMENT}
        )
    }
    for locality in (MATCHED_SENTENCE, ADJACENT_SENTENCE, DOCUMENT_ONLY):
        if locality in localities:
            return locality
    return fallback_locality


def _text_locality(
    value: str,
    *,
    matched_context: str,
    adjacent_context: str,
    document_context: str,
) -> str:
    matched_context = NEWS_DATELINE_PATTERN.sub("", matched_context)
    adjacent_context = NEWS_DATELINE_PATTERN.sub("", adjacent_context)
    compact_value = re.sub(r"[\s_:\-]+", "", str(value or ""))
    if _standalone_occurs(value, matched_context) or (
        compact_value != value and _standalone_occurs(compact_value, matched_context)
    ):
        return MATCHED_SENTENCE
    if _standalone_occurs(value, adjacent_context) or (
        compact_value != value and _standalone_occurs(compact_value, adjacent_context)
    ):
        return ADJACENT_SENTENCE
    normalized_value = normalize_keyword(value) or ""
    normalized_document = normalize_keyword(document_context) or ""
    if normalized_value and normalized_value in normalized_document:
        return DOCUMENT_ONLY
    return DOCUMENT_ONLY


def _standalone_occurs(value: str, text: str) -> bool:
    value = str(value or "").strip()
    text = str(text or "")
    if not value or not text:
        return False
    for match in re.finditer(re.escape(value), text, flags=re.IGNORECASE):
        start, end = match.span()
        if start > 0 and _is_word_character(text[start - 1]):
            continue
        if end >= len(text) or not _is_word_character(text[end]):
            return True
        tail = text[end:]
        if any(_particle_ends_at_boundary(tail, suffix) for suffix in KOREAN_PARTICLE_SUFFIXES):
            return True
    return False


def _particle_ends_at_boundary(tail: str, suffix: str) -> bool:
    if not tail.startswith(suffix):
        return False
    boundary = len(suffix)
    return boundary >= len(tail) or not _is_word_character(tail[boundary])


def _is_word_character(value: str) -> bool:
    return value.isalnum() or "가" <= value <= "힣"


def _is_malformed_topic(
    keyword: str,
    *,
    matched_context: str,
    adjacent_context: str,
    tokenizer: Tokenizer,
) -> bool:
    stripped = keyword.strip()
    if not stripped or " " not in stripped:
        return False
    raw_tokens = tokenizer.tokenize(stripped)
    if not raw_tokens:
        return False
    final_surface = stripped.rsplit(maxsplit=1)[-1]
    final_token = raw_tokens[-1]
    dependent_ending = final_token.tag.startswith(("J", "E", "X"))
    incomplete_fragment = final_surface in INCOMPLETE_FRAGMENT_TERMS
    literal_occurs = _standalone_occurs(stripped, matched_context) or _standalone_occurs(
        stripped,
        adjacent_context,
    )
    normalized = normalize_keyword(stripped) or ""
    normalized_local = normalize_keyword(f"{matched_context} {adjacent_context}") or ""
    joined_only = bool(normalized and normalized in normalized_local and not literal_occurs)
    return joined_only and (dependent_ending or incomplete_fragment)


def _text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

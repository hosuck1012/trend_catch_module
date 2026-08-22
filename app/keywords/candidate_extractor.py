from dataclasses import dataclass, replace
import html
import re

from app.keywords.keyword_normalizer import normalize_display, normalize_keyword
from app.keywords.phrase_extractor import is_noun, noun_phrases, specific_phrases
from app.keywords.protected_phrases import (
    PROTECTABLE_ENTITY_TYPES,
    matching_phrases,
    structural_phrases,
)
from app.keywords.tokenizer import Tokenizer


_SCRIPT_STYLE = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.I | re.S)
_HTML_TAG = re.compile(r"<[^>]+>")
_MARKDOWN_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_DOMAIN = re.compile(r"\b(?:[A-Za-z0-9-]+\.)+(?:com|net|org|co\.kr|kr|html?|php)\b", re.I)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SPECIAL = re.compile(r"([^\w\s가-힣])\1{1,}")
_SPACES = re.compile(r"\s+")
_HASHTAG = re.compile(r"#[0-9A-Za-z가-힣_-]+")


@dataclass(frozen=True)
class EntityEvidence:
    text: str
    entity_type: str
    confidence: float


@dataclass(frozen=True)
class CandidateEvidence:
    candidate_text: str
    normalized_candidate: str
    candidate_type: str
    extractor: str
    title_occurrence: int
    body_occurrence: int
    entity_type: str | None = None
    entity_confidence: float | None = None
    supporting_extractors: tuple[str, ...] = ()


def clean_analysis_text(value: str) -> str:
    value = html.unescape(value or "")
    value = _SCRIPT_STYLE.sub(" ", value)
    value = _MARKDOWN_LINK.sub(r" \1 ", value)
    value = _URL.sub(" ", value)
    value = _EMAIL.sub(" ", value)
    value = _DOMAIN.sub(" ", value)
    value = _HTML_TAG.sub(" ", value)
    value = _CONTROL.sub(" ", value)
    value = _SPECIAL.sub(r"\1", value)
    return _SPACES.sub(" ", value).strip()


def extract_candidates(
    *,
    title: str,
    body: str,
    tokenizer: Tokenizer,
    entities: list[EntityEvidence] | None = None,
) -> list[CandidateEvidence]:
    clean_title = clean_analysis_text(title)
    clean_body = clean_analysis_text(body)
    combined = f"{clean_title}\n{clean_body}".strip()
    raw: list[tuple[str, str, str, str | None, float | None]] = []

    for phrase in matching_phrases(combined):
        raw.append((phrase, "protected_phrase", "protected_phrase", None, None))
    for phrase in structural_phrases(clean_title, clean_body):
        raw.append((phrase, "protected_phrase", "structural_phrase", None, None))
    for entity in entities or []:
        candidate_type = (
            "protected_entity" if entity.entity_type in PROTECTABLE_ENTITY_TYPES else "entity"
        )
        raw.append((entity.text, candidate_type, "ner", entity.entity_type, entity.confidence))
    for match in _HASHTAG.finditer(combined):
        raw.append((match.group(0), "hashtag", "hashtag", None, None))

    for section, extractor in ((clean_title, "title_phrase"), (clean_body, "kiwi_phrase")):
        tokens = tokenizer.tokenize(section)
        for token in tokens:
            if is_noun(token):
                raw.append((token.text, "token", "kiwi_token", None, None))
        for phrase in specific_phrases(section, tokens):
            raw.append((phrase, "specific_phrase", "phrase_pattern", None, None))
        for phrase in noun_phrases(tokens):
            raw.append((phrase, "noun_phrase", extractor, None, None))

    merged: dict[str, CandidateEvidence] = {}
    for value, candidate_type, extractor, entity_type, confidence in raw:
        display = normalize_display(value)
        normalized = normalize_keyword(display)
        if not normalized:
            continue
        title_count = _normalized_count(clean_title, normalized)
        body_count = _normalized_count(clean_body, normalized)
        candidate = CandidateEvidence(
            candidate_text=display,
            normalized_candidate=normalized,
            candidate_type=candidate_type,
            extractor=extractor,
            title_occurrence=title_count,
            body_occurrence=body_count,
            entity_type=entity_type,
            entity_confidence=confidence,
            supporting_extractors=(extractor,),
        )
        existing = merged.get(normalized)
        if existing is None:
            merged[normalized] = candidate
            continue
        winner = candidate if _priority(candidate) > _priority(existing) else existing
        supporting_extractors = tuple(
            dict.fromkeys(
                (*existing.supporting_extractors, *candidate.supporting_extractors)
            )
        )
        merged[normalized] = replace(
            winner,
            supporting_extractors=supporting_extractors,
        )
    return list(merged.values())


def _normalized_count(text: str, normalized: str) -> int:
    compact = normalize_keyword(text) or ""
    return compact.count(normalized)


def _priority(candidate: CandidateEvidence) -> tuple[int, float, int]:
    order = {
        "protected_phrase": 5,
        "ner": 4,
        "structural_phrase": 4,
        "phrase_pattern": 4,
        "title_phrase": 3,
        "kiwi_phrase": 2,
        "hashtag": 2,
        "kiwi_token": 1,
    }
    return (order.get(candidate.extractor, 0), candidate.entity_confidence or 0.0, len(candidate.candidate_text))

from dataclasses import dataclass
from datetime import datetime, timedelta
import re

from app.config import Settings
from app.keywords.candidate_extractor import CandidateEvidence
from app.keywords.stopword_filter import rejection_reason


_ENGLISH = re.compile(r"^[A-Za-z]+$")
_VALID = re.compile(r"^[0-9A-Za-z가-힣 ]+$")
_NUMBER_WITH_UNIT = re.compile(r"^\d+(?:일|년|월|명|개|건|곳|원|억|만)?$")
_TRAVEL_TYPES = {"LOCATION", "PLACE", "EVENT", "FOOD"}
KEYWORD_HIGH_CONFIDENCE_SCORE = 60.0


def is_numeric_artifact(value: str) -> bool:
    return bool(_NUMBER_WITH_UNIT.fullmatch(value.replace(" ", "")))


@dataclass(frozen=True)
class QualityContext:
    document_count: int
    source_count: int
    latest_occurrence: datetime
    now: datetime


@dataclass(frozen=True)
class QualityDecision:
    quality_score: float
    accepted: bool
    rejection_reason: str | None


def evaluate_candidate(
    candidate: CandidateEvidence,
    context: QualityContext,
    settings: Settings,
) -> QualityDecision:
    value = candidate.candidate_text
    normalized = candidate.normalized_candidate
    immediate = rejection_reason(value, normalized)
    if immediate:
        return QualityDecision(0.0, False, immediate)
    if candidate.extractor != "protected_phrase" and " " in value:
        component_reason = next(
            (
                rejection_reason(part, part.lower())
                for part in value.split()
                if rejection_reason(part, part.lower())
            ),
            None,
        )
        if component_reason:
            return QualityDecision(0.0, False, component_reason)
    if normalized.isdigit() or is_numeric_artifact(normalized):
        return QualityDecision(0.0, False, "numeric_only")
    if not _VALID.fullmatch(value):
        return QualityDecision(0.0, False, "invalid_characters")
    korean_length = sum("가" <= char <= "힣" for char in normalized)
    english_length = sum(char.isascii() and char.isalpha() for char in normalized)
    if korean_length and len(normalized) < settings.keyword_min_length_ko:
        return QualityDecision(0.0, False, "too_short")
    if english_length and not korean_length and english_length < settings.keyword_min_length_en:
        return QualityDecision(0.0, False, "too_short")
    if len(value) > settings.keyword_max_length:
        return QualityDecision(0.0, False, "invalid_characters")
    if candidate.extractor == "ner" and context.document_count < 2:
        return QualityDecision(0.0, False, "low_frequency")
    if candidate.extractor in {"title_phrase", "kiwi_phrase"} and context.document_count < 2:
        proper_compound = all(part[:1].isupper() for part in value.split())
        if not candidate.entity_type and not proper_compound:
            return QualityDecision(0.0, False, "low_frequency")
    if (
        candidate.extractor in {"title_phrase", "kiwi_phrase"}
        and not candidate.title_occurrence
        and context.source_count < 2
    ):
        return QualityDecision(0.0, False, "unrelated_single_token")
    if _ENGLISH.fullmatch(value.replace(" ", "")) and " " not in value:
        proper_title = candidate.title_occurrence > 0 and value[:1].isupper()
        if context.document_count < 2 and not candidate.entity_type and not proper_title:
            return QualityDecision(0.0, False, "unrelated_single_token")

    score = 15.0
    if korean_length >= 4 and " " not in value:
        score += 5.0
    if candidate.title_occurrence:
        score += min(20.0, 20.0 * settings.keyword_title_weight / 1.5)
    if candidate.candidate_type in {"noun_phrase", "protected_phrase"} or " " in value:
        score += min(15.0, 15.0 * settings.keyword_phrase_weight / 1.4)
    if candidate.extractor == "protected_phrase":
        score += 10.0
    if candidate.entity_type:
        confidence = candidate.entity_confidence or 0.0
        score += min(20.0, confidence * 20.0 * settings.keyword_ner_weight / 1.8)
    score += min(15.0, max(context.document_count - 1, 0) / 3 * 15.0)
    score += min(10.0, max(context.source_count - 1, 0) * 10.0 * settings.keyword_source_diversity_weight / 1.2)
    if context.latest_occurrence >= context.now - timedelta(days=2):
        score += 10.0
    elif context.latest_occurrence >= context.now - timedelta(days=7):
        score += 5.0
    if candidate.entity_type in _TRAVEL_TYPES:
        score += 10.0
    if context.source_count == 1 and context.document_count >= 3:
        score -= 15.0
    score = round(max(0.0, min(100.0, score)), 2)
    accepted = score >= settings.keyword_min_quality_score
    return QualityDecision(score, accepted, None if accepted else "low_quality")

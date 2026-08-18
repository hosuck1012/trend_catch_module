from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from threading import Lock

from app.context_v2.embedding_adapter import EmbeddingAdapter
from app.context_v2.travel_taxonomy import TravelCategory


SEMANTIC_SCORER_VERSION = "v2-step2-local-1"
SCORE_BASE = 0.50
SIMILARITY_MARGIN_WEIGHT = 12.5
DOMINANT_SPECIFIC_NEGATIVE_PENALTY = 0.15
POSITIVE_SECTION = "positive"
NEGATIVE_SECTION = "negative"
NEGATIVE_ENTITY_HINTS = {
    "FINANCE": "BRAND",
    "LEGAL": "PERSON",
}
SPECIFIC_NEGATIVE_CATEGORIES = {"FINANCE", "LEGAL", "POLITICS", "ACCIDENT"}
DEFAULT_ANCHOR_PATH = Path(__file__).resolve().parents[2] / "data" / "travel_semantic_anchors.json"


@dataclass(frozen=True)
class SemanticAnchors:
    version: str
    positive: dict[str, list[str]]
    negative: dict[str, list[str]]
    content_hash: str


@dataclass(frozen=True)
class SemanticScore:
    best_positive_similarity: float
    positive_category: str
    best_negative_similarity: float
    negative_category: str
    semantic_travel_score: float
    semantic_status: str


def load_semantic_anchors(
    path: Path = DEFAULT_ANCHOR_PATH,
    *,
    expected_version: str | None = None,
) -> SemanticAnchors:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = str(payload.get("version", "")).strip()
    positive = _anchor_section(payload, POSITIVE_SECTION)
    negative = _anchor_section(payload, NEGATIVE_SECTION)
    expected_categories = {category.value for category in TravelCategory}
    if set(positive) != expected_categories:
        missing = sorted(expected_categories - set(positive))
        extra = sorted(set(positive) - expected_categories)
        raise ValueError(f"Invalid positive semantic categories: missing={missing}, extra={extra}")
    if not version:
        raise ValueError("Semantic anchor version is required")
    if expected_version is not None and version != expected_version:
        raise ValueError(
            f"Semantic anchor version mismatch: expected {expected_version}, found {version}"
        )
    canonical = json.dumps(
        {"version": version, "positive": positive, "negative": negative},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return SemanticAnchors(
        version=version,
        positive=positive,
        negative=negative,
        content_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def build_candidate_embedding_text(candidate: object, *, max_chars: int) -> str:
    context = candidate.keyword_context
    fields = [
        ("keyword", candidate.keyword),
        ("normalized_keyword", candidate.normalized_keyword),
        ("primary_entity", candidate.primary_entity),
        ("primary_entity_type", candidate.primary_entity_type),
        ("travel_category", candidate.travel_category),
        ("previous_sentence", context.previous_sentence),
        ("matched_sentence", context.matched_sentence),
        ("next_sentence", context.next_sentence),
        ("combined_context", context.combined_context),
    ]
    text = "\n".join(f"{name}: {str(value).strip()}" for name, value in fields if value)
    return text[:max_chars]


class SemanticScorer:
    def __init__(
        self,
        *,
        adapter: EmbeddingAdapter,
        anchors: SemanticAnchors,
        reject_threshold: float,
        review_threshold: float,
        strong_threshold: float,
    ) -> None:
        if not 0 <= reject_threshold <= review_threshold <= strong_threshold <= 1:
            raise ValueError("Semantic thresholds must be ordered values between 0 and 1")
        self.adapter = adapter
        self.anchors = anchors
        self.reject_threshold = reject_threshold
        self.review_threshold = review_threshold
        self.strong_threshold = strong_threshold
        self._anchor_embeddings: list[list[float]] | None = None
        self._positive_keys: list[str] = []
        self._negative_keys: list[str] = []
        self._anchor_lock = Lock()

    @property
    def cache_signature(self) -> str:
        return "|".join(
            (
                SEMANTIC_SCORER_VERSION,
                self.anchors.content_hash,
                f"{self.reject_threshold:.12g}",
                f"{self.review_threshold:.12g}",
                f"{self.strong_threshold:.12g}",
            )
        )

    def evaluate(self, candidate_texts: Sequence[str]) -> list[SemanticScore]:
        if not candidate_texts:
            return []
        anchor_embeddings = self._get_anchor_embeddings()
        candidate_embeddings = self.adapter.encode_passages(candidate_texts)
        positive_count = len(self._positive_keys)
        results: list[SemanticScore] = []
        for candidate_embedding in candidate_embeddings:
            similarities = [
                _cosine_similarity(candidate_embedding, anchor_embedding)
                for anchor_embedding in anchor_embeddings
            ]
            positive_values = similarities[:positive_count]
            negative_values = similarities[positive_count:]
            positive_index = max(range(len(positive_values)), key=positive_values.__getitem__)
            negative_index = max(range(len(negative_values)), key=negative_values.__getitem__)
            positive_similarity = positive_values[positive_index]
            negative_similarity = negative_values[negative_index]
            normalized_score = semantic_score_from_similarities(
                positive_similarity,
                negative_similarity,
                negative_category=self._negative_keys[negative_index],
            )
            results.append(
                SemanticScore(
                    best_positive_similarity=round(positive_similarity, 6),
                    positive_category=self._positive_keys[positive_index],
                    best_negative_similarity=round(negative_similarity, 6),
                    negative_category=self._negative_keys[negative_index],
                    semantic_travel_score=round(normalized_score * 100, 2),
                    semantic_status=classify_semantic_score(
                        normalized_score,
                        reject_threshold=self.reject_threshold,
                        review_threshold=self.review_threshold,
                        strong_threshold=self.strong_threshold,
                    ),
                )
            )
        return results

    def _get_anchor_embeddings(self) -> list[list[float]]:
        if self._anchor_embeddings is not None:
            return self._anchor_embeddings
        with self._anchor_lock:
            if self._anchor_embeddings is not None:
                return self._anchor_embeddings
            texts: list[str] = []
            for category, anchors in self.anchors.positive.items():
                texts.extend(
                    f"travel_category: {category}\nsemantic_anchor: {anchor}"
                    for anchor in anchors
                )
                self._positive_keys.extend([category] * len(anchors))
            for category, anchors in self.anchors.negative.items():
                entity_hint = NEGATIVE_ENTITY_HINTS.get(category)
                metadata = (
                    f"primary_entity_type: {entity_hint}\n"
                    if entity_hint is not None
                    else ""
                )
                texts.extend(
                    f"{metadata}negative_category: {category}\nsemantic_anchor: {anchor}"
                    for anchor in anchors
                )
                self._negative_keys.extend([category] * len(anchors))
            self._anchor_embeddings = self.adapter.encode_queries(texts)
        return self._anchor_embeddings

def _anchor_section(payload: Mapping[str, object], name: str) -> dict[str, list[str]]:
    raw = payload.get(name)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Semantic anchor section '{name}' is required")
    result: dict[str, list[str]] = {}
    for category, values in raw.items():
        if not isinstance(category, str) or not isinstance(values, list):
            raise ValueError(f"Invalid semantic anchors for section '{name}'")
        anchors = [value.strip() for value in values if isinstance(value, str) and value.strip()]
        if not anchors:
            raise ValueError(f"Semantic category '{category}' must contain anchors")
        result[category] = anchors
    return result


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match")
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def classify_semantic_score(
    normalized_score: float,
    *,
    reject_threshold: float,
    review_threshold: float,
    strong_threshold: float,
) -> str:
    if normalized_score < reject_threshold:
        return "semantic_rejected"
    if normalized_score < review_threshold:
        return "semantic_weak"
    if normalized_score < strong_threshold:
        return "semantic_review"
    return "semantic_strong"


def semantic_score_from_similarities(
    positive_similarity: float,
    negative_similarity: float,
    *,
    negative_category: str,
) -> float:
    """Map the narrow E5 cosine margin to 0..1 with an explicit negative penalty."""
    negative_penalty = (
        DOMINANT_SPECIFIC_NEGATIVE_PENALTY
        if negative_category in SPECIFIC_NEGATIVE_CATEGORIES
        and negative_similarity > positive_similarity
        else 0.0
    )
    return _clamp(
        SCORE_BASE
        + SIMILARITY_MARGIN_WEIGHT * (positive_similarity - negative_similarity)
        - negative_penalty,
        0.0,
        1.0,
    )

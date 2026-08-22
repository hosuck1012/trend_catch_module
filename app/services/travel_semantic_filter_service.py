from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date
import hashlib

from sqlalchemy.orm import Session

from app.config import get_settings
from app.context_v2.semantic_scorer import (
    SemanticScore,
    SemanticScorer,
    build_candidate_embedding_text,
)
from app.context_v2.semantic_precision import (
    CalibratedSemanticScore,
    SemanticPrecisionEvidence,
    build_semantic_precision_evidence,
    calibrate_semantic_score,
    load_generic_topic_terms,
)
from app.keywords.tokenizer import Tokenizer
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.repositories import travel_opportunity_repository as repo


SEMANTIC_RANKING_STATUSES = {"semantic_review", "semantic_strong"}


@dataclass(frozen=True)
class SemanticCandidatePreview:
    keyword: str
    normalized_keyword: str
    travel_category: str
    prefilter_status: str
    semantic_travel_score: float
    semantic_status: str
    semantic_positive_score: float
    semantic_positive_category: str
    semantic_negative_score: float
    semantic_negative_category: str
    semantic_margin: float
    semantic_confidence: float
    reasoning_codes: tuple[str, ...]


@dataclass(frozen=True)
class SemanticFilterResult:
    status: str
    dry_run: bool
    week_start: date | None
    processed: int
    semantic_rejected: int
    semantic_weak: int
    semantic_review: int
    semantic_strong: int
    estimated_gemini_candidates: int
    top_candidates: list[SemanticCandidatePreview]
    model_name: str
    scoring_version: str
    cache_hits: int


def semantic_filter_travel_opportunities(
    session: Session,
    *,
    scorer: SemanticScorer,
    week_start: date | None,
    dry_run: bool,
    force: bool,
    limit: int,
    topic_tokenizer: Tokenizer | None = None,
) -> SemanticFilterResult:
    settings = get_settings()
    resolved_start, _ = repo.resolve_week_range(session, week_start)
    model_name = scorer.adapter.model_name
    if not settings.travel_embedding_enabled or not scorer.adapter.enabled:
        return _empty_result(
            status="disabled",
            dry_run=dry_run,
            week_start=resolved_start,
            model_name=model_name,
            scoring_version=scorer.scoring_version,
        )
    if resolved_start is None:
        return _empty_result(
            status="dry_run" if dry_run else "ok",
            dry_run=dry_run,
            week_start=None,
            model_name=model_name,
            scoring_version=scorer.scoring_version,
        )

    candidates = repo.get_semantic_filter_candidates(
        session,
        week_start=resolved_start,
        limit=limit,
    )
    quality_signals = repo.get_semantic_keyword_quality_signals(
        session,
        candidates=candidates,
    )
    context_entities = repo.get_semantic_context_entities(
        session,
        candidates=candidates,
    )
    generic_terms = load_generic_topic_terms()
    precision_evidence = {
        candidate.id: build_semantic_precision_evidence(
            candidate,
            quality_signal=quality_signals.get(candidate.id),
            context_entities=context_entities.get(candidate.id, []),
            generic_terms=generic_terms,
            tokenizer=topic_tokenizer,
        )
        for candidate in candidates
    }
    pending_rows: list[TravelOpportunityCandidate] = []
    pending_texts: list[str] = []
    previews: list[SemanticCandidatePreview] = []
    status_counts: Counter[str] = Counter()
    cache_hits = 0
    now = repo.utc_now()
    values_by_id: dict[int, dict[str, object]] = {}

    for candidate in candidates:
        candidate_text = build_candidate_embedding_text(
            candidate,
            max_chars=settings.travel_embedding_max_context_chars,
        )
        input_hash = semantic_input_hash(
            model_name=model_name,
            context_hash=candidate.keyword_context.context_hash,
            anchor_version=scorer.anchors.version,
            scoring_version=scorer.scoring_version,
            candidate_text=candidate_text,
            scorer_signature=scorer.cache_signature,
            precision_signature=precision_evidence[candidate.id].cache_signature,
        )
        if not force and _is_cache_hit(candidate, input_hash=input_hash, model_name=model_name):
            cache_hits += 1
            calibrated = _calibrate_stored_candidate(
                candidate,
                evidence=precision_evidence[candidate.id],
                scorer=scorer,
            )
            preview = _preview_from_candidate(candidate, calibrated=calibrated)
            previews.append(preview)
            status_counts[preview.semantic_status] += 1
            continue
        pending_rows.append(candidate)
        pending_texts.append(candidate_text)

    semantic_scores = scorer.evaluate(pending_texts)
    for candidate, candidate_text, score in zip(
        pending_rows, pending_texts, semantic_scores, strict=True
    ):
        input_hash = semantic_input_hash(
            model_name=model_name,
            context_hash=candidate.keyword_context.context_hash,
            anchor_version=scorer.anchors.version,
            scoring_version=scorer.scoring_version,
            candidate_text=candidate_text,
            scorer_signature=scorer.cache_signature,
            precision_signature=precision_evidence[candidate.id].cache_signature,
        )
        calibrated = _calibrate_score(
            score,
            evidence=precision_evidence[candidate.id],
            scorer=scorer,
        )
        values_by_id[candidate.id] = {
            "embedding_model": model_name,
            "semantic_positive_score": score.best_positive_similarity,
            "semantic_positive_category": score.positive_category,
            "semantic_negative_score": score.best_negative_similarity,
            "semantic_negative_category": score.negative_category,
            "semantic_travel_score": calibrated.semantic_travel_score,
            "semantic_status": calibrated.semantic_status,
            "embedding_input_hash": input_hash,
            "semantic_calculated_at": now,
            "updated_at": now,
        }
        preview = _preview_from_score(candidate, score, calibrated=calibrated)
        previews.append(preview)
        status_counts[calibrated.semantic_status] += 1

    if not dry_run:
        repo.save_semantic_results(session, values_by_id=values_by_id)

    sorted_previews = sorted(
        previews,
        key=lambda item: (-item.semantic_travel_score, item.normalized_keyword),
    )
    distinct_previews: dict[str, SemanticCandidatePreview] = {}
    for preview in sorted_previews:
        distinct_previews.setdefault(preview.normalized_keyword, preview)
    top_candidates = list(distinct_previews.values())[:20]
    estimated = len(
        {
            preview.normalized_keyword
            for preview in previews
            if preview.semantic_status in SEMANTIC_RANKING_STATUSES
        }
    )
    return SemanticFilterResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        processed=len(candidates),
        semantic_rejected=status_counts["semantic_rejected"],
        semantic_weak=status_counts["semantic_weak"],
        semantic_review=status_counts["semantic_review"],
        semantic_strong=status_counts["semantic_strong"],
        estimated_gemini_candidates=estimated,
        top_candidates=top_candidates,
        model_name=model_name,
        scoring_version=scorer.scoring_version,
        cache_hits=cache_hits,
    )


def semantic_input_hash(
    *,
    model_name: str,
    context_hash: str,
    anchor_version: str,
    scoring_version: str,
    candidate_text: str,
    scorer_signature: str,
    precision_signature: str,
) -> str:
    payload = "\x1f".join(
        (
            model_name,
            context_hash,
            anchor_version,
            scoring_version,
            scorer_signature,
            precision_signature,
            candidate_text,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def serialize_semantic_filter_result(result: SemanticFilterResult) -> dict[str, object]:
    payload = asdict(result)
    payload["top_candidates"] = [asdict(item) for item in result.top_candidates]
    return payload


def _is_cache_hit(
    candidate: TravelOpportunityCandidate,
    *,
    input_hash: str,
    model_name: str,
) -> bool:
    return bool(
        candidate.embedding_model == model_name
        and candidate.embedding_input_hash == input_hash
        and candidate.semantic_status
        in {
            "semantic_rejected",
            "semantic_weak",
            "semantic_review",
            "semantic_strong",
            "rejected",
            "weak",
            "review",
            "strong",
        }
        and candidate.semantic_travel_score is not None
        and candidate.semantic_positive_score is not None
        and candidate.semantic_positive_category
        and candidate.semantic_negative_score is not None
        and candidate.semantic_negative_category
        and candidate.semantic_calculated_at is not None
    )


def _preview_from_score(
    candidate: TravelOpportunityCandidate,
    score: SemanticScore,
    *,
    calibrated: CalibratedSemanticScore,
) -> SemanticCandidatePreview:
    return SemanticCandidatePreview(
        keyword=candidate.keyword,
        normalized_keyword=candidate.normalized_keyword,
        travel_category=candidate.travel_category,
        prefilter_status=candidate.prefilter_status,
        semantic_travel_score=calibrated.semantic_travel_score,
        semantic_status=calibrated.semantic_status,
        semantic_positive_score=score.best_positive_similarity,
        semantic_positive_category=score.positive_category,
        semantic_negative_score=score.best_negative_similarity,
        semantic_negative_category=score.negative_category,
        semantic_margin=calibrated.semantic_margin,
        semantic_confidence=calibrated.semantic_confidence,
        reasoning_codes=calibrated.reasoning_codes,
    )


def _preview_from_candidate(
    candidate: TravelOpportunityCandidate,
    *,
    calibrated: CalibratedSemanticScore,
) -> SemanticCandidatePreview:
    return SemanticCandidatePreview(
        keyword=candidate.keyword,
        normalized_keyword=candidate.normalized_keyword,
        travel_category=candidate.travel_category,
        prefilter_status=candidate.prefilter_status,
        semantic_travel_score=float(candidate.semantic_travel_score),
        semantic_status=str(candidate.semantic_status),
        semantic_positive_score=float(candidate.semantic_positive_score),
        semantic_positive_category=str(candidate.semantic_positive_category),
        semantic_negative_score=float(candidate.semantic_negative_score),
        semantic_negative_category=str(candidate.semantic_negative_category),
        semantic_margin=calibrated.semantic_margin,
        semantic_confidence=calibrated.semantic_confidence,
        reasoning_codes=calibrated.reasoning_codes,
    )


def _calibrate_score(
    score: SemanticScore,
    *,
    evidence: SemanticPrecisionEvidence,
    scorer: SemanticScorer,
) -> CalibratedSemanticScore:
    return calibrate_semantic_score(
        positive_similarity=score.best_positive_similarity,
        positive_category=score.positive_category,
        negative_similarity=score.best_negative_similarity,
        negative_category=score.negative_category,
        evidence=evidence,
        reject_threshold=scorer.reject_threshold,
        review_threshold=scorer.review_threshold,
        strong_threshold=scorer.strong_threshold,
    )


def _calibrate_stored_candidate(
    candidate: TravelOpportunityCandidate,
    *,
    evidence: SemanticPrecisionEvidence,
    scorer: SemanticScorer,
) -> CalibratedSemanticScore:
    return calibrate_semantic_score(
        positive_similarity=float(candidate.semantic_positive_score),
        positive_category=str(candidate.semantic_positive_category),
        negative_similarity=float(candidate.semantic_negative_score),
        negative_category=str(candidate.semantic_negative_category),
        evidence=evidence,
        reject_threshold=scorer.reject_threshold,
        review_threshold=scorer.review_threshold,
        strong_threshold=scorer.strong_threshold,
    )


def _empty_result(
    *,
    status: str,
    dry_run: bool,
    week_start: date | None,
    model_name: str,
    scoring_version: str,
) -> SemanticFilterResult:
    return SemanticFilterResult(
        status=status,
        dry_run=dry_run,
        week_start=week_start,
        processed=0,
        semantic_rejected=0,
        semantic_weak=0,
        semantic_review=0,
        semantic_strong=0,
        estimated_gemini_candidates=0,
        top_candidates=[],
        model_name=model_name,
        scoring_version=scoring_version,
        cache_hits=0,
    )

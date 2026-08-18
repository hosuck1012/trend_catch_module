from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import json
from typing import Sequence

from sqlalchemy.orm import Session

from app.ai.gemini_adapter import GeminiAdapter, GeminiAdapterError
from app.ai.travel_evidence_builder import (
    TravelEvidencePackage,
    build_travel_evidence_package,
)
from app.ai.travel_opportunity_prompt import SYSTEM_INSTRUCTION
from app.ai.travel_opportunity_schemas import (
    DestinationCandidate,
    FinalTravelOpportunityAnalysis,
    TravelContentIdea,
)
from app.config import get_settings
from app.models.final_travel_opportunity import FinalTravelOpportunity
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.repositories import final_travel_opportunity_repository as repo
from app.repositories import travel_opportunity_repository as opportunity_repo
from app.repositories import travel_ranking_repository as ranking_repo
from app.services.keyword_normalization_service import normalize_keyword


@dataclass(frozen=True)
class FinalizeItem:
    keyword: str
    normalized_keyword: str
    candidate_id: int
    input_chars: int
    input_hash: str
    cache_hit: bool
    would_call: bool
    gemini_called: bool
    status: str
    final_decision: str | None = None
    final_travel_score: float | None = None
    travel_angle: str | None = None
    destination_candidates: list[dict[str, object]] | None = None
    content_ideas: list[dict[str, object]] | None = None
    evidence_refs: list[str] | None = None
    needs_external_verification: bool | None = None
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class FinalizeResult:
    status: str
    dry_run: bool
    week_start: date | None
    eligible_candidates: int
    expected_gemini_calls: int
    gemini_calls: int
    cache_hits: int
    completed: int
    partial: int
    errors: int
    skipped_budget: int
    model: str | None
    prompt_version: str
    items: list[FinalizeItem]


async def finalize_travel_opportunities(
    session: Session,
    *,
    week_start: date | None,
    keyword: str | None,
    limit: int,
    force: bool,
    dry_run: bool,
    adapter: GeminiAdapter,
) -> FinalizeResult:
    settings = get_settings()
    resolved_start, _ = opportunity_repo.resolve_week_range(session, week_start)
    model = adapter.model_name or settings.gemini_model
    prompt_version = settings.travel_gemini_prompt_version
    if resolved_start is None:
        return _empty_result(
            dry_run=dry_run,
            week_start=None,
            model=model or None,
            prompt_version=prompt_version,
        )
    normalized_keyword = normalize_keyword(keyword) if keyword else None
    weekly_budget = max(settings.travel_gemini_max_candidates_per_week, 0)
    candidate_limit = min(max(limit, 0), weekly_budget)
    candidates = repo.get_eligible_candidates(
        session,
        week_start=resolved_start,
        normalized_keyword=normalized_keyword,
        limit=candidate_limit,
    ) if candidate_limit else []
    usage = repo.weekly_usage_counts(session, week_start=resolved_start)
    remaining_calls = max(weekly_budget - usage["calls"], 0)

    prepared: list[
        tuple[TravelOpportunityCandidate, TravelEvidencePackage, FinalTravelOpportunity | None]
    ] = []
    for candidate in candidates:
        package = build_travel_evidence_package(
            session,
            candidate=candidate,
            settings=settings,
            model=model,
        )
        cached = None
        if settings.travel_gemini_cache_enabled and not force:
            cached = repo.get_cache_hit(
                session,
                normalized_keyword=candidate.normalized_keyword,
                week_start=candidate.week_start,
                model=model,
                prompt_version=prompt_version,
                input_hash=package.input_hash,
            )
        prepared.append((candidate, package, cached))

    expected_calls = min(
        sum(cached is None for _, _, cached in prepared),
        remaining_calls,
    )
    if dry_run:
        remaining_preview = remaining_calls
        items = []
        for candidate, package, cached in prepared:
            would_call = cached is None and remaining_preview > 0
            if would_call:
                remaining_preview -= 1
            items.append(
                FinalizeItem(
                    keyword=candidate.keyword,
                    normalized_keyword=candidate.normalized_keyword,
                    candidate_id=candidate.id,
                    input_chars=package.input_chars,
                    input_hash=package.input_hash,
                    cache_hit=cached is not None,
                    would_call=would_call,
                    gemini_called=False,
                    status="cached" if cached else "ready" if would_call else "budget_exhausted",
                )
            )
        return FinalizeResult(
            status="dry_run",
            dry_run=True,
            week_start=resolved_start,
            eligible_candidates=len(candidates),
            expected_gemini_calls=expected_calls,
            gemini_calls=0,
            cache_hits=sum(cached is not None for _, _, cached in prepared),
            completed=0,
            partial=0,
            errors=0,
            skipped_budget=sum(
                item.status == "budget_exhausted" for item in items
            ),
            model=model or None,
            prompt_version=prompt_version,
            items=items,
        )

    items: list[FinalizeItem] = []
    counters: Counter[str] = Counter()
    for candidate, package, cached in prepared:
        now = _utc_now()
        if cached is not None:
            repo.record_cache_hit(session, row=cached, now=now)
            counters["cache_hits"] += 1
            items.append(_item_from_row(candidate, package, cached, cache_hit=True))
            continue
        if remaining_calls <= 0:
            counters["skipped_budget"] += 1
            items.append(
                FinalizeItem(
                    keyword=candidate.keyword,
                    normalized_keyword=candidate.normalized_keyword,
                    candidate_id=candidate.id,
                    input_chars=package.input_chars,
                    input_hash=package.input_hash,
                    cache_hit=False,
                    would_call=False,
                    gemini_called=False,
                    status="budget_exhausted",
                )
            )
            continue
        request_count_before = int(getattr(adapter, "request_count", 0))
        try:
            generated = await adapter.generate_structured(
                user_prompt=package.user_prompt,
                response_model=FinalTravelOpportunityAnalysis,
                system_instruction=SYSTEM_INSTRUCTION,
            )
            if not isinstance(generated, FinalTravelOpportunityAnalysis):
                generated = FinalTravelOpportunityAnalysis.model_validate(generated)
            corrected, analysis_status = validate_final_analysis(
                generated,
                candidate=candidate,
                package=package,
            )
            row = repo.save_success(
                session,
                candidate=candidate,
                model=model,
                prompt_version=prompt_version,
                input_hash=package.input_hash,
                input_chars=package.input_chars,
                analysis=corrected,
                analysis_status=analysis_status,
                now=now,
            )
            counters[analysis_status] += 1
            counters["gemini_calls"] += 1
            remaining_calls -= 1
            items.append(_item_from_row(candidate, package, row, cache_hit=False))
        except Exception as exc:
            request_count_after = int(getattr(adapter, "request_count", 0))
            called = request_count_after > request_count_before or not hasattr(
                adapter, "request_count"
            )
            error_code = (
                exc.code if isinstance(exc, GeminiAdapterError) else "analysis_error"
            )
            repo.save_error(
                session,
                candidate=candidate,
                model=model,
                prompt_version=prompt_version,
                input_hash=package.input_hash,
                input_chars=package.input_chars,
                error_code=error_code,
                error_message=str(exc),
                now=now,
                count_as_call=called,
            )
            counters["errors"] += 1
            if called:
                counters["gemini_calls"] += 1
                remaining_calls -= 1
            items.append(
                FinalizeItem(
                    keyword=candidate.keyword,
                    normalized_keyword=candidate.normalized_keyword,
                    candidate_id=candidate.id,
                    input_chars=package.input_chars,
                    input_hash=package.input_hash,
                    cache_hit=False,
                    would_call=True,
                    gemini_called=called,
                    status="error",
                    error_code=error_code,
                    error_message=str(exc),
                )
            )
    return FinalizeResult(
        status="ok" if not counters["errors"] else "partial",
        dry_run=False,
        week_start=resolved_start,
        eligible_candidates=len(candidates),
        expected_gemini_calls=expected_calls,
        gemini_calls=counters["gemini_calls"],
        cache_hits=counters["cache_hits"],
        completed=counters["completed"],
        partial=counters["partial"],
        errors=counters["errors"],
        skipped_budget=counters["skipped_budget"],
        model=model or None,
        prompt_version=prompt_version,
        items=items,
    )


def validate_final_analysis(
    analysis: FinalTravelOpportunityAnalysis,
    *,
    candidate: TravelOpportunityCandidate,
    package: TravelEvidencePackage,
) -> tuple[FinalTravelOpportunityAnalysis, str]:
    valid_refs = []
    for evidence_ref in analysis.evidence_refs:
        if evidence_ref in package.valid_evidence_refs and evidence_ref not in valid_refs:
            valid_refs.append(evidence_ref)
    invalid_refs = len(valid_refs) != len(analysis.evidence_refs)

    destination_rows = []
    invalid_destinations: list[str] = []
    for destination in analysis.destination_candidates[:3]:
        verified = _destination_in_package(destination.name, package) and (
            destination.evidence_ref in package.valid_evidence_refs
        )
        if not verified:
            invalid_destinations.append(destination.name)
        destination_rows.append(
            destination.model_copy(update={"verified_from_input": verified})
        )

    content_ideas = []
    for idea in analysis.content_ideas[:3]:
        destination = idea.destination
        if destination and not _destination_in_package(destination, package):
            destination = "추가 검증 필요"
            if idea.destination not in invalid_destinations:
                invalid_destinations.append(idea.destination)
        content_ideas.append(idea.model_copy(update={"destination": destination}))

    confidence = analysis.confidence_score
    decision = analysis.final_decision
    needs_verification = analysis.needs_external_verification or bool(
        invalid_destinations
    )
    verification_queries = list(dict.fromkeys(analysis.verification_queries))[:3]
    cautions = list(dict.fromkeys(analysis.cautions))[:5]
    if invalid_destinations:
        caution = "입력 근거에서 확인되지 않은 장소를 최종 여행지로 확정하지 않았습니다."
        if caution not in cautions:
            cautions.append(caution)
        for name in invalid_destinations:
            query = f"{candidate.keyword} {name} 여행 근거"
            if query not in verification_queries and len(verification_queries) < 3:
                verification_queries.append(query)
    if not valid_refs:
        confidence = min(confidence, 40)
        needs_verification = True
        if decision == "accept":
            decision = "review"
    if "NEGATIVE_SEMANTIC_DOMINANT" in _json_list(
        candidate.evidence_codes_json
    ):
        decision = "reject"
    elif decision == "accept" and not _accept_allowed(
        candidate=candidate,
        final_travel_score=analysis.final_travel_score,
        confidence_score=confidence,
        has_valid_ref=bool(valid_refs),
    ):
        decision = "review"
    if candidate.evidence_gate == "NEEDS_EVIDENCE" and decision == "accept":
        decision = "review"
    if decision == "reject":
        destination_rows = []
        content_ideas = []

    corrected = analysis.model_copy(
        update={
            "keyword": candidate.keyword,
            "final_decision": decision,
            "destination_candidates": destination_rows,
            "content_ideas": content_ideas,
            "evidence_refs": valid_refs,
            "needs_external_verification": needs_verification,
            "verification_queries": verification_queries,
            "cautions": cautions[:5],
            "confidence_score": confidence,
        }
    )
    partial = invalid_refs or bool(invalid_destinations) or not valid_refs
    return corrected, "partial" if partial else "completed"


def serialize_finalize_result(result: FinalizeResult) -> dict[str, object]:
    payload = asdict(result)
    for item in payload["items"]:
        item["destination_candidates"] = item["destination_candidates"] or []
        item["content_ideas"] = item["content_ideas"] or []
        item["evidence_refs"] = item["evidence_refs"] or []
    return payload


def serialize_final(row: FinalTravelOpportunity) -> dict[str, object]:
    return {
        "id": row.id,
        "keyword": row.keyword,
        "normalized_keyword": row.normalized_keyword,
        "week_start": row.week_start,
        "week_end": row.week_end,
        "final_decision": row.final_decision,
        "final_travel_score": row.final_travel_score,
        "trend_context_summary": row.trend_context_summary,
        "why_now": row.why_now,
        "travel_angle": row.travel_angle,
        "destination_candidates": _json_list(row.destinations_json),
        "content_ideas": _json_list(row.content_ideas_json),
        "evidence_refs": _json_list(row.evidence_refs_json),
        "needs_external_verification": row.needs_external_verification,
        "verification_queries": _json_list(row.verification_queries_json),
        "cautions": _json_list(row.cautions_json),
        "confidence_score": row.confidence_score,
        "analysis_status": row.analysis_status,
        "model": row.gemini_model,
        "prompt_version": row.prompt_version,
        "generated_at": row.generated_at,
    }


def cost_report(session: Session, *, week_start: date | None) -> dict[str, object]:
    resolved_start, _ = opportunity_repo.resolve_week_range(session, week_start)
    funnel = ranking_repo.funnel_counts(session, week_start=resolved_start)
    if resolved_start is None:
        usage = {"calls": 0, "cache_hits": 0, "errors": 0}
        accepts = 0
    else:
        usage = repo.weekly_usage_counts(session, week_start=resolved_start)
        accepts = sum(
            row.final_decision == "accept"
            for row in repo.list_final_opportunities(
                session,
                week_start=resolved_start,
                decision=None,
                min_score=None,
                limit=10000,
            )
        )
    raw = funnel["raw"]
    eligible = funnel["gemini_eligible"]
    reduction = round((1 - eligible / raw) * 100, 2) if raw else 0.0
    return {
        "week_start": resolved_start,
        "raw_keyword_count": raw,
        "quality_keyword_count": funnel["quality"],
        "rule_candidate_count": funnel["rule"],
        "semantic_candidate_count": funnel["semantic"],
        "high_precision_candidate_count": funnel["high_precision"],
        "gemini_eligible_count": eligible,
        "final_accept_count": accepts,
        "gemini_calls_this_week": usage["calls"],
        "gemini_cache_hits": usage["cache_hits"],
        "gemini_errors": usage["errors"],
        "overall_llm_reduction_rate": reduction,
        "estimated_calls_per_year": _estimated_calls_per_year(session),
    }


def _accept_allowed(
    *,
    candidate: TravelOpportunityCandidate,
    final_travel_score: int,
    confidence_score: int,
    has_valid_ref: bool,
) -> bool:
    return bool(
        (candidate.high_precision_score or 0) >= 85
        and candidate.evidence_gate == "PASS"
        and has_valid_ref
        and final_travel_score >= 75
        and confidence_score >= 60
    )


def _destination_in_package(
    name: str,
    package: TravelEvidencePackage,
) -> bool:
    normalized_name = _normalize(name)
    if not normalized_name:
        return False
    if any(
        normalized_name == _normalize(allowed)
        for allowed in package.allowed_destination_names
    ):
        return True
    return any(
        normalized_name in _normalize(context_text)
        for context_text in package.context_texts
    )


def _item_from_row(
    candidate: TravelOpportunityCandidate,
    package: TravelEvidencePackage,
    row: FinalTravelOpportunity,
    *,
    cache_hit: bool,
) -> FinalizeItem:
    return FinalizeItem(
        keyword=candidate.keyword,
        normalized_keyword=candidate.normalized_keyword,
        candidate_id=candidate.id,
        input_chars=package.input_chars,
        input_hash=package.input_hash,
        cache_hit=cache_hit,
        would_call=not cache_hit,
        gemini_called=not cache_hit,
        status="cached" if cache_hit else row.analysis_status,
        final_decision=row.final_decision,
        final_travel_score=row.final_travel_score,
        travel_angle=row.travel_angle,
        destination_candidates=_json_list(row.destinations_json),
        content_ideas=_json_list(row.content_ideas_json),
        evidence_refs=_json_list(row.evidence_refs_json),
        needs_external_verification=row.needs_external_verification,
    )


def _estimated_calls_per_year(session: Session) -> float:
    history = repo.usage_history(session)
    if not history:
        return 0.0
    weeks = [row[0] for row in history]
    calls = sum(int(row[1] or 0) for row in history)
    observed_weeks = max(((max(weeks) - min(weeks)).days // 7) + 1, len(set(weeks)))
    return round(calls / observed_weeks * 52, 2)


def _empty_result(
    *,
    dry_run: bool,
    week_start: date | None,
    model: str | None,
    prompt_version: str,
) -> FinalizeResult:
    return FinalizeResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=week_start,
        eligible_candidates=0,
        expected_gemini_calls=0,
        gemini_calls=0,
        cache_hits=0,
        completed=0,
        partial=0,
        errors=0,
        skipped_budget=0,
        model=model,
        prompt_version=prompt_version,
        items=[],
    )


def _normalize(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

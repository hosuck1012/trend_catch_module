from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib

from app.ai.evidence_builder import (
    EvidenceInputTooLargeError,
    EvidencePackage,
    build_evidence_package,
)
from app.ai.gemini_adapter import GeminiAdapter, GeminiAdapterError
from app.ai.gemini_prompt import PROMPT_VERSION
from app.ai.response_validator import validate_explanation
from app.config import get_settings
from app.database import SessionLocal
from app.models.trend_ai_analysis import TrendAIAnalysis
from app.repositories.trend_ai_repository import (
    complete_analysis,
    fail_analysis,
    get_analysis_targets,
    get_cached_analysis,
    upsert_pending_analysis,
)
from app.services.keyword_normalization_service import normalize_keyword


class AIAnalysisTargetNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class AIAnalysisItemResult:
    keyword: str
    week_start: date
    analysis_id: int | None
    status: str
    error_code: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class AIAnalysisRunResult:
    status: str
    requested: int
    completed: int
    partial: int
    skipped: int
    errors: int
    model_name: str
    prompt_version: str
    results: tuple[AIAnalysisItemResult, ...]


async def generate_trend_analyses(
    *,
    keyword: str | None,
    limit: int,
    force: bool,
    week_start: date | None,
    adapter=None,
) -> AIAnalysisRunResult:
    settings = get_settings()
    gemini = adapter or GeminiAdapter()
    owns_adapter = adapter is None
    ensure_configured = getattr(gemini, "ensure_configured", None)
    if callable(ensure_configured):
        ensure_configured()
    model_name = str(getattr(gemini, "model_name", settings.gemini_model))
    effective_limit = min(
        max(limit, 1),
        max(settings.gemini_max_items_per_run, 1),
        5,
    )
    if keyword:
        effective_limit = 1
    normalized_keyword = normalize_keyword(keyword) if keyword else None
    if keyword and not normalized_keyword:
        raise AIAnalysisTargetNotFoundError("분석할 키워드를 찾을 수 없습니다.")
    with SessionLocal() as session:
        targets = get_analysis_targets(
            session,
            normalized_keyword=normalized_keyword,
            week_start=week_start,
            limit=effective_limit,
        )
    if keyword and not targets:
        raise AIAnalysisTargetNotFoundError("분석할 주간 트렌드 키워드를 찾을 수 없습니다.")

    items: list[AIAnalysisItemResult] = []
    completed = partial = skipped = errors = 0
    try:
        for trend in targets:
            now = _utc_now()
            analysis_id: int | None = None
            try:
                with SessionLocal() as session:
                    package = build_evidence_package(
                        session,
                        trend=trend,
                        normalized_keyword=trend.keyword,
                        model_name=model_name,
                        prompt_version=PROMPT_VERSION,
                    )
                    cached = None
                    if not force:
                        cached = get_cached_analysis(
                            session,
                            normalized_keyword=trend.keyword,
                            week_start=trend.week_start,
                            model_name=model_name,
                            prompt_version=PROMPT_VERSION,
                            input_hash=package.input_hash,
                            now=now,
                            cache_hours=settings.gemini_analysis_cache_hours,
                        )
                    if cached is not None:
                        skipped += 1
                        items.append(
                            AIAnalysisItemResult(
                                keyword=trend.keyword,
                                week_start=trend.week_start,
                                analysis_id=cached.id,
                                status="skipped",
                            )
                        )
                        continue
                    pending = upsert_pending_analysis(
                        session,
                        trend=trend,
                        normalized_keyword=trend.keyword,
                        model_name=model_name,
                        prompt_version=PROMPT_VERSION,
                        input_hash=package.input_hash,
                        now=now,
                    )
                    session.commit()
                    analysis_id = pending.id

                raw_explanation = await gemini.generate(user_prompt=package.user_prompt)
                raw_response = raw_explanation.model_dump(mode="json")
                validated = validate_explanation(raw_explanation, package)
                finished_at = _utc_now()
                with SessionLocal() as session:
                    row = session.get(TrendAIAnalysis, analysis_id)
                    if row is None:
                        raise RuntimeError("pending 분석 행을 찾을 수 없습니다.")
                    complete_analysis(
                        row,
                        analysis_status=validated.analysis_status,
                        explanation=validated.explanation,
                        raw_response=raw_response,
                        now=finished_at,
                    )
                    session.commit()
                if validated.analysis_status == "partial":
                    partial += 1
                else:
                    completed += 1
                items.append(
                    AIAnalysisItemResult(
                        keyword=trend.keyword,
                        week_start=trend.week_start,
                        analysis_id=analysis_id,
                        status=validated.analysis_status,
                    )
                )
            except EvidenceInputTooLargeError as exc:
                errors += 1
                row_id = _save_preparation_error(
                    trend=trend,
                    model_name=model_name,
                    error_code="input_too_large",
                    error_message=str(exc),
                )
                items.append(
                    AIAnalysisItemResult(
                        trend.keyword,
                        trend.week_start,
                        row_id,
                        "error",
                        "input_too_large",
                        str(exc),
                    )
                )
            except GeminiAdapterError as exc:
                errors += 1
                _mark_error(analysis_id, code=exc.code, message=str(exc))
                items.append(
                    AIAnalysisItemResult(
                        trend.keyword,
                        trend.week_start,
                        analysis_id,
                        "error",
                        exc.code,
                        str(exc),
                    )
                )
            except Exception:
                errors += 1
                safe_message = "AI 분석 처리 중 내부 오류가 발생했습니다."
                if isinstance(analysis_id, int):
                    _mark_error(analysis_id, code="internal_error", message=safe_message)
                items.append(
                    AIAnalysisItemResult(
                        trend.keyword,
                        trend.week_start,
                        analysis_id if isinstance(analysis_id, int) else None,
                        "error",
                        "internal_error",
                        safe_message,
                    )
                )
    finally:
        if owns_adapter:
            await gemini.close()

    return AIAnalysisRunResult(
        status="partial_success" if errors else "ok",
        requested=len(targets),
        completed=completed,
        partial=partial,
        skipped=skipped,
        errors=errors,
        model_name=model_name,
        prompt_version=PROMPT_VERSION,
        results=tuple(items),
    )


def _save_preparation_error(
    *,
    trend,
    model_name: str,
    error_code: str,
    error_message: str,
) -> int:
    now = _utc_now()
    fallback_hash = hashlib.sha256(
        f"{trend.id}:{model_name}:{PROMPT_VERSION}:{error_code}".encode("utf-8")
    ).hexdigest()
    with SessionLocal() as session:
        row = upsert_pending_analysis(
            session,
            trend=trend,
            normalized_keyword=trend.keyword,
            model_name=model_name,
            prompt_version=PROMPT_VERSION,
            input_hash=fallback_hash,
            now=now,
        )
        fail_analysis(
            row,
            error_code=error_code,
            error_message=error_message,
            now=now,
        )
        session.commit()
        return row.id


def _mark_error(analysis_id: int, *, code: str, message: str) -> None:
    with SessionLocal() as session:
        row = session.get(TrendAIAnalysis, analysis_id)
        if row is None:
            return
        fail_analysis(
            row,
            error_code=code,
            error_message=message,
            now=_utc_now(),
        )
        session.commit()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

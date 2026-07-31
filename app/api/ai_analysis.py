from collections.abc import AsyncGenerator
from datetime import date, datetime, timezone
import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.ai.gemini_adapter import (
    GeminiAdapter,
    GeminiConfigurationError,
)
from app.ai.gemini_prompt import PROMPT_VERSION
from app.config import get_settings
from app.database import get_db
from app.models.trend_ai_analysis import TrendAIAnalysis
from app.repositories.trend_ai_repository import (
    get_ai_status_counts,
    get_latest_analysis_for_keyword,
    get_source_contexts_for_analysis,
    list_analyses,
)
from app.schemas.trend_ai import (
    AIAnalysisGenerateResponse,
    AIAnalysisListResponse,
    AIAnalysisResponse,
    AIAnalysisStatusResponse,
)
from app.services.keyword_normalization_service import normalize_keyword
from app.services.trend_ai_analysis_service import (
    AIAnalysisTargetNotFoundError,
    generate_trend_analyses,
)


router = APIRouter(prefix="/api/ai-analysis", tags=["ai-analysis"])


async def get_gemini_adapter() -> AsyncGenerator[GeminiAdapter, None]:
    adapter = GeminiAdapter()
    try:
        yield adapter
    finally:
        await adapter.close()


@router.post("/generate", response_model=AIAnalysisGenerateResponse)
async def generate(
    keyword: str | None = Query(default=None, min_length=2, max_length=255),
    limit: int = Query(default=5, ge=1, le=20),
    force: bool = Query(default=False),
    week_start: date | None = Query(default=None),
    adapter: GeminiAdapter = Depends(get_gemini_adapter),
) -> dict[str, object]:
    try:
        result = await generate_trend_analyses(
            keyword=keyword,
            limit=limit,
            force=force,
            week_start=week_start,
            adapter=adapter,
        )
    except GeminiConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AIAnalysisTargetNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "status": result.status,
        "requested": result.requested,
        "completed": result.completed,
        "partial": result.partial,
        "skipped": result.skipped,
        "errors": result.errors,
        "model_name": result.model_name,
        "prompt_version": result.prompt_version,
        "results": [item.__dict__ for item in result.results],
    }


@router.get("/status", response_model=AIAnalysisStatusResponse)
def status(session: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    counts = get_ai_status_counts(
        session,
        now=datetime.now(timezone.utc).replace(tzinfo=None),
        cache_hours=settings.gemini_analysis_cache_hours,
    )
    return {
        "gemini_enabled": settings.gemini_enabled,
        "model_configured": bool(settings.gemini_model),
        "api_key_configured": bool(settings.gemini_api_key),
        "configured_model": settings.gemini_model or None,
        "completed_count": counts.completed_count,
        "partial_count": counts.partial_count,
        "error_count": counts.error_count,
        "cached_count": counts.cached_count,
        "last_generated_at": counts.last_generated_at,
        "prompt_version": PROMPT_VERSION,
    }


@router.get("/by-keyword/{normalized_keyword}", response_model=AIAnalysisResponse)
def by_keyword(
    normalized_keyword: str,
    week_start: date | None = Query(default=None),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    normalized = normalize_keyword(normalized_keyword)
    if not normalized:
        raise HTTPException(status_code=404, detail="AI 분석 결과를 찾을 수 없습니다.")
    analysis = get_latest_analysis_for_keyword(
        session,
        normalized_keyword=normalized,
        week_start=week_start,
    )
    if analysis is None:
        raise HTTPException(status_code=404, detail="AI 분석 결과를 찾을 수 없습니다.")
    contexts = get_source_contexts_for_analysis(session, analysis)
    return _analysis_response(analysis, contexts=contexts)


@router.get("", response_model=AIAnalysisListResponse)
def analyses(
    week_start: date | None = Query(default=None),
    status: Literal["pending", "completed", "partial", "skipped", "error"] | None = Query(default=None),
    travel_relevance_level: Literal["high", "medium", "low", "none"] | None = Query(default=None),
    min_travel_score: float | None = Query(default=None, ge=0, le=100),
    limit: int = Query(default=20, ge=1, le=100),
    session: Session = Depends(get_db),
) -> dict[str, object]:
    rows = list_analyses(
        session,
        week_start=week_start,
        status=status,
        travel_level=travel_relevance_level,
        min_travel_score=min_travel_score,
        limit=limit,
    )
    return {
        "total": len(rows),
        "items": [_analysis_response(row, contexts=[]) for row in rows],
    }


def _analysis_response(
    analysis: TrendAIAnalysis,
    *,
    contexts,
) -> dict[str, object]:
    return {
        "id": analysis.id,
        "keyword": analysis.keyword,
        "normalized_keyword": analysis.normalized_keyword,
        "week_start": analysis.week_start,
        "week_end": analysis.week_end,
        "analysis_status": analysis.analysis_status,
        "trend_summary": analysis.trend_summary,
        "rising_reason": analysis.rising_reason,
        "evidence_summary": _json_list(analysis.evidence_summary),
        "travel_relevance_score": analysis.travel_relevance_score,
        "travel_relevance_level": analysis.travel_relevance_level,
        "travel_relevance_reason": analysis.travel_relevance_reason,
        "recommended_destinations": _json_list(
            analysis.recommended_destinations_json
        ),
        "content_ideas": _json_list(analysis.content_ideas_json),
        "cautions": _json_list(analysis.cautions_json),
        "evidence_refs": _json_list(analysis.evidence_refs_json),
        "confidence_score": analysis.confidence_score,
        "model_name": analysis.model_name,
        "prompt_version": analysis.prompt_version,
        "generated_at": analysis.generated_at,
        "error_code": analysis.error_code,
        "error_message": analysis.error_message,
        "source_contexts": [
            {
                "provider": context.provider,
                "page_title": context.page_title,
                "page_url": context.page_url,
                "match_status": context.match_status,
            }
            for context in contexts
        ],
    }


def _json_list(raw: str | None) -> list:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return value if isinstance(value, list) else []

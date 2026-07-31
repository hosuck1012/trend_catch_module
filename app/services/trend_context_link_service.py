from datetime import datetime

from sqlalchemy.orm import Session

from app.context.provider_registry import TRAVEL_SUITABILITY, provider_source_score
from app.models.entity_context import EntityContext
from app.repositories.context_repository import (
    ContextTarget,
    recalculate_primary_context,
    upsert_trend_context_link,
)


def calculate_context_score(
    *,
    relation_score: float,
    match_score: float,
    provider: str,
    match_status: str,
    entity_type: str,
) -> float:
    score = (
        min(max(relation_score, 0.0), 100.0) * 0.45
        + min(max(match_score, 0.0), 1.0) * 100 * 0.35
        + provider_source_score(provider, match_status) * 0.10
        + TRAVEL_SUITABILITY.get(entity_type, 0.0) * 0.10
    )
    return round(min(max(score, 0.0), 100.0), 2)


def link_context_to_trend(
    session: Session,
    *,
    target: ContextTarget,
    context: EntityContext,
    now: datetime,
) -> None:
    context_score = calculate_context_score(
        relation_score=target.relation_score,
        match_score=context.match_score,
        provider=context.provider,
        match_status=context.match_status,
        entity_type=target.entity_type,
    )
    upsert_trend_context_link(
        session,
        target=target,
        context=context,
        context_score=context_score,
        now=now,
    )
    recalculate_primary_context(
        session,
        keyword=target.keyword,
        week_start=target.week_start,
    )

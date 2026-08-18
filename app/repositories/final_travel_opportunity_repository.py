from datetime import date, datetime
import json

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session, joinedload

from app.ai.travel_opportunity_schemas import FinalTravelOpportunityAnalysis
from app.models.final_travel_opportunity import FinalTravelOpportunity
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate


ELIGIBLE_RANKING_STATUSES = ("gemini_candidate", "priority_candidate")


def get_eligible_candidates(
    session: Session,
    *,
    week_start: date,
    normalized_keyword: str | None,
    limit: int,
) -> list[TravelOpportunityCandidate]:
    priority_order = case(
        (TravelOpportunityCandidate.ranking_status == "priority_candidate", 0),
        else_=1,
    )
    query = (
        select(TravelOpportunityCandidate)
        .options(joinedload(TravelOpportunityCandidate.keyword_context))
        .where(
            TravelOpportunityCandidate.week_start == week_start,
            TravelOpportunityCandidate.gemini_eligible.is_(True),
            TravelOpportunityCandidate.cluster_representative.is_(True),
            TravelOpportunityCandidate.ranking_status.in_(ELIGIBLE_RANKING_STATUSES),
        )
        .order_by(
            priority_order,
            TravelOpportunityCandidate.high_precision_score.desc(),
            TravelOpportunityCandidate.normalized_keyword.asc(),
        )
        .limit(max(limit * 20, limit))
    )
    if normalized_keyword:
        query = query.where(
            TravelOpportunityCandidate.normalized_keyword == normalized_keyword
        )
    rows = session.scalars(query).unique().all()
    distinct: dict[str, TravelOpportunityCandidate] = {}
    for row in rows:
        distinct.setdefault(row.normalized_keyword, row)
        if len(distinct) >= limit:
            break
    return list(distinct.values())


def get_analysis(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date,
    model: str,
    prompt_version: str,
) -> FinalTravelOpportunity | None:
    return session.scalar(
        select(FinalTravelOpportunity).where(
            FinalTravelOpportunity.normalized_keyword == normalized_keyword,
            FinalTravelOpportunity.week_start == week_start,
            FinalTravelOpportunity.gemini_model == model,
            FinalTravelOpportunity.prompt_version == prompt_version,
        )
    )


def get_cache_hit(
    session: Session,
    *,
    normalized_keyword: str,
    week_start: date,
    model: str,
    prompt_version: str,
    input_hash: str,
) -> FinalTravelOpportunity | None:
    return session.scalar(
        select(FinalTravelOpportunity).where(
            FinalTravelOpportunity.normalized_keyword == normalized_keyword,
            FinalTravelOpportunity.week_start == week_start,
            FinalTravelOpportunity.gemini_model == model,
            FinalTravelOpportunity.prompt_version == prompt_version,
            FinalTravelOpportunity.input_hash == input_hash,
            FinalTravelOpportunity.analysis_status.in_(
                ("completed", "partial", "cached")
            ),
        )
    )


def record_cache_hit(
    session: Session,
    *,
    row: FinalTravelOpportunity,
    now: datetime,
) -> None:
    row.analysis_status = "cached"
    row.cache_hit_count += 1
    row.updated_at = now
    session.commit()


def save_success(
    session: Session,
    *,
    candidate: TravelOpportunityCandidate,
    model: str,
    prompt_version: str,
    input_hash: str,
    input_chars: int,
    analysis: FinalTravelOpportunityAnalysis,
    analysis_status: str,
    now: datetime,
) -> FinalTravelOpportunity:
    row = get_analysis(
        session,
        normalized_keyword=candidate.normalized_keyword,
        week_start=candidate.week_start,
        model=model,
        prompt_version=prompt_version,
    )
    if row is None:
        row = FinalTravelOpportunity(
            normalized_keyword=candidate.normalized_keyword,
            week_start=candidate.week_start,
            gemini_model=model,
            prompt_version=prompt_version,
            generated_at=now,
            updated_at=now,
        )
        session.add(row)
    row.keyword = candidate.keyword
    row.week_end = candidate.week_end
    row.travel_opportunity_candidate_id = candidate.id
    row.input_hash = input_hash
    row.final_decision = analysis.final_decision
    row.final_travel_score = analysis.final_travel_score
    row.trend_context_summary = analysis.trend_context_summary
    row.why_now = analysis.why_now
    row.travel_angle = analysis.travel_angle
    row.destinations_json = _json_dump(
        [item.model_dump(mode="json") for item in analysis.destination_candidates]
    )
    row.content_ideas_json = _json_dump(
        [item.model_dump(mode="json") for item in analysis.content_ideas]
    )
    row.evidence_refs_json = _json_dump(analysis.evidence_refs)
    row.verification_queries_json = _json_dump(analysis.verification_queries)
    row.cautions_json = _json_dump(analysis.cautions)
    row.needs_external_verification = analysis.needs_external_verification
    row.confidence_score = analysis.confidence_score
    row.analysis_status = analysis_status
    row.gemini_call_count = (row.gemini_call_count or 0) + 1
    row.input_chars = input_chars
    row.error_code = None
    row.error_message = None
    row.generated_at = now
    row.updated_at = now
    session.commit()
    session.refresh(row)
    return row


def save_error(
    session: Session,
    *,
    candidate: TravelOpportunityCandidate,
    model: str,
    prompt_version: str,
    input_hash: str,
    input_chars: int,
    error_code: str,
    error_message: str,
    now: datetime,
    count_as_call: bool,
) -> FinalTravelOpportunity:
    row = get_analysis(
        session,
        normalized_keyword=candidate.normalized_keyword,
        week_start=candidate.week_start,
        model=model,
        prompt_version=prompt_version,
    )
    if row is None:
        row = FinalTravelOpportunity(
            normalized_keyword=candidate.normalized_keyword,
            week_start=candidate.week_start,
            gemini_model=model,
            prompt_version=prompt_version,
            cache_hit_count=0,
            gemini_call_count=0,
            generated_at=now,
            updated_at=now,
        )
        session.add(row)
    row.keyword = candidate.keyword
    row.week_end = candidate.week_end
    row.travel_opportunity_candidate_id = candidate.id
    row.input_hash = input_hash
    row.final_decision = "review"
    row.final_travel_score = 0
    row.trend_context_summary = "분석 오류로 결과를 생성하지 못했습니다."
    row.why_now = "분석 오류로 확인할 수 없습니다."
    row.travel_angle = "추가 분석이 필요합니다."
    row.destinations_json = "[]"
    row.content_ideas_json = "[]"
    row.evidence_refs_json = "[]"
    row.verification_queries_json = "[]"
    row.cautions_json = _json_dump(["Gemini 분석 오류가 발생했습니다."])
    row.needs_external_verification = True
    row.confidence_score = 0
    row.analysis_status = "error"
    row.gemini_call_count = (row.gemini_call_count or 0) + int(count_as_call)
    row.input_chars = input_chars
    row.error_code = error_code[:100]
    row.error_message = error_message[:1000]
    row.updated_at = now
    session.commit()
    session.refresh(row)
    return row


def list_final_opportunities(
    session: Session,
    *,
    week_start: date | None,
    decision: str | None,
    min_score: float | None,
    limit: int,
) -> list[FinalTravelOpportunity]:
    query = select(FinalTravelOpportunity)
    if week_start:
        query = query.where(FinalTravelOpportunity.week_start == week_start)
    if decision:
        query = query.where(FinalTravelOpportunity.final_decision == decision)
    if min_score is not None:
        query = query.where(FinalTravelOpportunity.final_travel_score >= min_score)
    return list(
        session.scalars(
            query.order_by(
                case(
                    (FinalTravelOpportunity.final_decision == "accept", 0),
                    (FinalTravelOpportunity.final_decision == "review", 1),
                    else_=2,
                ),
                FinalTravelOpportunity.final_travel_score.desc(),
                FinalTravelOpportunity.updated_at.desc(),
            ).limit(limit)
        ).all()
    )


def get_latest_final(
    session: Session,
    *,
    normalized_keyword: str,
) -> FinalTravelOpportunity | None:
    return session.scalar(
        select(FinalTravelOpportunity)
        .where(FinalTravelOpportunity.normalized_keyword == normalized_keyword)
        .order_by(
            FinalTravelOpportunity.week_start.desc(),
            FinalTravelOpportunity.updated_at.desc(),
        )
        .limit(1)
    )


def weekly_usage_counts(session: Session, *, week_start: date) -> dict[str, int]:
    row = session.execute(
        select(
            func.coalesce(func.sum(FinalTravelOpportunity.gemini_call_count), 0),
            func.coalesce(func.sum(FinalTravelOpportunity.cache_hit_count), 0),
            func.count(FinalTravelOpportunity.id).filter(
                FinalTravelOpportunity.analysis_status == "error"
            ),
        ).where(FinalTravelOpportunity.week_start == week_start)
    ).one()
    return {
        "calls": int(row[0]),
        "cache_hits": int(row[1]),
        "errors": int(row[2]),
    }


def usage_history(session: Session) -> list[tuple[date, int]]:
    return list(
        session.execute(
            select(
                FinalTravelOpportunity.week_start,
                func.sum(FinalTravelOpportunity.gemini_call_count),
            ).group_by(FinalTravelOpportunity.week_start)
        ).all()
    )


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

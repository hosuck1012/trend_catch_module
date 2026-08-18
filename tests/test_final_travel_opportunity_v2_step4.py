from datetime import date, datetime, timedelta

import pytest
from pydantic import ValidationError
from sqlalchemy import select

from app.ai.travel_evidence_builder import (
    TravelEvidencePackage,
    build_travel_evidence_package,
)
from app.ai.travel_opportunity_schemas import (
    DestinationCandidate,
    FinalTravelOpportunityAnalysis,
    TravelContentIdea,
)
from app.api.travel_opportunities import get_final_gemini_adapter
from app.config import get_settings
from app.main import app
from app.models.entity_mention import EntityMention
from app.models.keyword_candidate import KeywordCandidate
from app.models.entity_context import EntityContext
from app.models.final_travel_opportunity import FinalTravelOpportunity
from app.models.keyword_context import KeywordContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_context_link import TrendContextLink
from app.models.weekly_trend import WeeklyTrend
from app.services.final_travel_opportunity_service import validate_final_analysis
from dashboard.travel_opportunity_formatter import (
    final_destination_names,
    final_funnel_cards,
    split_final_opportunities,
)


WEEK_START = date(2026, 8, 10)
WEEK_END = WEEK_START + timedelta(days=6)
NOW = datetime(2026, 8, 13, 12, 0, 0)


class FakeGeminiAdapter:
    def __init__(self, response: FinalTravelOpportunityAnalysis | None = None) -> None:
        self.model_name = "gemini-3.6-flash"
        self.request_count = 0
        self.calls: list[str] = []
        self.response = response or _accept_response()

    async def generate_structured(self, *, user_prompt, response_model, system_instruction):
        self.request_count += 1
        self.calls.append(user_prompt)
        assert response_model is FinalTravelOpportunityAnalysis
        assert "외부 기억" in system_instruction
        return self.response

    async def close(self) -> None:
        return None


def test_only_eligible_cluster_representative_is_called(client, db_session) -> None:
    _seed_eligible(db_session)
    _add_candidate(db_session, "review후보", ranking_status="review", eligible=True)
    _add_candidate(
        db_session,
        "비대표후보",
        ranking_status="gemini_candidate",
        eligible=True,
        representative=False,
    )
    _add_candidate(
        db_session,
        "비대상후보",
        ranking_status="gemini_candidate",
        eligible=False,
    )
    fake = FakeGeminiAdapter()
    with _override_adapter(fake):
        response = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat()},
        )
    assert response.status_code == 200
    assert response.json()["eligible_candidates"] == 1
    assert len(fake.calls) == 1


def test_weekly_budget_caps_candidates(client, db_session) -> None:
    _seed_eligible(db_session)
    for index in range(4):
        _add_candidate(db_session, f"후보{index}")
    fake = FakeGeminiAdapter()
    with _override_adapter(fake):
        response = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat(), "limit": 20},
        )
    assert response.status_code == 200
    assert response.json()["eligible_candidates"] == 3
    assert len(fake.calls) == 3


def test_zero_candidate_and_dry_run_never_call_gemini(client, db_session) -> None:
    db_session.add(_trend("후보없음"))
    db_session.commit()
    fake = FakeGeminiAdapter()
    with _override_adapter(fake):
        empty = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat()},
        )
    _seed_eligible(db_session)
    with _override_adapter(fake):
        dry_run = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat(), "dry_run": True},
        )
    assert empty.json()["gemini_calls"] == 0
    assert dry_run.json()["expected_gemini_calls"] == 1
    assert dry_run.json()["gemini_calls"] == 0
    assert fake.calls == []


def test_cache_hit_reuses_same_input_hash(client, db_session) -> None:
    _seed_eligible(db_session)
    fake = FakeGeminiAdapter()
    with _override_adapter(fake):
        first = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat(), "force": False},
        )
        second = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat(), "force": False},
        )
    assert first.json()["gemini_calls"] == 1
    assert second.json()["cache_hits"] == 1
    assert second.json()["gemini_calls"] == 0
    assert first.json()["items"][0]["input_hash"] == second.json()["items"][0]["input_hash"]
    assert len(fake.calls) == 1


def test_input_hash_changes_with_context(db_session) -> None:
    candidate = _seed_eligible(db_session)
    settings = get_settings()
    first = build_travel_evidence_package(
        db_session,
        candidate=candidate,
        settings=settings,
        model="gemini-3.6-flash",
    )
    context = db_session.scalar(select(KeywordContext).limit(1))
    context.matched_sentence = "광안리 축제 일정이 변경되어 새로운 문맥이 추가됐다."
    context.combined_context = context.matched_sentence
    db_session.commit()
    second = build_travel_evidence_package(
        db_session,
        candidate=candidate,
        settings=settings,
        model="gemini-3.6-flash",
    )
    assert first.input_hash != second.input_hash


def test_evidence_package_limits_context_and_generates_ids(monkeypatch, db_session) -> None:
    candidate = _seed_eligible(db_session, context_count=4)
    monkeypatch.setenv("TRAVEL_GEMINI_MAX_CONTEXTS", "2")
    monkeypatch.setenv("TRAVEL_GEMINI_MAX_EVIDENCE_DOCS", "2")
    get_settings.cache_clear()
    package = build_travel_evidence_package(
        db_session,
        candidate=candidate,
        settings=get_settings(),
        model="gemini-3.6-flash",
    )
    assert len(package.payload["contexts"]) == 2
    assert len(package.payload["evidence"]["source_titles"]) == 2
    assert "RANKING-V2" in package.valid_evidence_refs
    assert any(ref.startswith("CTX-") for ref in package.valid_evidence_refs)
    assert any(ref.startswith("DOC-") for ref in package.valid_evidence_refs)
    assert any(ref.startswith("ENTITY-") for ref in package.valid_evidence_refs)
    assert package.input_chars <= get_settings().travel_gemini_max_input_chars


def test_evidence_package_includes_only_matched_or_manual_context(db_session) -> None:
    candidate = _seed_eligible(db_session)
    matched = _entity_context("wikipedia_ko", "matched", "matched")
    manual = _entity_context("manual", "manual", "manual")
    error = _entity_context("wikipedia_ko", "error", "error")
    db_session.add_all([matched, manual, error])
    db_session.flush()
    for context in (matched, manual, error):
        db_session.add(
            TrendContextLink(
                keyword=candidate.normalized_keyword,
                week_start=WEEK_START,
                week_end=WEEK_END,
                entity_context_id=context.id,
                normalized_entity=context.normalized_entity,
                entity_type=context.entity_type,
                context_score=90,
                is_primary=context is matched,
                created_at=NOW,
                updated_at=NOW,
            )
        )
    db_session.commit()
    package = build_travel_evidence_package(
        db_session,
        candidate=candidate,
        settings=get_settings(),
        model="gemini-3.6-flash",
    )
    refs = package.valid_evidence_refs
    assert f"CONTEXT-{matched.id}" in refs
    assert f"CONTEXT-{manual.id}" in refs
    assert f"CONTEXT-{error.id}" not in refs


def test_invalid_evidence_refs_removed_and_confidence_capped(db_session) -> None:
    candidate = _seed_eligible(db_session)
    package = _empty_package(valid_refs=frozenset())
    analysis = _accept_response(evidence_refs=["FAKE-999"], confidence=90)
    corrected, status = validate_final_analysis(
        analysis,
        candidate=candidate,
        package=package,
    )
    assert corrected.evidence_refs == []
    assert corrected.confidence_score == 40
    assert corrected.final_decision == "review"
    assert status == "partial"


def test_destination_validation_detects_hallucination(db_session) -> None:
    candidate = _seed_eligible(db_session)
    package = _empty_package(
        valid_refs=frozenset({"CTX-1"}),
        allowed_names=frozenset({"광안리"}),
        context_texts=("광안리에서 축제가 열린다.",),
    )
    analysis = _accept_response(
        destinations=[_destination("군산", "CTX-1")],
        content_destination="군산",
        evidence_refs=["CTX-1"],
    )
    corrected, status = validate_final_analysis(
        analysis,
        candidate=candidate,
        package=package,
    )
    assert corrected.destination_candidates[0].verified_from_input is False
    assert corrected.content_ideas[0].destination == "추가 검증 필요"
    assert corrected.needs_external_verification is True
    assert corrected.final_decision == "accept"
    assert status == "partial"


def test_evidence_pass_accept_and_needs_evidence_accept_block(db_session) -> None:
    candidate = _seed_eligible(db_session)
    package = _grounded_package()
    accepted, _ = validate_final_analysis(
        _accept_response(),
        candidate=candidate,
        package=package,
    )
    assert accepted.final_decision == "accept"
    assert accepted.destination_candidates[0].verified_from_input is True

    candidate.evidence_gate = "NEEDS_EVIDENCE"
    blocked, _ = validate_final_analysis(
        _accept_response(),
        candidate=candidate,
        package=package,
    )
    assert blocked.final_decision == "review"


def test_finance_reject_and_final_decision_correction(db_session) -> None:
    candidate = _seed_eligible(db_session)
    candidate.evidence_codes_json = '["NEGATIVE_SEMANTIC_DOMINANT"]'
    rejected, _ = validate_final_analysis(
        _accept_response(),
        candidate=candidate,
        package=_grounded_package(),
    )
    assert rejected.final_decision == "reject"
    assert rejected.content_ideas == []

    candidate.evidence_codes_json = "[]"
    candidate.high_precision_score = 82
    corrected, _ = validate_final_analysis(
        _accept_response(),
        candidate=candidate,
        package=_grounded_package(),
    )
    assert corrected.final_decision == "review"


def test_content_idea_schema_limits_and_formats() -> None:
    idea = TravelContentIdea(
        title="광안리 축제 1박 2일",
        concept="축제 관람과 해변 방문을 결합한다.",
        destination="광안리",
        format="travel_course",
        target_audience="축제 여행자",
        why_it_works="입력 근거에 장소와 행사가 함께 있다.",
    )
    assert idea.format == "travel_course"
    with pytest.raises(ValidationError):
        TravelContentIdea(
            title="잘못된 형식",
            concept="설명",
            destination="광안리",
            format="podcast",
            target_audience="여행자",
            why_it_works="설명",
        )


def test_final_api_cost_report_and_no_automatic_gemini_call(client, db_session) -> None:
    _seed_eligible(db_session)
    fake = FakeGeminiAdapter()
    with _override_adapter(fake):
        dry = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat(), "dry_run": True},
        )
        cost_before = client.get(
            "/api/travel-opportunities/cost-report",
            params={"week_start": WEEK_START.isoformat()},
        )
        finals_before = client.get("/api/travel-opportunities/final")
        run = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat()},
        )
        finals_after = client.get("/api/travel-opportunities/final")
        detail = client.get("/api/travel-opportunities/final/부산불꽃축제")
        cost_after = client.get(
            "/api/travel-opportunities/cost-report",
            params={"week_start": WEEK_START.isoformat()},
        )
    assert dry.status_code == cost_before.status_code == finals_before.status_code == 200
    assert fake.request_count == 1
    assert run.json()["completed"] == 1
    assert finals_before.json()["total"] == 0
    assert finals_after.json()["total"] == 1
    assert detail.json()["final_decision"] == "accept"
    assert cost_after.json()["gemini_calls_this_week"] == 1
    assert cost_after.json()["quality_keyword_count"] == 1
    assert cost_after.json()["gemini_eligible_count"] == 1
    assert cost_after.json()["overall_llm_reduction_rate"] >= 0


def test_one_candidate_failure_does_not_stop_next_candidate(client, db_session) -> None:
    _seed_eligible(db_session)
    _add_candidate(db_session, "두번째후보")

    class FailOnceAdapter(FakeGeminiAdapter):
        async def generate_structured(self, **kwargs):
            self.request_count += 1
            if self.request_count == 1:
                raise RuntimeError("first candidate failed")
            self.calls.append(kwargs["user_prompt"])
            return self.response

    fake = FailOnceAdapter()
    with _override_adapter(fake):
        response = client.post(
            "/api/travel-opportunities/finalize",
            params={"week_start": WEEK_START.isoformat(), "limit": 2},
        )
    assert response.status_code == 200
    assert response.json()["errors"] == 1
    assert response.json()["completed"] + response.json()["partial"] == 1
    assert fake.request_count == 2


def test_dashboard_final_formatters() -> None:
    accepted = {
        "final_decision": "accept",
        "destination_candidates": [
            {"name": "광안리", "verified_from_input": True},
            {"name": "군산", "verified_from_input": False},
        ],
    }
    review = {"final_decision": "review", "destination_candidates": []}
    assert final_destination_names(accepted) == ["광안리"]
    assert split_final_opportunities([review, accepted]) == ([accepted], [review])
    cards = final_funnel_cards(
        {
            "raw_keyword_count": 100,
            "quality_keyword_count": 20,
            "rule_candidate_count": 5,
            "semantic_candidate_count": 3,
            "high_precision_candidate_count": 2,
            "gemini_eligible_count": 1,
            "final_accept_count": 1,
        }
    )
    assert [card["value"] for card in cards] == [100, 20, 5, 3, 2, 1, 1]


class _override_adapter:
    def __init__(self, fake: FakeGeminiAdapter) -> None:
        self.fake = fake

    def __enter__(self):
        async def dependency_override():
            yield self.fake

        app.dependency_overrides[get_final_gemini_adapter] = dependency_override
        return self.fake

    def __exit__(self, *_args):
        app.dependency_overrides.pop(get_final_gemini_adapter, None)


def _seed_eligible(session, *, context_count: int = 2) -> TravelOpportunityCandidate:
    session.add(_trend("부산불꽃축제"))
    contexts = []
    for index in range(context_count):
        source = "newsis_rss" if index % 2 == 0 else "youtube"
        document = SourceDocument(
            source=source,
            source_id=f"festival-{index}",
            title=f"광안리 부산불꽃축제 개최 소식 {index}",
            text="원문 전체는 Gemini 입력에 사용하지 않는다. " * 100,
            published_at=NOW + timedelta(minutes=index),
            collected_at=NOW,
            views=100 if source == "youtube" else None,
            likes=10 if source == "youtube" else None,
            comments=2 if source == "youtube" else None,
            url=f"https://example.test/festival/{index}",
        )
        session.add(document)
        session.flush()
        session.add(
            KeywordOccurrence(
                document_id=document.id,
                keyword="부산불꽃축제",
                normalized_keyword="부산불꽃축제",
                source=source,
                occurred_at=document.published_at,
                keyword_quality_score=95,
                pipeline_version="v2",
            )
        )
        session.add(
            KeywordCandidate(
                document_id=document.id,
                candidate_text="부산불꽃축제",
                normalized_candidate="부산불꽃축제",
                candidate_type="noun_phrase",
                extractor="test",
                quality_score=95,
                accepted=True,
                rejection_reason=None,
                title_occurrence=1,
                body_occurrence=1,
                entity_type="EVENT",
                entity_confidence=0.95,
                created_at=NOW + timedelta(days=30),
                pipeline_version="v2",
            )
        )
        context = KeywordContext(
            document_id=document.id,
            keyword="부산불꽃축제",
            normalized_keyword="부산불꽃축제",
            previous_sentence="부산 지역의 가을 행사 소식이다.",
            matched_sentence="부산 광안리에서 부산불꽃축제가 열린다.",
            next_sentence="대중교통 방문 정보가 함께 안내됐다.",
            combined_context="부산 지역의 가을 행사 소식이다. 부산 광안리에서 부산불꽃축제가 열린다. 대중교통 방문 정보가 함께 안내됐다.",
            occurrence_index=0,
            source=source,
            published_at=document.published_at,
            context_hash=f"festival-context-{index}",
            created_at=NOW,
            updated_at=NOW,
        )
        session.add(context)
        session.flush()
        contexts.append(context)
        session.add_all(
            [
                _mention(document.id, "부산불꽃축제", "EVENT", index * 20),
                _mention(document.id, "부산", "LOCATION", index * 20 + 5),
                _mention(document.id, "광안리", "PLACE", index * 20 + 10),
            ]
        )
    candidate = _candidate(
        keyword="부산불꽃축제",
        context_id=contexts[0].id,
    )
    session.add(candidate)
    session.commit()
    session.refresh(candidate)
    return candidate


def _add_candidate(
    session,
    keyword: str,
    *,
    ranking_status: str = "gemini_candidate",
    eligible: bool = True,
    representative: bool = True,
) -> TravelOpportunityCandidate:
    context = session.scalar(select(KeywordContext).limit(1))
    candidate = _candidate(keyword=keyword, context_id=context.id)
    candidate.ranking_status = ranking_status
    candidate.gemini_eligible = eligible
    candidate.cluster_representative = representative
    session.add(candidate)
    session.commit()
    return candidate


def _candidate(*, keyword: str, context_id: int) -> TravelOpportunityCandidate:
    return TravelOpportunityCandidate(
        keyword=keyword,
        normalized_keyword=keyword,
        week_start=WEEK_START,
        week_end=WEEK_END,
        keyword_context_id=context_id,
        primary_entity=keyword,
        primary_entity_type="EVENT",
        travel_category="FESTIVAL",
        entity_prior_score=35,
        positive_context_score=30,
        negative_context_penalty=0,
        trend_evidence_score=20,
        source_diversity_score=15,
        travel_pre_score=95,
        prefilter_status="strong",
        matched_positive_terms_json='["축제","개최","방문"]',
        matched_negative_terms_json="[]",
        reasoning_codes_json="[]",
        semantic_travel_score=96,
        semantic_status="semantic_strong",
        semantic_positive_category="FESTIVAL",
        semantic_negative_category=None,
        trend_strength_score=94,
        context_clarity_score=90,
        travel_convertibility_score=96,
        evidence_confidence_score=90,
        high_precision_score=93,
        evidence_gate="PASS",
        evidence_codes_json='["MULTI_SOURCE_CONFIRMATION","EVENT_LOCATION_PAIR"]',
        evidence_document_count=2,
        evidence_source_count=2,
        ranking_status="priority_candidate",
        rank_in_week=1,
        ranking_version="v2-step3-local-1",
        calculated_at=NOW,
        cluster_id=f"cluster-{keyword}",
        cluster_representative=True,
        gemini_eligible=True,
        created_at=NOW,
        updated_at=NOW,
    )


def _trend(keyword: str) -> WeeklyTrend:
    return WeeklyTrend(
        keyword=keyword,
        week_start=WEEK_START,
        week_end=WEEK_END,
        weekly_mentions=8,
        previous_weekly_mentions=2,
        active_days=5,
        source_count=2,
        growth_rate=1.0,
        peak_day_share=0.3,
        persistence_score=90,
        diversity_score=100,
        freshness_score=90,
        volume_score=95,
        growth_score=100,
        trend_score=94,
        keyword_quality_score=95,
        search_interest_score=85,
        search_interest_available=True,
        search_provider_count=1,
        one_day_spike_penalty=0,
        spam_penalty=0,
        final_score=95,
        status="weekly_trend",
        calculated_at=NOW,
        pipeline_version="v2",
    )


def _mention(document_id: int, text: str, entity_type: str, start: int) -> EntityMention:
    return EntityMention(
        document_id=document_id,
        text=text,
        normalized_text=text.replace(" ", ""),
        entity_type=entity_type,
        confidence=0.98,
        extractor="rule",
        start_char=start,
        end_char=start + len(text),
        source="newsis_rss",
        occurred_at=NOW,
        created_at=NOW,
    )


def _entity_context(provider: str, status: str, suffix: str) -> EntityContext:
    return EntityContext(
        normalized_entity=f"광안리{suffix}",
        entity_text="광안리",
        entity_type="PLACE",
        provider=provider,
        page_id=suffix,
        page_title="광안리",
        page_url=f"https://example.test/{provider}/{suffix}",
        summary="광안리에 관한 저장된 근거다.",
        description=None,
        match_score=0.9 if status != "error" else 0,
        match_status=status,
        source_language="ko",
        license_name=None,
        attribution_text=None,
        revision_id=None,
        retrieved_at=NOW,
        updated_at=NOW,
    )


def _accept_response(
    *,
    destinations: list[DestinationCandidate] | None = None,
    content_destination: str = "광안리",
    evidence_refs: list[str] | None = None,
    confidence: int = 90,
) -> FinalTravelOpportunityAnalysis:
    return FinalTravelOpportunityAnalysis(
        keyword="부산불꽃축제",
        final_decision="accept",
        final_travel_score=91,
        trend_context_summary="광안리에서 열리는 축제가 다시 언급되고 있다.",
        why_now="축제 개최 문맥이 여러 근거에 나타났다.",
        travel_angle="축제 관람과 부산 여행을 결합할 수 있다.",
        destination_candidates=destinations or [_destination("광안리", "RANKING-V2")],
        content_ideas=[
            TravelContentIdea(
                title="부산불꽃축제 광안리 1박 2일",
                concept="축제 관람과 해변 방문을 결합한다.",
                destination=content_destination,
                format="travel_course",
                target_audience="축제 여행자",
                why_it_works="행사와 장소 근거가 함께 제공됐다.",
            )
        ],
        evidence_refs=evidence_refs or ["RANKING-V2"],
        needs_external_verification=False,
        verification_queries=[],
        cautions=[],
        confidence_score=confidence,
    )


def _destination(name: str, evidence_ref: str) -> DestinationCandidate:
    return DestinationCandidate(
        name=name,
        entity_type="PLACE",
        reason="입력 문맥에 장소가 명시됐다.",
        evidence_ref=evidence_ref,
        verified_from_input=True,
    )


def _grounded_package() -> TravelEvidencePackage:
    return _empty_package(
        valid_refs=frozenset({"RANKING-V2"}),
        allowed_names=frozenset({"광안리"}),
        context_texts=("광안리에서 부산불꽃축제가 열린다.",),
    )


def _empty_package(
    *,
    valid_refs: frozenset[str] = frozenset({"RANKING-V2"}),
    allowed_names: frozenset[str] = frozenset({"광안리"}),
    context_texts: tuple[str, ...] = ("광안리에서 축제가 열린다.",),
) -> TravelEvidencePackage:
    return TravelEvidencePackage(
        payload={},
        user_prompt="prompt",
        input_hash="a" * 64,
        input_chars=100,
        valid_evidence_refs=valid_refs,
        allowed_destination_names=allowed_names,
        context_texts=context_texts,
    )

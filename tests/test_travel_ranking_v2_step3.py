from datetime import date, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import func, select

from app.models.entity_context import EntityContext
from app.models.entity_mention import EntityMention
from app.models.keyword_context import KeywordContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.services.travel_ranking_service import (
    RankedCandidate,
    annualized_estimate,
    assess_evidence,
    assign_clusters,
    assign_gemini_budget,
    calculate_high_precision_score,
    classify_ranking,
    score_context_clarity,
    score_travel_convertibility,
    score_trend_strength,
)


WEEK_START = date(2026, 8, 3)
WEEK_END = WEEK_START + timedelta(days=6)
NOW = datetime(2026, 8, 7, 12, 0, 0)


def test_trend_strength_uses_available_metrics() -> None:
    trend = _trend("부산불꽃축제", search_score=80)
    score = score_trend_strength(trend, document_count=3)
    assert 0 <= score <= 100
    assert score >= 80


def test_missing_search_interest_is_reweighted_not_replaced_with_fifty() -> None:
    missing = _trend("검색없음", search_score=None)
    zero = _trend("검색0", search_score=0)
    missing_score = score_trend_strength(missing, document_count=3)
    zero_score = score_trend_strength(zero, document_count=3)
    assert missing_score > zero_score


def test_context_clarity_rewards_keyword_and_cross_document_context() -> None:
    contexts = [
        _context(1, "부산불꽃축제", "부산 광안리에서 부산불꽃축제가 열린다."),
        _context(2, "부산불꽃축제", "부산 광안리에서 부산불꽃축제가 개최된다."),
    ]
    score = score_context_clarity(
        keyword="부산불꽃축제",
        contexts=contexts,
        mentions=[_mention(1, "부산불꽃축제", "EVENT", 0.95)],
    )
    assert score >= 60


def test_duplicate_context_in_same_document_is_penalized() -> None:
    unique = [
        _context(1, "성수팝업", "성수동에서 성수팝업 행사가 열린다."),
        _context(2, "성수팝업", "성수동에서 성수팝업 행사가 개최된다."),
    ]
    duplicate = [unique[0], _context(1, "성수팝업", unique[0].matched_sentence)]
    unique_score = score_context_clarity(keyword="성수팝업", contexts=unique, mentions=[])
    duplicate_score = score_context_clarity(keyword="성수팝업", contexts=duplicate, mentions=[])
    assert duplicate_score < unique_score


def test_travel_convertibility_combines_rule_semantic_category_and_entity() -> None:
    row = _candidate_stub(
        travel_category="FILM_LOCATION",
        semantic_score=92,
        semantic_positive="FILM_LOCATION",
    )
    score = score_travel_convertibility(rows=[row], entity_types={"CONTENT_TITLE"})
    assert score >= 80


def test_evidence_pass_for_event_location_with_multiple_sources() -> None:
    contexts = [
        _context(1, "부산불꽃축제", "부산 광안리에서 부산불꽃축제가 개최된다.", source="newsis_rss"),
        _context(2, "부산불꽃축제", "부산 광안리에서 부산불꽃축제가 열린다.", source="youtube"),
        _context(3, "부산불꽃축제", "광안리 부산불꽃축제 개최 소식이다.", source="newsis_rss"),
    ]
    evidence = assess_evidence(
        rows=[_candidate_stub(travel_category="FESTIVAL")],
        contexts=contexts,
        document_count=3,
        source_count=2,
        mentions=[
            _mention(1, "부산불꽃축제", "EVENT", 0.98),
            _mention(1, "부산", "LOCATION", 0.98),
            _mention(1, "광안리", "PLACE", 0.98),
        ],
        trend_links=[],
        entity_contexts=[],
    )
    assert evidence.gate == "PASS"
    assert "EVENT_LOCATION_PAIR" in evidence.codes
    assert evidence.score >= 65


def test_needs_evidence_for_film_without_location() -> None:
    evidence = assess_evidence(
        rows=[_candidate_stub(travel_category="FILM_LOCATION")],
        contexts=[_context(1, "8월의크리스마스", "8월의 크리스마스는 허진호 감독의 영화다.")],
        document_count=1,
        source_count=1,
        mentions=[_mention(1, "8월의 크리스마스", "CONTENT_TITLE", 0.95)],
        trend_links=[],
        entity_contexts=[],
    )
    assert evidence.gate == "NEEDS_EVIDENCE"
    assert "NO_LOCATION_EVIDENCE" in evidence.codes
    assert "CONTENT_TITLE_CONTEXT" in evidence.codes


def test_negative_semantic_dominant_is_rejected() -> None:
    evidence = assess_evidence(
        rows=[
            _candidate_stub(
                travel_category="OTHER",
                semantic_negative="FINANCE",
            )
        ],
        contexts=[_context(1, "삼성전자주가", "삼성전자 주가와 투자 전망을 분석한다.")],
        document_count=1,
        source_count=1,
        mentions=[_mention(1, "삼성전자", "BRAND", 0.95)],
        trend_links=[],
        entity_contexts=[],
    )
    assert evidence.gate == "REJECT"
    assert "NEGATIVE_SEMANTIC_DOMINANT" in evidence.codes


def test_matched_and_manual_context_are_evidence_but_wikipedia_error_is_not() -> None:
    base = dict(
        rows=[_candidate_stub(travel_category="LANDMARK")],
        contexts=[_context(1, "광안리", "광안리는 부산의 해변 명소다.")],
        document_count=1,
        source_count=1,
        mentions=[_mention(1, "광안리", "PLACE", 0.95)],
        trend_links=[],
    )
    matched = assess_evidence(
        **base,
        entity_contexts=[_entity_context("wikipedia_ko", "matched")],
    )
    manual = assess_evidence(
        **base,
        entity_contexts=[_entity_context("manual", "manual")],
    )
    error = assess_evidence(
        **base,
        entity_contexts=[_entity_context("wikipedia_ko", "error")],
    )
    assert "MATCHED_CONTEXT" in matched.codes
    assert "MANUAL_CONTEXT" in manual.codes
    assert "MATCHED_CONTEXT" not in error.codes
    assert error.score < matched.score < manual.score


def test_high_precision_formula_and_ranking_boundaries() -> None:
    assert calculate_high_precision_score(
        trend_strength=100,
        context_clarity=50,
        travel_convertibility=100,
        evidence_confidence=50,
    ) == 80
    assert classify_ranking(69.99, "PASS") == "rejected"
    assert classify_ranking(70, "PASS") == "review"
    assert classify_ranking(85, "PASS") == "gemini_candidate"
    assert classify_ranking(90, "PASS") == "priority_candidate"
    assert classify_ranking(95, "NEEDS_EVIDENCE") == "gemini_candidate"
    assert classify_ranking(99, "REJECT") == "rejected"


def test_weekly_budget_prioritizes_priority_then_gemini_and_does_not_fill() -> None:
    candidates = [
        _ranked("priority", 91, "priority_candidate"),
        _ranked("gemini-high", 89, "gemini_candidate"),
        _ranked("gemini-low", 86, "gemini_candidate"),
        _ranked("review", 84, "review"),
    ]
    assign_gemini_budget(candidates, max_candidates=2)
    assert [item.normalized_keyword for item in candidates if item.gemini_eligible] == [
        "priority",
        "gemini-high",
    ]


def test_duplicate_cluster_and_representative_selection() -> None:
    base = _ranked("폭싹속았수다", 88, "gemini_candidate")
    suffix = _ranked("폭싹속았수다촬영지", 91, "priority_candidate")
    suffix._document_ids = {3}
    spaced = _ranked("제주폭싹속았수다촬영지", 87, "gemini_candidate")
    spaced._entity_keys = {("제주", "LOCATION"), ("폭싹속았수다", "CONTENT_TITLE")}
    suffix._entity_keys = {("제주", "LOCATION"), ("폭싹속았수다", "CONTENT_TITLE")}
    candidates = [base, suffix, spaced]
    assign_clusters(candidates)
    assert len({item.cluster_id for item in candidates}) == 1
    assert [item.normalized_keyword for item in candidates if item.cluster_representative] == [
        "폭싹속았수다촬영지"
    ]


def test_annual_estimate_and_insufficient_history(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.services.travel_ranking_service.repo.ranked_history",
        lambda _session: [
            (date(2026, 1, 5), "gemini_candidate"),
            (date(2026, 2, 23), "priority_candidate"),
        ],
    )
    estimate, insufficient = annualized_estimate(SimpleNamespace())
    assert estimate == 13.0
    assert insufficient is False

    monkeypatch.setattr(
        "app.services.travel_ranking_service.repo.ranked_history",
        lambda _session: [(date(2026, 1, 5), "review")],
    )
    estimate, insufficient = annualized_estimate(SimpleNamespace())
    assert estimate == 0.0
    assert insufficient is True


def test_rank_api_handles_zero_candidates(client, db_session) -> None:
    db_session.add(_trend("후보없음", search_score=None))
    db_session.commit()
    response = client.post(
        "/api/travel-opportunities/rank",
        params={"week_start": WEEK_START.isoformat(), "dry_run": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["processed"] == 0
    assert payload["estimated_gemini_calls"] == 0
    assert payload["top_candidates"] == []


def test_rank_dry_run_keeps_db_unchanged_and_never_calls_gemini(
    monkeypatch, client, db_session
) -> None:
    _seed_rankable_candidates(db_session)
    before = db_session.scalar(
        select(func.count(TravelOpportunityCandidate.id)).where(
            TravelOpportunityCandidate.ranking_version.is_not(None)
        )
    ) or 0

    async def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("Gemini must not run during Step 3 ranking")

    monkeypatch.setattr("app.ai.gemini_adapter.GeminiAdapter.generate", forbidden_generate)
    response = client.post(
        "/api/travel-opportunities/rank",
        params={"week_start": WEEK_START.isoformat(), "dry_run": True},
    )
    db_session.expire_all()
    after = db_session.scalar(
        select(func.count(TravelOpportunityCandidate.id)).where(
            TravelOpportunityCandidate.ranking_version.is_not(None)
        )
    ) or 0
    assert response.status_code == 200
    payload = response.json()
    assert payload["processed"] == 3
    assert payload["evidence_pass"] >= 1
    assert payload["needs_evidence"] >= 1
    assert payload["evidence_reject"] >= 1
    assert payload["funnel"]["raw_keyword"] == 3
    assert payload["funnel"]["keyword_quality_passed"] == 3
    assert payload["funnel"]["rule_candidate"] == 3
    assert payload["funnel"]["semantic_candidate"] == 3
    assert before == after == 0


def test_persisted_rank_and_calibration_dashboard_summary(client, db_session) -> None:
    _seed_rankable_candidates(db_session)
    ranked = client.post(
        "/api/travel-opportunities/rank",
        params={"week_start": WEEK_START.isoformat(), "dry_run": False, "force": True},
    )
    report = client.get(
        "/api/travel-opportunities/calibration-report",
        params={"week_start": WEEK_START.isoformat()},
    )
    assert ranked.status_code == 200
    assert report.status_code == 200
    payload = report.json()
    assert payload["ranking_version"] == "v2-step3-local-1"
    assert payload["total_semantic_candidates"] == 3
    assert sum(payload["evidence_gate_counts"].values()) == 3
    assert payload["funnel"]["gemini_eligible"] <= payload["weekly_gemini_budget"]
    assert payload["overall_reduction_rate"] >= 0
    assert payload["top_20_candidates"]
    assert {
        "trend_strength_score",
        "context_clarity_score",
        "travel_convertibility_score",
        "evidence_confidence_score",
        "evidence_codes",
        "gemini_eligible",
    }.issubset(payload["top_20_candidates"][0])


def _seed_rankable_candidates(session) -> None:
    specifications = [
        (
            "부산불꽃축제",
            "FESTIVAL",
            "FESTIVAL",
            None,
            [
                ("newsis_rss", "부산 광안리에서 부산불꽃축제가 성대하게 개최된다."),
                ("youtube", "부산 광안리에서 부산불꽃축제가 올해 다시 열린다."),
                ("newsis_rss", "광안리 부산불꽃축제 개최와 공연 소식이 전해졌다."),
            ],
        ),
        (
            "8월의크리스마스",
            "FILM_LOCATION",
            "FILM_LOCATION",
            None,
            [("newsis_rss", "8월의 크리스마스는 허진호 감독의 대표 영화 작품이다.")],
        ),
        (
            "삼성전자주가",
            "OTHER",
            None,
            "FINANCE",
            [("youtube", "삼성전자 주가와 투자 실적 전망을 분석한다.")],
        ),
    ]
    for keyword, category, positive, negative, document_specs in specifications:
        trend = _trend(keyword, search_score=85)
        session.add(trend)
        session.flush()
        for index, (source, sentence) in enumerate(document_specs):
            document = SourceDocument(
                source=source,
                source_id=f"{keyword}-{index}",
                title=sentence,
                text=f"관련 배경을 설명한다. {sentence} 방문 정보와 개최 장소를 안내한다.",
                published_at=NOW + timedelta(minutes=index),
                collected_at=NOW,
                views=100 if source == "youtube" else None,
                likes=10 if source == "youtube" else None,
                comments=2 if source == "youtube" else None,
                url=f"https://example.test/{keyword}/{index}",
            )
            session.add(document)
            session.flush()
            session.add(
                KeywordOccurrence(
                    document_id=document.id,
                    keyword=keyword,
                    normalized_keyword=keyword,
                    source=source,
                    occurred_at=document.published_at,
                    keyword_quality_score=95,
                    pipeline_version="v2",
                )
            )
            context = KeywordContext(
                document_id=document.id,
                keyword=keyword,
                normalized_keyword=keyword,
                previous_sentence="관련 배경과 지역 정보를 설명한다.",
                matched_sentence=sentence,
                next_sentence="방문 정보와 개최 장소를 안내한다.",
                combined_context=f"관련 배경과 지역 정보를 설명한다. {sentence} 방문 정보와 개최 장소를 안내한다.",
                occurrence_index=0,
                source=source,
                published_at=document.published_at,
                context_hash=f"{keyword}-{index}",
                created_at=NOW,
                updated_at=NOW,
            )
            session.add(context)
            session.flush()
            entity_type = "EVENT" if category == "FESTIVAL" else "CONTENT_TITLE" if category == "FILM_LOCATION" else "BRAND"
            session.add(_mention(document.id, keyword, entity_type, 0.96))
            if category == "FESTIVAL":
                session.add(_mention(document.id, "부산", "LOCATION", 0.98, suffix=f"-{index}-loc"))
                session.add(_mention(document.id, "광안리", "PLACE", 0.98, suffix=f"-{index}-place"))
            session.add(
                TravelOpportunityCandidate(
                    keyword=keyword,
                    normalized_keyword=keyword,
                    week_start=WEEK_START,
                    week_end=WEEK_END,
                    keyword_context_id=context.id,
                    primary_entity=keyword,
                    primary_entity_type=entity_type,
                    travel_category=category,
                    entity_prior_score=35,
                    positive_context_score=30 if category != "OTHER" else 0,
                    negative_context_penalty=40 if negative else 0,
                    trend_evidence_score=20,
                    source_diversity_score=15,
                    travel_pre_score=96 if category == "FESTIVAL" else 88 if category == "FILM_LOCATION" else 70,
                    prefilter_status="strong" if category != "OTHER" else "review",
                    matched_positive_terms_json='["축제", "개최", "방문"]' if category == "FESTIVAL" else '["영화", "작품"]' if category == "FILM_LOCATION" else "[]",
                    matched_negative_terms_json='["주가", "투자"]' if negative else "[]",
                    reasoning_codes_json="[]",
                    semantic_travel_score=96 if category == "FESTIVAL" else 92 if category == "FILM_LOCATION" else 75,
                    semantic_status="semantic_strong" if category != "OTHER" else "semantic_review",
                    semantic_positive_category=positive,
                    semantic_negative_category=negative,
                    created_at=NOW,
                    updated_at=NOW,
                )
            )
    session.commit()


def _trend(keyword: str, *, search_score: float | None) -> WeeklyTrend:
    return WeeklyTrend(
        keyword=keyword,
        week_start=WEEK_START,
        week_end=WEEK_END,
        weekly_mentions=8,
        previous_weekly_mentions=2,
        active_days=5,
        source_count=3,
        growth_rate=1.0,
        peak_day_share=0.3,
        persistence_score=90,
        diversity_score=100,
        freshness_score=90,
        volume_score=95,
        growth_score=100,
        trend_score=94,
        keyword_quality_score=95,
        search_interest_score=search_score,
        search_interest_available=search_score is not None,
        search_provider_count=1 if search_score is not None else 0,
        one_day_spike_penalty=0,
        spam_penalty=0,
        final_score=95,
        status="weekly_trend",
        calculated_at=NOW,
        pipeline_version="v2",
    )


def _context(
    document_id: int,
    keyword: str,
    sentence: str,
    *,
    source: str = "newsis_rss",
) -> KeywordContext:
    return KeywordContext(
        id=document_id,
        document_id=document_id,
        keyword=keyword,
        normalized_keyword=keyword.replace(" ", ""),
        previous_sentence="관련 지역과 배경을 설명한다.",
        matched_sentence=sentence,
        next_sentence="방문과 개최 정보를 이어서 설명한다.",
        combined_context=f"관련 지역과 배경을 설명한다. {sentence} 방문과 개최 정보를 이어서 설명한다.",
        occurrence_index=0,
        source=source,
        published_at=NOW,
        context_hash=f"hash-{document_id}",
        created_at=NOW,
        updated_at=NOW,
    )


def _mention(
    document_id: int,
    text: str,
    entity_type: str,
    confidence: float,
    *,
    suffix: str = "",
) -> EntityMention:
    return EntityMention(
        document_id=document_id,
        text=text,
        normalized_text=text.replace(" ", "") + suffix,
        entity_type=entity_type,
        confidence=confidence,
        extractor="rule",
        start_char=None,
        end_char=None,
        source="newsis_rss",
        occurred_at=NOW,
        created_at=NOW,
    )


def _candidate_stub(
    *,
    travel_category: str,
    semantic_score: float = 90,
    semantic_positive: str | None = None,
    semantic_negative: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        travel_category=travel_category,
        travel_pre_score=90.0,
        semantic_travel_score=semantic_score,
        semantic_positive_category=semantic_positive,
        semantic_negative_category=semantic_negative,
        negative_context_penalty=40.0 if semantic_negative else 0.0,
        matched_positive_terms_json='["방문", "축제", "개최"]',
    )


def _entity_context(provider: str, match_status: str) -> EntityContext:
    return EntityContext(
        normalized_entity="광안리",
        entity_text="광안리",
        entity_type="PLACE",
        provider=provider,
        page_id="1",
        page_title="광안리",
        page_url=f"https://example.test/{provider}/{match_status}",
        summary="부산의 해변 명소",
        description=None,
        match_score=0.9,
        match_status=match_status,
        source_language="ko",
        license_name=None,
        attribution_text=None,
        revision_id=None,
        retrieved_at=NOW,
        updated_at=NOW,
    )


def _ranked(keyword: str, score: float, status: str) -> RankedCandidate:
    return RankedCandidate(
        keyword=keyword,
        normalized_keyword=keyword,
        week_start=WEEK_START,
        travel_category="FILM_LOCATION",
        semantic_category="FILM_LOCATION",
        semantic_status="semantic_strong",
        semantic_travel_score=90,
        travel_pre_score=90,
        trend_strength_score=90,
        context_clarity_score=90,
        travel_convertibility_score=90,
        evidence_confidence_score=90,
        high_precision_score=score,
        evidence_gate="PASS",
        evidence_codes=[],
        evidence_document_count=2,
        evidence_source_count=2,
        ranking_status=status,
        cluster_representative=True,
    )

from datetime import date, datetime

from sqlalchemy import func, select

from app.context_v2.context_extractor import extract_keyword_contexts
from app.context_v2.sentence_splitter import split_sentences
from app.context_v2.travel_rules import EntitySignal, TrendSignal, evaluate_travel_rules, load_terms
from app.models.entity_mention import EntityMention
from app.models.keyword_candidate import KeywordCandidate
from app.models.keyword_context import KeywordContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.weekly_trend import WeeklyTrend
from app.repositories.travel_opportunity_repository import summarize_v2
from app.repositories import travel_opportunity_repository as travel_repo
from app.services.keyword_context_service import build_keyword_contexts
from app.services.travel_prefilter_service import prefilter_travel_opportunities


WEEK_START = date(2026, 7, 27)
WEEK_END = date(2026, 8, 2)
NOW = datetime(2026, 8, 1, 9, 0, 0)
CANDIDATE_CREATED_AT = datetime(2026, 8, 19, 9, 0, 0)


def test_korean_sentence_splitter_handles_quotes_and_newlines() -> None:
    text = "첫 문장입니다.\n이번 주 영화는 '8월의 크리스마스'입니다. 다음 문장입니다!"
    assert split_sentences(text) == [
        "첫 문장입니다.",
        "이번 주 영화는 '8월의 크리스마스'입니다.",
        "다음 문장입니다!",
    ]


def test_english_sentence_splitter() -> None:
    text = 'First sentence. "Second sentence?" Third sentence!'
    assert split_sentences(text) == ["First sentence.", '"Second sentence?"', "Third sentence!"]


def test_previous_matched_next_extraction() -> None:
    contexts = extract_keyword_contexts(
        text="이전 문장입니다. 현재 8월의 크리스마스 영화입니다. 다음 문장입니다.",
        keyword="8월의 크리스마스",
        normalized_keyword="8월의크리스마스",
        sentences_before=1,
        sentences_after=1,
        max_chars=1500,
    )
    assert contexts[0].previous_sentence == "이전 문장입니다."
    assert contexts[0].matched_sentence == "현재 8월의 크리스마스 영화입니다."
    assert contexts[0].next_sentence == "다음 문장입니다."


def test_context_extraction_at_document_start_and_end() -> None:
    start = extract_keyword_contexts(
        text="부산불꽃축제가 열립니다. 다음 문장입니다.",
        keyword="부산불꽃축제",
        normalized_keyword="부산불꽃축제",
        sentences_before=1,
        sentences_after=1,
        max_chars=1500,
    )[0]
    end = extract_keyword_contexts(
        text="이전 문장입니다. 마지막은 두바이 초콜릿입니다.",
        keyword="두바이 초콜릿",
        normalized_keyword="두바이초콜릿",
        sentences_before=1,
        sentences_after=1,
        max_chars=1500,
    )[0]
    assert start.previous_sentence is None
    assert end.next_sentence is None


def test_multiple_occurrences_and_context_hash_deduplication() -> None:
    contexts = extract_keyword_contexts(
        text="성수동 카페가 인기입니다. 두바이 초콜릿이 유행입니다. 다른 문장입니다. 두바이 초콜릿 디저트가 확산됩니다.",
        keyword="두바이 초콜릿",
        normalized_keyword="두바이초콜릿",
        sentences_before=1,
        sentences_after=1,
        max_chars=1500,
    )
    assert len(contexts) == 2
    assert len({context.context_hash for context in contexts}) == 2


def test_context_length_limit() -> None:
    contexts = extract_keyword_contexts(
        text=f"{'가' * 300}. 키워드가 포함된 문장입니다. {'나' * 300}.",
        keyword="키워드",
        normalized_keyword="키워드",
        sentences_before=1,
        sentences_after=1,
        max_chars=80,
    )
    assert len(contexts[0].combined_context) <= 80
    assert "키워드" in contexts[0].combined_context


def test_content_title_film_rule_review_or_better() -> None:
    result = evaluate_travel_rules(
        keyword="8월의 크리스마스",
        context="1998년 개봉한 허진호 감독의 영화 작품이 다시 소개되고 있다.",
        entities=[EntitySignal("8월의 크리스마스", "8월의크리스마스", "CONTENT_TITLE", 95, True)],
        trend=TrendSignal(weekly_mentions=5, final_score=90, source_count=2, document_count=2),
    )
    assert result.travel_category == "FILM_LOCATION"
    assert result.prefilter_status in {"review", "strong"}
    assert "CONTENT_TITLE_WITH_FILM_CONTEXT" in result.reasoning_codes


def test_event_festival_rule_can_be_strong() -> None:
    result = evaluate_travel_rules(
        keyword="부산불꽃축제",
        context="부산 광안리에서 열리는 부산불꽃축제가 올해도 개최된다.",
        entities=[
            EntitySignal("부산불꽃축제", "부산불꽃축제", "EVENT", 99, True),
            EntitySignal("부산", "부산", "LOCATION", 90, False),
            EntitySignal("광안리", "광안리", "PLACE", 90, False),
        ],
        trend=TrendSignal(weekly_mentions=8, final_score=95, source_count=3, document_count=3),
    )
    assert result.travel_category == "FESTIVAL"
    assert result.prefilter_status == "strong"


def test_food_place_and_region_meme_rules() -> None:
    food = evaluate_travel_rules(
        keyword="두바이 초콜릿",
        context="두바이 초콜릿 디저트가 성수동 카페를 중심으로 유행하고 있다.",
        entities=[EntitySignal("두바이 초콜릿", "두바이초콜릿", "FOOD", 90, True)],
        trend=TrendSignal(final_score=80, source_count=2, document_count=2),
    )
    place = evaluate_travel_rules(
        keyword="광안리",
        context="광안리 해변 여행과 방문 관심이 늘고 있다.",
        entities=[EntitySignal("광안리", "광안리", "PLACE", 90, True)],
        trend=TrendSignal(final_score=70, source_count=2, document_count=2),
    )
    meme = evaluate_travel_rules(
        keyword="거제야호",
        context="거제야호 밈으로 거제 여행 관심이 높아졌다.",
        entities=[
            EntitySignal("거제야호", "거제야호", "MEME", 90, True),
            EntitySignal("거제", "거제", "LOCATION", 80, False),
        ],
        trend=TrendSignal(final_score=80, source_count=2, document_count=2),
    )
    assert food.travel_category == "FOOD"
    assert "FOOD_TREND" in food.reasoning_codes
    assert place.travel_category in {"NATURE", "LOCAL_CULTURE", "LANDMARK"}
    assert "PLACE_TREND" in place.reasoning_codes
    assert meme.travel_category == "REGIONAL_MEME"


def test_brand_finance_and_person_legal_reject() -> None:
    brand = evaluate_travel_rules(
        keyword="삼성전자",
        context="삼성전자 2분기 영업이익과 주가 전망이 발표됐다.",
        entities=[EntitySignal("삼성전자", "삼성전자", "BRAND", 90, True)],
        trend=TrendSignal(final_score=99, source_count=3, document_count=3),
    )
    person = evaluate_travel_rules(
        keyword="어떤 연예인",
        context="해당 연예인의 법적 분쟁 관련 재판이 진행됐다.",
        entities=[EntitySignal("어떤 연예인", "어떤연예인", "PERSON", 90, True)],
        trend=TrendSignal(final_score=99, source_count=3, document_count=3),
    )
    assert brand.prefilter_status == "rejected"
    assert "FINANCE_CONTEXT" in brand.reasoning_codes
    assert person.prefilter_status == "rejected"
    assert "LEGAL_CONTEXT" in person.reasoning_codes


def test_positive_negative_scoring_and_score_range() -> None:
    positive = evaluate_travel_rules(
        keyword="테스트 축제",
        context="지역 축제 공연 전시 체험 관광 방문 명소 카페 시장 해변 섬 산 공원",
        entities=[EntitySignal("테스트 축제", "테스트축제", "EVENT", 90, True)],
        trend=TrendSignal(final_score=100, source_count=5, document_count=5),
    )
    negative = evaluate_travel_rules(
        keyword="테스트",
        context="주가 실적 투자 재판 소송 사고",
        entities=[EntitySignal("테스트", "테스트", "BRAND", 90, True)],
        trend=TrendSignal(final_score=100, source_count=5, document_count=5),
    )
    assert positive.positive_context_score == 30
    assert negative.negative_context_penalty == 40
    assert 0 <= positive.travel_pre_score <= 100
    assert 0 <= negative.travel_pre_score <= 100


def test_prefilter_status_boundaries() -> None:
    rejected = evaluate_travel_rules(
        keyword="무신호",
        context="일반 문장입니다.",
        entities=[],
        trend=TrendSignal(final_score=0, source_count=1, document_count=1),
    )
    weak = evaluate_travel_rules(
        keyword="광안리",
        context="광안리 소식입니다.",
        entities=[EntitySignal("광안리", "광안리", "PLACE", 90, True)],
        trend=TrendSignal(final_score=50, source_count=1, document_count=1),
    )
    review = evaluate_travel_rules(
        keyword="8월의 크리스마스",
        context="영화 개봉 감독 작품",
        entities=[EntitySignal("8월의 크리스마스", "8월의크리스마스", "CONTENT_TITLE", 90, True)],
        trend=TrendSignal(final_score=75, source_count=2, document_count=2),
    )
    strong = evaluate_travel_rules(
        keyword="부산불꽃축제",
        context="부산 축제 개최 여행 방문 명소 공연",
        entities=[EntitySignal("부산불꽃축제", "부산불꽃축제", "EVENT", 90, True)],
        trend=TrendSignal(final_score=100, source_count=3, document_count=3),
    )
    assert rejected.prefilter_status == "rejected"
    assert weak.prefilter_status == "weak"
    assert review.prefilter_status == "review"
    assert strong.prefilter_status == "strong"


def test_build_context_dry_run_and_db_unchanged(client, db_session) -> None:
    _seed_travel(db_session)
    before = db_session.scalar(select(func.count(KeywordContext.id))) or 0
    response = client.post("/api/travel-opportunities/build-contexts?dry_run=true")
    db_session.expire_all()
    assert response.status_code == 200
    assert response.json()["contexts_would_create"] >= 1
    assert (db_session.scalar(select(func.count(KeywordContext.id))) or 0) == before


def test_context_hash_duplicate_prevention(db_session) -> None:
    _seed_travel(db_session)
    first = build_keyword_contexts(db_session, week_start=WEEK_START, limit=50, force=False, dry_run=False)
    second = build_keyword_contexts(db_session, week_start=WEEK_START, limit=50, force=False, dry_run=False)
    assert first.contexts_would_create >= 1
    assert second.duplicate_contexts >= first.contexts_would_create


def test_prefilter_dry_run_db_unchanged_and_gemini_not_called(monkeypatch, client, db_session) -> None:
    _seed_travel(db_session)
    before = db_session.scalar(select(func.count(TravelOpportunityCandidate.id))) or 0

    async def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("Gemini must not run during V2 prefilter")

    monkeypatch.setattr("app.ai.gemini_adapter.GeminiAdapter.generate", forbidden_generate)
    response = client.post("/api/travel-opportunities/prefilter?dry_run=true")
    db_session.expire_all()
    assert response.status_code == 200
    payload = response.json()
    assert payload["processed"] >= 1
    assert "reduction_rate" in payload
    assert payload["quality_keyword_count"] == 2
    assert (db_session.scalar(select(func.count(TravelOpportunityCandidate.id))) or 0) == before


def test_build_contexts_process_all_traverses_multiple_pages(client, db_session) -> None:
    _seed_travel(db_session)

    response = client.post(
        "/api/travel-opportunities/build-contexts"
        "?dry_run=false&force=false&limit=1&process_all=true"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["batches"] == 2
    assert payload["has_more"] is False
    assert payload["next_cursor"] is None
    assert payload["created"] == payload["contexts_would_create"]
    assert payload["created"] >= 2
    context_documents = set(db_session.scalars(select(KeywordContext.document_id)).all())
    assert len(context_documents) == 2


def test_build_contexts_manual_cursor_keeps_stable_dataset(client, db_session) -> None:
    _seed_travel(db_session)

    first = client.post(
        f"/api/travel-opportunities/build-contexts?week_start={WEEK_START.isoformat()}"
        "&dry_run=false&force=false&limit=1"
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["has_more"] is True
    assert first_payload["next_cursor"] is not None

    second = client.post(
        f"/api/travel-opportunities/build-contexts?week_start={WEEK_START.isoformat()}"
        f"&dry_run=false&force=false&limit=1&after_id={first_payload['next_cursor']}"
    )
    assert second.status_code == 200
    assert second.json()["has_more"] is False
    context_documents = set(db_session.scalars(select(KeywordContext.document_id)).all())
    assert len(context_documents) == 2


def test_build_contexts_paginated_dry_run_and_empty_page_do_not_write(client, db_session) -> None:
    _seed_travel(db_session)
    before = db_session.scalar(select(func.count(KeywordContext.id))) or 0

    dry_run = client.post(
        f"/api/travel-opportunities/build-contexts?week_start={WEEK_START.isoformat()}"
        "&dry_run=true&force=false&limit=1&process_all=true"
    )
    empty = client.post(
        f"/api/travel-opportunities/build-contexts?week_start={WEEK_START.isoformat()}"
        "&dry_run=true&force=false&limit=1&after_id=999999"
    )

    assert dry_run.status_code == 200
    assert dry_run.json()["batches"] == 2
    assert empty.status_code == 200
    assert empty.json()["batches"] == 0
    assert empty.json()["has_more"] is False
    assert (db_session.scalar(select(func.count(KeywordContext.id))) or 0) == before


def test_rule_cursor_pages_are_ordered_and_do_not_overlap(db_session) -> None:
    _seed_travel(db_session)
    build_keyword_contexts(
        db_session,
        week_start=WEEK_START,
        limit=1,
        force=False,
        dry_run=False,
        process_all=True,
    )

    first, cursor, has_more = travel_repo.get_keyword_contexts_page(
        db_session,
        week_start=WEEK_START,
        week_end=WEEK_END,
        candidate_week_start=WEEK_START,
        after_id=None,
        limit=1,
        force=False,
    )
    second, _next_cursor, _has_more = travel_repo.get_keyword_contexts_page(
        db_session,
        week_start=WEEK_START,
        week_end=WEEK_END,
        candidate_week_start=WEEK_START,
        after_id=cursor,
        limit=1,
        force=False,
    )

    assert has_more is True
    assert cursor == first[-1].id
    assert second[0].id > first[-1].id


def test_prefilter_process_all_is_complete_and_restart_safe(client, db_session) -> None:
    _seed_travel(db_session)
    build_keyword_contexts(
        db_session,
        week_start=WEEK_START,
        limit=2,
        force=False,
        dry_run=False,
        process_all=True,
    )
    context_total = db_session.scalar(select(func.count(KeywordContext.id))) or 0

    first = client.post(
        f"/api/travel-opportunities/prefilter?week_start={WEEK_START.isoformat()}"
        "&dry_run=false&force=false&limit=3&process_all=true"
    )
    assert first.status_code == 200
    first_payload = first.json()
    assert first_payload["processed"] == context_total
    assert first_payload["created"] == context_total
    assert first_payload["updated"] == 0
    assert first_payload["has_more"] is False
    assert first_payload["batches"] == (context_total + 2) // 3

    evaluated, materialized, remaining = travel_repo.count_rule_materialization_coverage(
        db_session,
        week_start=WEEK_START,
        week_end=WEEK_END,
        candidate_week_start=WEEK_START,
    )
    assert (evaluated, materialized, remaining) == (context_total, context_total, 0)
    row_count = db_session.scalar(select(func.count(TravelOpportunityCandidate.id))) or 0
    distinct_contexts = db_session.scalar(
        select(func.count(func.distinct(TravelOpportunityCandidate.keyword_context_id)))
    ) or 0
    timestamps = dict(
        db_session.execute(
            select(TravelOpportunityCandidate.id, TravelOpportunityCandidate.updated_at)
        ).all()
    )
    assert row_count == distinct_contexts == context_total

    second = client.post(
        f"/api/travel-opportunities/prefilter?week_start={WEEK_START.isoformat()}"
        "&dry_run=false&force=false&limit=2&process_all=true"
    )
    assert second.status_code == 200
    second_payload = second.json()
    assert second_payload["processed"] == 0
    assert second_payload["created"] == 0
    assert second_payload["updated"] == 0
    assert second_payload["skipped"] == context_total
    assert second_payload["batches"] == 0
    assert (db_session.scalar(select(func.count(TravelOpportunityCandidate.id))) or 0) == row_count
    assert dict(
        db_session.execute(
            select(TravelOpportunityCandidate.id, TravelOpportunityCandidate.updated_at)
        ).all()
    ) == timestamps


def test_prefilter_force_preview_counts_mixed_rows_and_force_false_preserves_semantic(
    db_session,
) -> None:
    _seed_travel(db_session)
    build_keyword_contexts(
        db_session,
        week_start=WEEK_START,
        limit=2,
        force=False,
        dry_run=False,
        process_all=True,
    )
    context_total = db_session.scalar(select(func.count(KeywordContext.id))) or 0
    first = prefilter_travel_opportunities(
        db_session,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=1,
    )
    assert first.created == 1
    existing = db_session.scalar(select(TravelOpportunityCandidate))
    assert existing is not None
    existing.semantic_status = "semantic_review"
    existing.semantic_travel_score = 77.0
    existing.ranking_status = "review"
    db_session.commit()

    preview = prefilter_travel_opportunities(
        db_session,
        week_start=WEEK_START,
        dry_run=True,
        force=True,
        limit=10,
        process_all=True,
    )
    assert preview.processed == context_total
    assert preview.would_update == 1
    assert preview.would_create == context_total - 1

    resumed = prefilter_travel_opportunities(
        db_session,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=2,
        process_all=True,
    )
    assert resumed.created == context_total - 1
    db_session.refresh(existing)
    assert existing.semantic_status == "semantic_review"
    assert existing.semantic_travel_score == 77.0
    assert existing.ranking_status == "review"


def test_prefilter_empty_cursor_page_is_safe(client, db_session) -> None:
    _seed_travel(db_session)
    build_keyword_contexts(
        db_session,
        week_start=WEEK_START,
        limit=10,
        force=False,
        dry_run=False,
        process_all=True,
    )

    response = client.post(
        f"/api/travel-opportunities/prefilter?week_start={WEEK_START.isoformat()}"
        "&dry_run=true&force=false&limit=2&after_id=999999"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["processed"] == 0
    assert payload["batches"] == 0
    assert payload["has_more"] is False
    assert payload["next_cursor"] is None


def test_summary_calculation_for_dashboard(client, db_session) -> None:
    _seed_travel(db_session)
    build_keyword_contexts(db_session, week_start=WEEK_START, limit=50, force=False, dry_run=False)
    prefilter_travel_opportunities(db_session, week_start=WEEK_START, dry_run=False, force=True, limit=50)
    response = client.get(f"/api/travel-opportunities/summary?week_start={WEEK_START.isoformat()}")
    assert response.status_code == 200
    summary = response.json()
    assert summary["raw_keyword_count"] == 2
    assert summary["raw_keyword_occurrences"] == 2
    assert summary["keyword_candidate_total"] == 2
    assert summary["keyword_candidate_accepted_rows"] == 2
    assert summary["distinct_accepted_keywords"] == 2
    assert summary["quality_keyword_count"] == 2
    assert summary["weekly_trend_count"] == 2
    assert summary["context_candidate_count"] >= 2
    assert summary["estimated_gemini_calls"] == summary["strong_candidate_count"]
    assert 0 <= summary["llm_reduction_rate"] <= 100


def test_summary_quality_counts_are_zero_without_candidates(db_session) -> None:
    db_session.add_all(
        [
            _trend("주간키워드1", 90, 2),
            _trend("주간키워드2", 80, 2),
        ]
    )
    db_session.commit()

    summary = summarize_v2(db_session, week_start=WEEK_START)

    assert summary["keyword_candidate_total"] == 0
    assert summary["keyword_candidate_accepted_rows"] == 0
    assert summary["distinct_accepted_keywords"] == 0
    assert summary["quality_keyword_count"] == 0
    assert summary["raw_keyword_count"] == 2
    assert summary["raw_keyword_occurrences"] == 0
    assert summary["weekly_trend_count"] == 2


def test_summary_quality_counts_use_document_week_and_distinct_keywords(db_session) -> None:
    target_one = _source_document("target-one", NOW)
    target_two = _source_document("target-two", datetime(2026, 7, 28, 9, 0, 0))
    before_week = _source_document("before-week", datetime(2026, 7, 20, 9, 0, 0))
    after_week = _source_document("after-week", datetime(2026, 8, 3, 0, 0, 0))
    db_session.add_all([target_one, target_two, before_week, after_week])
    db_session.flush()
    db_session.add_all(
        [
            _candidate(target_one.id, "중복키워드", accepted=True),
            _candidate(target_two.id, "중복키워드", accepted=True),
            _candidate(target_two.id, "고유키워드", accepted=True),
            _candidate(target_two.id, "거절키워드", accepted=False),
            _candidate(before_week.id, "이전주간키워드", accepted=True),
            _candidate(after_week.id, "다음주간키워드", accepted=True),
            _trend("WeeklyTrend와별도", 90, 2),
        ]
    )
    db_session.commit()

    summary = summarize_v2(db_session, week_start=WEEK_START)

    assert summary["keyword_candidate_total"] == 4
    assert summary["keyword_candidate_accepted_rows"] == 3
    assert summary["distinct_accepted_keywords"] == 2
    assert summary["quality_keyword_count"] == 2
    assert summary["weekly_trend_count"] == 1


def test_positive_and_negative_term_files_have_terms() -> None:
    assert len(load_terms("travel_positive_terms.json")) >= 30
    assert len(load_terms("travel_negative_terms.json")) >= 20


def _seed_travel(session) -> None:
    film = SourceDocument(
        source="newsis_rss",
        source_id="film-doc",
        title="8월의 크리스마스 재조명",
        text="1998년 개봉한 허진호 감독의 영화 '8월의 크리스마스'가 다시 소개되고 있다.",
        published_at=NOW,
        collected_at=NOW,
        views=None,
        likes=None,
        comments=None,
        url="https://example.test/film",
    )
    finance = SourceDocument(
        source="youtube",
        source_id="finance-doc",
        title="삼성전자 실적",
        text="삼성전자 2분기 영업이익과 주가 전망이 발표됐다.",
        published_at=NOW,
        collected_at=NOW,
        views=100,
        likes=1,
        comments=1,
        url="https://example.test/finance",
    )
    session.add_all([film, finance])
    session.flush()
    session.add_all(
        [
            KeywordOccurrence(
                document_id=film.id,
                keyword="8월의 크리스마스",
                normalized_keyword="8월의크리스마스",
                source="newsis_rss",
                occurred_at=NOW,
                keyword_quality_score=90,
                pipeline_version="v2",
            ),
            KeywordOccurrence(
                document_id=finance.id,
                keyword="삼성전자",
                normalized_keyword="삼성전자",
                source="youtube",
                occurred_at=NOW,
                keyword_quality_score=90,
                pipeline_version="v2",
            ),
        ]
    )
    session.add_all(
        [
            _candidate(film.id, "8월의크리스마스", accepted=True),
            _candidate(finance.id, "삼성전자", accepted=True),
        ]
    )
    session.add_all(
        [
            _trend("8월의크리스마스", 90, 2),
            _trend("삼성전자", 90, 3),
        ]
    )
    session.add_all(
        [
            EntityMention(
                document_id=film.id,
                text="8월의 크리스마스",
                normalized_text="8월의크리스마스",
                entity_type="CONTENT_TITLE",
                confidence=0.95,
                extractor="rule",
                start_char=28,
                end_char=37,
                source="newsis_rss",
                occurred_at=NOW,
                created_at=NOW,
            ),
            EntityMention(
                document_id=finance.id,
                text="삼성전자",
                normalized_text="삼성전자",
                entity_type="BRAND",
                confidence=0.95,
                extractor="rule",
                start_char=0,
                end_char=4,
                source="youtube",
                occurred_at=NOW,
                created_at=NOW,
            ),
        ]
    )
    session.commit()


def _source_document(source_id: str, published_at: datetime) -> SourceDocument:
    return SourceDocument(
        source="mock",
        source_id=source_id,
        title=source_id,
        text=f"{source_id} 본문",
        published_at=published_at,
        collected_at=CANDIDATE_CREATED_AT,
        views=None,
        likes=None,
        comments=None,
        url=f"https://example.test/{source_id}",
    )


def _candidate(document_id: int, normalized: str, *, accepted: bool) -> KeywordCandidate:
    return KeywordCandidate(
        document_id=document_id,
        candidate_text=normalized,
        normalized_candidate=normalized,
        candidate_type="noun_phrase",
        extractor="test",
        quality_score=90 if accepted else 10,
        accepted=accepted,
        rejection_reason=None if accepted else "low_quality",
        title_occurrence=1,
        body_occurrence=1,
        entity_type=None,
        entity_confidence=None,
        created_at=CANDIDATE_CREATED_AT,
        pipeline_version="v2",
    )


def _trend(keyword: str, final_score: float, source_count: int) -> WeeklyTrend:
    return WeeklyTrend(
        keyword=keyword,
        week_start=WEEK_START,
        week_end=WEEK_END,
        weekly_mentions=5,
        previous_weekly_mentions=1,
        active_days=3,
        source_count=source_count,
        growth_rate=2.0,
        peak_day_share=0.4,
        persistence_score=70,
        diversity_score=60,
        freshness_score=80,
        volume_score=75,
        growth_score=70,
        trend_score=75,
        keyword_quality_score=90,
        search_interest_score=None,
        search_interest_available=False,
        search_provider_count=0,
        one_day_spike_penalty=0,
        spam_penalty=0,
        final_score=final_score,
        status="weekly_trend",
        calculated_at=NOW,
        pipeline_version="v2",
    )

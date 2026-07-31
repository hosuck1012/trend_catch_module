from datetime import date, datetime
import json

from app.api.ai_analysis import get_gemini_adapter
from app.config import get_settings
from app.main import app
from app.models.entity_context import EntityContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.trend_ai_analysis import TrendAIAnalysis
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from dashboard.state import clear_dashboard_cache
from dashboard.views import _select_ai_candidates


WEEK_START = date(2026, 7, 27)
WEEK_END = date(2026, 8, 2)
NOW = datetime(2026, 8, 1, 9, 0, 0)


def test_dashboard_overview_and_existing_api_compatibility(client, db_session) -> None:
    _seed_dashboard(db_session)

    response = client.get("/api/dashboard/overview")
    health = client.get("/health")
    weekly = client.get("/api/trends/weekly")

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_week"] == WEEK_START.isoformat()
    assert len(payload["metric_cards"]) == 6
    assert payload["top_trends"][0]["keyword"] == "거제야호"
    assert payload["source_distribution"] == [
        {"name": "newsis_rss", "count": 1},
        {"name": "youtube", "count": 1},
    ]
    assert health.status_code == 200
    assert weekly.status_code == 200


def test_dashboard_trend_list_filters(client, db_session) -> None:
    _seed_dashboard(db_session)

    by_week = client.get("/api/dashboard/trends", params={"week_start": WEEK_START})
    by_query = client.get("/api/dashboard/trends", params={"query": "거제"})
    by_source = client.get("/api/dashboard/trends", params={"source": "youtube"})
    by_score = client.get("/api/dashboard/trends", params={"min_final_score": 80})
    by_travel = client.get("/api/dashboard/trends", params={"min_travel_score": 80})
    by_ai = client.get("/api/dashboard/trends", params={"ai_status": "completed"})
    watchlist = client.get("/api/dashboard/trends", params={"watchlist_only": True})

    assert by_week.json()["total"] == 2
    assert [item["keyword"] for item in by_query.json()["items"]] == ["거제야호"]
    assert [item["keyword"] for item in by_source.json()["items"]] == ["거제야호"]
    assert [item["keyword"] for item in by_score.json()["items"]] == ["거제야호"]
    assert [item["keyword"] for item in by_travel.json()["items"]] == ["거제야호"]
    assert [item["keyword"] for item in by_ai.json()["items"]] == ["거제야호"]
    assert [item["keyword"] for item in watchlist.json()["items"]] == ["제주도여행"]


def test_dashboard_trend_detail_handles_nullable_fields(client, db_session) -> None:
    _seed_dashboard(db_session)

    response = client.get("/api/dashboard/trends/거제야호")

    assert response.status_code == 200
    payload = response.json()
    assert payload["trend"]["trend_score"] == 72.0
    assert payload["trend"]["acceleration"] is None
    assert payload["entities"][0]["entity_text"] == "거제시"
    assert payload["contexts"][0]["page_title"] == "거제시"
    assert payload["ai_analysis"]["model_name"] == "gemini-3.6-flash"
    assert payload["documents"][0]["snippet"].startswith("거제 여행")


def test_dashboard_empty_data_and_unknown_week(client) -> None:
    response = client.get("/api/dashboard/overview")

    assert response.status_code == 200
    assert response.json()["selected_week"] is None
    assert response.json()["top_trends"] == []


def test_dashboard_falls_back_to_latest_week_with_actual_data(client, db_session) -> None:
    _seed_dashboard(db_session)

    response = client.get(
        "/api/dashboard/overview",
        params={"week_start": "2026-08-03"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_week"] == "2026-08-03"
    assert payload["selected_week"] == WEEK_START.isoformat()
    assert payload["week_fallback_used"] is True
    assert payload["top_trends"]


def test_low_quality_and_suspicious_keywords_are_hidden_by_default(
    client,
    db_session,
) -> None:
    _seed_dashboard(db_session)
    low = _trend("낮은품질", final_score=90, status="weekly_trend", source_count=1)
    low.keyword_quality_score = 20
    low_specificity = _trend(
        "보통일반어", final_score=91, status="weekly_trend", source_count=1
    )
    low_specificity.keyword_quality_score = 55
    suspicious = _trend("com", final_score=95, status="weekly_trend", source_count=1)
    db_session.add_all([low, low_specificity, suspicious])
    db_session.commit()

    default = client.get("/api/dashboard/trends")
    with_low_quality = client.get(
        "/api/dashboard/trends",
        params={"include_low_quality": True},
    )

    default_keywords = {item["keyword"] for item in default.json()["items"]}
    all_keywords = {item["keyword"] for item in with_low_quality.json()["items"]}
    assert "낮은품질" not in default_keywords
    assert "보통일반어" not in default_keywords
    assert "com" not in default_keywords
    assert {"낮은품질", "보통일반어", "com"}.issubset(all_keywords)


def test_pipeline_separates_wikipedia_matched_and_errors(client, db_session) -> None:
    _seed_dashboard(db_session)

    response = client.get("/api/dashboard/overview")

    wikipedia = next(
        item for item in response.json()["pipeline_status"] if item["key"] == "wikipedia"
    )
    assert wikipedia["status"] == "partial"
    assert wikipedia["details"]["matched"] == 1
    assert wikipedia["details"]["manual"] == 0
    assert wikipedia["details"]["ambiguous"] == 0
    assert wikipedia["details"]["unmatched"] == 0
    assert wikipedia["details"]["error"] == 1
    assert wikipedia["details"]["errors"] == 1


def test_ai_status_does_not_expose_api_key(client, monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "never-expose-this-secret")
    get_settings.cache_clear()

    response = client.get("/api/ai-analysis/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["api_key_configured"] is True
    assert "never-expose-this-secret" not in response.text
    assert "gemini_api_key" not in payload
    get_settings.cache_clear()


def test_ai_candidate_selection_is_limited_and_uses_requested_priority() -> None:
    trends = [
        {
            "normalized_keyword": f"키워드{index}",
            "final_score": score,
            "keyword_quality_score": quality,
            "source_count": sources,
            "document_count": documents,
            "suspicious": False,
        }
        for index, (score, quality, sources, documents) in enumerate(
            [
                (90, 80, 1, 2),
                (90, 90, 1, 2),
                (90, 90, 2, 1),
                (90, 90, 2, 3),
                (80, 99, 9, 9),
                (70, 99, 9, 9),
            ]
        )
    ]
    trends.append(
        {
            "normalized_keyword": "의심어",
            "final_score": 100,
            "keyword_quality_score": 100,
            "source_count": 10,
            "document_count": 10,
            "suspicious": True,
        }
    )

    selected = _select_ai_candidates(trends)

    assert [item["normalized_keyword"] for item in selected] == [
        "키워드3",
        "키워드2",
        "키워드1",
        "키워드0",
        "키워드4",
    ]


def test_dashboard_refresh_clears_all_streamlit_data_cache(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr("dashboard.state.st.cache_data.clear", lambda: calls.append(True))

    clear_dashboard_cache()

    assert calls == [True]


def test_dashboard_reads_do_not_invoke_gemini(client, db_session) -> None:
    _seed_dashboard(db_session)
    calls = 0

    async def forbidden_adapter():
        nonlocal calls
        calls += 1
        raise AssertionError("Gemini must not run during dashboard reads")

    app.dependency_overrides[get_gemini_adapter] = forbidden_adapter
    try:
        response = client.get("/api/dashboard/overview")
    finally:
        app.dependency_overrides.pop(get_gemini_adapter, None)

    assert response.status_code == 200
    assert calls == 0


def _seed_dashboard(session) -> None:
    first = SourceDocument(
        source="youtube",
        source_id="dash-youtube",
        title="거제야호로 떠나는 거제 여행",
        text="거제 여행 관련 영상의 짧은 설명입니다.",
        published_at=NOW,
        collected_at=NOW,
        views=100,
        likes=10,
        comments=2,
        url="https://example.test/youtube",
    )
    second = SourceDocument(
        source="newsis_rss",
        source_id="dash-news",
        title="제주도여행 관심",
        text="제주 여행 관련 뉴스 요약입니다.",
        published_at=datetime(2026, 7, 31, 8, 0, 0),
        collected_at=NOW,
        views=None,
        likes=None,
        comments=None,
        url="https://example.test/news",
    )
    session.add_all([first, second])
    session.flush()
    session.add_all(
        [
            KeywordOccurrence(
                document_id=first.id,
                keyword="거제야호",
                normalized_keyword="거제야호",
                source="youtube",
                occurred_at=NOW,
            ),
            KeywordOccurrence(
                document_id=second.id,
                keyword="제주도여행",
                normalized_keyword="제주도여행",
                source="newsis_rss",
                occurred_at=datetime(2026, 7, 31, 8, 0, 0),
            ),
        ]
    )
    session.add_all(
        [
            _trend("거제야호", final_score=82.5, status="weekly_trend", source_count=1),
            _trend("제주도여행", final_score=55.0, status="watchlist", source_count=1),
        ]
    )
    session.add(
        TrendEntityLink(
            keyword="거제야호",
            week_start=WEEK_START,
            week_end=WEEK_END,
            entity_text="거제시",
            normalized_entity="거제시",
            entity_type="LOCATION",
            mention_count=3,
            document_count=1,
            source_count=1,
            average_confidence=0.95,
            relation_score=91.0,
            is_primary=True,
            calculated_at=NOW,
        )
    )
    matched = _context("matched", "https://ko.wikipedia.org/wiki/test", "거제시")
    error = _context("error", "https://ko.wikipedia.org/wiki/error", "오류 후보")
    session.add_all([matched, error])
    session.flush()
    session.add(
        TrendContextLink(
            keyword="거제야호",
            week_start=WEEK_START,
            week_end=WEEK_END,
            entity_context_id=matched.id,
            normalized_entity="거제시",
            entity_type="LOCATION",
            context_score=92.0,
            is_primary=True,
            created_at=NOW,
            updated_at=NOW,
        )
    )
    session.add(
        TrendAIAnalysis(
            keyword="거제야호",
            normalized_keyword="거제야호",
            week_start=WEEK_START,
            week_end=WEEK_END,
            model_name="gemini-3.6-flash",
            prompt_version="dashboard-test-v1",
            analysis_status="completed",
            trend_summary="거제 관련 콘텐츠 관심이 증가했습니다.",
            rising_reason="입력 문서와 점수에서 상승 신호가 확인됩니다.",
            evidence_summary=json.dumps(["주간 점수 상승"], ensure_ascii=False),
            travel_relevance_score=88.0,
            travel_relevance_level="high",
            travel_relevance_reason="LOCATION 객체가 연결됐습니다.",
            recommended_destinations_json=json.dumps([], ensure_ascii=False),
            content_ideas_json=json.dumps([], ensure_ascii=False),
            cautions_json=json.dumps([], ensure_ascii=False),
            evidence_refs_json=json.dumps(["SCORE-WEEKLY"], ensure_ascii=False),
            confidence_score=84.0,
            input_hash="a" * 64,
            raw_response_json="{}",
            error_code=None,
            error_message=None,
            generated_at=NOW,
            updated_at=NOW,
        )
    )
    session.commit()


def _trend(keyword: str, *, final_score: float, status: str, source_count: int) -> WeeklyTrend:
    return WeeklyTrend(
        keyword=keyword,
        week_start=WEEK_START,
        week_end=WEEK_END,
        weekly_mentions=8,
        previous_weekly_mentions=3,
        active_days=4,
        source_count=source_count,
        growth_rate=1.5,
        peak_day_share=0.3,
        persistence_score=70,
        diversity_score=60,
        freshness_score=90,
        volume_score=75,
        growth_score=80,
        trend_score=72,
        keyword_quality_score=90,
        search_interest_score=68,
        search_interest_available=True,
        search_provider_count=1,
        one_day_spike_penalty=0,
        spam_penalty=0,
        final_score=final_score,
        status=status,
        calculated_at=NOW,
        pipeline_version="v2",
    )


def _context(status: str, url: str, title: str) -> EntityContext:
    return EntityContext(
        normalized_entity=title,
        entity_text=title,
        entity_type="LOCATION",
        provider="wikipedia_ko",
        page_id=None,
        page_title=title,
        page_url=url,
        summary="한국어 위키백과 요약" if status == "matched" else "요청 오류",
        description=None,
        match_score=0.9 if status == "matched" else 0.0,
        match_status=status,
        source_language="ko",
        license_name=None,
        attribution_text="한국어 위키백과",
        revision_id=None,
        retrieved_at=NOW,
        updated_at=NOW,
    )

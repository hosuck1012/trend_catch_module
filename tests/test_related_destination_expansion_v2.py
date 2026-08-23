from datetime import date, datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import func, select

from app.models.entity_context import EntityContext
from app.models.keyword_context import KeywordContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_context_link import TrendContextLink
from app.models.weekly_trend import WeeklyTrend
from app.services.related_destination_expansion_service import (
    is_related_destination_context,
)
from app.services.travel_semantic_filter_service import _merge_embedding_context


WEEK_START = date(2026, 7, 25)
WEEK_END = date(2026, 7, 31)
NOW = datetime(2026, 7, 31, 12, 0, 0)
NORMALIZED_KEYWORD = "f1더무비"


def test_related_destination_expansion_is_dry_run_safe_and_idempotent(
    client, db_session
) -> None:
    _seed_f1_trend(db_session)

    preview = client.post(
        "/api/travel-opportunities/expand-destinations",
        params={"week_start": WEEK_START.isoformat(), "dry_run": True},
    )

    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["matched_keywords"] == 1
    assert preview_payload["destinations_matched"] == 3
    assert preview_payload["contexts_created"] == 3
    assert preview_payload["links_created"] == 3
    assert db_session.scalar(select(func.count(EntityContext.id))) == 0
    assert db_session.scalar(select(func.count(TrendContextLink.id))) == 0

    created = client.post(
        "/api/travel-opportunities/expand-destinations",
        params={"week_start": WEEK_START.isoformat(), "dry_run": False},
    )
    repeated = client.post(
        "/api/travel-opportunities/expand-destinations",
        params={"week_start": WEEK_START.isoformat(), "dry_run": False},
    )

    assert created.status_code == 200
    assert created.json()["contexts_created"] == 3
    assert created.json()["links_created"] == 3
    assert repeated.status_code == 200
    assert repeated.json()["contexts_created"] == 0
    assert repeated.json()["links_created"] == 0
    assert repeated.json()["skipped"] == 3
    contexts = list(db_session.scalars(select(EntityContext)).all())
    assert {context.entity_text for context in contexts} == {
        "코리아 인터내셔널 서킷",
        "인제스피디움",
        "BMW 드라이빙 센터",
    }
    assert all(context.page_url.startswith("https://") for context in contexts)
    assert all(context.match_status == "manual" for context in contexts)


def test_related_destination_evidence_reaches_prefilter_ranking_and_api(
    client, db_session
) -> None:
    _seed_f1_trend(db_session)
    expanded = client.post(
        "/api/travel-opportunities/expand-destinations",
        params={"week_start": WEEK_START.isoformat(), "dry_run": False},
    )
    prefiltered = client.post(
        "/api/travel-opportunities/prefilter",
        params={
            "week_start": WEEK_START.isoformat(),
            "dry_run": False,
            "force": True,
            "process_all": True,
            "limit": 50,
        },
    )

    assert expanded.status_code == 200
    assert prefiltered.status_code == 200
    candidates = list(
        db_session.scalars(
            select(TravelOpportunityCandidate).where(
                TravelOpportunityCandidate.normalized_keyword == NORMALIZED_KEYWORD
            )
        ).all()
    )
    assert len(candidates) == 3
    assert {candidate.travel_category for candidate in candidates} == {"SPORTS_EVENT"}
    assert all(candidate.prefilter_status in {"weak", "review", "strong"} for candidate in candidates)
    assert all(candidate.primary_entity_type == "CONTENT_TITLE" for candidate in candidates)
    assert all(
        "RELATED_DESTINATION_VERIFIED" in candidate.reasoning_codes_json
        for candidate in candidates
    )

    for candidate in candidates:
        candidate.semantic_travel_score = 82.0
        candidate.semantic_status = "semantic_strong"
        candidate.semantic_positive_score = 0.86
        candidate.semantic_positive_category = "SPORTS_EVENT"
        candidate.semantic_negative_score = 0.80
        candidate.semantic_negative_category = "GENERAL_NON_TRAVEL"
    db_session.commit()

    ranked = client.post(
        "/api/travel-opportunities/rank",
        params={
            "week_start": WEEK_START.isoformat(),
            "dry_run": False,
            "force": True,
            "limit": 100,
        },
    )
    detail = client.get(f"/api/travel-opportunities/{NORMALIZED_KEYWORD}")
    report = client.get(
        "/api/travel-opportunities/calibration-report",
        params={"week_start": WEEK_START.isoformat()},
    )
    candidate_list = client.get(
        "/api/travel-opportunities",
        params={"week_start": WEEK_START.isoformat(), "limit": 100},
    )

    assert ranked.status_code == 200
    ranked_item = next(
        item
        for item in ranked.json()["top_candidates"]
        if item["normalized_keyword"] == NORMALIZED_KEYWORD
    )
    assert ranked_item["ranking_status"] == "review"
    assert ranked_item["evidence_gate"] == "NEEDS_EVIDENCE"
    assert "RELATED_DESTINATION_CONTEXT" in ranked_item["evidence_codes"]
    assert len(ranked_item["related_destinations"]) == 3
    assert detail.status_code == 200
    assert len(detail.json()["related_destinations"]) == 3
    assert candidate_list.status_code == 200
    list_item = next(
        item
        for item in candidate_list.json()["items"]
        if item["normalized_keyword"] == NORMALIZED_KEYWORD
    )
    assert len(list_item["related_destinations"]) == 3
    report_item = next(
        item
        for item in report.json()["top_20_candidates"]
        if item["normalized_keyword"] == NORMALIZED_KEYWORD
    )
    assert len(report_item["related_destinations"]) == 3

    for candidate in candidates:
        candidate.high_precision_score = 1.0
        candidate.evidence_codes_json = '["STALE"]'
    db_session.commit()
    reranked = client.post(
        "/api/travel-opportunities/rank",
        params={
            "week_start": WEEK_START.isoformat(),
            "dry_run": False,
            "limit": 100,
        },
    )
    db_session.expire_all()
    refreshed = db_session.scalar(
        select(TravelOpportunityCandidate).where(
            TravelOpportunityCandidate.normalized_keyword == NORMALIZED_KEYWORD
        )
    )
    assert reranked.status_code == 200
    assert refreshed is not None
    assert refreshed.high_precision_score != 1.0
    assert "RELATED_DESTINATION_CONTEXT" in refreshed.evidence_codes_json


def test_expansion_requires_the_configured_trend_source(client, db_session) -> None:
    _seed_f1_trend(db_session, occurrence_source="youtube")

    response = client.post(
        "/api/travel-opportunities/expand-destinations",
        params={"week_start": WEEK_START.isoformat(), "dry_run": False},
    )

    assert response.status_code == 200
    assert response.json()["matched_keywords"] == 0
    assert response.json()["destinations_matched"] == 0


def test_invalid_destination_context_is_not_eligible() -> None:
    context = SimpleNamespace(
        page_id="travel-destination:MOTORSPORT:test",
        match_status="error",
    )

    assert is_related_destination_context(context) is False


def test_supplemental_embedding_context_stays_within_budget() -> None:
    merged = _merge_embedding_context(
        "기본 문맥 " * 50,
        "공식 정보 확인 여행지 서킷 방문 관람 체험 " * 20,
        max_chars=120,
    )

    assert len(merged) <= 120
    assert "기본 문맥" in merged
    assert "공식 정보 확인 여행지" in merged


def _seed_f1_trend(session, *, occurrence_source: str = "google_yis_2025_kr") -> None:
    session.add(
        WeeklyTrend(
            keyword=NORMALIZED_KEYWORD,
            week_start=WEEK_START,
            week_end=WEEK_END,
            weekly_mentions=3,
            previous_weekly_mentions=0,
            active_days=3,
            source_count=1,
            growth_rate=3.0,
            peak_day_share=1 / 3,
            persistence_score=85,
            diversity_score=25,
            freshness_score=90,
            volume_score=80,
            growth_score=90,
            trend_score=85,
            keyword_quality_score=90,
            search_interest_score=None,
            search_interest_available=False,
            search_provider_count=0,
            one_day_spike_penalty=0,
            spam_penalty=0,
            final_score=88,
            status="watchlist",
            calculated_at=NOW,
            pipeline_version="v2",
        )
    )
    for index in range(3):
        published_at = NOW - timedelta(days=index * 2)
        document = SourceDocument(
            source=occurrence_source,
            source_id=f"f1-seed-{index}",
            title="Google Year in Search 2025 Korea: F1 더 무비",
            text="F1 더 무비는 대한민국 영화 분야 인기 검색어입니다.",
            published_at=published_at,
            collected_at=NOW,
            views=None,
            likes=None,
            comments=None,
            url="https://trends.withgoogle.com/year-in-search/2025/kr/",
        )
        session.add(document)
        session.flush()
        session.add(
            KeywordOccurrence(
                document_id=document.id,
                keyword="F1 더 무비",
                normalized_keyword=NORMALIZED_KEYWORD,
                source=occurrence_source,
                occurred_at=published_at,
                keyword_quality_score=90,
                pipeline_version="v2",
            )
        )
        session.add(
            KeywordContext(
                document_id=document.id,
                keyword="F1 더 무비",
                normalized_keyword=NORMALIZED_KEYWORD,
                previous_sentence=None,
                matched_sentence="F1 더 무비는 대한민국 영화 분야 인기 검색어입니다.",
                next_sentence=None,
                combined_context="F1 더 무비는 대한민국 영화 분야 인기 검색어입니다.",
                occurrence_index=0,
                source=occurrence_source,
                published_at=published_at,
                context_hash=f"f1-context-{index}",
                created_at=NOW,
                updated_at=NOW,
            )
        )
    session.commit()

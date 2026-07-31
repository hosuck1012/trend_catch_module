from datetime import date, datetime, timedelta

import pytest
from sqlalchemy import select

from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.weekly_trend import WeeklyTrend
from app.repositories.trend_repository import OccurrenceRow, count_weekly_trends_for_week
from app.services.trend_calculation_service import calculate_scores


def test_recalculate_api_returns_error_without_keyword_occurrences(client) -> None:
    response = client.post("/api/trends/recalculate")

    assert response.status_code == 400
    assert "키워드 발생 데이터가 없습니다" in response.json()["detail"]


def test_week_range_uses_latest_occurrence_date(client) -> None:
    _collect_and_extract(client)

    response = client.post("/api/trends/recalculate")

    assert response.status_code == 200
    assert response.json()["week_start"] == "2026-07-22"
    assert response.json()["week_end"] == "2026-07-28"


def test_keyword_metric_calculations_for_mock_keyword(client, db_session) -> None:
    _collect_and_extract(client)
    client.post("/api/trends/recalculate")

    trend = _get_trend(db_session, "거제야호")

    assert trend.weekly_mentions == 22
    assert trend.previous_weekly_mentions == 8
    assert trend.active_days == 7
    assert trend.source_count == 2
    assert trend.growth_rate == pytest.approx(1.75)
    assert trend.peak_day_share == pytest.approx(4 / 22, abs=0.0001)
    assert trend.persistence_score == 100
    assert 0 <= trend.final_score <= 100


def test_score_calculation_has_expected_period_counts() -> None:
    base = date(2026, 7, 30)
    week_start = base - timedelta(days=6)
    previous_week_start = week_start - timedelta(days=7)
    previous_week_end = week_start - timedelta(days=1)
    rows = [
        OccurrenceRow("sample", "youtube", previous_week_start),
        OccurrenceRow("sample", "youtube", previous_week_end),
        OccurrenceRow("sample", "youtube", week_start),
        OccurrenceRow("sample", "naver_news", week_start + timedelta(days=2)),
        OccurrenceRow("sample", "youtube", base),
    ]

    scores = calculate_scores(
        rows=rows,
        week_start=week_start,
        week_end=base,
        previous_week_start=previous_week_start,
        previous_week_end=previous_week_end,
    )

    assert scores[0].weekly_mentions == 3
    assert scores[0].previous_weekly_mentions == 2
    assert scores[0].active_days == 3
    assert scores[0].source_count == 2
    assert scores[0].growth_rate == pytest.approx(0.5)
    assert scores[0].peak_day_share == pytest.approx(1 / 3, abs=0.0001)
    assert scores[0].persistence_score == pytest.approx(42.86)
    assert 0 <= scores[0].final_score <= 100


def test_recalculate_does_not_duplicate_weekly_trends(client, db_session) -> None:
    _collect_and_extract(client)
    first = client.post("/api/trends/recalculate").json()
    count_after_first = count_weekly_trends_for_week(
        db_session,
        date.fromisoformat(first["week_start"]),
    )

    second = client.post("/api/trends/recalculate").json()
    count_after_second = count_weekly_trends_for_week(
        db_session,
        date.fromisoformat(second["week_start"]),
    )

    assert count_after_first == count_after_second


def test_mock_keyword_statuses(client, db_session) -> None:
    _collect_and_extract(client)
    client.post("/api/trends/recalculate")

    assert _get_trend(db_session, "거제야호").status == "weekly_trend"
    assert _get_trend(db_session, "폭싹속았수다촬영지").status == "weekly_trend"
    assert _get_trend(db_session, "두바이초콜릿챌린지").status == "watchlist"
    assert _get_trend(db_session, "제주도여행").status == "stable"


def test_weekly_trends_api(client) -> None:
    _collect_extract_and_recalculate(client)

    response = client.get("/api/trends/weekly")

    body = response.json()
    assert response.status_code == 200
    assert body["week_start"] == "2026-07-22"
    assert body["week_end"] == "2026-07-28"
    assert body["total"] > 0
    assert all(item["status"] == "weekly_trend" for item in body["items"])
    assert "거제야호" in {item["keyword"] for item in body["items"]}
    assert "폭싹속았수다촬영지" in {item["keyword"] for item in body["items"]}


def test_watchlist_api(client) -> None:
    _collect_extract_and_recalculate(client)

    response = client.get("/api/trends/watchlist")

    body = response.json()
    assert response.status_code == 200
    assert body["total"] > 0
    assert all(item["status"] == "watchlist" for item in body["items"])
    assert "두바이초콜릿챌린지" in {item["keyword"] for item in body["items"]}


def test_summary_api(client) -> None:
    _collect_extract_and_recalculate(client)

    response = client.get("/api/trends/summary")

    body = response.json()
    assert response.status_code == 200
    assert body["week_start"] == "2026-07-22"
    assert body["week_end"] == "2026-07-28"
    assert body["weekly_trend_count"] > 0
    assert body["watchlist_count"] > 0
    assert body["stable_count"] > 0
    assert body["top_weekly_trend"] is not None
    assert body["top_watchlist"] is not None


def test_summary_api_before_recalculation_returns_message(client) -> None:
    response = client.get("/api/trends/summary")

    assert response.status_code == 404
    assert "POST /api/trends/recalculate" in response.json()["detail"]


def test_weekly_api_orders_by_final_score_desc(client) -> None:
    _collect_extract_and_recalculate(client)

    response = client.get("/api/trends/weekly?limit=100")
    scores = [item["final_score"] for item in response.json()["items"]]

    assert response.status_code == 200
    assert scores == sorted(scores, reverse=True)


def test_watchlist_api_orders_by_mentions_peak_share_and_score(client) -> None:
    _collect_extract_and_recalculate(client)

    response = client.get("/api/trends/watchlist?limit=100")
    items = response.json()["items"]

    assert response.status_code == 200
    assert items == sorted(
        items,
        key=lambda item: (
            -item["weekly_mentions"],
            -item["peak_day_share"],
            -item["final_score"],
        ),
    )


def test_limit_is_validated(client) -> None:
    _collect_extract_and_recalculate(client)

    assert client.get("/api/trends/weekly?limit=0").status_code == 422
    assert client.get("/api/trends/watchlist?limit=101").status_code == 422


def test_custom_occurrences_can_be_recalculated(client, db_session) -> None:
    for offset in range(3):
        _add_occurrence(
            db_session,
            keyword="직접입력",
            source="youtube" if offset % 2 == 0 else "naver_news",
            occurred_at=datetime(2026, 7, 28 - offset, 9, 0, 0),
            source_id=f"custom-{offset}",
        )
    db_session.commit()

    response = client.post("/api/trends/recalculate")

    assert response.status_code == 200
    assert _get_trend(db_session, "직접입력").weekly_mentions == 3


def _collect_and_extract(client) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")


def _collect_extract_and_recalculate(client) -> None:
    _collect_and_extract(client)
    client.post("/api/trends/recalculate")


def _get_trend(db_session, keyword: str) -> WeeklyTrend:
    trend = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == keyword))
    assert trend is not None
    return trend


def _add_occurrence(
    db_session,
    *,
    keyword: str,
    source: str,
    occurred_at: datetime,
    source_id: str,
) -> None:
    document = SourceDocument(
        source=source,
        source_id=source_id,
        title=keyword,
        text=keyword,
        published_at=occurred_at,
        collected_at=occurred_at,
        views=None,
        likes=None,
        comments=None,
        url=None,
    )
    db_session.add(document)
    db_session.flush()
    db_session.add(
        KeywordOccurrence(
            document_id=document.id,
            keyword=keyword,
            normalized_keyword=keyword,
            source=source,
            occurred_at=occurred_at,
        )
    )
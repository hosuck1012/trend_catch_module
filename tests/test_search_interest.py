from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from app.models.search_interest_observation import SearchInterestObservation
from app.models.search_validation_result import SearchValidationResult
from app.models.weekly_trend import WeeklyTrend
from app.repositories.search_interest_repository import ObservationPoint
from app.services.google_year_in_search_seed_service import (
    GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE,
)
from app.services.search_interest_scoring_service import (
    ProviderScore,
    calculate_provider_score,
    calculate_search_growth_rate,
    combine_provider_scores,
)
from app.services.trend_scoring_service import calculate_final_score


SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"


def _upload_csv(client, endpoint: str, content: bytes, filename: str = "sample.csv"):
    return client.post(
        endpoint,
        files={"file": (filename, content, "text/csv")},
        data={"geo": "KR"},
    )


def _csv(*rows: str) -> bytes:
    return ("keyword,date,value,geo\n" + "\n".join(rows) + "\n").encode("utf-8")


def _prepare_trends(client) -> None:
    assert client.post("/api/collect/mock").status_code == 200
    assert client.post("/api/keywords/extract").status_code == 200
    assert client.post("/api/trends/recalculate").status_code == 200


def _sample_bytes(provider: str) -> bytes:
    return (SAMPLES_DIR / f"{provider}_sample.csv").read_bytes()


def _points(provider: str, previous: list[float], current: list[float]) -> list[ObservationPoint]:
    week_start = date(2026, 7, 22)
    values = []
    for offset, value in enumerate(previous):
        values.append(
            ObservationPoint(provider, "거제야호", week_start - timedelta(days=7 - offset), value)
        )
    for offset, value in enumerate(current):
        values.append(
            ObservationPoint(provider, "거제야호", week_start + timedelta(days=offset), value)
        )
    return values


def _score(value: float, *, current: float = 50, previous: float = 25) -> ProviderScore:
    return ProviderScore(
        current_average=current,
        previous_average=previous,
        growth_rate=1.0,
        growth_score=100.0,
        persistence_score=100.0,
        provider_score=value,
    )


def test_utf8_csv_import(client, db_session) -> None:
    response = _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _csv("거제야호,2026-07-22,55,KR"),
    )

    assert response.status_code == 200
    assert response.json()["inserted_rows"] == 1
    assert db_session.scalar(select(func.count(SearchInterestObservation.id))) == 1


def test_utf8_bom_csv_import(client) -> None:
    content = "\ufeffkeyword,date,value,geo\n거제야호,2026-07-22,55,KR\n".encode("utf-8")

    response = _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        content,
    )

    assert response.status_code == 200
    assert response.json()["inserted_rows"] == 1


def test_missing_required_csv_column_returns_error(client) -> None:
    response = _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        "keyword,date\n거제야호,2026-07-22\n".encode("utf-8"),
    )

    assert response.status_code == 422
    assert "value" in response.json()["detail"]


def test_invalid_date_row_is_skipped(client) -> None:
    response = _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _csv(
            "거제야호,invalid,55,KR",
            "거제야호,2026-07-22,60,KR",
        ),
    )

    body = response.json()
    assert response.status_code == 200
    assert body["received_rows"] == 2
    assert body["inserted_rows"] == 1
    assert body["skipped_rows"] == 1
    assert body["errors"][0]["row"] == 2


def test_interest_value_range_is_validated(client) -> None:
    response = _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _csv(
            "거제야호,2026-07-21,-1,KR",
            "거제야호,2026-07-22,101,KR",
            "거제야호,2026-07-23,50,KR",
        ),
    )

    body = response.json()
    assert response.status_code == 200
    assert body["inserted_rows"] == 1
    assert body["skipped_rows"] == 2
    assert len(body["errors"]) == 2


def test_same_csv_reupload_does_not_duplicate(client, db_session) -> None:
    content = _csv("거제야호,2026-07-22,55,KR")

    first = _upload_csv(client, "/api/search-interest/import/google-trends", content)
    second = _upload_csv(client, "/api/search-interest/import/google-trends", content)

    assert first.json()["inserted_rows"] == 1
    assert second.json()["inserted_rows"] == 0
    assert second.json()["skipped_rows"] == 1
    assert db_session.scalar(select(func.count(SearchInterestObservation.id))) == 1


def test_google_provider_is_saved(client, db_session) -> None:
    _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _csv("거제야호,2026-07-22,55,KR"),
    )

    observation = db_session.scalar(select(SearchInterestObservation))
    assert observation is not None
    assert observation.provider == "google_trends"
    assert observation.source_type == "csv"


def test_google_year_in_search_seed_imports_official_keywords(client, db_session) -> None:
    _prepare_trends(client)

    first = client.post("/api/search-interest/import/google-year-in-search-2025-kr")
    second = client.post("/api/search-interest/import/google-year-in-search-2025-kr")

    body = first.json()
    assert first.status_code == 200
    assert body["provider"] == GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE
    assert body["year"] == 2025
    assert body["geo"] == "KR"
    assert body["categories"] >= 10
    assert body["received_keywords"] > 100
    assert body["inserted_occurrences"] > 0
    assert second.status_code == 200
    assert second.json()["inserted_occurrences"] == 0

    response = client.post("/api/trends/recalculate")
    db_session.expire_all()
    seeded = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == "상하이"))
    assert response.status_code == 200
    assert seeded is not None
    assert seeded.status == "watchlist"
    assert seeded.source_count == 1
    assert seeded.keyword_quality_score == 90


def test_naver_provider_is_saved(client, db_session) -> None:
    _upload_csv(
        client,
        "/api/search-interest/import/naver-datalab",
        _csv("거제야호,2026-07-22,55,KR"),
    )

    observation = db_session.scalar(select(SearchInterestObservation))
    assert observation is not None
    assert observation.provider == "naver_datalab"


def test_manual_observations_are_inserted_and_updated(client, db_session) -> None:
    request = {
        "provider": "google_trends",
        "keyword": "거제 야호",
        "geo": "KR",
        "observations": [
            {"date": "2026-07-21", "value": 12},
            {"date": "2026-07-22", "value": 55},
        ],
    }
    first = client.post("/api/search-interest/manual", json=request)
    request["observations"] = [{"date": "2026-07-22", "value": 65}]
    second = client.post("/api/search-interest/manual", json=request)
    db_session.expire_all()

    updated = db_session.scalar(
        select(SearchInterestObservation).where(
            SearchInterestObservation.observed_date == date(2026, 7, 22)
        )
    )
    assert first.status_code == 200
    assert first.json()["inserted_rows"] == 2
    assert second.json()["updated_rows"] == 1
    assert updated is not None
    assert updated.interest_value == 65
    assert updated.normalized_keyword == "거제야호"
    assert updated.source_type == "manual"


def test_manual_observations_reject_blank_geo(client) -> None:
    response = client.post(
        "/api/search-interest/manual",
        json={
            "provider": "google_trends",
            "keyword": "거제야호",
            "geo": " ",
            "observations": [{"date": "2026-07-22", "value": 55}],
        },
    )

    assert response.status_code == 422
    assert "geo" in response.json()["detail"]


def test_current_average_calculation() -> None:
    score = calculate_provider_score(
        _points("google_trends", [10] * 7, [40, 50, 60, 70, 80, 90, 100]),
        week_start=date(2026, 7, 22),
        week_end=date(2026, 7, 28),
    )

    assert score.current_average == 70


def test_previous_average_calculation() -> None:
    score = calculate_provider_score(
        _points("google_trends", [10, 20, 30, 40, 50, 60, 70], [80] * 7),
        week_start=date(2026, 7, 22),
        week_end=date(2026, 7, 28),
    )

    assert score.previous_average == 40


def test_search_growth_rate_calculation() -> None:
    assert calculate_search_growth_rate(60, 20) == 2
    assert calculate_search_growth_rate(10, 0) == 1
    assert calculate_search_growth_rate(0, 0) == 0


def test_provider_score_calculation() -> None:
    score = calculate_provider_score(
        _points("google_trends", [20] * 7, [60] * 7),
        week_start=date(2026, 7, 22),
        week_end=date(2026, 7, 28),
    )

    assert score.growth_score == 100
    assert score.persistence_score == 100
    assert score.provider_score == 84


def test_two_provider_combined_score() -> None:
    combined = combine_provider_scores(
        {
            "google_trends": _score(80),
            "naver_datalab": _score(70),
        }
    )

    assert combined.combined_score == 75
    assert combined.provider_count == 2
    assert combined.coverage_score == 100


def test_single_provider_combined_score() -> None:
    combined = combine_provider_scores({"google_trends": _score(82.4)})

    assert combined.combined_score == 82.4
    assert combined.provider_count == 1
    assert combined.coverage_score == 50


def test_no_provider_data_keeps_search_score_null(client, db_session) -> None:
    _prepare_trends(client)

    response = client.post("/api/search-interest/recalculate")
    db_session.expire_all()
    trend = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == "거제야호"))
    validation = db_session.scalar(
        select(SearchValidationResult).where(SearchValidationResult.keyword == "거제야호")
    )

    assert response.status_code == 200
    assert trend is not None and trend.search_interest_score is None
    assert trend.search_interest_available is False
    assert validation is not None
    assert validation.combined_score is None
    assert validation.provider_count == 0


def test_weekly_trend_search_interest_score_is_updated(client, db_session) -> None:
    _prepare_trends(client)
    _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _sample_bytes("google_trends"),
    )

    response = client.post("/api/search-interest/recalculate")
    db_session.expire_all()
    trend = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == "거제야호"))

    assert response.status_code == 200
    assert trend is not None
    assert trend.search_interest_score == 78


def test_final_score_is_recalculated(client, db_session) -> None:
    _prepare_trends(client)
    trend = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == "거제야호"))
    assert trend is not None
    before = trend.final_score
    _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _sample_bytes("google_trends"),
    )

    client.post("/api/search-interest/recalculate")
    db_session.refresh(trend)
    expected = round(
        calculate_final_score(
            volume_score=trend.volume_score,
            growth_score=trend.growth_score,
            persistence_score=trend.persistence_score,
            diversity_score=trend.diversity_score,
            search_interest_score=78,
            freshness_score=trend.freshness_score,
            one_day_spike_penalty=trend.one_day_spike_penalty,
            spam_penalty=trend.spam_penalty,
        ),
        2,
    )

    assert trend.final_score == expected
    assert trend.final_score != before


def test_search_interest_status_api(client) -> None:
    _prepare_trends(client)
    _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _sample_bytes("google_trends"),
    )
    _upload_csv(
        client,
        "/api/search-interest/import/naver-datalab",
        _sample_bytes("naver_datalab"),
    )
    client.post("/api/search-interest/recalculate")

    response = client.get("/api/search-interest/status")
    body = response.json()

    assert response.status_code == 200
    assert body["google_trends"]["observations"] == 56
    assert body["google_trends"]["keywords"] == 4
    assert body["naver_datalab"]["observations"] == 56
    assert body["validated_weekly_trends"] == 4
    assert body["unvalidated_weekly_trends"] > 0


def test_search_interest_keyword_detail_api(client) -> None:
    _prepare_trends(client)
    _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _sample_bytes("google_trends"),
    )
    _upload_csv(
        client,
        "/api/search-interest/import/naver-datalab",
        _sample_bytes("naver_datalab"),
    )
    client.post("/api/search-interest/recalculate")

    response = client.get("/api/search-interest/거제야호")
    body = response.json()

    assert response.status_code == 200
    assert body["keyword"] == "거제야호"
    assert body["provider_count"] == 2
    assert body["coverage_score"] == 100
    assert body["combined_score"] == 78.8
    assert client.get("/api/search-interest/없는키워드").status_code == 404


def test_weekly_trend_response_contains_search_fields(client) -> None:
    _prepare_trends(client)
    _upload_csv(
        client,
        "/api/search-interest/import/google-trends",
        _sample_bytes("google_trends"),
    )
    client.post("/api/search-interest/recalculate")

    response = client.get("/api/trends/weekly?limit=100")
    item = next(item for item in response.json()["items"] if item["keyword"] == "거제야호")

    assert item["search_interest_score"] == 78
    assert item["search_provider_count"] == 1
    assert item["search_coverage_score"] == 50
    assert item["google_trends_score"] == 78
    assert item["naver_datalab_score"] is None

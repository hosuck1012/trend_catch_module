from datetime import date, timedelta

from sqlalchemy import func, inspect, select

from app.models.source_document import SourceDocument
from app.database import engine
from app.services.mock_data_service import MOCK_BASE_DATE, collect_mock_data


def test_tables_are_created() -> None:
    table_names = set(inspect(engine).get_table_names())
    assert {
        "source_documents",
        "keyword_occurrences",
        "weekly_trends",
    }.issubset(table_names)


def test_collect_mock_api_success(client) -> None:
    response = client.post("/api/collect/mock")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "inserted_documents": 120,
        "skipped_documents": 0,
        "date_range": {
            "start": "2026-07-15",
            "end": "2026-07-28",
        },
    }


def test_mock_data_is_generated_for_14_day_range(db_session) -> None:
    collect_mock_data(db_session)

    published_at_values = db_session.scalars(select(SourceDocument.published_at)).all()
    published_dates = {published_at.date() for published_at in published_at_values}

    assert len(published_dates) == 14
    assert min(published_dates) == MOCK_BASE_DATE - timedelta(days=13)
    assert max(published_dates) == MOCK_BASE_DATE


def test_collect_mock_api_is_idempotent(client) -> None:
    first_response = client.post("/api/collect/mock")
    second_response = client.post("/api/collect/mock")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["inserted_documents"] == 120
    assert first_response.json()["skipped_documents"] == 0
    assert second_response.json()["inserted_documents"] == 0
    assert second_response.json()["skipped_documents"] == 120


def test_documents_count_api(client) -> None:
    client.post("/api/collect/mock")

    response = client.get("/api/documents/count")

    assert response.status_code == 200
    assert response.json() == {
        "total_documents": 120,
        "youtube": 60,
        "naver_news": 60,
    }


def test_mock_keywords_exist_in_source_documents(db_session) -> None:
    collect_mock_data(db_session)

    for keyword in ("거제야호", "두바이초콜릿챌린지", "제주도여행", "폭싹속았수다촬영지"):
        document_count = db_session.scalar(
            select(func.count(SourceDocument.id)).where(
                SourceDocument.title.contains(keyword),
                SourceDocument.text.contains(keyword),
            )
        )

        assert document_count is not None
        assert document_count > 0


def test_spike_keyword_has_single_peak_day(db_session) -> None:
    collect_mock_data(db_session)

    rows = db_session.execute(
        select(
            func.date(SourceDocument.published_at),
            func.count(SourceDocument.id),
        )
        .where(SourceDocument.title.contains("두바이초콜릿챌린지"))
        .group_by(func.date(SourceDocument.published_at))
    ).all()

    daily_counts = {date.fromisoformat(day): count for day, count in rows}
    assert max(daily_counts.values()) == 18
    assert sum(count for count in daily_counts.values() if count <= 1) == 2

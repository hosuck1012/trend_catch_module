from datetime import datetime

from sqlalchemy import func, select

from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.services.keyword_extraction_service import extract_keywords_from_text
from app.services.keyword_normalization_service import normalize_keyword


REQUIRED_KEYWORDS = ("거제야호", "두바이초콜릿챌린지", "제주도여행", "폭싹속았수다촬영지")


def test_keyword_normalization() -> None:
    assert normalize_keyword("거제 야호") == "거제야호"
    assert normalize_keyword("거제야호!") == "거제야호"
    assert normalize_keyword("#거제야호") == "거제야호"
    assert normalize_keyword("폭싹 속았수다 촬영지") == "폭싹속았수다촬영지"
    assert normalize_keyword("두바이-초콜릿-챌린지") == "두바이초콜릿챌린지"
    assert normalize_keyword("Netflix") == "netflix"
    assert normalize_keyword("!") is None
    assert normalize_keyword("가") is None


def test_hashtag_extraction() -> None:
    extracted = extract_keywords_from_text("#거제야호 오늘 #제주도여행")
    normalized = {keyword.normalized_keyword for keyword in extracted}
    originals = {keyword.keyword for keyword in extracted}

    assert "거제야호" in normalized
    assert "제주도여행" in normalized
    assert "#거제야호" in originals


def test_korean_keyword_and_phrase_extraction() -> None:
    extracted = extract_keywords_from_text("폭싹 속았수다 촬영지 두바이초콜릿챌린지")
    normalized = {keyword.normalized_keyword for keyword in extracted}

    assert "폭싹속았수다촬영지" in normalized
    assert "두바이초콜릿챌린지" in normalized


def test_stop_words_are_removed() -> None:
    extracted = extract_keywords_from_text("오늘 영상 뉴스 관련 이번 인기 화제 거제야호")
    normalized = {keyword.normalized_keyword for keyword in extracted}

    assert "오늘" not in normalized
    assert "영상" not in normalized
    assert "뉴스" not in normalized
    assert "관련" not in normalized
    assert "거제야호" in normalized


def test_numeric_only_keywords_are_removed() -> None:
    extracted = extract_keywords_from_text("2026 12345 KPOP2026")
    normalized = {keyword.normalized_keyword for keyword in extracted}

    assert "2026" not in normalized
    assert "12345" not in normalized
    assert "kpop2026" in normalized


def test_keywords_extract_api_success(client) -> None:
    client.post("/api/collect/mock")

    response = client.post("/api/keywords/extract")

    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["processed_documents"] == 120
    assert body["skipped_documents"] == 0
    assert body["inserted_occurrences"] > 120


def test_keywords_extract_api_is_idempotent(client) -> None:
    client.post("/api/collect/mock")
    first_response = client.post("/api/keywords/extract")
    second_response = client.post("/api/keywords/extract")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["inserted_occurrences"] > 0
    assert second_response.json()["processed_documents"] == 0
    assert second_response.json()["inserted_occurrences"] == 0
    assert second_response.json()["skipped_documents"] == 120


def test_top_keywords_api(client) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")

    response = client.get("/api/keywords/top?limit=5")

    body = response.json()
    assert response.status_code == 200
    assert body["total_occurrences"] > 0
    assert body["unique_keywords"] > 0
    assert len(body["items"]) == 5
    assert body["items"][0]["mentions"] >= body["items"][1]["mentions"]
    assert {"youtube", "naver_news"}.issubset(set(body["items"][0]["sources"]))


def test_keyword_detail_api(client) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")

    response = client.get("/api/keywords/거제야호")

    body = response.json()
    assert response.status_code == 200
    assert body["keyword"] == "거제야호"
    assert body["mentions"] == 30
    assert body["active_days"] == 14
    assert body["source_count"] == 2
    assert body["sources"] == ["naver_news", "youtube"]
    assert body["first_seen"] == "2026-07-15T09:00:00"
    assert body["last_seen"] == "2026-07-28T12:23:00"


def test_unknown_keyword_detail_api_returns_404(client) -> None:
    response = client.get("/api/keywords/없는키워드")

    assert response.status_code == 404


def test_required_keywords_exist_as_occurrences(client, db_session) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")

    for keyword in REQUIRED_KEYWORDS:
        count = db_session.scalar(
            select(func.count(KeywordOccurrence.id)).where(
                KeywordOccurrence.normalized_keyword == keyword
            )
        )
        assert count is not None
        assert count > 0


def test_existing_partial_occurrences_do_not_block_extraction(client, db_session) -> None:
    document = SourceDocument(
        source="youtube",
        source_id="partial-doc",
        title="거제야호 #제주도여행",
        text="폭싹 속았수다 촬영지 두바이초콜릿챌린지",
        published_at=datetime(2026, 7, 28, 12, 0, 0),
        collected_at=datetime(2026, 7, 28, 13, 0, 0),
        views=1000,
        likes=100,
        comments=10,
        url=None,
    )
    db_session.add(document)
    db_session.flush()
    db_session.add(
        KeywordOccurrence(
            document_id=document.id,
            keyword="거제야호",
            normalized_keyword="거제야호",
            source=document.source,
            occurred_at=document.published_at,
        )
    )
    db_session.commit()

    response = client.post("/api/keywords/extract")

    normalized = set(
        db_session.scalars(
            select(KeywordOccurrence.normalized_keyword).where(
                KeywordOccurrence.document_id == document.id
            )
        ).all()
    )
    assert response.status_code == 200
    assert response.json()["processed_documents"] == 1
    assert "거제야호" in normalized
    assert "제주도여행" in normalized
    assert "폭싹속았수다촬영지" in normalized
    assert "두바이초콜릿챌린지" in normalized

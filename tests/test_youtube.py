from datetime import datetime

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.models.source_document import SourceDocument


class MockAsyncClient:
    payload = {"items": []}
    status_code = 200
    exception = None
    calls = []

    def __init__(self, *args, **kwargs):
        self.timeout = kwargs.get("timeout")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url, params):
        self.__class__.calls.append({"url": url, "params": params})
        if self.__class__.exception is not None:
            raise self.__class__.exception
        request = httpx.Request("GET", url)
        return httpx.Response(
            self.__class__.status_code,
            json=self.__class__.payload,
            request=request,
        )


def youtube_payload(*, include_missing_statistics=False):
    first_statistics = {"viewCount": "12345", "likeCount": "678", "commentCount": "90"}
    second_statistics = {} if include_missing_statistics else {
        "viewCount": "200",
        "likeCount": "20",
        "commentCount": "2",
    }
    return {
        "items": [
            {
                "id": "video-1",
                "snippet": {
                    "title": "First Korean travel trend",
                    "description": "A detailed description",
                    "tags": ["Seoul", "Travel", "Korea"],
                    "publishedAt": "2026-07-30T10:15:00Z",
                },
                "statistics": first_statistics,
            },
            {
                "id": "video-2",
                "snippet": {
                    "title": "Second Korean travel trend",
                    "description": "",
                    "publishedAt": "2026-07-29T09:00:00Z",
                },
                "statistics": second_statistics,
            },
        ]
    }


def set_youtube_key(monkeypatch, value="test-key"):
    monkeypatch.setenv("YOUTUBE_API_KEY", value)
    monkeypatch.setenv("YOUTUBE_REGION_CODE", "KR")
    monkeypatch.setenv("YOUTUBE_MAX_RESULTS", "50")
    get_settings.cache_clear()


def install_mock_client(monkeypatch, payload=None, status_code=200, exception=None):
    MockAsyncClient.payload = payload if payload is not None else {"items": []}
    MockAsyncClient.status_code = status_code
    MockAsyncClient.exception = exception
    MockAsyncClient.calls = []
    monkeypatch.setattr("app.collectors.youtube_collector.httpx.AsyncClient", MockAsyncClient)
    return MockAsyncClient


def test_youtube_api_key_missing_does_not_call_external_api(client, monkeypatch) -> None:
    monkeypatch.setenv("YOUTUBE_API_KEY", "")
    get_settings.cache_clear()
    mock_client = install_mock_client(monkeypatch, youtube_payload())

    response = client.post("/api/collect/youtube")

    assert response.status_code == 503
    assert response.json()["detail"] == "YouTube API 키가 설정되지 않았습니다."
    assert mock_client.calls == []


def test_youtube_normal_response_parsing(client, monkeypatch, db_session) -> None:
    set_youtube_key(monkeypatch)
    mock_client = install_mock_client(monkeypatch, youtube_payload())

    response = client.post("/api/collect/youtube?region_code=KR&max_results=2")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["source"] == "youtube"
    assert body["region_code"] == "KR"
    assert body["requested_results"] == 2
    assert body["received_results"] == 2
    assert body["inserted_documents"] == 2
    assert body["skipped_documents"] == 0
    assert mock_client.calls[0]["params"]["part"] == "snippet,statistics"
    assert mock_client.calls[0]["params"]["chart"] == "mostPopular"
    assert mock_client.calls[0]["params"]["regionCode"] == "KR"
    assert mock_client.calls[0]["params"]["maxResults"] == 2

    documents = db_session.scalars(select(SourceDocument).order_by(SourceDocument.source_id)).all()
    assert len(documents) == 2
    assert documents[0].url == "https://www.youtube.com/watch?v=video-1"


def test_youtube_title_description_and_tags_are_saved(client, monkeypatch, db_session) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())

    response = client.post("/api/collect/youtube?max_results=2")

    document = db_session.scalar(select(SourceDocument).where(SourceDocument.source_id == "video-1"))
    assert response.status_code == 200
    assert document is not None
    assert document.title == "First Korean travel trend"
    assert "A detailed description" in document.text
    assert "Seoul Travel Korea" in document.text
    assert document.published_at == datetime(2026, 7, 30, 10, 15, 0)


def test_youtube_statistics_are_converted_to_integers(client, monkeypatch, db_session) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())

    response = client.post("/api/collect/youtube?max_results=2")

    document = db_session.scalar(select(SourceDocument).where(SourceDocument.source_id == "video-1"))
    assert response.status_code == 200
    assert document is not None
    assert document.views == 12345
    assert document.likes == 678
    assert document.comments == 90


def test_youtube_missing_statistics_are_handled(client, monkeypatch, db_session) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload(include_missing_statistics=True))

    response = client.post("/api/collect/youtube?max_results=2")

    document = db_session.scalar(select(SourceDocument).where(SourceDocument.source_id == "video-2"))
    assert response.status_code == 200
    assert document is not None
    assert document.views is None
    assert document.likes is None
    assert document.comments is None


def test_youtube_duplicate_collection_is_skipped(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())

    first_response = client.post("/api/collect/youtube?max_results=2")
    second_response = client.post("/api/collect/youtube?max_results=2")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["inserted_documents"] == 2
    assert first_response.json()["skipped_documents"] == 0
    assert second_response.json()["inserted_documents"] == 0
    assert second_response.json()["skipped_documents"] == 2


def test_youtube_region_code_validation(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())

    response = client.post("/api/collect/youtube?region_code=kr&max_results=2")

    assert response.status_code == 422


def test_youtube_max_results_validation(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())

    low_response = client.post("/api/collect/youtube?max_results=0")
    high_response = client.post("/api/collect/youtube?max_results=51")

    assert low_response.status_code == 422
    assert high_response.status_code == 422


def test_youtube_api_400_response(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, {"error": {"message": "bad request"}}, status_code=400)

    response = client.post("/api/collect/youtube?max_results=2")

    assert response.status_code == 502
    assert response.json()["detail"] == "YouTube API 키, API 활성화 상태 또는 quota를 확인하세요."


def test_youtube_api_403_response(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, {"error": {"message": "forbidden"}}, status_code=403)

    response = client.post("/api/collect/youtube?max_results=2")

    assert response.status_code == 502
    assert response.json()["detail"] == "YouTube API 키, API 활성화 상태 또는 quota를 확인하세요."


def test_youtube_timeout_response(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, exception=httpx.TimeoutException("timeout"))

    response = client.post("/api/collect/youtube?max_results=2")

    assert response.status_code == 503
    assert response.json()["detail"] == "YouTube API 요청 시간이 초과되었습니다."


def test_youtube_collect_api_response(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())

    response = client.post("/api/collect/youtube?region_code=KR&max_results=2")

    assert response.status_code == 200
    assert set(response.json()) == {
        "status",
        "source",
        "region_code",
        "requested_results",
        "received_results",
        "inserted_documents",
        "skipped_documents",
        "collected_at",
    }


def test_recent_youtube_documents_api(client, monkeypatch) -> None:
    set_youtube_key(monkeypatch)
    install_mock_client(monkeypatch, youtube_payload())
    client.post("/api/collect/youtube?max_results=2")

    response = client.get("/api/documents/recent?source=youtube&limit=1")

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["source"] == "youtube"
    assert body["items"][0]["source_id"] == "video-1"
    assert body["items"][0]["views"] == 12345

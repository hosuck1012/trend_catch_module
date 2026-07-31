from datetime import datetime

import httpx
from sqlalchemy import select

from app.collectors.newsis_rss_collector import NEWSIS_RSS_FEED_URLS, NEWSIS_RSS_USER_AGENT
from app.config import get_settings
from app.models.source_document import SourceDocument
from app.services.newsis_rss_collection_service import (
    clean_rss_text,
    make_newsis_source_id,
    parse_newsis_published_at,
    prepare_newsis_rss_document,
)


class MockNewsisAsyncClient:
    responses = {}
    exception = None
    calls = []
    init_kwargs = []

    def __init__(self, *args, **kwargs):
        self.__class__.init_kwargs.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        self.__class__.calls.append({"url": url})
        if self.__class__.exception is not None:
            raise self.__class__.exception
        response = self.__class__.responses[url]
        if isinstance(response, Exception):
            raise response
        status_code, content = response
        return httpx.Response(
            status_code,
            content=content.encode("utf-8"),
            request=httpx.Request("GET", url),
        )


def install_mock_client(monkeypatch, responses=None, exception=None):
    MockNewsisAsyncClient.responses = responses or {}
    MockNewsisAsyncClient.exception = exception
    MockNewsisAsyncClient.calls = []
    MockNewsisAsyncClient.init_kwargs = []
    monkeypatch.setattr(
        "app.collectors.newsis_rss_collector.httpx.AsyncClient",
        MockNewsisAsyncClient,
    )
    return MockNewsisAsyncClient


def set_newsis_defaults(monkeypatch, feeds="culture", timeout="15"):
    monkeypatch.setenv("NEWSIS_RSS_FEEDS", feeds)
    monkeypatch.setenv("NEWSIS_RSS_TIMEOUT_SECONDS", timeout)
    get_settings.cache_clear()


def rss_xml(items):
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Newsis Test</title>
    {''.join(items)}
  </channel>
</rss>"""


def rss_item(
    *,
    title="뉴시스 제목",
    link="https://www.newsis.com/view/?id=NISX20260731_0001",
    guid="NISX20260731_0001",
    description="본문 요약",
    pub_date="Fri, 31 Jul 2026 09:30:00 +0900",
):
    guid_xml = f"<guid>{guid}</guid>" if guid is not None else ""
    description_xml = f"<description>{description}</description>" if description is not None else ""
    pub_date_xml = f"<pubDate>{pub_date}</pubDate>" if pub_date is not None else ""
    return f"""
<item>
  <title>{title}</title>
  <link>{link}</link>
  {guid_xml}
  {description_xml}
  {pub_date_xml}
</item>"""


def response_map(**feed_xml):
    return {
        NEWSIS_RSS_FEED_URLS[feed]: (200, xml)
        for feed, xml in feed_xml.items()
    }


def test_normal_rss_xml_parsing(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    mock_client = install_mock_client(
        monkeypatch,
        response_map(culture=rss_xml([rss_item()])),
    )

    response = client.post("/api/collect/newsis-rss")

    assert response.status_code == 200
    assert response.json()["received_items"] == 1
    assert mock_client.calls[0]["url"] == NEWSIS_RSS_FEED_URLS["culture"]
    assert mock_client.init_kwargs[0]["headers"]["User-Agent"] == NEWSIS_RSS_USER_AGENT


def test_title_is_saved(client, monkeypatch, db_session) -> None:
    set_newsis_defaults(monkeypatch)
    install_mock_client(
        monkeypatch,
        response_map(culture=rss_xml([rss_item(title="문화 RSS 제목")])),
    )

    response = client.post("/api/collect/newsis-rss")

    document = db_session.scalar(select(SourceDocument).where(SourceDocument.source == "newsis_rss"))
    assert response.status_code == 200
    assert document is not None
    assert document.title == "문화 RSS 제목"


def test_description_or_summary_is_saved() -> None:
    collected_at = datetime(2026, 7, 31, 1, 0, 0)

    description_doc = prepare_newsis_rss_document(
        {
            "title": "제목",
            "link": "https://www.newsis.com/view/?id=desc",
            "guid": "desc",
            "description": "description text",
            "published": "Fri, 31 Jul 2026 09:30:00 +0900",
        },
        collected_at,
    )
    summary_doc = prepare_newsis_rss_document(
        {
            "title": "제목",
            "link": "https://www.newsis.com/view/?id=summary",
            "guid": "summary",
            "summary": "summary text",
            "published": "Fri, 31 Jul 2026 09:30:00 +0900",
        },
        collected_at,
    )

    assert description_doc is not None
    assert summary_doc is not None
    assert description_doc.text == "description text"
    assert summary_doc.text == "summary text"


def test_html_tags_are_removed() -> None:
    assert clean_rss_text("<p>첫 문장</p><br><strong>둘째</strong>") == "첫 문장 둘째"


def test_html_entities_are_restored() -> None:
    assert clean_rss_text("A&amp;B &lt;테스트&gt;") == "A&B <테스트>"


def test_published_date_is_converted() -> None:
    published_at, was_missing = parse_newsis_published_at(
        {"published": "Fri, 31 Jul 2026 09:30:00 +0900"},
        datetime(2026, 7, 31, 1, 0, 0),
    )

    assert published_at == datetime(2026, 7, 31, 0, 30, 0)
    assert was_missing is False


def test_missing_date_uses_safe_fallback() -> None:
    collected_at = datetime(2026, 7, 31, 1, 0, 0)

    published_at, was_missing = parse_newsis_published_at({}, collected_at)

    assert published_at == collected_at
    assert was_missing is True


def test_invalid_date_uses_safe_fallback() -> None:
    collected_at = datetime(2026, 7, 31, 1, 0, 0)

    published_at, was_missing = parse_newsis_published_at(
        {"published": "not-a-date"},
        collected_at,
    )

    assert published_at == collected_at
    assert was_missing is True


def test_guid_based_source_id_generation() -> None:
    document = prepare_newsis_rss_document(
        {
            "title": "제목",
            "link": "https://www.newsis.com/view/?id=link",
            "guid": "guid-value",
        },
        datetime(2026, 7, 31, 1, 0, 0),
    )

    assert document is not None
    assert document.source_id == make_newsis_source_id("guid-value")


def test_link_based_source_id_generation_when_guid_missing() -> None:
    link = "https://www.newsis.com/view/?id=link-only"

    document = prepare_newsis_rss_document(
        {"title": "제목", "link": link},
        datetime(2026, 7, 31, 1, 0, 0),
    )

    assert document is not None
    assert document.source_id == make_newsis_source_id(link)


def test_entries_without_title_or_link_are_skipped(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    missing_title = rss_item(title="", link="https://www.newsis.com/view/?id=missing-title")
    missing_link = rss_item(title="제목", link="")
    install_mock_client(
        monkeypatch,
        response_map(culture=rss_xml([missing_title, missing_link])),
    )

    response = client.post("/api/collect/newsis-rss")

    assert response.status_code == 200
    assert response.json()["inserted_documents"] == 0
    assert response.json()["skipped_documents"] == 2


def test_duplicate_article_across_multiple_feeds_is_skipped(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch, feeds="culture,entertain")
    duplicate = rss_item(
        title="중복 기사",
        link="https://www.newsis.com/view/?id=duplicate",
        guid="duplicate-guid",
    )
    install_mock_client(
        monkeypatch,
        response_map(
            culture=rss_xml([duplicate]),
            entertain=rss_xml([duplicate]),
        ),
    )

    response = client.post("/api/collect/newsis-rss")

    assert response.status_code == 200
    assert response.json()["received_items"] == 2
    assert response.json()["inserted_documents"] == 1
    assert response.json()["skipped_documents"] == 1


def test_duplicate_collection_is_skipped(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    install_mock_client(
        monkeypatch,
        response_map(culture=rss_xml([rss_item(guid="repeat-guid")])),
    )

    first_response = client.post("/api/collect/newsis-rss")
    second_response = client.post("/api/collect/newsis-rss")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json()["inserted_documents"] == 1
    assert second_response.json()["inserted_documents"] == 0
    assert second_response.json()["skipped_documents"] == 1


def test_partial_feed_failure_is_reported(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch, feeds="culture,entertain")
    install_mock_client(
        monkeypatch,
        {
            NEWSIS_RSS_FEED_URLS["culture"]: (200, rss_xml([rss_item()])),
            NEWSIS_RSS_FEED_URLS["entertain"]: (500, "server error"),
        },
    )

    response = client.post("/api/collect/newsis-rss")

    body = response.json()
    assert response.status_code == 200
    assert body["inserted_documents"] == 1
    assert body["failed_feeds"][0]["feed"] == "entertain"


def test_all_feed_failures_return_error(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch, feeds="culture,entertain")
    install_mock_client(
        monkeypatch,
        {
            NEWSIS_RSS_FEED_URLS["culture"]: (500, "server error"),
            NEWSIS_RSS_FEED_URLS["entertain"]: (500, "server error"),
        },
    )

    response = client.post("/api/collect/newsis-rss")

    assert response.status_code == 502
    assert response.json()["detail"]["message"] == "모든 뉴시스 RSS 피드 수집에 실패했습니다."


def test_timeout_is_reported(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    install_mock_client(monkeypatch, exception=httpx.TimeoutException("timeout"))

    response = client.post("/api/collect/newsis-rss")

    assert response.status_code == 502
    assert "초과" in response.json()["detail"]["failed_feeds"][0]["error"]


def test_invalid_xml_is_reported(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    install_mock_client(
        monkeypatch,
        response_map(culture="this is not xml"),
    )

    response = client.post("/api/collect/newsis-rss")

    assert response.status_code == 502
    assert "XML" in response.json()["detail"]["failed_feeds"][0]["error"]


def test_disallowed_feed_name_is_validated(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    mock_client = install_mock_client(monkeypatch, response_map(culture=rss_xml([rss_item()])))

    response = client.post("/api/collect/newsis-rss?feeds=https://example.com/rss.xml")

    assert response.status_code == 422
    assert mock_client.calls == []


def test_collect_newsis_rss_api_response(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    install_mock_client(
        monkeypatch,
        response_map(culture=rss_xml([rss_item()])),
    )

    response = client.post("/api/collect/newsis-rss?feeds=culture")

    assert response.status_code == 200
    assert response.json()["source"] == "newsis_rss"
    assert response.json()["feeds"] == ["culture"]
    assert response.json()["requested_feeds"] == 1


def test_recent_newsis_rss_documents_api(client, monkeypatch) -> None:
    set_newsis_defaults(monkeypatch)
    install_mock_client(
        monkeypatch,
        response_map(
            culture=rss_xml(
                [
                    rss_item(
                        title="오래된 기사",
                        link="https://www.newsis.com/view/?id=old",
                        guid="old-guid",
                        pub_date="Thu, 30 Jul 2026 09:30:00 +0900",
                    ),
                    rss_item(
                        title="최신 기사",
                        link="https://www.newsis.com/view/?id=new",
                        guid="new-guid",
                        pub_date="Fri, 31 Jul 2026 09:30:00 +0900",
                    ),
                ]
            )
        ),
    )
    client.post("/api/collect/newsis-rss")

    response = client.get("/api/documents/recent?source=newsis_rss&limit=1")

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["source"] == "newsis_rss"
    assert body["items"][0]["title"] == "최신 기사"
    assert body["items"][0]["url"] == "https://www.newsis.com/view/?id=new"

import asyncio
from datetime import date, datetime

import httpx
import pytest
from sqlalchemy import func, select

from app.api.context import get_wikipedia_client
from app.config import get_settings
from app.context.context_matcher import (
    ScoredCandidate,
    generate_search_queries,
    rank_candidates,
    score_candidate,
)
from app.context.context_normalizer import clean_plain_text
from app.context.wikipedia_client import (
    WIKIMEDIA_CONTACT_REQUIRED_MESSAGE,
    WIKIMEDIA_POLICY_REJECTION_MESSAGE,
    WikimediaConfigurationError,
    WikimediaPolicyError,
    WikipediaClient,
    WikipediaClientError,
    WikipediaPageSummary,
    WikipediaSearchResult,
    build_wikimedia_user_agent,
)
from app.main import app
from app.models.entity_context import EntityContext
from app.models.trend_context_link import TrendContextLink
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.repositories.context_repository import (
    ContextTarget,
    EntityContextValues,
    recalculate_primary_context,
    upsert_entity_context,
)
from app.services.context_enrichment_service import (
    clear_wikipedia_search_cache,
    enrich_contexts,
)
from app.services.trend_context_link_service import (
    calculate_context_score,
    link_context_to_trend,
)


WEEK_START = date(2026, 7, 25)
WEEK_END = date(2026, 7, 31)
TEST_CONTACT_URL = "https://example.test/trend-catch-module"
TEST_USER_AGENT = f"TrendCatchModule/0.1 ({TEST_CONTACT_URL})"


@pytest.fixture(autouse=True)
def clean_wikipedia_search_cache() -> None:
    clear_wikipedia_search_cache()
    yield
    clear_wikipedia_search_cache()


@pytest.fixture(autouse=True)
def configured_wikimedia_contact(monkeypatch) -> None:
    monkeypatch.setenv("WIKIMEDIA_CLIENT_NAME", "TrendCatchModule")
    monkeypatch.setenv("WIKIMEDIA_CLIENT_VERSION", "0.1")
    monkeypatch.setenv("WIKIMEDIA_CONTACT_URL", TEST_CONTACT_URL)
    monkeypatch.delenv("WIKIMEDIA_CONTACT_EMAIL", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_contact_url_builds_wikimedia_user_agent() -> None:
    user_agent = build_wikimedia_user_agent(
        client_name="TrendCatchModule",
        client_version="0.1",
        contact_url="https://example.test/project",
        contact_email="",
    )

    assert user_agent == "TrendCatchModule/0.1 (https://example.test/project)"


def test_contact_email_builds_wikimedia_user_agent() -> None:
    user_agent = build_wikimedia_user_agent(
        client_name="TrendCatchModule",
        client_version="0.1",
        contact_url="",
        contact_email="maintainer@example.test",
    )

    assert user_agent == "TrendCatchModule/0.1 (maintainer@example.test)"


def test_contact_url_takes_priority_over_email() -> None:
    user_agent = build_wikimedia_user_agent(
        client_name="TrendCatchModule",
        client_version="0.1",
        contact_url="https://example.test/project",
        contact_email="maintainer@example.test",
    )

    assert user_agent.endswith("(https://example.test/project)")
    assert "maintainer@" not in user_agent


@pytest.mark.parametrize(
    ("contact_url", "contact_email", "expected_setting"),
    [
        ("ftp://example.test/project", "", "WIKIMEDIA_CONTACT_URL"),
        ("", "invalid-email", "WIKIMEDIA_CONTACT_EMAIL"),
    ],
)
def test_invalid_wikimedia_contact_is_rejected(
    contact_url,
    contact_email,
    expected_setting,
) -> None:
    with pytest.raises(WikimediaConfigurationError) as exc_info:
        build_wikimedia_user_agent(
            client_name="TrendCatchModule",
            client_version="0.1",
            contact_url=contact_url,
            contact_email=contact_email,
        )

    assert expected_setting in str(exc_info.value)


def test_missing_contact_blocks_request(monkeypatch) -> None:
    monkeypatch.delenv("WIKIMEDIA_CONTACT_URL", raising=False)
    monkeypatch.delenv("WIKIMEDIA_CONTACT_EMAIL", raising=False)
    get_settings.cache_clear()
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"query": {"search": []}})

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        with pytest.raises(WikimediaConfigurationError) as exc_info:
            await client.search("거제시")
        await async_client.aclose()
        return str(exc_info.value)

    message = asyncio.run(run())

    assert message == WIKIMEDIA_CONTACT_REQUIRED_MESSAGE
    assert calls == 0


def test_wikipedia_search_request_parameters_and_korean_parsing() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = dict(request.url.params)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            json={
                "query": {
                    "search": [
                        {
                            "pageid": 123,
                            "title": "거제시",
                            "snippet": "<span>경상남도의 도시</span>",
                        }
                    ]
                }
            },
        )

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        result = await client.search("거제시")
        await async_client.aclose()
        return result

    results = asyncio.run(run())

    assert captured["params"] == {
        "action": "query",
        "list": "search",
        "srsearch": "거제시",
        "srlimit": "5",
        "format": "json",
        "formatversion": "2",
        "utf8": "1",
    }
    assert captured["headers"]["user-agent"] == TEST_USER_AGENT
    assert captured["headers"]["api-user-agent"] == TEST_USER_AGENT
    assert captured["headers"]["accept"] == "application/json"
    assert captured["headers"]["accept-language"] == "ko-KR,ko;q=0.9,en;q=0.5"
    assert captured["headers"]["user-agent"] != "python-httpx"
    assert results[0].title == "거제시"
    assert results[0].snippet == "경상남도의 도시"


def test_wikipedia_page_extract_and_redirect_parsing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["prop"] == "extracts|info|pageprops"
        assert request.url.params["redirects"] == "1"
        return httpx.Response(
            200,
            json={
                "query": {
                    "redirects": [{"from": "제주도", "to": "제주특별자치도"}],
                    "pages": [
                        {
                            "pageid": 10,
                            "title": "제주특별자치도",
                            "fullurl": "https://ko.wikipedia.org/wiki/제주특별자치도",
                            "extract": "<b>제주</b>는 대한민국의 섬이다.[1]",
                            "lastrevid": 99,
                            "pageprops": {"wikibase-shortdesc": "대한민국의 특별자치도"},
                        }
                    ],
                }
            },
        )

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        result = await client.get_page_summary("제주도")
        await async_client.aclose()
        return result

    result = asyncio.run(run())

    assert result is not None
    assert result.title == "제주특별자치도"
    assert result.extract == "제주는 대한민국의 섬이다."
    assert result.redirect_from == "제주도"
    assert result.revision_id == "99"


def test_wikipedia_summary_max_length(monkeypatch) -> None:
    monkeypatch.setenv("WIKIPEDIA_SUMMARY_MAX_CHARS", "10")
    from app.config import get_settings

    get_settings.cache_clear()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"query": {"pages": [{"title": "긴 문서", "extract": "가" * 30}]}},
        )

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        result = await client.get_page_summary("긴 문서")
        await async_client.aclose()
        return result

    result = asyncio.run(run())
    assert result is not None
    assert len(result.extract) == 10


def test_wikipedia_429_retries_twice() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"query": {"search": []}})

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        await client.search("거제시")
        await async_client.aclose()

    asyncio.run(run())
    assert calls == 3


def test_robot_policy_403_is_converted_without_retry() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            403,
            text="Please respect our robot policy and provide contact information.",
        )

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        with pytest.raises(WikimediaPolicyError) as exc_info:
            await client.search("거제시")
        await async_client.aclose()
        return str(exc_info.value)

    message = asyncio.run(run())

    assert message == WIKIMEDIA_POLICY_REJECTION_MESSAGE
    assert calls == 1


@pytest.mark.parametrize("status_code", [500, 502, 503, 504])
def test_wikipedia_retryable_server_errors_retry_twice(
    status_code,
    monkeypatch,
) -> None:
    calls = 0
    delays = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("app.context.wikipedia_client.asyncio.sleep", fake_sleep)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(status_code, headers={"Retry-After": "0"})
        return httpx.Response(200, json={"query": {"search": []}})

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        await client.search("거제시")
        await async_client.aclose()

    asyncio.run(run())

    assert calls == 3
    assert delays == [0.0, 0.0]


def test_wikipedia_timeout_retries_twice(monkeypatch) -> None:
    calls = 0

    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("app.context.wikipedia_client.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ReadTimeout("timeout", request=request)
        return httpx.Response(200, json={"query": {"search": []}})

    async def run():
        async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = WikipediaClient(async_client)
        await client.search("거제시")
        await async_client.aclose()

    asyncio.run(run())
    assert calls == 3


def test_clean_plain_text_removes_html_script_and_references() -> None:
    text = clean_plain_text("<script>alert(1)</script><b>제주</b> 설명[1]")
    assert text == "제주 설명"


def test_exact_title_match_score_is_matched() -> None:
    candidate = _score(title="거제시", entity="거제시", entity_type="LOCATION")
    assert candidate.match_score >= 0.70
    assert candidate.match_status == "matched"


def test_place_search_queries_include_related_location_and_deduplicate() -> None:
    queries = generate_search_queries(
        entity_text="금오름",
        normalized_entity="금오름",
        entity_type="PLACE",
        related_locations=["제주", "제주"],
    )

    assert queries == ["금오름", "제주 금오름"]
    assert len(queries) <= 3


def test_normalized_title_match_adds_score() -> None:
    candidate = _score(
        title="폭싹속았수다",
        entity="폭싹 속았수다",
        entity_type="CONTENT_TITLE",
    )
    assert candidate.match_score >= 0.20


def test_close_top_candidates_are_ambiguous() -> None:
    rows = [
        ScoredCandidate("1", "A", "https://example.com/a", "", None, 0.80, "matched"),
        ScoredCandidate("2", "B", "https://example.com/b", "", None, 0.77, "matched"),
    ]
    ranked = rank_candidates(rows, matched_threshold=0.70)
    assert ranked[0].match_status == "ambiguous"


def test_low_score_candidate_is_unmatched() -> None:
    candidate = _score(title="전혀 다른 문서", entity="거제시", entity_type="LOCATION")
    assert candidate.match_score < 0.50
    assert candidate.match_status == "unmatched"


def test_matched_only_fetches_summary(db_session) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    fake = FakeWikipediaClient()

    result = asyncio.run(
        enrich_contexts(
            limit=1,
            force=False,
            entity_type=None,
            provider="wikipedia_ko",
            wikipedia_client=fake,
        )
    )

    assert result.matched == 1
    assert fake.summary_calls == ["거제시"]


def test_unmatched_does_not_fetch_summary(db_session) -> None:
    _seed_target(db_session, entity_text="알수없는밈", normalized_entity="알수없는밈", entity_type="MEME")
    fake = FakeWikipediaClient(unmatched=True)

    result = asyncio.run(
        enrich_contexts(
            limit=1,
            force=False,
            entity_type=None,
            provider="wikipedia_ko",
            wikipedia_client=fake,
        )
    )

    assert result.unmatched == 1
    assert fake.summary_calls == []


def test_entity_context_upsert_prevents_duplicate(db_session) -> None:
    values = _context_values()
    first, first_action = upsert_entity_context(db_session, values)
    db_session.commit()
    second, second_action = upsert_entity_context(db_session, values)
    db_session.commit()

    assert first.id == second.id
    assert first_action == "created"
    assert second_action == "updated"
    assert db_session.scalar(select(func.count(EntityContext.id))) == 1


def test_force_false_uses_cache_and_force_true_requeries(db_session) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    fake = FakeWikipediaClient()

    first = asyncio.run(_enrich(fake, force=False))
    second = asyncio.run(_enrich(fake, force=False))
    third = asyncio.run(_enrich(fake, force=True))

    assert first.processed_entities == 1
    assert second.skipped_entities == 1
    assert third.processed_entities == 1
    assert fake.search_calls == 2


def test_partial_wikipedia_failure_continues(db_session) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    _add_entity_link(
        db_session,
        keyword="거제야호",
        entity_text="실패객체",
        normalized_entity="실패객체",
        entity_type="PLACE",
        relation_score=70,
    )
    fake = FakeWikipediaClient(fail_entity="실패객체")

    result = asyncio.run(
        enrich_contexts(
            limit=2,
            force=True,
            entity_type=None,
            provider="wikipedia_ko",
            wikipedia_client=fake,
        )
    )

    assert result.status == "partial_success"
    assert result.errors == 1
    assert result.matched == 1


def test_context_score_formula() -> None:
    score = calculate_context_score(
        relation_score=80,
        match_score=0.9,
        provider="wikipedia_ko",
        match_status="matched",
        entity_type="PLACE",
    )
    assert score == 87.5


def test_primary_context_excludes_ambiguous(db_session) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    target = ContextTarget(
        "거제야호", WEEK_START, WEEK_END, "거제시", "거제시", "LOCATION", 80, ()
    )
    now = datetime.now()
    matched = _add_context(db_session, status="matched", url="https://example.com/matched")
    ambiguous = _add_context(
        db_session,
        status="ambiguous",
        url="https://example.com/ambiguous",
        score=0.99,
    )
    link_context_to_trend(db_session, target=target, context=matched, now=now)
    link_context_to_trend(db_session, target=target, context=ambiguous, now=now)
    recalculate_primary_context(db_session, keyword="거제야호", week_start=WEEK_START)
    db_session.commit()

    primary = db_session.scalar(
        select(TrendContextLink).where(TrendContextLink.is_primary.is_(True))
    )
    assert primary is not None
    assert primary.entity_context_id == matched.id


def test_context_enrich_and_candidates_apis(client, db_session) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    fake = FakeWikipediaClient()
    _override_wikipedia(fake)
    try:
        candidate_response = client.get("/api/context/candidates/거제시")
        enrich_response = client.post("/api/context/enrich?limit=1")
    finally:
        app.dependency_overrides.pop(get_wikipedia_client, None)

    assert candidate_response.status_code == 200
    assert candidate_response.json()["candidates"][0]["page_title"] == "거제시"
    assert enrich_response.status_code == 200
    assert enrich_response.json()["matched"] == 1
    assert fake.search_calls == 1


def test_wikipedia_disabled_candidates_response(client, monkeypatch) -> None:
    monkeypatch.setenv("WIKIPEDIA_ENABLED", "false")
    get_settings.cache_clear()
    fake = FakeWikipediaClient()
    _override_wikipedia(fake)
    try:
        response = client.get("/api/context/candidates/거제시")
    finally:
        app.dependency_overrides.pop(get_wikipedia_client, None)

    assert response.status_code == 200
    assert response.json()["status"] == "disabled"
    assert fake.search_calls == 0


def test_context_apis_report_missing_wikimedia_contact(
    client,
    db_session,
    monkeypatch,
) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    monkeypatch.delenv("WIKIMEDIA_CONTACT_URL", raising=False)
    monkeypatch.delenv("WIKIMEDIA_CONTACT_EMAIL", raising=False)
    get_settings.cache_clear()

    candidate_response = client.get("/api/context/candidates/거제시")
    enrich_response = client.post("/api/context/enrich?limit=1")

    assert candidate_response.status_code == 503
    assert candidate_response.json()["detail"] == WIKIMEDIA_CONTACT_REQUIRED_MESSAGE
    assert enrich_response.status_code == 503
    assert enrich_response.json()["detail"] == WIKIMEDIA_CONTACT_REQUIRED_MESSAGE


def test_manual_context_api_cleans_html_and_links_trend(client, db_session) -> None:
    _seed_target(db_session, entity_text="거제야호 밈", normalized_entity="거제야호 밈", entity_type="MEME")
    response = client.post("/api/context/manual", json=_manual_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["context"]["match_status"] == "manual"
    assert body["context"]["summary"] == "사용자 작성 요약"
    assert "원문 전체" in body["guidance"]
    assert db_session.scalar(select(func.count(TrendContextLink.id))) == 1


def test_manual_summary_length_and_url_validation(client, db_session) -> None:
    _seed_target(db_session, entity_text="거제야호 밈", normalized_entity="거제야호 밈", entity_type="MEME")
    too_long = _manual_payload()
    too_long["summary"] = "가" * 1001
    invalid_url = _manual_payload()
    invalid_url["page_url"] = "ftp://example.com/page"

    assert client.post("/api/context/manual", json=too_long).status_code == 422
    assert client.post("/api/context/manual", json=invalid_url).status_code == 422


def test_patch_and_delete_manual_context(client, db_session) -> None:
    _seed_target(db_session, entity_text="거제야호 밈", normalized_entity="거제야호 밈", entity_type="MEME")
    created = client.post("/api/context/manual", json=_manual_payload()).json()
    context_id = created["context"]["id"]

    patched = client.patch(
        f"/api/context/manual/{context_id}",
        json={"summary": "수정된 요약"},
    )
    deleted = client.delete(f"/api/context/manual/{context_id}")

    assert patched.status_code == 200
    assert patched.json()["context"]["summary"] == "수정된 요약"
    assert deleted.status_code == 200
    assert db_session.get(EntityContext, context_id) is None


def test_automatic_wikipedia_context_cannot_be_modified(client, db_session) -> None:
    context = _add_context(db_session, status="matched", url="https://example.com/wiki")

    patch_response = client.patch(
        f"/api/context/manual/{context.id}",
        json={"summary": "수정 시도"},
    )
    delete_response = client.delete(f"/api/context/manual/{context.id}")

    assert patch_response.status_code == 403
    assert delete_response.status_code == 403


def test_context_by_keyword_status_summary_and_existing_extensions(client, db_session) -> None:
    _seed_target(db_session, entity_text="거제시", normalized_entity="거제시")
    fake = FakeWikipediaClient()
    asyncio.run(_enrich(fake, force=False))

    detail = client.get("/api/context/by-keyword/거제야호")
    status = client.get("/api/context/status")
    summary = client.get("/api/context/summary")
    entity_detail = client.get("/api/entities/by-keyword/거제야호")
    weekly = client.get("/api/trends/weekly")

    assert detail.status_code == 200
    assert detail.json()["primary_context"]["page_title"] == "거제시"
    assert status.json()["matched"] == 1
    assert summary.json()["items"][0]["provider"] == "wikipedia_ko"
    assert summary.json()["items"][0]["page_url"]
    assert entity_detail.json()["context_available"] is True
    assert weekly.json()["items"][0]["context_available"] is True


class FakeWikipediaClient:
    def __init__(self, *, unmatched: bool = False, fail_entity: str | None = None) -> None:
        self.unmatched = unmatched
        self.fail_entity = fail_entity
        self.search_calls = 0
        self.summary_calls: list[str] = []

    async def search(self, query: str) -> list[WikipediaSearchResult]:
        self.search_calls += 1
        if self.fail_entity and self.fail_entity in query:
            raise WikipediaClientError("mock failure")
        if self.unmatched:
            return [
                WikipediaSearchResult(
                    "999",
                    "전혀 다른 문서",
                    "https://ko.wikipedia.org/wiki/전혀_다른_문서",
                    "관련 없는 설명",
                    None,
                )
            ]
        title = "거제시" if "거제" in query else query.split()[0]
        return [
            WikipediaSearchResult(
                "123",
                title,
                f"https://ko.wikipedia.org/wiki/{title}",
                f"{title}는 대한민국의 도시 지역이다.",
                None,
            )
        ]

    async def get_page_summary(self, title: str) -> WikipediaPageSummary:
        self.summary_calls.append(title)
        return WikipediaPageSummary(
            "123",
            title,
            f"https://ko.wikipedia.org/wiki/{title}",
            f"{title}에 대한 짧은 요약",
            "대한민국의 도시",
            "456",
            None,
        )

    def search_url(self, query: str) -> str:
        return f"https://ko.wikipedia.org/w/index.php?search={query}"


def _score(*, title: str, entity: str, entity_type: str) -> ScoredCandidate:
    return score_candidate(
        entity_text=entity,
        normalized_entity=entity,
        entity_type=entity_type,
        context_text=entity,
        page_id="1",
        page_title=title,
        page_url="https://example.com/page",
        snippet="",
        redirect_title=None,
    )


def _seed_target(
    session,
    *,
    entity_text: str,
    normalized_entity: str,
    entity_type: str = "LOCATION",
) -> None:
    trend = WeeklyTrend(
        keyword="거제야호",
        week_start=WEEK_START,
        week_end=WEEK_END,
        weekly_mentions=3,
        previous_weekly_mentions=1,
        active_days=2,
        source_count=2,
        growth_rate=2.0,
        peak_day_share=0.5,
        persistence_score=30,
        diversity_score=50,
        freshness_score=100,
        volume_score=40,
        growth_score=100,
        search_interest_score=50,
        one_day_spike_penalty=0,
        spam_penalty=0,
        final_score=60,
        status="weekly_trend",
        calculated_at=datetime.now(),
    )
    session.add(trend)
    session.add(
        TrendEntityLink(
            keyword="거제야호",
            week_start=WEEK_START,
            week_end=WEEK_END,
            entity_text=entity_text,
            normalized_entity=normalized_entity,
            entity_type=entity_type,
            mention_count=2,
            document_count=2,
            source_count=2,
            average_confidence=0.95,
            relation_score=90,
            is_primary=True,
            calculated_at=datetime.now(),
        )
    )
    session.commit()


def _add_entity_link(
    session,
    *,
    keyword: str,
    entity_text: str,
    normalized_entity: str,
    entity_type: str,
    relation_score: float,
) -> None:
    session.add(
        TrendEntityLink(
            keyword=keyword,
            week_start=WEEK_START,
            week_end=WEEK_END,
            entity_text=entity_text,
            normalized_entity=normalized_entity,
            entity_type=entity_type,
            mention_count=1,
            document_count=1,
            source_count=1,
            average_confidence=0.8,
            relation_score=relation_score,
            is_primary=False,
            calculated_at=datetime.now(),
        )
    )
    session.commit()


def _context_values() -> EntityContextValues:
    now = datetime.now()
    return EntityContextValues(
        normalized_entity="거제시",
        entity_text="거제시",
        entity_type="LOCATION",
        provider="wikipedia_ko",
        page_id="123",
        page_title="거제시",
        page_url="https://ko.wikipedia.org/wiki/거제시",
        summary="거제시 요약",
        description="대한민국의 도시",
        match_score=0.9,
        match_status="matched",
        source_language="ko",
        license_name="CC BY-SA",
        attribution_text="한국어 위키백과",
        revision_id="456",
        retrieved_at=now,
        updated_at=now,
    )


def _add_context(
    session,
    *,
    status: str,
    url: str,
    score: float = 0.9,
) -> EntityContext:
    values = _context_values()
    values = EntityContextValues(
        **{
            **values.__dict__,
            "page_url": url,
            "match_status": status,
            "match_score": score,
        }
    )
    context, _ = upsert_entity_context(session, values)
    session.commit()
    return context


def _manual_payload() -> dict[str, object]:
    return {
        "provider": "namuwiki_manual",
        "entity_text": "거제야호 밈",
        "entity_type": "MEME",
        "page_title": "거제야호",
        "page_url": "https://namu.wiki/w/거제야호",
        "summary": "<script>bad()</script><b>사용자 작성 요약</b>",
        "keyword": "거제야호",
        "week_start": WEEK_START.isoformat(),
    }


async def _enrich(fake: FakeWikipediaClient, *, force: bool):
    return await enrich_contexts(
        limit=1,
        force=force,
        entity_type=None,
        provider="wikipedia_ko",
        wikipedia_client=fake,
    )


def _override_wikipedia(fake: FakeWikipediaClient) -> None:
    async def override():
        yield fake

    app.dependency_overrides[get_wikipedia_client] = override

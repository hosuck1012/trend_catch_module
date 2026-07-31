from dataclasses import dataclass
from typing import Any

import feedparser
import httpx


NEWSIS_RSS_FEED_URLS: dict[str, str] = {
    "sokbo": "https://www.newsis.com/RSS/sokbo.xml",
    "culture": "https://www.newsis.com/RSS/culture.xml",
    "entertain": "https://www.newsis.com/RSS/entertain.xml",
    "country": "https://www.newsis.com/RSS/country.xml",
    "society": "https://www.newsis.com/RSS/society.xml",
    "industry": "https://www.newsis.com/RSS/industry.xml",
    "politics": "https://www.newsis.com/RSS/politics.xml",
    "international": "https://www.newsis.com/RSS/international.xml",
    "economy": "https://www.newsis.com/RSS/economy.xml",
    "bank": "https://www.newsis.com/RSS/bank.xml",
    "health": "https://www.newsis.com/RSS/health.xml",
    "met": "https://www.newsis.com/RSS/met.xml",
    "sports": "https://www.newsis.com/RSS/sports.xml",
    "square": "https://www.newsis.com/RSS/square.xml",
}

NEWSIS_RSS_USER_AGENT = "TrendCatchModule/0.1"


class NewsisRssCollectorError(Exception):
    pass


class NewsisRssNetworkError(NewsisRssCollectorError):
    pass


class NewsisRssTimeoutError(NewsisRssNetworkError):
    pass


class NewsisRssHttpError(NewsisRssCollectorError):
    pass


class NewsisRssParseError(NewsisRssCollectorError):
    pass


@dataclass(frozen=True)
class NewsisRssFeedPayload:
    feed: str
    url: str
    entries: list[dict[str, Any]]


def allowed_newsis_rss_feeds() -> tuple[str, ...]:
    return tuple(NEWSIS_RSS_FEED_URLS.keys())


def newsis_rss_url_for_feed(feed: str) -> str:
    return NEWSIS_RSS_FEED_URLS[feed]


async def fetch_newsis_rss_feed(
    feed: str,
    *,
    timeout_seconds: int,
) -> NewsisRssFeedPayload:
    url = newsis_rss_url_for_feed(feed)
    timeout = httpx.Timeout(float(timeout_seconds))
    headers = {"User-Agent": NEWSIS_RSS_USER_AGENT}

    try:
        async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
            response = await client.get(url)
    except httpx.TimeoutException as exc:
        raise NewsisRssTimeoutError(f"{feed} RSS 요청 시간이 초과되었습니다.") from exc
    except httpx.RequestError as exc:
        raise NewsisRssNetworkError(f"{feed} RSS 서버에 연결할 수 없습니다.") from exc

    if response.status_code >= 400:
        raise NewsisRssHttpError(
            f"{feed} RSS 요청이 HTTP {response.status_code} 상태로 실패했습니다."
        )

    parsed = feedparser.parse(response.content)
    if getattr(parsed, "bozo", False):
        bozo_exception = getattr(parsed, "bozo_exception", None)
        raise NewsisRssParseError(f"{feed} RSS XML 형식이 올바르지 않습니다: {bozo_exception}")

    entries = [dict(entry) for entry in getattr(parsed, "entries", [])]
    if not entries:
        raise NewsisRssParseError(f"{feed} RSS entries가 비어 있습니다.")

    return NewsisRssFeedPayload(feed=feed, url=url, entries=entries)

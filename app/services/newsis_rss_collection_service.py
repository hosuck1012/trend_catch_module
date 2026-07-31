import hashlib
import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.collectors.newsis_rss_collector import (
    NewsisRssCollectorError,
    allowed_newsis_rss_feeds,
    fetch_newsis_rss_feed,
)
from app.config import get_settings
from app.models.source_document import SourceDocument


NEWSIS_RSS_SOURCE = "newsis_rss"
HTML_TAG_PATTERN = re.compile(r"</?[A-Za-z][^>]*>")
WHITESPACE_PATTERN = re.compile(r"\s+")


class NewsisRssCollectionInputError(ValueError):
    pass


class NewsisRssAllFeedsFailedError(RuntimeError):
    def __init__(self, failed_feeds: list["NewsisRssFailedFeed"]) -> None:
        self.failed_feeds = failed_feeds
        super().__init__("모든 뉴시스 RSS 피드 수집에 실패했습니다.")


@dataclass(frozen=True)
class NewsisRssFailedFeed:
    feed: str
    error: str


@dataclass(frozen=True)
class NewsisRssPreparedDocument:
    source_id: str
    title: str
    text: str
    published_at: datetime
    published_at_was_missing: bool
    url: str
    guid: str | None


@dataclass(frozen=True)
class NewsisRssCollectionResult:
    source: str
    feeds: list[str]
    requested_feeds: int
    received_items: int
    inserted_documents: int
    skipped_documents: int
    failed_feeds: list[NewsisRssFailedFeed]
    collected_at: datetime


async def collect_newsis_rss_documents(
    session: Session,
    *,
    feeds: str | None = None,
) -> NewsisRssCollectionResult:
    settings = get_settings()
    resolved_feeds = resolve_newsis_rss_feeds(feeds, settings.newsis_rss_feeds)
    timeout_seconds = max(1, settings.newsis_rss_timeout_seconds)
    collected_at = datetime.now(timezone.utc).replace(tzinfo=None)
    received_items = 0
    inserted_documents = 0
    skipped_documents = 0
    failed_feeds: list[NewsisRssFailedFeed] = []
    seen_source_ids: set[str] = set()
    seen_urls: set[str] = set()
    successful_feeds = 0

    for feed in resolved_feeds:
        try:
            payload = await fetch_newsis_rss_feed(feed, timeout_seconds=timeout_seconds)
        except NewsisRssCollectorError as exc:
            failed_feeds.append(NewsisRssFailedFeed(feed=feed, error=str(exc)))
            continue

        successful_feeds += 1
        received_items += len(payload.entries)

        for entry in payload.entries:
            prepared = prepare_newsis_rss_document(entry, collected_at)
            if prepared is None:
                skipped_documents += 1
                continue
            if prepared.source_id in seen_source_ids or prepared.url in seen_urls:
                skipped_documents += 1
                continue
            seen_source_ids.add(prepared.source_id)
            seen_urls.add(prepared.url)
            if _document_exists(session, prepared):
                skipped_documents += 1
                continue

            session.add(_to_source_document(prepared, collected_at))
            session.flush()
            inserted_documents += 1

    if successful_feeds == 0:
        session.rollback()
        raise NewsisRssAllFeedsFailedError(failed_feeds)

    session.commit()
    return NewsisRssCollectionResult(
        source=NEWSIS_RSS_SOURCE,
        feeds=resolved_feeds,
        requested_feeds=len(resolved_feeds),
        received_items=received_items,
        inserted_documents=inserted_documents,
        skipped_documents=skipped_documents,
        failed_feeds=failed_feeds,
        collected_at=collected_at,
    )


def resolve_newsis_rss_feeds(feeds: str | None, default_feeds: str) -> list[str]:
    raw_value = feeds if feeds is not None and feeds.strip() else default_feeds
    resolved: list[str] = []
    seen: set[str] = set()
    for part in raw_value.split(","):
        feed = part.strip().lower()
        if not feed:
            continue
        if feed not in allowed_newsis_rss_feeds():
            raise NewsisRssCollectionInputError(f"허용되지 않은 뉴시스 RSS feed입니다: {feed}")
        if feed in seen:
            continue
        seen.add(feed)
        resolved.append(feed)
    if not resolved:
        raise NewsisRssCollectionInputError("수집할 뉴시스 RSS feed가 없습니다.")
    return resolved


def prepare_newsis_rss_document(
    entry: dict[str, Any],
    collected_at: datetime,
) -> NewsisRssPreparedDocument | None:
    title = clean_rss_text(_entry_value(entry, "title"))
    url = clean_rss_text(_entry_value(entry, "link"))
    if not title or not url:
        return None

    guid = clean_rss_text(
        _entry_value(entry, "guid")
        or _entry_value(entry, "id")
        or _entry_value(entry, "newsis_id")
    )
    description = clean_rss_text(
        _entry_value(entry, "description") or _entry_value(entry, "summary")
    )
    published_at, was_missing = parse_newsis_published_at(entry, collected_at)
    source_id = make_newsis_source_id(guid or url)
    return NewsisRssPreparedDocument(
        source_id=source_id,
        title=title,
        text=description,
        published_at=published_at,
        published_at_was_missing=was_missing,
        url=url,
        guid=guid or None,
    )


def clean_rss_text(value: Any) -> str:
    if value is None:
        return ""
    text = html.unescape(str(value))
    text = HTML_TAG_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def parse_newsis_published_at(
    entry: dict[str, Any],
    collected_at: datetime,
) -> tuple[datetime, bool]:
    published_value = _entry_value(entry, "published") or _entry_value(entry, "pubDate")
    if published_value is None or not str(published_value).strip():
        return collected_at, True

    try:
        parsed = parsedate_to_datetime(str(published_value))
    except (TypeError, ValueError, IndexError, OverflowError):
        return collected_at, True

    if parsed.tzinfo is None:
        return parsed, False
    return parsed.astimezone(timezone.utc).replace(tzinfo=None), False


def make_newsis_source_id(identifier: str) -> str:
    normalized_identifier = identifier.strip()
    return hashlib.sha256(normalized_identifier.encode("utf-8")).hexdigest()


def _entry_value(entry: dict[str, Any], key: str) -> Any:
    value = entry.get(key)
    if value is not None:
        return value
    return entry.get(key.lower())


def _document_exists(session: Session, prepared: NewsisRssPreparedDocument) -> bool:
    existing_id = session.scalar(
        select(SourceDocument.id).where(
            SourceDocument.source == NEWSIS_RSS_SOURCE,
            or_(
                SourceDocument.source_id == prepared.source_id,
                SourceDocument.url == prepared.url,
            ),
        )
    )
    return existing_id is not None


def _to_source_document(
    prepared: NewsisRssPreparedDocument,
    collected_at: datetime,
) -> SourceDocument:
    return SourceDocument(
        source=NEWSIS_RSS_SOURCE,
        source_id=prepared.source_id,
        title=prepared.title,
        text=prepared.text,
        published_at=prepared.published_at,
        collected_at=collected_at,
        views=None,
        likes=None,
        comments=None,
        url=prepared.url,
    )

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.collectors.youtube_collector import fetch_most_popular_videos
from app.config import get_settings
from app.models.source_document import SourceDocument


class YouTubeCollectionInputError(ValueError):
    pass


class YouTubeApiKeyMissingError(RuntimeError):
    pass


@dataclass(frozen=True)
class YouTubeCollectionResult:
    source: str
    region_code: str
    requested_results: int
    received_results: int
    inserted_documents: int
    skipped_documents: int
    collected_at: datetime


async def collect_youtube_popular_videos(
    session: Session,
    *,
    region_code: str | None = None,
    max_results: int | None = None,
) -> YouTubeCollectionResult:
    settings = get_settings()
    resolved_region_code = _resolve_region_code(region_code, settings.youtube_region_code)
    resolved_max_results = _resolve_max_results(max_results, settings.youtube_max_results)
    api_key = settings.youtube_api_key.strip()
    if not api_key:
        raise YouTubeApiKeyMissingError("YouTube API 키가 설정되지 않았습니다.")

    collected_at = datetime.now(timezone.utc).replace(tzinfo=None)
    payload = await fetch_most_popular_videos(
        api_key=api_key,
        region_code=resolved_region_code,
        max_results=resolved_max_results,
    )
    raw_items = payload.get("items")
    items = raw_items if isinstance(raw_items, list) else []

    inserted_documents = 0
    skipped_documents = 0
    seen_source_ids: set[str] = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        source_id = item.get("id")
        if not isinstance(source_id, str) or not source_id.strip():
            continue
        source_id = source_id.strip()
        if source_id in seen_source_ids:
            skipped_documents += 1
            continue
        seen_source_ids.add(source_id)

        existing_id = session.scalar(
            select(SourceDocument.id).where(
                SourceDocument.source == "youtube",
                SourceDocument.source_id == source_id,
            )
        )
        if existing_id is not None:
            skipped_documents += 1
            continue

        document = _source_document_from_item(item, source_id, collected_at)
        session.add(document)
        inserted_documents += 1

    session.commit()
    return YouTubeCollectionResult(
        source="youtube",
        region_code=resolved_region_code,
        requested_results=resolved_max_results,
        received_results=len(items),
        inserted_documents=inserted_documents,
        skipped_documents=skipped_documents,
        collected_at=collected_at,
    )


def _resolve_region_code(region_code: str | None, default_region_code: str) -> str:
    value = (region_code or default_region_code or "KR").strip()
    if not re.fullmatch(r"[A-Z]{2}", value):
        raise YouTubeCollectionInputError("region_code는 영문 대문자 두 글자 형식이어야 합니다.")
    return value


def _resolve_max_results(max_results: int | None, default_max_results: int) -> int:
    value = max_results if max_results is not None else default_max_results
    if value < 1 or value > 50:
        raise YouTubeCollectionInputError("max_results는 1 이상 50 이하이어야 합니다.")
    return value


def _source_document_from_item(
    item: dict[str, Any],
    source_id: str,
    collected_at: datetime,
) -> SourceDocument:
    snippet = item.get("snippet") if isinstance(item.get("snippet"), dict) else {}
    statistics = item.get("statistics") if isinstance(item.get("statistics"), dict) else {}

    title = _safe_string(snippet.get("title")) or "Untitled YouTube video"
    description = _safe_string(snippet.get("description"))
    tags = snippet.get("tags")
    tag_text = " ".join(str(tag) for tag in tags if tag is not None) if isinstance(tags, list) else ""
    text = "\n\n".join(part for part in (description, tag_text) if part).strip()

    return SourceDocument(
        source="youtube",
        source_id=source_id,
        title=title,
        text=text,
        published_at=_parse_youtube_datetime(snippet.get("publishedAt")),
        collected_at=collected_at,
        views=_safe_int(statistics.get("viewCount")),
        likes=_safe_int(statistics.get("likeCount")),
        comments=_safe_int(statistics.get("commentCount")),
        url=f"https://www.youtube.com/watch?v={source_id}",
    )


def _safe_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_youtube_datetime(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        return datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc).replace(tzinfo=None)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)

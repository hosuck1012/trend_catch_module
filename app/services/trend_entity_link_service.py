from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from app.models.entity_mention import EntityMention
from app.repositories.entity_repository import (
    get_keyword_document_ids,
    get_latest_linkable_trends,
    get_mentions_for_documents,
    replace_trend_entity_links,
)
from app.services.keyword_normalization_service import normalize_keyword


@dataclass(frozen=True)
class TrendEntityLinkResult:
    status: str
    week_start: str | None
    week_end: str | None
    processed_keywords: int
    linked_keywords: int
    inserted_links: int
    primary_links: int


def link_trends_to_entities(session: Session) -> TrendEntityLinkResult:
    trends = get_latest_linkable_trends(session)
    if not trends:
        return TrendEntityLinkResult("skipped", None, None, 0, 0, 0, 0)

    inserted_links = 0
    linked_keywords = 0
    primary_links = 0
    calculated_at = datetime.now()
    for trend in trends:
        document_ids = get_keyword_document_ids(
            session,
            keyword=trend.keyword,
            week_start=trend.week_start,
            week_end=trend.week_end,
        )
        mentions = get_mentions_for_documents(session, document_ids)
        grouped = _group_mentions(mentions, trend.keyword)
        links = _calculate_links(
            keyword=trend.keyword,
            week_start=trend.week_start,
            week_end=trend.week_end,
            document_ids=document_ids,
            total_source_count=max(trend.source_count, 1),
            groups=grouped,
            calculated_at=calculated_at,
        )
        replace_trend_entity_links(
            session,
            keyword=trend.keyword,
            week_start=trend.week_start,
            links=links,
        )
        if links:
            linked_keywords += 1
            inserted_links += len(links)
            primary_links += sum(bool(link["is_primary"]) for link in links)

    first = trends[0]
    return TrendEntityLinkResult(
        status="ok",
        week_start=first.week_start.isoformat(),
        week_end=first.week_end.isoformat(),
        processed_keywords=len(trends),
        linked_keywords=linked_keywords,
        inserted_links=inserted_links,
        primary_links=primary_links,
    )


def _group_mentions(
    mentions: list[EntityMention], keyword: str
) -> dict[tuple[str, str], list[EntityMention]]:
    normalized_keyword = normalize_keyword(keyword) or keyword.lower()
    grouped: dict[tuple[str, str], list[EntityMention]] = defaultdict(list)
    for mention in mentions:
        mention_as_keyword = normalize_keyword(mention.normalized_text)
        if mention_as_keyword == normalized_keyword:
            continue
        grouped[(mention.normalized_text, mention.entity_type)].append(mention)
    return grouped


def _calculate_links(
    *,
    keyword: str,
    week_start,
    week_end,
    document_ids: list[int],
    total_source_count: int,
    groups: dict[tuple[str, str], list[EntityMention]],
    calculated_at: datetime,
) -> list[dict[str, object]]:
    if not groups:
        return []
    max_mentions = max(len(items) for items in groups.values())
    total_documents = max(len(set(document_ids)), 1)
    links: list[dict[str, object]] = []
    for (normalized_entity, entity_type), mentions in groups.items():
        mention_count = len(mentions)
        document_count = len({mention.document_id for mention in mentions})
        source_count = len({mention.source for mention in mentions})
        average_confidence = sum(item.confidence for item in mentions) / mention_count
        relation_score = (
            mention_count / max_mentions * 100 * 0.35
            + document_count / total_documents * 100 * 0.25
            + min(source_count / total_source_count, 1.0) * 100 * 0.20
            + average_confidence * 100 * 0.20
        )
        if entity_type in {"LOCATION", "PLACE"}:
            relation_score *= 1.10
        links.append(
            {
                "keyword": keyword,
                "week_start": week_start,
                "week_end": week_end,
                "entity_text": mentions[0].text,
                "normalized_entity": normalized_entity,
                "entity_type": entity_type,
                "mention_count": mention_count,
                "document_count": document_count,
                "source_count": source_count,
                "average_confidence": round(average_confidence, 4),
                "relation_score": round(min(100.0, relation_score), 2),
                "is_primary": False,
                "calculated_at": calculated_at,
            }
        )
    travel_links = [
        link for link in links if link["entity_type"] in {"LOCATION", "PLACE"}
    ]
    primary = max(travel_links or links, key=lambda item: item["relation_score"])
    primary["is_primary"] = True
    return sorted(links, key=lambda item: item["relation_score"], reverse=True)

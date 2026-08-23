from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
import json
from pathlib import Path
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.entity_context import EntityContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.trend_context_link import TrendContextLink
from app.models.weekly_trend import WeeklyTrend
from app.repositories import context_repository
from app.repositories import travel_opportunity_repository
from app.repositories.context_repository import ContextTarget, EntityContextValues
from app.services.keyword_normalization_service import normalize_keyword
from app.services.trend_context_link_service import link_context_to_trend


CATALOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "travel_destination_expansion_catalog.json"
)
DESTINATION_PAGE_ID_PREFIX = "travel-destination:"
DESTINATION_RELATION_TYPE = "content_theme_destination"


@dataclass(frozen=True)
class RelatedDestinationDefinition:
    destination_id: str
    name: str
    region: str
    entity_type: str
    activity: str
    visitability: str
    official_url: str
    source_title: str
    match_confidence: float


@dataclass(frozen=True)
class DestinationExpansionRule:
    keyword: str
    normalized_keyword: str
    required_source: str | None
    source_category: str | None
    theme_code: str
    theme_label: str
    travel_category: str
    keyword_entity_type: str
    theme_confidence: float
    destinations: tuple[RelatedDestinationDefinition, ...]


@dataclass(frozen=True)
class DestinationExpansionCatalog:
    version: str
    rules: tuple[DestinationExpansionRule, ...]


@dataclass(frozen=True)
class RelatedDestinationPreview:
    keyword: str
    theme_code: str
    theme_label: str
    destination_name: str
    region: str
    activity: str
    official_url: str
    relation_score: float


@dataclass(frozen=True)
class RelatedDestinationExpansionResult:
    status: str
    dry_run: bool
    week_start: date | None
    week_end: date | None
    catalog_version: str
    matched_keywords: int
    destinations_matched: int
    contexts_created: int
    contexts_updated: int
    links_created: int
    links_updated: int
    skipped: int
    previews: list[RelatedDestinationPreview]


@lru_cache(maxsize=4)
def load_destination_expansion_catalog(
    path: Path = CATALOG_PATH,
) -> DestinationExpansionCatalog:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    version = str(payload.get("version", "")).strip()
    raw_rules = payload.get("rules")
    if not version or not isinstance(raw_rules, list):
        raise ValueError("Destination expansion catalog requires version and rules")
    rules: list[DestinationExpansionRule] = []
    seen_keywords: set[str] = set()
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            raise ValueError("Destination expansion rules must be objects")
        keyword = str(raw_rule.get("keyword", "")).strip()
        normalized_keyword = normalize_keyword(keyword)
        if not keyword or not normalized_keyword or normalized_keyword in seen_keywords:
            raise ValueError("Destination expansion rule keywords must be unique")
        seen_keywords.add(normalized_keyword)
        destinations = tuple(
            _destination_from_payload(raw_destination)
            for raw_destination in raw_rule.get("destinations", [])
        )
        if not destinations:
            raise ValueError(f"Destination expansion rule has no destinations: {keyword}")
        rules.append(
            DestinationExpansionRule(
                keyword=keyword,
                normalized_keyword=normalized_keyword,
                required_source=_optional_string(raw_rule.get("required_source")),
                source_category=_optional_string(raw_rule.get("source_category")),
                theme_code=str(raw_rule.get("theme_code", "")).strip().upper(),
                theme_label=str(raw_rule.get("theme_label", "")).strip(),
                travel_category=str(raw_rule.get("travel_category", "")).strip().upper(),
                keyword_entity_type=str(
                    raw_rule.get("keyword_entity_type", "CONTENT_TITLE")
                ).strip().upper(),
                theme_confidence=_probability(raw_rule.get("theme_confidence")),
                destinations=destinations,
            )
        )
    return DestinationExpansionCatalog(version=version, rules=tuple(rules))


def expand_related_destinations(
    session: Session,
    *,
    week_start: date | None,
    dry_run: bool,
    force: bool,
    limit: int,
) -> RelatedDestinationExpansionResult:
    resolved_start, resolved_end = travel_opportunity_repository.resolve_week_range(
        session, week_start
    )
    catalog = load_destination_expansion_catalog()
    if resolved_start is None or resolved_end is None:
        return _empty_result(catalog=catalog, dry_run=dry_run)

    trends = list(
        session.scalars(
            select(WeeklyTrend)
            .where(WeeklyTrend.week_start == resolved_start)
            .order_by(
                WeeklyTrend.final_score.desc(),
                WeeklyTrend.keyword_quality_score.desc(),
                WeeklyTrend.keyword.asc(),
            )
        ).all()
    )
    trends_by_keyword = {trend.keyword: trend for trend in trends}
    source_keywords = _source_keywords(
        session,
        rules=catalog.rules,
        week_start=resolved_start,
        week_end=resolved_end,
    )
    matched_rules = [
        rule
        for rule in catalog.rules
        if rule.normalized_keyword in trends_by_keyword
        and _source_category_matches(rule)
        and (
            rule.required_source is None
            or rule.normalized_keyword in source_keywords.get(rule.required_source, set())
        )
    ][:limit]

    contexts_created = 0
    contexts_updated = 0
    links_created = 0
    links_updated = 0
    skipped = 0
    previews: list[RelatedDestinationPreview] = []
    now = travel_opportunity_repository.utc_now()
    for rule in matched_rules:
        trend = trends_by_keyword[rule.normalized_keyword]
        for destination in rule.destinations:
            relation_score = round(
                rule.theme_confidence * destination.match_confidence * 100,
                2,
            )
            previews.append(
                RelatedDestinationPreview(
                    keyword=trend.keyword,
                    theme_code=rule.theme_code,
                    theme_label=rule.theme_label,
                    destination_name=destination.name,
                    region=destination.region,
                    activity=destination.activity,
                    official_url=destination.official_url,
                    relation_score=relation_score,
                )
            )
            page_id = destination_page_id(rule.theme_code, destination.destination_id)
            metadata = {
                "relation_type": DESTINATION_RELATION_TYPE,
                "catalog_version": catalog.version,
                "theme_code": rule.theme_code,
                "theme_label": rule.theme_label,
                "travel_category": rule.travel_category,
                "keyword_entity_type": rule.keyword_entity_type,
                "source_category": rule.source_category,
                "region": destination.region,
                "activity": destination.activity,
                "visitability": destination.visitability,
            }
            values = EntityContextValues(
                normalized_entity=normalize_keyword(destination.name) or destination.name,
                entity_text=destination.name,
                entity_type=destination.entity_type,
                provider="manual",
                page_id=page_id,
                page_title=destination.source_title,
                page_url=destination.official_url,
                summary=(
                    f"{destination.name}은(는) {rule.theme_label} 테마의 검증된 연관 여행지입니다. "
                    f"지역: {destination.region}. 체험: {destination.activity}. "
                    f"방문 조건: {destination.visitability}."
                ),
                description=json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                match_score=destination.match_confidence,
                match_status="manual",
                source_language="ko",
                license_name=None,
                attribution_text="공식 운영기관 또는 공인 협회 자료를 수동 검증",
                revision_id=catalog.version,
                retrieved_at=now,
                updated_at=now,
            )
            existing_context = _existing_context(session, values)
            existing_link = (
                _existing_link(
                    session,
                    keyword=trend.keyword,
                    week_start=resolved_start,
                    context_id=existing_context.id,
                )
                if existing_context is not None
                else None
            )
            context_changed = existing_context is None or force or not _context_matches(
                existing_context, values
            )
            desired_context_score = _desired_context_score(
                relation_score=relation_score,
                match_score=destination.match_confidence,
                entity_type=destination.entity_type,
            )
            link_changed = (
                existing_link is None
                or force
                or existing_link.week_end != resolved_end
                or existing_link.entity_type != destination.entity_type
                or existing_link.normalized_entity != values.normalized_entity
                or abs(existing_link.context_score - desired_context_score) > 0.001
            )
            if existing_context is None:
                contexts_created += 1
            elif context_changed:
                contexts_updated += 1
            if existing_link is None:
                links_created += 1
            elif link_changed:
                links_updated += 1
            if not context_changed and not link_changed:
                skipped += 1
            if dry_run:
                continue
            context = existing_context
            if context_changed:
                context, _ = context_repository.upsert_entity_context(session, values)
            assert context is not None
            if link_changed:
                target = ContextTarget(
                    keyword=trend.keyword,
                    week_start=resolved_start,
                    week_end=resolved_end,
                    entity_text=destination.name,
                    normalized_entity=values.normalized_entity,
                    entity_type=destination.entity_type,
                    relation_score=relation_score,
                    related_locations=(destination.region,),
                )
                link_context_to_trend(session, target=target, context=context, now=now)

    if not dry_run and (contexts_created or contexts_updated or links_created or links_updated):
        session.commit()
    return RelatedDestinationExpansionResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        week_end=resolved_end,
        catalog_version=catalog.version,
        matched_keywords=len(matched_rules),
        destinations_matched=len(previews),
        contexts_created=contexts_created,
        contexts_updated=contexts_updated,
        links_created=links_created,
        links_updated=links_updated,
        skipped=skipped,
        previews=previews,
    )


def serialize_expansion_result(
    result: RelatedDestinationExpansionResult,
) -> dict[str, object]:
    payload = asdict(result)
    payload["previews"] = [asdict(item) for item in result.previews]
    return payload


def destination_page_id(theme_code: str, destination_id: str) -> str:
    return f"{DESTINATION_PAGE_ID_PREFIX}{theme_code.upper()}:{destination_id}"


def is_related_destination_context(context: EntityContext) -> bool:
    return bool(
        context.page_id
        and context.page_id.startswith(DESTINATION_PAGE_ID_PREFIX)
        and context.match_status in {"matched", "manual"}
    )


def related_destination_metadata(context: EntityContext) -> dict[str, object]:
    if not is_related_destination_context(context) or not context.description:
        return {}
    try:
        value = json.loads(context.description)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def related_destination_summary(contexts: list[EntityContext]) -> str:
    rows = []
    for context in contexts:
        metadata = related_destination_metadata(context)
        theme_label = str(metadata.get("theme_label", "연관 여행"))
        activity = str(metadata.get("activity", context.summary))
        rows.append(
            f"{theme_label} 공식 정보 확인 여행지 {context.entity_text}: {activity}. "
            "방문·관람·체험을 목적으로 하는 여행."
        )
    return " ".join(rows)


def serialize_related_destination(context: EntityContext) -> dict[str, object]:
    metadata = related_destination_metadata(context)
    return {
        "name": context.entity_text,
        "entity_type": context.entity_type,
        "region": metadata.get("region"),
        "theme_code": metadata.get("theme_code"),
        "theme_label": metadata.get("theme_label"),
        "activity": metadata.get("activity"),
        "visitability": metadata.get("visitability"),
        "official_url": context.page_url,
        "source_title": context.page_title,
        "match_score": context.match_score,
    }


def _destination_from_payload(raw: object) -> RelatedDestinationDefinition:
    if not isinstance(raw, dict):
        raise ValueError("Destination definitions must be objects")
    official_url = str(raw.get("official_url", "")).strip()
    parsed = urlparse(official_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Destination official_url must be HTTP(S)")
    entity_type = str(raw.get("entity_type", "PLACE")).strip().upper()
    if entity_type not in {"PLACE", "LOCATION"}:
        raise ValueError("Related destinations must be PLACE or LOCATION")
    destination = RelatedDestinationDefinition(
        destination_id=str(raw.get("destination_id", "")).strip(),
        name=str(raw.get("name", "")).strip(),
        region=str(raw.get("region", "")).strip(),
        entity_type=entity_type,
        activity=str(raw.get("activity", "")).strip(),
        visitability=str(raw.get("visitability", "")).strip(),
        official_url=official_url,
        source_title=str(raw.get("source_title", "")).strip(),
        match_confidence=_probability(raw.get("match_confidence")),
    )
    if not all(
        (
            destination.destination_id,
            destination.name,
            destination.region,
            destination.activity,
            destination.visitability,
            destination.source_title,
        )
    ):
        raise ValueError("Destination definitions require complete evidence fields")
    return destination


def _source_keywords(
    session: Session,
    *,
    rules: tuple[DestinationExpansionRule, ...],
    week_start: date,
    week_end: date,
) -> dict[str, set[str]]:
    sources = {rule.required_source for rule in rules if rule.required_source}
    result: dict[str, set[str]] = {}
    for source in sources:
        assert source is not None
        result[source] = set(
            session.scalars(
                select(KeywordOccurrence.normalized_keyword)
                .where(
                    KeywordOccurrence.source == source,
                    KeywordOccurrence.occurred_at
                    >= datetime.combine(week_start, datetime.min.time()),
                    KeywordOccurrence.occurred_at
                    < datetime.combine(
                        week_end + timedelta(days=1), datetime.min.time()
                    ),
                )
                .distinct()
            ).all()
        )
    return result


def _source_category_matches(rule: DestinationExpansionRule) -> bool:
    if not rule.source_category:
        return True
    if rule.required_source != "google_yis_2025_kr":
        return True
    from app.services.google_year_in_search_seed_service import (
        YEAR_IN_SEARCH_2025_KR_KEYWORDS,
    )

    return any(
        item.category == rule.source_category
        and normalize_keyword(item.keyword) == rule.normalized_keyword
        for item in YEAR_IN_SEARCH_2025_KR_KEYWORDS
    )


def _existing_context(
    session: Session, values: EntityContextValues
) -> EntityContext | None:
    return session.scalar(
        select(EntityContext).where(
            EntityContext.normalized_entity == values.normalized_entity,
            EntityContext.entity_type == values.entity_type,
            EntityContext.provider == values.provider,
            EntityContext.page_url == values.page_url,
        )
    )


def _existing_link(
    session: Session,
    *,
    keyword: str,
    week_start: date,
    context_id: int,
) -> TrendContextLink | None:
    return session.scalar(
        select(TrendContextLink).where(
            TrendContextLink.keyword == keyword,
            TrendContextLink.week_start == week_start,
            TrendContextLink.entity_context_id == context_id,
        )
    )


def _context_matches(context: EntityContext, values: EntityContextValues) -> bool:
    stable_fields = (
        "entity_text",
        "page_id",
        "page_title",
        "summary",
        "description",
        "match_score",
        "match_status",
        "source_language",
        "attribution_text",
        "revision_id",
    )
    return all(getattr(context, field) == getattr(values, field) for field in stable_fields)


def _desired_context_score(
    *, relation_score: float, match_score: float, entity_type: str
) -> float:
    from app.context.provider_registry import TRAVEL_SUITABILITY, provider_source_score

    score = (
        min(max(relation_score, 0.0), 100.0) * 0.45
        + min(max(match_score, 0.0), 1.0) * 100 * 0.35
        + provider_source_score("manual", "manual") * 0.10
        + TRAVEL_SUITABILITY.get(entity_type, 0.0) * 0.10
    )
    return round(min(max(score, 0.0), 100.0), 2)


def _probability(value: object) -> float:
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError("Confidence values must be between 0 and 1")
    return result


def _optional_string(value: object) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _empty_result(
    *, catalog: DestinationExpansionCatalog, dry_run: bool
) -> RelatedDestinationExpansionResult:
    return RelatedDestinationExpansionResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=None,
        week_end=None,
        catalog_version=catalog.version,
        matched_keywords=0,
        destinations_matched=0,
        contexts_created=0,
        contexts_updated=0,
        links_created=0,
        links_updated=0,
        skipped=0,
        previews=[],
    )

from dataclasses import dataclass
from datetime import date, datetime, timezone

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.context.context_matcher import (
    ScoredCandidate,
    generate_search_queries,
    rank_candidates,
    score_candidate,
)
from app.context.context_normalizer import (
    clean_plain_text,
    validate_http_url,
)
from app.context.provider_registry import MANUAL_PROVIDERS, WIKIPEDIA_PROVIDER
from app.context.wikipedia_client import WikipediaClient, WikipediaClientError
from app.database import SessionLocal
from app.models.entity_context import EntityContext
from app.ner.entity_dictionary import canonical_location_for
from app.ner.entity_resolver import normalize_entity_text
from app.repositories.context_repository import (
    ContextTarget,
    EntityContextValues,
    delete_manual_context,
    get_cached_context,
    get_context_by_id,
    get_context_target_by_entity,
    get_context_targets,
    get_links_for_context,
    get_relation_score,
    get_weekly_trend,
    recalculate_primary_context,
    upsert_entity_context,
)
from app.services.keyword_normalization_service import normalize_keyword
from app.services.trend_context_link_service import link_context_to_trend


SEARCH_CACHE_TTL_SECONDS = 24 * 60 * 60
_WIKIPEDIA_SEARCH_CACHE: dict[str, tuple[datetime, list]] = {}


class ContextEnrichmentError(RuntimeError):
    pass


class ContextInputError(ValueError):
    pass


class ContextNotFoundError(LookupError):
    pass


class ContextPermissionError(PermissionError):
    pass


@dataclass(frozen=True)
class ContextEnrichmentResult:
    status: str
    provider: str
    processed_entities: int
    skipped_entities: int
    matched: int
    ambiguous: int
    unmatched: int
    errors: int
    created_contexts: int
    updated_contexts: int


@dataclass(frozen=True)
class ManualContextResult:
    context: EntityContext
    keyword: str
    week_start: date
    context_score: float


async def enrich_contexts(
    *,
    limit: int,
    force: bool,
    entity_type: str | None,
    provider: str,
    wikipedia_client: WikipediaClient | None = None,
) -> ContextEnrichmentResult:
    settings = get_settings()
    if provider != WIKIPEDIA_PROVIDER:
        raise ContextInputError(f"자동 보강을 지원하지 않는 provider입니다: {provider}")
    if not settings.wikipedia_enabled:
        return ContextEnrichmentResult(
            status="disabled",
            provider=provider,
            processed_entities=0,
            skipped_entities=0,
            matched=0,
            ambiguous=0,
            unmatched=0,
            errors=0,
            created_contexts=0,
            updated_contexts=0,
        )

    effective_limit = min(limit, max(settings.context_max_entities_per_run, 1))
    with SessionLocal() as session:
        targets = get_context_targets(
            session,
            entity_type=entity_type,
            limit=effective_limit,
        )

    client = wikipedia_client or WikipediaClient()
    owns_client = wikipedia_client is None
    search_cache: dict[str, list] = {}
    processed = skipped = matched = ambiguous = unmatched = errors = 0
    created = updated = 0
    try:
        ensure_configured = getattr(client, "ensure_configured", None)
        if callable(ensure_configured):
            ensure_configured()
        for target in targets:
            now = _utc_now()
            if not force:
                with SessionLocal() as session:
                    cached = get_cached_context(
                        session,
                        normalized_entity=target.normalized_entity,
                        entity_type=target.entity_type,
                        provider=provider,
                        now=now,
                    )
                    if cached is not None:
                        link_context_to_trend(
                            session,
                            target=target,
                            context=cached,
                            now=now,
                        )
                        session.commit()
                        skipped += 1
                        continue

            processed += 1
            try:
                candidates = await find_wikipedia_candidates(
                    target=target,
                    client=client,
                    search_cache=search_cache,
                    force_refresh=force,
                )
                values = await _context_values_from_candidates(
                    target=target,
                    candidates=candidates,
                    client=client,
                    now=now,
                )
            except WikipediaClientError:
                errors += 1
                values = _error_context_values(target=target, client=client, now=now)

            with SessionLocal() as session:
                context, action = upsert_entity_context(session, values)
                link_context_to_trend(
                    session,
                    target=target,
                    context=context,
                    now=now,
                )
                session.commit()
            if action == "created":
                created += 1
            else:
                updated += 1
            if values.match_status == "matched":
                matched += 1
            elif values.match_status == "ambiguous":
                ambiguous += 1
            elif values.match_status == "unmatched":
                unmatched += 1
    finally:
        if owns_client:
            await client.close()

    return ContextEnrichmentResult(
        status="partial_success" if errors else "ok",
        provider=provider,
        processed_entities=processed,
        skipped_entities=skipped,
        matched=matched,
        ambiguous=ambiguous,
        unmatched=unmatched,
        errors=errors,
        created_contexts=created,
        updated_contexts=updated,
    )


async def find_wikipedia_candidates(
    *,
    target: ContextTarget,
    client: WikipediaClient,
    search_cache: dict[str, list] | None = None,
    force_refresh: bool = False,
) -> list[ScoredCandidate]:
    settings = get_settings()
    queries = generate_search_queries(
        entity_text=target.entity_text,
        normalized_entity=target.normalized_entity,
        entity_type=target.entity_type,
        related_locations=list(target.related_locations),
    )
    cache = search_cache if search_cache is not None else {}
    results_by_url = {}
    for query in queries:
        if query not in cache:
            cache[query] = await _search_with_cache(
                client,
                query,
                force_refresh=force_refresh,
            )
        for result in cache[query]:
            results_by_url.setdefault(result.page_url, result)

    scored = [
        score_candidate(
            entity_text=target.entity_text,
            normalized_entity=target.normalized_entity,
            entity_type=target.entity_type,
            context_text=" ".join(
                (target.keyword, *target.related_locations)
            ),
            page_id=result.page_id,
            page_title=result.title,
            page_url=result.page_url,
            snippet=result.snippet,
            redirect_title=result.redirect_title,
        )
        for result in results_by_url.values()
    ]
    return rank_candidates(
        scored,
        matched_threshold=settings.context_match_threshold,
    )


def clear_wikipedia_search_cache() -> None:
    _WIKIPEDIA_SEARCH_CACHE.clear()


async def _search_with_cache(
    client: WikipediaClient,
    query: str,
    *,
    force_refresh: bool,
) -> list:
    now = _utc_now()
    cache_key = f"{getattr(client, 'endpoint', WIKIPEDIA_PROVIDER)}|{query}"
    cached = _WIKIPEDIA_SEARCH_CACHE.get(cache_key)
    if (
        not force_refresh
        and cached is not None
        and (now - cached[0]).total_seconds() < SEARCH_CACHE_TTL_SECONDS
    ):
        return cached[1]
    results = await client.search(query)
    _WIKIPEDIA_SEARCH_CACHE[cache_key] = (now, results)
    return results


async def get_candidates_for_entity(
    *,
    normalized_entity: str,
    client: WikipediaClient,
) -> tuple[ContextTarget | None, list[ScoredCandidate]]:
    with SessionLocal() as session:
        target = get_context_target_by_entity(session, normalized_entity)
    if target is None:
        return None, []
    return target, await find_wikipedia_candidates(target=target, client=client)


def create_manual_context(
    session: Session,
    *,
    provider: str,
    entity_text: str,
    entity_type: str,
    page_title: str,
    page_url: str,
    summary: str,
    keyword: str,
    week_start: date,
) -> ManualContextResult:
    if provider not in MANUAL_PROVIDERS:
        raise ContextInputError("provider는 namuwiki_manual 또는 manual이어야 합니다.")
    clean_entity = clean_plain_text(entity_text)
    canonical = canonical_location_for(clean_entity) or clean_entity
    normalized_entity = normalize_entity_text(clean_entity, canonical)
    normalized_keyword = normalize_keyword(keyword)
    clean_title = clean_plain_text(page_title)
    clean_summary = clean_plain_text(summary, max_chars=1000)
    try:
        clean_url = validate_http_url(page_url)
    except ValueError as exc:
        raise ContextInputError(str(exc)) from exc
    if not normalized_entity or not clean_title or not clean_summary:
        raise ContextInputError("entity_text, page_title, summary는 비어 있을 수 없습니다.")
    if normalized_keyword is None:
        raise ContextInputError("유효한 keyword가 필요합니다.")
    trend = get_weekly_trend(
        session,
        keyword=normalized_keyword,
        week_start=week_start,
    )
    if trend is None:
        raise ContextNotFoundError("해당 주차의 WeeklyTrend를 찾을 수 없습니다.")
    relation_score = get_relation_score(
        session,
        keyword=normalized_keyword,
        week_start=week_start,
        normalized_entity=normalized_entity,
        entity_type=entity_type,
    )
    now = _utc_now()
    try:
        context, _action = upsert_entity_context(
            session,
            EntityContextValues(
                normalized_entity=normalized_entity,
                entity_text=clean_entity,
                entity_type=entity_type,
                provider=provider,
                page_id=None,
                page_title=clean_title,
                page_url=clean_url,
                summary=clean_summary,
                description=None,
                match_score=1.0,
                match_status="manual",
                source_language="ko",
                license_name=None,
                attribution_text="사용자 제공 맥락",
                revision_id=None,
                retrieved_at=now,
                updated_at=now,
            ),
        )
        target = ContextTarget(
            keyword=normalized_keyword,
            week_start=trend.week_start,
            week_end=trend.week_end,
            entity_text=clean_entity,
            normalized_entity=normalized_entity,
            entity_type=entity_type,
            relation_score=relation_score,
            related_locations=(),
        )
        link_context_to_trend(
            session,
            target=target,
            context=context,
            now=now,
        )
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ContextInputError("동일한 수동 맥락이 이미 존재합니다.") from exc
    link = next(
        link
        for link in context.trend_links
        if link.keyword == normalized_keyword and link.week_start == week_start
    )
    return ManualContextResult(
        context=context,
        keyword=normalized_keyword,
        week_start=week_start,
        context_score=link.context_score,
    )


def update_manual_context(
    session: Session,
    *,
    context_id: int,
    page_title: str | None,
    page_url: str | None,
    summary: str | None,
) -> EntityContext:
    context = get_context_by_id(session, context_id)
    if context is None:
        raise ContextNotFoundError("맥락 정보를 찾을 수 없습니다.")
    if context.provider not in MANUAL_PROVIDERS:
        raise ContextPermissionError("자동 Wikipedia 맥락은 수동 API로 수정할 수 없습니다.")
    if page_title is not None:
        cleaned = clean_plain_text(page_title)
        if not cleaned:
            raise ContextInputError("page_title은 비어 있을 수 없습니다.")
        context.page_title = cleaned
    if page_url is not None:
        try:
            context.page_url = validate_http_url(page_url)
        except ValueError as exc:
            raise ContextInputError(str(exc)) from exc
    if summary is not None:
        cleaned = clean_plain_text(summary, max_chars=1000)
        if not cleaned:
            raise ContextInputError("summary는 비어 있을 수 없습니다.")
        context.summary = cleaned
    context.updated_at = _utc_now()
    try:
        session.commit()
    except IntegrityError as exc:
        session.rollback()
        raise ContextInputError("동일한 수동 맥락이 이미 존재합니다.") from exc
    return context


def remove_manual_context(session: Session, *, context_id: int) -> None:
    context = get_context_by_id(session, context_id)
    if context is None:
        raise ContextNotFoundError("맥락 정보를 찾을 수 없습니다.")
    if context.provider not in MANUAL_PROVIDERS:
        raise ContextPermissionError("자동 Wikipedia 맥락은 수동 API로 삭제할 수 없습니다.")
    affected_links = get_links_for_context(session, context_id)
    delete_manual_context(session, context)
    session.flush()
    for keyword, week_start in affected_links:
        recalculate_primary_context(
            session,
            keyword=keyword,
            week_start=week_start,
        )
    session.commit()


async def _context_values_from_candidates(
    *,
    target: ContextTarget,
    candidates: list[ScoredCandidate],
    client: WikipediaClient,
    now: datetime,
) -> EntityContextValues:
    if not candidates:
        queries = generate_search_queries(
            entity_text=target.entity_text,
            normalized_entity=target.normalized_entity,
            entity_type=target.entity_type,
            related_locations=list(target.related_locations),
        )
        query = queries[0] if queries else target.entity_text
        return EntityContextValues(
            normalized_entity=target.normalized_entity,
            entity_text=target.entity_text,
            entity_type=target.entity_type,
            provider=WIKIPEDIA_PROVIDER,
            page_id=None,
            page_title=target.entity_text,
            page_url=client.search_url(query),
            summary="",
            description=None,
            match_score=0.0,
            match_status="unmatched",
            source_language=get_settings().wikipedia_language,
            license_name="CC BY-SA",
            attribution_text="한국어 위키백과",
            revision_id=None,
            retrieved_at=now,
            updated_at=now,
        )

    top = candidates[0]
    page_title = top.page_title
    page_url = top.page_url
    page_id = top.page_id
    summary = ""
    description = top.snippet or None
    revision_id = None
    if top.match_status == "matched":
        page = await client.get_page_summary(top.page_title)
        if page is None:
            raise WikipediaClientError("Wikipedia 요약 페이지를 찾을 수 없습니다.")
        page_title = page.title
        page_url = page.page_url
        page_id = page.page_id or page_id
        summary = page.extract
        description = page.description or description
        revision_id = page.revision_id
    return EntityContextValues(
        normalized_entity=target.normalized_entity,
        entity_text=target.entity_text,
        entity_type=target.entity_type,
        provider=WIKIPEDIA_PROVIDER,
        page_id=page_id,
        page_title=page_title,
        page_url=page_url,
        summary=summary,
        description=description,
        match_score=top.match_score,
        match_status=top.match_status,
        source_language=get_settings().wikipedia_language,
        license_name="CC BY-SA",
        attribution_text="한국어 위키백과",
        revision_id=revision_id,
        retrieved_at=now,
        updated_at=now,
    )


def _error_context_values(
    *,
    target: ContextTarget,
    client: WikipediaClient,
    now: datetime,
) -> EntityContextValues:
    queries = generate_search_queries(
        entity_text=target.entity_text,
        normalized_entity=target.normalized_entity,
        entity_type=target.entity_type,
        related_locations=list(target.related_locations),
    )
    query = queries[0] if queries else target.entity_text
    return EntityContextValues(
        normalized_entity=target.normalized_entity,
        entity_text=target.entity_text,
        entity_type=target.entity_type,
        provider=WIKIPEDIA_PROVIDER,
        page_id=None,
        page_title=target.entity_text,
        page_url=client.search_url(query),
        summary="",
        description=None,
        match_score=0.0,
        match_status="error",
        source_language=get_settings().wikipedia_language,
        license_name="CC BY-SA",
        attribution_text="한국어 위키백과",
        revision_id=None,
        retrieved_at=now,
        updated_at=now,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

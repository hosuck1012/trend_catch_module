from dataclasses import asdict, dataclass
from datetime import date

from sqlalchemy.orm import Session

from app.config import get_settings
from app.context_v2.context_extractor import extract_keyword_contexts
from app.repositories import travel_opportunity_repository as repo


@dataclass(frozen=True)
class ContextExample:
    document_id: int
    keyword: str
    previous_sentence: str | None
    matched_sentence: str
    next_sentence: str | None
    combined_context: str


@dataclass(frozen=True)
class BuildContextsResult:
    status: str
    dry_run: bool
    week_start: date | None
    week_end: date | None
    keywords_processed: int
    documents_processed: int
    contexts_found: int
    contexts_would_create: int
    duplicate_contexts: int
    context_examples: list[ContextExample]
    created: int = 0
    skipped: int = 0
    next_cursor: int | None = None
    has_more: bool = False
    batches: int = 0
    errors: int = 0
    existing_valid: int = 0
    stale_contexts: int = 0
    removed: int = 0
    unmatched_accepted_pairs: int = 0


def build_keyword_contexts(
    session: Session,
    *,
    week_start: date | None,
    limit: int,
    force: bool,
    dry_run: bool,
    after_id: int | None = None,
    process_all: bool = False,
) -> BuildContextsResult:
    settings = get_settings()
    resolved_start, resolved_end = repo.resolve_week_range(session, week_start)
    if not settings.travel_opportunity_v2_enabled:
        return BuildContextsResult("disabled", dry_run, resolved_start, resolved_end, 0, 0, 0, 0, 0, [])
    keywords = repo.get_quality_keywords(
        session,
        week_start=resolved_start,
        week_end=resolved_end,
        limit=None,
    )
    examples: list[ContextExample] = []
    full_sync = bool(process_all and after_id is None and resolved_start and resolved_end)
    # An empty quality-keyword set can be transient (for example, while an
    # upstream rebuild is incomplete). It is not sufficient evidence that every
    # existing context in the active week is stale.
    cleanup_enabled = bool(full_sync and keywords)
    existing_before = (
        repo.get_keyword_contexts_in_range(
            session,
            week_start=resolved_start,
            week_end=resolved_end,
        )
        if cleanup_enabled
        else []
    )
    valid_context_keys: set[tuple[int, str, str]] = set()
    accepted_pairs: set[tuple[int, str]] = set()
    matched_pairs: set[tuple[int, str]] = set()
    keyword_values: set[str] = set()
    document_ids: set[int] = set()
    contexts_found = 0
    contexts_would_create = 0
    duplicate_contexts = 0
    created = 0
    batches = 0
    cursor = after_id
    next_cursor: int | None = None
    has_more = False
    while True:
        rows, page_next_cursor, page_has_more = repo.get_keyword_occurrence_documents_page(
            session,
            normalized_keywords=keywords,
            week_start=resolved_start,
            week_end=resolved_end,
            after_id=cursor,
            limit=limit,
        )
        if not rows:
            has_more = False
            next_cursor = None
            break
        batches += 1
        now = repo.utc_now()
        pending: list[dict[str, object]] = []
        for occurrence, document in rows:
            keyword_values.add(occurrence.normalized_keyword)
            document_ids.add(document.id)
            pair = (document.id, occurrence.normalized_keyword)
            accepted_pairs.add(pair)
            extracted = extract_keyword_contexts(
                text=f"{document.title}\n{document.text}".strip(),
                keyword=occurrence.keyword,
                normalized_keyword=occurrence.normalized_keyword,
                sentences_before=settings.context_sentences_before,
                sentences_after=settings.context_sentences_after,
                max_chars=settings.context_max_chars,
            )
            contexts_found += len(extracted)
            if extracted:
                matched_pairs.add(pair)
            for context in extracted:
                valid_context_keys.add(
                    (document.id, context.normalized_keyword, context.context_hash)
                )
                payload = {
                    "document_id": document.id,
                    "keyword": context.keyword,
                    "normalized_keyword": context.normalized_keyword,
                    "previous_sentence": context.previous_sentence,
                    "matched_sentence": context.matched_sentence,
                    "next_sentence": context.next_sentence,
                    "combined_context": context.combined_context,
                    "occurrence_index": context.occurrence_index,
                    "source": document.source,
                    "published_at": document.published_at,
                    "context_hash": context.context_hash,
                    "created_at": now,
                    "updated_at": now,
                }
                pending.append(payload)
                if len(examples) < 5:
                    examples.append(
                        ContextExample(
                            document_id=document.id,
                            keyword=context.keyword,
                            previous_sentence=context.previous_sentence,
                            matched_sentence=context.matched_sentence,
                            next_sentence=context.next_sentence,
                            combined_context=context.combined_context,
                        )
                    )
        keys = [
            (int(row["document_id"]), str(row["normalized_keyword"]), str(row["context_hash"]))
            for row in pending
        ]
        # Context identity is content-addressed. Even force runs must not insert the
        # same key again because doing so would violate the unique constraint and
        # could cascade into already materialized downstream evidence.
        existing = repo.existing_context_keys(session, triples=keys)
        to_create = [
            row
            for row in pending
            if (int(row["document_id"]), str(row["normalized_keyword"]), str(row["context_hash"]))
            not in existing
        ]
        contexts_would_create += len(to_create)
        duplicate_contexts += len(pending) - len(to_create)
        if not dry_run and to_create:
            repo.add_keyword_contexts(session, to_create)
            created += len(to_create)
        has_more = page_has_more
        next_cursor = page_next_cursor
        if not process_all or not page_has_more:
            break
        if page_next_cursor is None or page_next_cursor <= (cursor or 0):
            raise RuntimeError("Context pagination cursor did not advance")
        if batches >= 100_000:
            raise RuntimeError("Context pagination exceeded safety limit")
        cursor = page_next_cursor
    existing_valid = sum(
        (row.document_id, row.normalized_keyword, row.context_hash) in valid_context_keys
        for row in existing_before
    )
    stale_ids = [
        row.id
        for row in existing_before
        if (row.document_id, row.normalized_keyword, row.context_hash)
        not in valid_context_keys
    ]
    removed = 0
    if not dry_run and stale_ids:
        removed = repo.delete_keyword_contexts(session, context_ids=stale_ids)
    if not dry_run and (created or removed):
        # Context creation and stale cleanup form one synchronization unit. A
        # later-page failure must not leave an incomplete partial rebuild.
        session.commit()
    return BuildContextsResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        week_end=resolved_end,
        keywords_processed=len(keyword_values),
        documents_processed=len(document_ids),
        contexts_found=contexts_found,
        contexts_would_create=contexts_would_create,
        duplicate_contexts=duplicate_contexts,
        context_examples=examples,
        created=created,
        skipped=duplicate_contexts,
        next_cursor=next_cursor,
        has_more=has_more,
        batches=batches,
        existing_valid=existing_valid,
        stale_contexts=len(stale_ids),
        removed=removed,
        unmatched_accepted_pairs=len(accepted_pairs - matched_pairs),
    )


def serialize_build_result(result: BuildContextsResult) -> dict[str, object]:
    payload = asdict(result)
    payload["context_examples"] = [asdict(item) for item in result.context_examples]
    return payload

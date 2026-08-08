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


def build_keyword_contexts(
    session: Session,
    *,
    week_start: date | None,
    limit: int,
    force: bool,
    dry_run: bool,
) -> BuildContextsResult:
    settings = get_settings()
    resolved_start, resolved_end = repo.resolve_week_range(session, week_start)
    if not settings.travel_opportunity_v2_enabled:
        return BuildContextsResult("disabled", dry_run, resolved_start, resolved_end, 0, 0, 0, 0, 0, [])
    keywords = repo.get_quality_keywords(
        session,
        week_start=resolved_start,
        week_end=resolved_end,
        limit=limit,
    )
    rows = repo.get_keyword_occurrence_documents(
        session,
        normalized_keywords=keywords,
        week_start=resolved_start,
        week_end=resolved_end,
        limit=limit,
    )
    now = repo.utc_now()
    pending: list[dict[str, object]] = []
    examples: list[ContextExample] = []
    contexts_found = 0
    for occurrence, document in rows:
        extracted = extract_keyword_contexts(
            text=f"{document.title}\n{document.text}".strip(),
            keyword=occurrence.keyword,
            normalized_keyword=occurrence.normalized_keyword,
            sentences_before=settings.context_sentences_before,
            sentences_after=settings.context_sentences_after,
            max_chars=settings.context_max_chars,
        )
        contexts_found += len(extracted)
        for context in extracted:
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
    existing = repo.existing_context_keys(session, triples=keys) if not force else set()
    to_create = [
        row
        for row in pending
        if (int(row["document_id"]), str(row["normalized_keyword"]), str(row["context_hash"]))
        not in existing
    ]
    if not dry_run and to_create:
        repo.add_keyword_contexts(session, to_create)
    return BuildContextsResult(
        status="dry_run" if dry_run else "ok",
        dry_run=dry_run,
        week_start=resolved_start,
        week_end=resolved_end,
        keywords_processed=len(keywords),
        documents_processed=len({document.id for _occurrence, document in rows}),
        contexts_found=contexts_found,
        contexts_would_create=len(to_create),
        duplicate_contexts=len(pending) - len(to_create),
        context_examples=examples,
    )


def serialize_build_result(result: BuildContextsResult) -> dict[str, object]:
    payload = asdict(result)
    payload["context_examples"] = [asdict(item) for item in result.context_examples]
    return payload

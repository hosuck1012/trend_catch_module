import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from time import perf_counter

from app.config import get_settings
from app.database import SessionLocal
from app.keywords.tokenizer import KiwiTokenizerAdapter, ResilientTokenizer, Tokenizer
from app.ner.entity_dictionary import extract_dictionary_entities
from app.ner.entity_labels import (
    ENTITY_LABEL_DESCRIPTIONS,
    NER_LABEL_VERSION,
    NER_PIPELINE_VERSION,
    EntityCandidate,
    EntityType,
)
from app.ner.entity_resolver import resolve_entities
from app.ner.entity_rules import extract_rule_entities
from app.ner.gliner_adapter import GlinerAdapter, GlinerAdapterError, gliner_adapter
from app.ner.text_chunking import TextChunk, build_ner_chunks
from app.repositories.entity_repository import (
    DocumentSnapshot,
    get_document_snapshots_by_ids,
    get_document_snapshots_page,
    get_entity_extraction_states,
    get_recent_document_snapshots,
    replace_document_entities_with_state,
)


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")
MAX_BATCH_LOOPS = 10_000


@dataclass(frozen=True)
class EntityExtractionResult:
    status: str
    model_status: str
    processed_documents: int
    skipped_documents: int
    inserted_entities: int
    entity_counts: dict[str, int]
    extractor_counts: dict[str, int]
    errors: list[str]
    model_error: str | None
    dry_run: bool = False
    mentions_detected: int = 0
    documents_with_entities: int = 0
    created_entities: int = 0
    updated_documents: int = 0
    duplicates_merged: int = 0
    batches: int = 0
    next_cursor: int | None = None
    has_more: bool = False
    pipeline_version: str = NER_PIPELINE_VERSION
    label_version: str = NER_LABEL_VERSION
    model_load_count: int = 0
    processing_seconds: float = 0.0


@dataclass(frozen=True)
class DocumentEntityAnalysis:
    document: DocumentSnapshot
    input_hash: str
    candidates: list[EntityCandidate]
    raw_candidate_count: int


async def extract_entities(
    *,
    limit: int,
    force: bool,
    source: str | None,
    since_days: int,
    adapter: GlinerAdapter | None = None,
    dry_run: bool = False,
    after_id: int | None = None,
    batch_size: int = 25,
    process_all: bool = False,
    document_ids: list[int] | None = None,
) -> EntityExtractionResult:
    started = perf_counter()
    settings = get_settings()
    active_adapter = adapter or gliner_adapter
    tokenizer: Tokenizer = ResilientTokenizer(KiwiTokenizerAdapter())
    since = datetime.now() - timedelta(days=since_days)
    page_size = max(1, min(batch_size, 100))
    cursor = max(0, after_id or 0)
    remaining = max(1, limit)

    entity_counts: Counter[str] = Counter()
    extractor_counts: Counter[str] = Counter()
    processed = 0
    skipped = 0
    detected = 0
    inserted = 0
    updated_documents = 0
    documents_with_entities = 0
    duplicates_merged = 0
    batches = 0
    errors: list[str] = []
    model_error: str | None = None
    has_more = False
    selected_ids = list(dict.fromkeys(document_ids or []))

    while batches < MAX_BATCH_LOOPS:
        with SessionLocal() as session:
            if selected_ids:
                documents = get_document_snapshots_by_ids(session, selected_ids)
                selected_ids = []
            elif process_all or after_id is not None:
                requested = page_size if process_all else min(page_size, remaining)
                documents = get_document_snapshots_page(
                    session,
                    since=since,
                    source=source,
                    after_id=cursor,
                    limit=requested,
                )
            elif batches == 0:
                effective_limit = min(limit, max(1, settings.ner_max_documents_per_run))
                documents = get_recent_document_snapshots(
                    session,
                    since=since,
                    source=source,
                    limit=effective_limit,
                )
            else:
                documents = []
            states = get_entity_extraction_states(
                session, [document.id for document in documents]
            )

        if not documents:
            break
        batches += 1
        cursor = max(cursor, max(document.id for document in documents))
        prepared: list[tuple[DocumentSnapshot, str, list[TextChunk]]] = []
        for document in documents:
            clean_title = _clean_text(document.title)
            clean_body = _clean_text(document.text)
            input_hash = ner_input_hash(
                title=clean_title,
                body=clean_body,
                model_name=settings.ner_model_name,
                model_enabled=settings.ner_enabled,
                threshold=settings.ner_threshold,
                max_chars=settings.ner_text_max_chars,
            )
            state = states.get(document.id)
            model_complete = state is not None and (
                state.model_succeeded or not settings.ner_enabled
            )
            if (
                not force
                and state is not None
                and state.input_hash == input_hash
                and state.pipeline_version == NER_PIPELINE_VERSION
                and state.label_version == NER_LABEL_VERSION
                and state.model_name == settings.ner_model_name
                and model_complete
            ):
                skipped += 1
                continue
            chunks = build_ner_chunks(
                clean_title,
                clean_body,
                max_chars=settings.ner_text_max_chars,
            )
            if not chunks:
                skipped += 1
                continue
            prepared.append((document, input_hash, chunks))

        analyses: list[DocumentEntityAnalysis] = []
        batch_model_error: str | None = None
        if prepared:
            analyses, batch_model_error = await _analyze_documents(
                prepared,
                adapter=active_adapter,
                tokenizer=tokenizer,
                model_enabled=settings.ner_enabled and model_error is None,
                model_batch_size=max(1, settings.ner_batch_size),
            )
        if batch_model_error and model_error is None:
            model_error = batch_model_error
            errors.append(batch_model_error)

        for analysis in analyses:
            processed += 1
            detected += len(analysis.candidates)
            if analysis.candidates:
                documents_with_entities += 1
            duplicates_merged += max(
                0, analysis.raw_candidate_count - len(analysis.candidates)
            )
            for candidate in analysis.candidates:
                entity_counts[candidate.entity_type.value] += 1
                extractor_counts[candidate.extractor] += 1
            if dry_run:
                continue
            with SessionLocal() as session:
                count = replace_document_entities_with_state(
                    session,
                    document=analysis.document,
                    candidates=analysis.candidates,
                    input_hash=analysis.input_hash,
                    model_name=settings.ner_model_name,
                    pipeline_version=NER_PIPELINE_VERSION,
                    label_version=NER_LABEL_VERSION,
                    model_succeeded=not settings.ner_enabled or model_error is None,
                    processed_at=datetime.now(),
                )
            inserted += count
            updated_documents += 1

        if document_ids is not None or (not process_all and after_id is None):
            break
        if not process_all:
            remaining -= len(documents)
            if remaining <= 0:
                has_more = len(documents) == page_size
                break
        if len(documents) < page_size:
            break
    else:
        errors.append("pagination safety limit exceeded")

    model_status = active_adapter.get_status()
    status = "partial_success" if errors and processed else "failed" if errors else "ok"
    return EntityExtractionResult(
        status=status,
        model_status=model_status.status,
        processed_documents=processed,
        skipped_documents=skipped,
        inserted_entities=inserted,
        entity_counts={
            entity_type.value: entity_counts[entity_type.value]
            for entity_type in EntityType
        },
        extractor_counts={
            extractor: extractor_counts[extractor]
            for extractor in ("gliner", "dictionary", "rule", "merged")
        },
        errors=errors,
        model_error=model_error,
        dry_run=dry_run,
        mentions_detected=detected,
        documents_with_entities=documents_with_entities,
        created_entities=inserted,
        updated_documents=updated_documents,
        duplicates_merged=duplicates_merged,
        batches=batches,
        next_cursor=cursor if batches else after_id,
        has_more=has_more,
        model_load_count=model_status.model_load_count,
        processing_seconds=round(perf_counter() - started, 3),
    )


async def _analyze_documents(
    prepared: list[tuple[DocumentSnapshot, str, list[TextChunk]]],
    *,
    adapter: GlinerAdapter,
    tokenizer: Tokenizer,
    model_enabled: bool,
    model_batch_size: int,
) -> tuple[list[DocumentEntityAnalysis], str | None]:
    raw_by_document: dict[int, list[EntityCandidate]] = defaultdict(list)
    chunks: list[tuple[int, TextChunk]] = []
    for document, _input_hash, document_chunks in prepared:
        for chunk in document_chunks:
            deterministic = [
                *extract_dictionary_entities(chunk.text),
                *extract_rule_entities(chunk.text, tokenizer=tokenizer),
            ]
            raw_by_document[document.id].extend(
                _shift_candidate(candidate, chunk.start_char)
                for candidate in deterministic
            )
            chunks.append((document.id, chunk))

    model_error: str | None = None
    if model_enabled and chunks:
        try:
            for start in range(0, len(chunks), model_batch_size):
                batch = chunks[start : start + model_batch_size]
                predictions = await adapter.predict([chunk.text for _, chunk in batch])
                for (document_id, chunk), candidates in zip(batch, predictions, strict=True):
                    raw_by_document[document_id].extend(
                        _shift_candidate(candidate, chunk.start_char)
                        for candidate in candidates
                    )
        except GlinerAdapterError as exc:
            model_error = str(exc)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {str(exc)}"[:1000]

    analyses: list[DocumentEntityAnalysis] = []
    for document, input_hash, _chunks in prepared:
        raw = raw_by_document[document.id]
        analyses.append(
            DocumentEntityAnalysis(
                document=document,
                input_hash=input_hash,
                candidates=resolve_entities(raw),
                raw_candidate_count=len(raw),
            )
        )
    return analyses, model_error


def ner_input_hash(
    *,
    title: str,
    body: str,
    model_name: str,
    model_enabled: bool,
    threshold: float,
    max_chars: int,
) -> str:
    payload = {
        "title": title,
        "body": body,
        "model": model_name,
        "model_enabled": model_enabled,
        "pipeline_version": NER_PIPELINE_VERSION,
        "label_version": NER_LABEL_VERSION,
        "labels": ENTITY_LABEL_DESCRIPTIONS,
        "threshold": threshold,
        "max_chars": max_chars,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_document_text(title: str, body: str, *, max_chars: int) -> str:
    clean_title = _clean_text(title)
    clean_body = _clean_text(body)
    if not clean_title and not clean_body:
        return ""
    if not clean_title:
        return clean_body[:max_chars]
    if not clean_body:
        return clean_title[:max_chars]
    title_part = clean_title[:max_chars]
    remaining = max_chars - len(title_part) - 1
    if remaining <= 0:
        return title_part
    return f"{title_part}\n{clean_body[:remaining]}"


def _shift_candidate(candidate: EntityCandidate, offset: int) -> EntityCandidate:
    return replace(
        candidate,
        start_char=(candidate.start_char + offset if candidate.start_char is not None else None),
        end_char=(candidate.end_char + offset if candidate.end_char is not None else None),
    )


def _clean_text(value: str) -> str:
    cleaned = html.unescape(HTML_TAG_PATTERN.sub(" ", value or ""))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return EXCESS_NEWLINES_PATTERN.sub("\n\n", cleaned).strip()

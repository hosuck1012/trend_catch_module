import html
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta

from app.config import get_settings
from app.database import SessionLocal
from app.ner.entity_dictionary import extract_dictionary_entities
from app.ner.entity_labels import EntityCandidate, EntityType
from app.ner.entity_resolver import resolve_entities
from app.ner.entity_rules import extract_rule_entities
from app.ner.gliner_adapter import GlinerAdapter, GlinerAdapterError, gliner_adapter
from app.repositories.entity_repository import (
    DocumentSnapshot,
    get_recent_document_snapshots,
    replace_document_entities,
)


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"[^\S\r\n]+")
EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")


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


async def extract_entities(
    *,
    limit: int,
    force: bool,
    source: str | None,
    since_days: int,
    adapter: GlinerAdapter | None = None,
) -> EntityExtractionResult:
    settings = get_settings()
    effective_limit = min(limit, max(1, settings.ner_max_documents_per_run))
    since = datetime.now() - timedelta(days=since_days)
    with SessionLocal() as session:
        documents = get_recent_document_snapshots(
            session,
            since=since,
            source=source,
            limit=effective_limit,
        )

    prepared: list[tuple[DocumentSnapshot, str, list[EntityCandidate]]] = []
    skipped = 0
    for document in documents:
        if document.has_entities and not force:
            skipped += 1
            continue
        text = prepare_document_text(
            document.title,
            document.text,
            max_chars=settings.ner_text_max_chars,
        )
        if not text:
            skipped += 1
            continue
        deterministic = [
            *extract_dictionary_entities(text),
            *extract_rule_entities(text),
        ]
        prepared.append((document, text, deterministic))

    active_adapter = adapter or gliner_adapter
    predictions: list[list[EntityCandidate]] = [[] for _ in prepared]
    model_error: str | None = None
    if settings.ner_enabled and prepared:
        try:
            for start in range(0, len(prepared), max(1, settings.ner_batch_size)):
                batch = prepared[start : start + max(1, settings.ner_batch_size)]
                batch_predictions = await active_adapter.predict([item[1] for item in batch])
                predictions[start : start + len(batch)] = batch_predictions
        except GlinerAdapterError as exc:
            model_error = str(exc)
        except Exception as exc:
            model_error = f"{type(exc).__name__}: {str(exc)}"[:1000]

    entity_counts: Counter[str] = Counter()
    extractor_counts: Counter[str] = Counter()
    inserted = 0
    for index, (document, _text, deterministic) in enumerate(prepared):
        resolved = resolve_entities([*deterministic, *predictions[index]])
        with SessionLocal() as session:
            count = replace_document_entities(
                session,
                document=document,
                candidates=resolved,
                force=force,
            )
        inserted += count
        if count:
            for candidate in resolved:
                entity_counts[candidate.entity_type.value] += 1
                extractor_counts[candidate.extractor] += 1

    model_status = active_adapter.get_status().status
    if model_error and prepared:
        status = "partial_success" if inserted else "failed"
    else:
        status = "ok"
    return EntityExtractionResult(
        status=status,
        model_status=model_status,
        processed_documents=len(prepared),
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
        errors=[model_error] if model_error else [],
        model_error=model_error,
    )


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


def _clean_text(value: str) -> str:
    cleaned = html.unescape(HTML_TAG_PATTERN.sub(" ", value or ""))
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return EXCESS_NEWLINES_PATTERN.sub("\n\n", cleaned).strip()

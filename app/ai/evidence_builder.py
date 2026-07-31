from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from sqlalchemy.orm import Session

from app.ai.gemini_prompt import SYSTEM_INSTRUCTION, build_user_prompt
from app.config import get_settings
from app.context.context_normalizer import clean_plain_text
from app.models.weekly_trend import WeeklyTrend
from app.repositories.trend_ai_repository import (
    count_keyword_documents,
    get_candidate_documents,
    get_eligible_context_links,
    get_entity_links,
)
from app.services.keyword_normalization_service import normalize_keyword


TRAVEL_ENTITY_TYPES = {"LOCATION", "PLACE", "EVENT", "FOOD"}


class EvidenceInputTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class EvidencePackage:
    payload: dict[str, object]
    user_prompt: str
    input_hash: str
    valid_refs: frozenset[str]
    travel_entities: tuple[dict[str, object], ...]
    input_truncated: bool

    @property
    def input_chars(self) -> int:
        return len(SYSTEM_INSTRUCTION) + len(self.user_prompt)


def build_evidence_package(
    session: Session,
    *,
    trend: WeeklyTrend,
    normalized_keyword: str,
    model_name: str,
    prompt_version: str,
    max_documents: int | None = None,
    max_contexts: int | None = None,
    max_input_chars: int | None = None,
) -> EvidencePackage:
    settings = get_settings()
    document_limit = max(1, max_documents or settings.gemini_max_documents)
    context_limit = max(0, max_contexts if max_contexts is not None else settings.gemini_max_contexts)
    input_limit = max(1, max_input_chars or settings.gemini_max_input_chars)

    candidates = get_candidate_documents(
        session,
        normalized_keyword=normalized_keyword,
        week_start=trend.week_start,
        week_end=trend.week_end,
        limit=max(document_limit * 4, document_limit),
    )
    selected_documents = _select_documents(
        candidates,
        normalized_keyword=normalized_keyword,
        limit=document_limit,
    )
    entity_links = get_entity_links(
        session,
        keyword=trend.keyword,
        week_start=trend.week_start,
    )
    context_links = get_eligible_context_links(
        session,
        keyword=trend.keyword,
        week_start=trend.week_start,
    )[:context_limit]

    document_count = count_keyword_documents(
        session,
        normalized_keyword=normalized_keyword,
        week_start=trend.week_start,
        week_end=trend.week_end,
    )
    trend_payload = {
        "keyword": trend.keyword,
        "week_start": trend.week_start.isoformat(),
        "week_end": trend.week_end.isoformat(),
        "weekly_mentions": trend.weekly_mentions,
        "previous_weekly_mentions": trend.previous_weekly_mentions,
        "growth_rate": trend.growth_rate,
        "source_count": trend.source_count,
        "document_count": document_count,
        "search_interest_score": trend.search_interest_score,
        "final_score": trend.final_score,
        "watchlist": trend.status == "watchlist",
    }
    documents = [
        {
            "ref": f"DOC-{document.id}",
            "id": document.id,
            "source": document.source,
            "title": clean_plain_text(document.title, max_chars=300),
            "published_at": document.published_at.isoformat(),
            "snippet": clean_plain_text(document.text, max_chars=500),
            "url": document.url,
            "modified_at": document.collected_at.isoformat(),
        }
        for document in selected_documents
    ]
    entities = [
        {
            "ref": f"ENTITY-{link.normalized_entity}",
            "entity_text": link.entity_text,
            "normalized_entity": link.normalized_entity,
            "entity_type": link.entity_type,
            "relation_score": link.relation_score,
            "mention_count": link.mention_count,
            "source_count": link.source_count,
            "is_primary": link.is_primary,
            "calculated_at": link.calculated_at.isoformat(),
        }
        for link in entity_links[:20]
    ]
    contexts = [
        {
            "ref": f"CONTEXT-{link.entity_context.id}",
            "id": link.entity_context.id,
            "normalized_entity": link.normalized_entity,
            "entity_type": link.entity_type,
            "provider": link.entity_context.provider,
            "page_title": link.entity_context.page_title,
            "page_url": link.entity_context.page_url,
            "summary": clean_plain_text(link.entity_context.summary, max_chars=700),
            "match_status": link.entity_context.match_status,
            "context_score": link.context_score,
            "is_primary": link.is_primary,
            "updated_at": link.entity_context.updated_at.isoformat(),
        }
        for link in context_links
    ]
    payload: dict[str, object] = {
        "keyword": trend.keyword,
        "trend": trend_payload,
        "documents": documents,
        "entities": entities,
        "contexts": contexts,
        "input_truncated": False,
    }
    was_truncated = _fit_payload(payload, max_chars=input_limit)
    user_prompt = build_user_prompt(payload)
    if len(SYSTEM_INSTRUCTION) + len(user_prompt) > input_limit:
        raise EvidenceInputTooLargeError(
            "GEMINI_MAX_INPUT_CHARS가 필수 근거를 담기에 너무 작습니다."
        )
    valid_refs = {"SCORE-WEEKLY"}
    for section in ("documents", "entities", "contexts"):
        valid_refs.update(str(item["ref"]) for item in payload[section])
    travel_entities = tuple(
        item for item in payload["entities"] if item["entity_type"] in TRAVEL_ENTITY_TYPES
    )
    input_hash = _build_input_hash(
        payload=payload,
        model_name=model_name,
        prompt_version=prompt_version,
    )
    return EvidencePackage(
        payload=payload,
        user_prompt=user_prompt,
        input_hash=input_hash,
        valid_refs=frozenset(valid_refs),
        travel_entities=travel_entities,
        input_truncated=was_truncated,
    )


def _select_documents(documents, *, normalized_keyword: str, limit: int):
    ordered = sorted(
        documents,
        key=lambda document: (
            not _title_exact_match(document.title, normalized_keyword),
            -_timestamp(document.published_at),
            -document.id,
        ),
    )
    selected = []
    seen_sources = set()
    for document in ordered:
        if document.source in seen_sources:
            continue
        selected.append(document)
        seen_sources.add(document.source)
        if len(selected) == limit:
            return selected
    selected_ids = {document.id for document in selected}
    for document in ordered:
        if document.id in selected_ids:
            continue
        selected.append(document)
        if len(selected) == limit:
            break
    return selected


def _title_exact_match(title: str, normalized_keyword: str) -> bool:
    normalized_title = normalize_keyword(title) or ""
    return normalized_title == normalized_keyword or normalized_keyword in normalized_title


def _timestamp(value: datetime) -> float:
    try:
        return value.timestamp()
    except (OSError, OverflowError, ValueError):
        return 0.0


def _fit_payload(payload: dict[str, object], *, max_chars: int) -> bool:
    truncated = False

    def too_long() -> bool:
        return len(SYSTEM_INSTRUCTION) + len(build_user_prompt(payload)) > max_chars

    for snippet_limit in (320, 200, 120, 60):
        if not too_long():
            break
        for document in payload["documents"]:
            snippet = str(document.get("snippet") or "")
            if len(snippet) > snippet_limit:
                document["snippet"] = snippet[:snippet_limit].rstrip()
                truncated = True

    minimum_documents = min(2, len(payload["documents"]))
    while too_long() and len(payload["documents"]) > minimum_documents:
        payload["documents"].pop()
        truncated = True

    while too_long() and len(payload["entities"]) > 1:
        removable = next(
            (
                index
                for index in range(len(payload["entities"]) - 1, -1, -1)
                if not payload["entities"][index].get("is_primary")
            ),
            None,
        )
        if removable is None:
            break
        payload["entities"].pop(removable)
        truncated = True

    while too_long() and len(payload["contexts"]) > 1:
        removable = next(
            (
                index
                for index in range(len(payload["contexts"]) - 1, -1, -1)
                if not payload["contexts"][index].get("is_primary")
            ),
            None,
        )
        if removable is None:
            break
        payload["contexts"].pop(removable)
        truncated = True

    if too_long():
        for context in payload["contexts"]:
            summary = str(context.get("summary") or "")
            if len(summary) > 240:
                context["summary"] = summary[:240].rstrip()
                truncated = True
    payload["input_truncated"] = truncated
    return truncated


def _build_input_hash(
    *,
    payload: dict[str, object],
    model_name: str,
    prompt_version: str,
) -> str:
    canonical = json.dumps(
        {
            "model_name": model_name,
            "prompt_version": prompt_version,
            "evidence": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

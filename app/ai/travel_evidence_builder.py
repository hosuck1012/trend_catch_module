from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

from sqlalchemy.orm import Session

from app.ai.travel_opportunity_prompt import SYSTEM_INSTRUCTION, build_user_prompt
from app.config import Settings
from app.models.entity_context import EntityContext
from app.models.entity_mention import EntityMention
from app.models.keyword_context import KeywordContext
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.models.trend_entity_link import TrendEntityLink
from app.repositories import travel_ranking_repository as ranking_repo


class TravelEvidenceInputTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class TravelEvidencePackage:
    payload: dict[str, object]
    user_prompt: str
    input_hash: str
    input_chars: int
    valid_evidence_refs: frozenset[str]
    allowed_destination_names: frozenset[str]
    context_texts: tuple[str, ...]


def build_travel_evidence_package(
    session: Session,
    *,
    candidate: TravelOpportunityCandidate,
    settings: Settings,
    model: str,
) -> TravelEvidencePackage:
    keyword = candidate.normalized_keyword
    contexts = ranking_repo.get_keyword_contexts(
        session,
        keywords=[keyword],
        week_start=candidate.week_start,
    )
    contexts = _select_contexts(
        contexts,
        limit=max(1, settings.travel_gemini_max_contexts),
    )
    document_ids = ranking_repo.get_document_ids_by_keyword(
        session,
        keywords=[keyword],
        week_start=candidate.week_start,
    ).get(keyword, set())
    document_ids.update(context.document_id for context in contexts)
    documents = ranking_repo.get_documents(session, document_ids)
    document_rows = sorted(
        documents.values(),
        key=lambda row: (row.published_at, row.id),
        reverse=True,
    )[: max(1, settings.travel_gemini_max_evidence_docs)]
    selected_document_ids = {row.id for row in document_rows}
    mentions = _relevant_mentions(
        contexts=contexts,
        mentions=ranking_repo.get_entity_mentions(session, selected_document_ids),
    )
    trend_links = ranking_repo.get_trend_entity_links(
        session,
        keywords=[keyword],
        week_start=candidate.week_start,
    )
    entity_contexts = _eligible_entity_contexts(
        ranking_repo.get_entity_contexts(
            session,
            keywords=[keyword],
            week_start=candidate.week_start,
        ).get(keyword, [])
    )

    context_payload = []
    for context in contexts:
        evidence_id = f"CTX-{context.id}"
        context_payload.append(
            {
                "evidence_id": evidence_id,
                "source": context.source,
                "previous_sentence": _trim(context.previous_sentence, 500),
                "matched_sentence": _trim(context.matched_sentence, 700),
                "next_sentence": _trim(context.next_sentence, 500),
            }
        )

    document_payload = []
    for document in document_rows:
        evidence_id = f"DOC-{document.id}"
        document_payload.append(
            {
                "evidence_id": evidence_id,
                "source": document.source,
                "title": _trim(document.title, 400),
            }
        )

    entity_payload = []
    seen_entities: set[tuple[str, str]] = set()
    for mention in mentions:
        key = (_normalize(mention.normalized_text), mention.entity_type)
        if not key[0] or key in seen_entities:
            continue
        seen_entities.add(key)
        evidence_id = f"ENTITY-{mention.id}"
        entity_payload.append(
            {
                "evidence_id": evidence_id,
                "name": mention.text,
                "normalized_name": mention.normalized_text,
                "entity_type": mention.entity_type,
                "confidence": round(mention.confidence, 4),
                "origin": "entity_mention",
            }
        )
    for link in trend_links:
        key = (_normalize(link.normalized_entity), link.entity_type)
        if not key[0] or key in seen_entities:
            continue
        seen_entities.add(key)
        evidence_id = f"ENTITY-LINK-{link.id}"
        entity_payload.append(
            {
                "evidence_id": evidence_id,
                "name": link.entity_text,
                "normalized_name": link.normalized_entity,
                "entity_type": link.entity_type,
                "confidence": round(link.average_confidence, 4),
                "origin": "trend_entity_link",
            }
        )

    entity_context_payload = []
    for context in entity_contexts:
        evidence_id = f"CONTEXT-{context.id}"
        entity_context_payload.append(
            {
                "evidence_id": evidence_id,
                "provider": context.provider,
                "match_status": context.match_status,
                "entity_name": context.entity_text,
                "entity_type": context.entity_type,
                "page_title": context.page_title,
                "summary": _trim(context.summary, 700),
            }
        )

    payload: dict[str, object] = {
        "candidate": {
            "keyword": candidate.keyword,
            "normalized_keyword": candidate.normalized_keyword,
            "travel_category": candidate.travel_category,
            "high_precision_score": candidate.high_precision_score,
            "ranking_status": candidate.ranking_status,
            "evidence_gate": candidate.evidence_gate,
        },
        "scores": {
            "trend_strength_score": candidate.trend_strength_score,
            "context_clarity_score": candidate.context_clarity_score,
            "travel_convertibility_score": candidate.travel_convertibility_score,
            "evidence_confidence_score": candidate.evidence_confidence_score,
        },
        "contexts": context_payload,
        "entities": entity_payload,
        "evidence": {
            "evidence_codes": _json_list(candidate.evidence_codes_json),
            "document_count": candidate.evidence_document_count,
            "source_count": candidate.evidence_source_count,
            "source_titles": document_payload,
            "matched_or_manual_contexts": entity_context_payload,
            "ranking_ref": "RANKING-V2",
        },
    }
    user_prompt = _fit_prompt(
        payload,
        max_chars=max(settings.travel_gemini_max_input_chars, 1000),
    )
    canonical = json.dumps(
        {
            "model": model,
            "prompt_version": settings.travel_gemini_prompt_version,
            "payload": payload,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    input_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    fitted_refs = _evidence_refs_from_payload(payload)
    fitted_names = _allowed_names_from_payload(payload)
    fitted_contexts = payload["contexts"]
    assert isinstance(fitted_contexts, list)
    context_texts = tuple(
        " ".join(
            str(row.get(key) or "")
            for key in ("previous_sentence", "matched_sentence", "next_sentence")
        )
        for row in fitted_contexts
        if isinstance(row, dict)
    )
    return TravelEvidencePackage(
        payload=payload,
        user_prompt=user_prompt,
        input_hash=input_hash,
        input_chars=len(SYSTEM_INSTRUCTION) + len(user_prompt),
        valid_evidence_refs=frozenset(fitted_refs),
        allowed_destination_names=frozenset(fitted_names),
        context_texts=context_texts,
    )


def _select_contexts(
    contexts: list[KeywordContext], *, limit: int
) -> list[KeywordContext]:
    selected: list[KeywordContext] = []
    seen_sources: set[str] = set()
    for context in contexts:
        if context.source not in seen_sources:
            selected.append(context)
            seen_sources.add(context.source)
        if len(selected) >= limit:
            return selected
    selected_ids = {context.id for context in selected}
    for context in contexts:
        if context.id not in selected_ids:
            selected.append(context)
        if len(selected) >= limit:
            break
    return selected


def _relevant_mentions(
    *,
    contexts: list[KeywordContext],
    mentions: list[EntityMention],
) -> list[EntityMention]:
    context_by_document: dict[int, str] = {}
    for context in contexts:
        context_by_document[context.document_id] = (
            context_by_document.get(context.document_id, "")
            + " "
            + _normalize(context.combined_context)
        )
    result = []
    for mention in mentions:
        text = _normalize(mention.text)
        normalized = _normalize(mention.normalized_text)
        context_text = context_by_document.get(mention.document_id, "")
        if (text and text in context_text) or (normalized and normalized in context_text):
            result.append(mention)
    return result


def _eligible_entity_contexts(
    contexts: Iterable[EntityContext],
) -> list[EntityContext]:
    return [
        context
        for context in contexts
        if context.match_status == "manual"
        or (
            context.provider == "wikipedia_ko"
            and context.match_status == "matched"
        )
    ]


def _fit_prompt(payload: dict[str, object], *, max_chars: int) -> str:
    prompt = build_user_prompt(payload)
    while len(SYSTEM_INSTRUCTION) + len(prompt) > max_chars:
        evidence = payload["evidence"]
        assert isinstance(evidence, dict)
        source_titles = evidence["source_titles"]
        external_contexts = evidence["matched_or_manual_contexts"]
        contexts = payload["contexts"]
        entities = payload["entities"]
        assert isinstance(source_titles, list)
        assert isinstance(external_contexts, list)
        assert isinstance(contexts, list)
        assert isinstance(entities, list)
        if len(source_titles) > 1:
            source_titles.pop()
        elif external_contexts:
            external_contexts.pop()
        elif len(contexts) > 1:
            contexts.pop()
        elif len(entities) > 1:
            entities.pop()
        elif not _shrink_strings(payload):
            raise TravelEvidenceInputTooLargeError(
                "Gemini evidence package exceeds TRAVEL_GEMINI_MAX_INPUT_CHARS."
            )
        prompt = build_user_prompt(payload)
    return prompt


def _evidence_refs_from_payload(payload: dict[str, object]) -> set[str]:
    refs = {"RANKING-V2"}
    evidence = payload.get("evidence")
    collections = [payload.get("contexts"), payload.get("entities")]
    if isinstance(evidence, dict):
        collections.extend(
            (
                evidence.get("source_titles"),
                evidence.get("matched_or_manual_contexts"),
            )
        )
    for collection in collections:
        if not isinstance(collection, list):
            continue
        for row in collection:
            if isinstance(row, dict) and row.get("evidence_id"):
                refs.add(str(row["evidence_id"]))
    return refs


def _allowed_names_from_payload(payload: dict[str, object]) -> set[str]:
    names: set[str] = set()
    entities = payload.get("entities")
    if isinstance(entities, list):
        for row in entities:
            if not isinstance(row, dict):
                continue
            if (
                row.get("origin") != "trend_entity_link"
                and row.get("entity_type") not in {"LOCATION", "PLACE"}
            ):
                continue
            names.update(
                str(row[key])
                for key in ("name", "normalized_name")
                if row.get(key)
            )
    evidence = payload.get("evidence")
    if isinstance(evidence, dict):
        contexts = evidence.get("matched_or_manual_contexts")
        if isinstance(contexts, list):
            for row in contexts:
                if not isinstance(row, dict):
                    continue
                names.update(
                    str(row[key])
                    for key in ("entity_name", "page_title")
                    if row.get(key)
                )
    return {name for name in names if name.strip()}


def _shrink_strings(value: object) -> bool:
    changed = False
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and len(item) > 160:
                value[key] = item[:157] + "..."
                changed = True
            else:
                changed = _shrink_strings(item) or changed
    elif isinstance(value, list):
        for item in value:
            changed = _shrink_strings(item) or changed
    return changed


def _trim(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _normalize(value: str) -> str:
    return "".join(character.casefold() for character in value if character.isalnum())


def _json_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return []
    return [str(item) for item in value] if isinstance(value, list) else []

import html
import re
from dataclasses import replace

from app.ner.entity_labels import ENTITY_SPECIFICITY, EntityCandidate


EXTRACTOR_PRIORITY = {"gliner": 1, "merged": 2, "rule": 3, "dictionary": 4}
QUOTE_CHARS = "\"'‘’“”〈〉《》「」『』"


def normalize_entity_text(text: str, canonical_text: str | None = None) -> str | None:
    value = canonical_text if canonical_text else text
    value = html.unescape(value).strip()
    value = value.lstrip("#").strip()
    value = value.strip(QUOTE_CHARS).strip()
    value = re.sub(r"\s+", " ", value)
    value = value.lower()
    if len(value) <= 1 or value.isdigit():
        return None
    return value


def resolve_entities(candidates: list[EntityCandidate]) -> list[EntityCandidate]:
    normalized: list[EntityCandidate] = []
    for candidate in candidates:
        normalized_text = normalize_entity_text(candidate.text, candidate.canonical_text)
        if normalized_text is None:
            continue
        normalized.append(replace(candidate, normalized_text=normalized_text))

    grouped: dict[tuple[int | None, int | None, str], list[EntityCandidate]] = {}
    for candidate in normalized:
        key = (candidate.start_char, candidate.end_char, candidate.normalized_text or "")
        grouped.setdefault(key, []).append(candidate)

    resolved: list[EntityCandidate] = []
    for group in grouped.values():
        winner = max(group, key=_candidate_rank)
        same_type = [item for item in group if item.entity_type == winner.entity_type]
        if len({item.extractor for item in same_type}) > 1:
            winner = replace(
                winner,
                confidence=max(item.confidence for item in same_type),
                extractor="merged",
            )
        resolved.append(winner)

    deduplicated: list[EntityCandidate] = []
    for candidate in sorted(resolved, key=_sort_key):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(deduplicated)
                if _same_span(existing, candidate)
                or (
                    existing.normalized_text == candidate.normalized_text
                    and _spans_overlap(existing, candidate)
                )
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(candidate)
            continue
        if _candidate_rank(candidate) > _candidate_rank(deduplicated[duplicate_index]):
            deduplicated[duplicate_index] = candidate
    return sorted(deduplicated, key=_sort_key)


def _candidate_rank(candidate: EntityCandidate) -> tuple[float, int, int, int]:
    return (
        candidate.confidence,
        EXTRACTOR_PRIORITY.get(candidate.extractor, 0),
        ENTITY_SPECIFICITY[candidate.entity_type],
        len(candidate.text),
    )


def _spans_overlap(left: EntityCandidate, right: EntityCandidate) -> bool:
    if left.start_char is None or right.start_char is None:
        return True
    if left.end_char is None or right.end_char is None:
        return True
    return left.start_char < right.end_char and right.start_char < left.end_char


def _same_span(left: EntityCandidate, right: EntityCandidate) -> bool:
    return (
        left.start_char is not None
        and left.end_char is not None
        and left.start_char == right.start_char
        and left.end_char == right.end_char
    )


def _sort_key(candidate: EntityCandidate) -> tuple[int, int, str]:
    return (
        candidate.start_char if candidate.start_char is not None else 10**9,
        -(candidate.end_char or 0),
        candidate.normalized_text or "",
    )

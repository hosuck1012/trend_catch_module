from dataclasses import dataclass
import hashlib

from app.context_v2.sentence_splitter import find_keyword_occurrences, split_sentences_with_spans


@dataclass(frozen=True)
class ExtractedKeywordContext:
    keyword: str
    normalized_keyword: str
    previous_sentence: str | None
    matched_sentence: str
    next_sentence: str | None
    combined_context: str
    occurrence_index: int
    context_hash: str


def extract_keyword_contexts(
    *,
    text: str,
    keyword: str,
    normalized_keyword: str,
    sentences_before: int,
    sentences_after: int,
    max_chars: int,
) -> list[ExtractedKeywordContext]:
    spans = split_sentences_with_spans(text)
    if not spans:
        return []
    occurrences = find_keyword_occurrences(text, keyword)
    if not occurrences and normalized_keyword != keyword:
        occurrences = find_keyword_occurrences(text, normalized_keyword)
    contexts: list[ExtractedKeywordContext] = []
    seen_hashes: set[str] = set()
    for occurrence_index, (start, _end) in enumerate(occurrences):
        sentence_index = _sentence_index_for_position(spans, start)
        if sentence_index is None:
            continue
        previous_start = max(0, sentence_index - max(sentences_before, 0))
        next_end = min(len(spans), sentence_index + max(sentences_after, 0) + 1)
        previous = " ".join(span.text for span in spans[previous_start:sentence_index]) or None
        matched = spans[sentence_index].text
        next_sentence = " ".join(span.text for span in spans[sentence_index + 1:next_end]) or None
        combined = _limited_context(previous, matched, next_sentence, max_chars)
        context_hash = hash_context(normalized_keyword, combined)
        if context_hash in seen_hashes:
            continue
        seen_hashes.add(context_hash)
        contexts.append(
            ExtractedKeywordContext(
                keyword=keyword,
                normalized_keyword=normalized_keyword,
                previous_sentence=previous,
                matched_sentence=matched,
                next_sentence=next_sentence,
                combined_context=combined,
                occurrence_index=occurrence_index,
                context_hash=context_hash,
            )
        )
    return contexts


def hash_context(normalized_keyword: str, combined_context: str) -> str:
    payload = f"{normalized_keyword}\n{combined_context}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sentence_index_for_position(spans, position: int) -> int | None:
    for index, span in enumerate(spans):
        if span.start <= position < span.end:
            return index
    return None


def _limited_context(
    previous: str | None,
    matched: str,
    next_sentence: str | None,
    max_chars: int,
) -> str:
    parts = [part for part in (previous, matched, next_sentence) if part]
    combined = " ".join(parts).strip()
    if max_chars <= 0 or len(combined) <= max_chars:
        return combined
    if len(matched) >= max_chars:
        return matched[:max_chars].rstrip()
    remaining = max_chars - len(matched)
    before_budget = remaining // 2
    after_budget = remaining - before_budget
    before = (previous or "")[-before_budget:].strip() if before_budget else ""
    after = (next_sentence or "")[:after_budget].strip() if after_budget else ""
    return " ".join(part for part in (before, matched, after) if part)[:max_chars].rstrip()

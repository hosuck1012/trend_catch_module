from dataclasses import dataclass
import re

from app.keywords.keyword_normalizer import normalize_keyword


_SENTENCE_ENDINGS = {".", "?", "!", "。", "！", "？"}
_CLOSING_QUOTES = {'"', "'", "’", "”", "』", "」", "》", ")", "]"}
_NORMALIZED_MATCH_BOUNDARIES = {*_SENTENCE_ENDINGS, "\n", "\r"}


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int


def split_sentences_with_spans(text: str) -> list[SentenceSpan]:
    if not text:
        return []
    spans: list[SentenceSpan] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        boundary = char == "\n" or char in _SENTENCE_ENDINGS
        if boundary:
            end = index if char == "\n" else index + 1
            while end < length and text[end] in _CLOSING_QUOTES:
                end += 1
            _append_span(spans, text, start, end)
            start = end
            while start < length and text[start].isspace():
                start += 1
            index = start
            continue
        index += 1
    _append_span(spans, text, start, length)
    return spans


def split_sentences(text: str) -> list[str]:
    return [span.text for span in split_sentences_with_spans(text)]


def _append_span(spans: list[SentenceSpan], text: str, start: int, end: int) -> None:
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    spans.append(SentenceSpan(stripped, start + leading, start + trailing))


def find_keyword_occurrences(text: str, keyword: str) -> list[tuple[int, int]]:
    if not text or not keyword:
        return []
    pattern = re.escape(keyword)
    flags = re.IGNORECASE if keyword.isascii() else 0
    matches = [(match.start(), match.end()) for match in re.finditer(pattern, text, flags)]
    if matches:
        return matches
    compact = keyword.replace(" ", "")
    compact_pattern = re.escape(compact)
    compact_matches = [
        (match.start(), match.end())
        for match in re.finditer(compact_pattern, text, flags)
    ]
    if compact_matches:
        return compact_matches
    return _normalized_occurrences(text, keyword)


def _normalized_occurrences(text: str, keyword: str) -> list[tuple[int, int]]:
    normalized_keyword = normalize_keyword(keyword) or ""
    if not normalized_keyword:
        return []
    normalized_chars: list[str] = []
    source_positions: list[int] = []
    for index, char in enumerate(text):
        if char in _NORMALIZED_MATCH_BOUNDARIES:
            normalized_chars.append("\0")
            source_positions.append(index)
            continue
        normalized = normalize_keyword(char)
        if not normalized:
            continue
        normalized_chars.append(normalized)
        source_positions.extend([index] * len(normalized))
    normalized_text = "".join(normalized_chars)
    return [
        (source_positions[match.start()], source_positions[match.end() - 1] + 1)
        for match in re.finditer(re.escape(normalized_keyword), normalized_text)
    ]

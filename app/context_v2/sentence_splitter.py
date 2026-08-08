from dataclasses import dataclass
import re


_SENTENCE_ENDINGS = {".", "?", "!", "。", "！", "？"}
_CLOSING_QUOTES = {'"', "'", "’", "”", "』", "」", "》", ")", "]"}


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
    if matches or " " not in keyword:
        return matches
    compact_pattern = re.escape(keyword.replace(" ", ""))
    return [(match.start(), match.end()) for match in re.finditer(compact_pattern, text, flags)]

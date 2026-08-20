from dataclasses import dataclass
import re


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\n+")
WORD = re.compile(r"\S+")
MODEL_SAFE_MAX_CHARS = 800


@dataclass(frozen=True)
class TextChunk:
    text: str
    start_char: int
    kind: str


def build_ner_chunks(title: str, body: str, *, max_chars: int) -> list[TextChunk]:
    # GLiNER multi-v2.1 truncates at 384 model tokens. Korean news text in the
    # operating corpus reaches that limit well before 1,500 characters, so the
    # runtime setting is also bounded by a conservative model-safe ceiling.
    limit = max(100, min(max_chars, MODEL_SAFE_MAX_CHARS))
    chunks: list[TextChunk] = []
    if title:
        chunks.extend(_split_long_span(title, 0, limit, "title"))
    if not body:
        return chunks

    body_offset = len(title) + 1 if title else 0
    sentences = _sentence_spans(body)
    index = 0
    while index < len(sentences):
        sentence_start, sentence_end = sentences[index]
        if sentence_end - sentence_start > limit:
            chunks.extend(
                _split_long_span(
                    body[sentence_start:sentence_end],
                    body_offset + sentence_start,
                    limit,
                    "body",
                )
            )
            index += 1
            continue

        end_index = index
        chunk_end = sentence_end
        while end_index + 1 < len(sentences):
            next_end = sentences[end_index + 1][1]
            if next_end - sentence_start > limit:
                break
            end_index += 1
            chunk_end = next_end
        value = body[sentence_start:chunk_end].strip()
        if value:
            leading = len(body[sentence_start:chunk_end]) - len(
                body[sentence_start:chunk_end].lstrip()
            )
            chunks.append(
                TextChunk(value, body_offset + sentence_start + leading, "body")
            )
        if end_index >= len(sentences) - 1:
            break
        index = end_index if end_index > index else index + 1
    return chunks


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_BOUNDARY.finditer(text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _split_long_span(
    text: str,
    base_offset: int,
    limit: int,
    kind: str,
) -> list[TextChunk]:
    words = list(WORD.finditer(text))
    if not words:
        return []
    chunks: list[TextChunk] = []
    index = 0
    while index < len(words):
        start = words[index].start()
        end_index = index
        end = words[index].end()
        while end_index + 1 < len(words) and words[end_index + 1].end() - start <= limit:
            end_index += 1
            end = words[end_index].end()
        if end - start > limit:
            for offset in range(start, end, limit):
                value = text[offset : min(end, offset + limit)]
                chunks.append(TextChunk(value, base_offset + offset, kind))
        else:
            chunks.append(TextChunk(text[start:end], base_offset + start, kind))
        index = end_index + 1
    return chunks

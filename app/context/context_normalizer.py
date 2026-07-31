import html
import re
import unicodedata
from html.parser import HTMLParser
from urllib.parse import urlparse


REFERENCE_PATTERN = re.compile(r"\[(?:\d+|편집|출처\s*필요)\]")
WHITESPACE_PATTERN = re.compile(r"\s+")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() in {"script", "style"}:
            self._ignored_depth += 1
        elif tag.lower() in {"br", "p", "div", "li", "section", "article"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag.lower() in {"p", "div", "li", "section", "article"}:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def clean_plain_text(value: str, *, max_chars: int | None = None) -> str:
    parser = _PlainTextParser()
    parser.feed(value or "")
    parser.close()
    cleaned = html.unescape("".join(parser.parts))
    cleaned = REFERENCE_PATTERN.sub(" ", cleaned)
    cleaned = WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    if max_chars is not None:
        cleaned = cleaned[:max_chars].rstrip()
    return cleaned


def normalize_context_text(value: str, *, strip_parenthetical: bool = False) -> str:
    cleaned = clean_plain_text(value)
    cleaned = unicodedata.normalize("NFKC", cleaned).lower().strip()
    if strip_parenthetical:
        cleaned = re.sub(r"\s*\([^)]*\)\s*$", "", cleaned)
    return "".join(TOKEN_PATTERN.findall(cleaned))


def context_tokens(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        cleaned = clean_plain_text(value)
        tokens.update(token.lower() for token in TOKEN_PATTERN.findall(cleaned) if len(token) > 1)
        normalized = normalize_context_text(cleaned, strip_parenthetical=True)
        if len(normalized) > 1:
            tokens.add(normalized)
    return tokens


def validate_http_url(value: str) -> str:
    cleaned = value.strip()
    parsed = urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("page_url은 유효한 http 또는 https URL이어야 합니다.")
    return cleaned

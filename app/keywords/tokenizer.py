from dataclasses import dataclass
from typing import Protocol
import re


@dataclass(frozen=True)
class Token:
    text: str
    tag: str
    start: int
    end: int


class Tokenizer(Protocol):
    def tokenize(self, text: str) -> list[Token]: ...


class KiwiTokenizerAdapter:
    """Lazy Kiwi adapter. Constructing the service never loads tokenizer data."""

    def __init__(self, kiwi=None) -> None:
        self._kiwi = kiwi

    def tokenize(self, text: str) -> list[Token]:
        if self._kiwi is None:
            from kiwipiepy import Kiwi

            self._kiwi = Kiwi()
        result: list[Token] = []
        for item in self._kiwi.tokenize(text):
            start = int(item.start)
            result.append(Token(item.form, item.tag, start, start + int(item.len)))
        return result


_FALLBACK_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*|[가-힣]+|\d+")
_KOREAN_ENDINGS = (
    "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "이며", "하고",
    "가", "이", "은", "는", "을", "를", "에", "의", "와", "과", "도", "만", "로",
)


class RegexFallbackTokenizer:
    def tokenize(self, text: str) -> list[Token]:
        tokens: list[Token] = []
        for match in _FALLBACK_PATTERN.finditer(text):
            value = match.group(0)
            tag = "SL" if value[0].isascii() and value[0].isalpha() else "SN" if value.isdigit() else "NNG"
            if tag == "NNG":
                value = _strip_korean_ending(value)
            if value:
                tokens.append(Token(value, tag, match.start(), match.start() + len(value)))
        return tokens


def _strip_korean_ending(value: str) -> str:
    for ending in _KOREAN_ENDINGS:
        minimum_root = 1 if len(ending) > 1 else 3
        if (
            value not in {"두바이"}
            and value.endswith(ending)
            and len(value) - len(ending) >= minimum_root
        ):
            return value[: -len(ending)]
    return value


class ResilientTokenizer:
    def __init__(self, primary: Tokenizer, fallback: Tokenizer | None = None) -> None:
        self.primary = primary
        self.fallback = fallback or RegexFallbackTokenizer()

    def tokenize(self, text: str) -> list[Token]:
        try:
            return self.primary.tokenize(text)
        except (ImportError, OSError, RuntimeError):
            return self.fallback.tokenize(text)

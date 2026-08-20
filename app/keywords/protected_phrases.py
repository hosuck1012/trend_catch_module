from functools import lru_cache
import json
from pathlib import Path
import re

from app.keywords.keyword_normalizer import normalize_keyword


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "protected_phrases.json"
PROTECTABLE_ENTITY_TYPES = {"CONTENT_TITLE", "EVENT", "FOOD", "MEME"}
_QUOTED_PATTERNS = (
    re.compile(r'"([^"\n]{2,60})"'),
    re.compile(r"'([^'\n]{2,60})'"),
    re.compile(r"‘([^’\n]{2,60})’"),
    re.compile(r"“([^”\n]{2,60})”"),
    re.compile(r"「([^」\n]{2,60})」"),
    re.compile(r"『([^』\n]{2,60})』"),
)
_TITLE_CUE_VALUE = (
    r"영화|드라마|예능|연극|뮤지컬|전시(?!장)|공연(?!장)|"
    r"콘서트|축제|앨범|정규|신곡|신작|작품|개인전|특별전|"
    r"팝업|촬영지|촬영장"
)
_LEADING_TITLE_CUE = re.compile(rf"(?:{_TITLE_CUE_VALUE})\s*$")
_TRAILING_TITLE_CUE = re.compile(
    rf"^\s*(?:은|는|이|가|을|를|의)?\s*(?:{_TITLE_CUE_VALUE})"
)
_LONG_STATEMENT_END = re.compile(
    r"(?:했다|한다|된다|있다|없다|않다|겠다|습니다|입니다|합니다)$"
)


@lru_cache(maxsize=1)
def configured_phrases() -> tuple[str, ...]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    values = payload.get("phrases", payload) if isinstance(payload, dict) else payload
    return tuple(str(value).strip() for value in values if str(value).strip())


def matching_phrases(text: str) -> list[str]:
    normalized_text = normalize_keyword(text) or ""
    return [
        phrase
        for phrase in configured_phrases()
        if (normalize_keyword(phrase) or "") in normalized_text
    ]


def structural_phrases(title: str, body: str) -> list[str]:
    values: list[str] = []
    for text in (title, body):
        for pattern in _QUOTED_PATTERNS:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if _has_adjacent_title_cue(text, match, value):
                    values.append(value)
    deduplicated: dict[str, str] = {}
    for value in values:
        normalized = normalize_keyword(value) or ""
        if 2 <= len(normalized) <= 40:
            deduplicated.setdefault(normalized, value)
    return list(deduplicated.values())


def _has_adjacent_title_cue(text: str, match: re.Match[str], value: str) -> bool:
    normalized = normalize_keyword(value) or ""
    if len(normalized) < 3 or (len(normalized) >= 20 and _LONG_STATEMENT_END.search(normalized)):
        return False
    before = text[max(0, match.start() - 12) : match.start()]
    after = text[match.end() : min(len(text), match.end() + 12)]
    return bool(_LEADING_TITLE_CUE.search(before) or _TRAILING_TITLE_CUE.search(after))

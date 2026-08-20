import json
from functools import lru_cache
from pathlib import Path
import re

from app.keywords.keyword_normalizer import normalize_display, normalize_keyword
from app.keywords.phrase_extractor import is_noun
from app.keywords.phrase_signals import matching_phrase_suffix
from app.keywords.stopword_filter import is_generic
from app.keywords.tokenizer import KiwiTokenizerAdapter, ResilientTokenizer, Token, Tokenizer
from app.ner.entity_labels import EntityCandidate, EntityType


PLACE_SUFFIXES = (
    "해수욕장",
    "스타디움",
    "박물관",
    "미술관",
    "전망대",
    "경기장",
    "리조트",
    "공항",
    "공원",
    "궁전",
    "사찰",
    "성당",
    "카페",
    "호텔",
    "관광특구",
    "특구",
    "스파",
    "오름",
    "궁",
    "섬",
)
EVENT_SUFFIXES = (
    "불꽃축제",
    "페스티벌",
    "팝업스토어",
    "콘서트",
    "음악회",
    "마라톤",
    "박람회",
    "개인전",
    "특별전",
    "엑스포",
    "축제",
    "공연",
    "전시",
    "대회",
)
MEME_SUFFIXES = ("챌린지", "밤핑", "유행어", "열풍", "밈", "짤")
FOOD_SUFFIXES = ("초콜릿", "디저트", "커피", "음료", "라떼", "케이크", "빙수")
RULE_PARTICLES = ("에서", "으로", "은", "는", "이", "가", "을", "를", "에", "로", "와", "과", "도", "의")
GENERIC_PREFIXES = {
    "개",
    "회",
    "년",
    "올해",
    "지난해",
    "상반기",
    "하반기",
    "이번",
    "해당",
    "이어",
    "운영",
}
EVENT_COLLISIONS = {"전당대회"}
REPORTING_QUOTE_GLUE = {
    "고 밝혔다",
    "고 전했다",
    "라며",
    "이라며",
    "며",
    "이어",
}

PROTECTED_ENTITY_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "ner_protected_entities.json"
)
PROTECTED_PARTICLES = "은|는|이|가|을|를|에|에서|으로|로|와|과|도|의"

def extract_rule_entities(
    text: str,
    *,
    tokenizer: Tokenizer | None = None,
) -> list[EntityCandidate]:
    active_tokenizer = tokenizer or _default_tokenizer()
    tokens = active_tokenizer.tokenize(text)
    candidates: list[EntityCandidate] = []
    candidates.extend(
        _token_suffix_candidates(
            text, tokens, PLACE_SUFFIXES, EntityType.PLACE, 0.80, max_terms=4
        )
    )
    candidates.extend(
        _token_suffix_candidates(
            text, tokens, EVENT_SUFFIXES, EntityType.EVENT, 0.84, max_terms=6
        )
    )
    candidates.extend(
        _token_suffix_candidates(
            text, tokens, MEME_SUFFIXES, EntityType.MEME, 0.84, max_terms=4
        )
    )
    candidates.extend(
        _token_suffix_candidates(
            text, tokens, FOOD_SUFFIXES, EntityType.FOOD, 0.78, max_terms=3
        )
    )
    candidates.extend(_content_title_candidates(text))
    candidates.extend(_person_context_candidates(text, tokens))
    candidates.extend(_protected_entity_candidates(text))
    return candidates


@lru_cache(maxsize=1)
def _default_tokenizer() -> Tokenizer:
    return ResilientTokenizer(KiwiTokenizerAdapter())


def _token_suffix_candidates(
    text: str,
    tokens: list[Token],
    suffixes: tuple[str, ...],
    entity_type: EntityType,
    confidence: float,
    *,
    max_terms: int,
) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    for end_index, token in enumerate(tokens):
        if not (is_noun(token) or token.tag == "SN"):
            continue
        terminal_text, removed = _strip_rule_particle(token.text)
        suffix = _matching_allowed_suffix(terminal_text, suffixes)
        if not suffix:
            continue
        run_start = end_index
        while run_start > 0 and end_index - run_start + 1 < max_terms:
            previous = tokens[run_start - 1]
            gap = text[previous.end : tokens[run_start].start]
            if not (is_noun(previous) or previous.tag == "SN") or gap.strip():
                break
            run_start -= 1
        starts = {
            run_start,
            max(run_start, end_index - 2),
            max(run_start, end_index - 1),
            end_index,
        }
        for start_index in sorted(starts):
            window = tokens[start_index : end_index + 1]
            while len(window) > 1:
                first_normalized = normalize_keyword(window[0].text) or ""
                if not first_normalized or not _is_generic_prefix(
                    window[0].text, first_normalized
                ):
                    break
                window = window[1:]
            terminal_end = window[-1].end - removed
            value = normalize_display(text[window[0].start : terminal_end])
            normalized = normalize_keyword(value) or ""
            if (
                not value
                or not normalized
                or normalized == suffix
                or is_generic(value, normalized)
                or (
                    entity_type == EntityType.EVENT
                    and normalized in EVENT_COLLISIONS
                )
                or (
                    entity_type in {EntityType.EVENT, EntityType.MEME}
                    and not _has_topic_anchor(window, suffix)
                )
            ):
                continue
            candidates.append(
                EntityCandidate(
                    text=value,
                    entity_type=entity_type,
                    confidence=confidence,
                    extractor="rule",
                    start_char=window[0].start,
                    end_char=terminal_end,
                )
            )
    return candidates


def _matching_allowed_suffix(value: str, suffixes: tuple[str, ...]) -> str | None:
    normalized = normalize_keyword(value) or ""
    configured = matching_phrase_suffix(value)
    for suffix in sorted(suffixes, key=len, reverse=True):
        normalized_suffix = normalize_keyword(suffix) or ""
        if normalized.endswith(normalized_suffix):
            if normalized_suffix == "전시" and configured != normalized_suffix:
                continue
            return normalized_suffix
    return None


def _strip_rule_particle(value: str) -> tuple[str, int]:
    for particle in RULE_PARTICLES:
        if value.endswith(particle) and len(value) - len(particle) >= 2:
            return value[: -len(particle)], len(particle)
    return value, 0


def _is_generic_prefix(value: str, normalized: str) -> bool:
    return normalized in GENERIC_PREFIXES or is_generic(value, normalized)


def _has_topic_anchor(window: list[Token], suffix: str) -> bool:
    terminal = normalize_keyword(window[-1].text) or ""
    return (
        terminal != suffix
        or any(token.tag.startswith("NNP") or token.tag == "SL" for token in window)
        or any(
            left.end == right.start
            and left.tag != "SN"
            and not _is_generic_prefix(
                left.text, normalize_keyword(left.text) or ""
            )
            and not _is_generic_prefix(
                right.text, normalize_keyword(right.text) or ""
            )
            for left, right in zip(window, window[1:])
        )
    )


def _content_title_candidates(text: str) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    quoted_patterns = (
        re.compile(r'"([^"\n]{2,50})"'),
        re.compile(r"'([^'\n]{2,50})'"),
        re.compile(r"‘([^’\n]{2,50})’"),
        re.compile(r"“([^”\n]{2,50})”"),
        re.compile(r"〈([^〉\n]{2,50})〉"),
        re.compile(r"《([^》\n]{2,50})》"),
        re.compile(r"「([^」\n]{2,50})」"),
        re.compile(r"『([^』\n]{2,50})』"),
    )
    cue = r"영화|드라마|예능|노래|웹툰|책|작품|프로그램|전시|공연|촬영지"
    leading_cue = re.compile(rf"(?:{cue})\s*$")
    trailing_cue = re.compile(rf"^\s*(?:은|는|이|가|을|를|의)?\s*(?:{cue})")
    for pattern in quoted_patterns:
        for match in pattern.finditer(text):
            value = match.group(1).strip()
            before = text[max(0, match.start() - 16) : match.start()]
            after = text[match.end() : min(len(text), match.end() + 16)]
            if (
                not _valid_quoted_title(value)
                or not (leading_cue.search(before) or trailing_cue.search(after))
            ):
                continue
            candidates.append(
                EntityCandidate(
                    text=value,
                    entity_type=EntityType.CONTENT_TITLE,
                    confidence=0.8,
                    extractor="rule",
                    start_char=match.start(1),
                    end_char=match.end(1),
                )
            )
    filming_context = re.compile(
        r"(?<![0-9A-Za-z가-힣])([가-힣0-9A-Za-z]{2,15}(?:\s+[가-힣0-9A-Za-z]{2,15}){0,2})\s+촬영지"
    )
    for match in filming_context.finditer(text):
        candidates.append(
            EntityCandidate(
                text=match.group(1),
                entity_type=EntityType.CONTENT_TITLE,
                confidence=0.82,
                extractor="rule",
                start_char=match.start(1),
                end_char=match.end(1),
            )
        )
    return candidates


def _valid_quoted_title(value: str) -> bool:
    normalized = normalize_keyword(value) or ""
    if not normalized or normalized in REPORTING_QUOTE_GLUE:
        return False
    # Sentence punctuation followed by more prose indicates that a closing
    # quote was paired with the next opening quote. Title punctuation remains
    # valid when it is not followed by another phrase.
    if re.search(r"[.!?]\s+\S", value):
        return False
    return True


def _person_context_candidates(
    text: str,
    tokens: list[Token],
) -> list[EntityCandidate]:
    pattern = re.compile(
        r"(?<![0-9A-Za-z가-힣])([가-힣]{2,10})(?:가|이|은|는)\s+"
        r"(?=[^.!?]{0,40}(?:콘서트|공연|팬미팅))"
    )
    proper_spans = {
        (token.start, token.end)
        for token in tokens
        if token.tag.startswith("NNP")
    }
    candidates: list[EntityCandidate] = []
    for match in pattern.finditer(text):
        span = (match.start(1), match.end(1))
        if span not in proper_spans:
            continue
        candidates.append(
            EntityCandidate(
                text=match.group(1),
                entity_type=EntityType.PERSON,
                confidence=0.84,
                extractor="rule",
                start_char=match.start(1),
                end_char=match.end(1),
            )
        )
    return candidates


@lru_cache(maxsize=1)
def _protected_entity_entries() -> tuple[tuple[str, EntityType], ...]:
    payload = json.loads(PROTECTED_ENTITY_PATH.read_text(encoding="utf-8"))
    return tuple(
        (str(item["text"]).strip(), EntityType(str(item["type"])))
        for item in payload.get("entities", [])
        if str(item.get("text", "")).strip()
    )


def _protected_entity_candidates(text: str) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    for value, entity_type in _protected_entity_entries():
        terms = [term for term in re.split(r"\s+", value) if term]
        protected_body = r"(?:\s*[:：·-]?\s*)".join(
            re.escape(term) for term in terms
        )
        pattern = re.compile(
            rf"(?<![0-9A-Za-z가-힣]){protected_body}"
            rf"(?=$|[^0-9A-Za-z가-힣]|(?:{PROTECTED_PARTICLES})(?=$|[^0-9A-Za-z가-힣]))",
            re.IGNORECASE,
        )
        for match in pattern.finditer(text):
            candidates.append(
                EntityCandidate(
                    text=match.group(0),
                    entity_type=entity_type,
                    confidence=0.9,
                    extractor="rule",
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
    return candidates

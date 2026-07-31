import re

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
    "시장",
    "공원",
    "궁전",
    "사찰",
    "성당",
    "카페",
    "호텔",
    "오름",
    "역",
    "궁",
    "섬",
    "산",
)
EVENT_SUFFIXES = (
    "불꽃축제",
    "페스티벌",
    "콘서트",
    "마라톤",
    "박람회",
    "축제",
    "공연",
    "전시",
)
MEME_SUFFIXES = ("챌린지", "유행어", "열풍", "밈", "짤")
FOOD_SUFFIXES = ("초콜릿", "디저트", "커피", "음료", "라떼", "케이크", "빙수")

TOKEN = r"[0-9A-Za-z가-힣]"
PARTICLE_LOOKAHEAD = r"(?=$|[\s,.!?;:'\"()\[\]{}]|은|는|이|가|을|를|에|에서|으로|로|와|과|도|의)"


def extract_rule_entities(text: str) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    candidates.extend(
        _suffix_candidates(text, PLACE_SUFFIXES, EntityType.PLACE, 0.80, prefix_max=24)
    )
    candidates.extend(
        _suffix_candidates(text, EVENT_SUFFIXES, EntityType.EVENT, 0.82, prefix_max=24)
    )
    candidates.extend(_meme_candidates(text))
    candidates.extend(
        _suffix_candidates(text, FOOD_SUFFIXES, EntityType.FOOD, 0.74, prefix_max=12)
    )
    candidates.extend(_content_title_candidates(text))
    candidates.extend(_person_context_candidates(text))
    return candidates


def _suffix_candidates(
    text: str,
    suffixes: tuple[str, ...],
    entity_type: EntityType,
    confidence: float,
    *,
    prefix_max: int,
) -> list[EntityCandidate]:
    suffix_pattern = "|".join(re.escape(suffix) for suffix in suffixes)
    pattern = re.compile(
        rf"(?<!{TOKEN}){TOKEN}{{0,{prefix_max}}}(?:{suffix_pattern}){PARTICLE_LOOKAHEAD}"
    )
    return [
        EntityCandidate(
            text=match.group(0),
            entity_type=entity_type,
            confidence=confidence,
            extractor="rule",
            start_char=match.start(),
            end_char=match.end(),
        )
        for match in pattern.finditer(text)
        if len(match.group(0)) >= 2
    ]


def _meme_candidates(text: str) -> list[EntityCandidate]:
    suffix_pattern = "|".join(re.escape(suffix) for suffix in MEME_SUFFIXES)
    pattern = re.compile(
        rf"(?<!{TOKEN}){TOKEN}{{2,20}}(?:\s+{TOKEN}{{2,20}})?\s*(?:{suffix_pattern})"
        rf"{PARTICLE_LOOKAHEAD}"
    )
    return [
        EntityCandidate(
            text=match.group(0),
            entity_type=EntityType.MEME,
            confidence=0.83,
            extractor="rule",
            start_char=match.start(),
            end_char=match.end(),
        )
        for match in pattern.finditer(text)
    ]


def _content_title_candidates(text: str) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    quoted = re.compile(r"[\"'‘’“”〈《「『](.{2,50}?)[\"'‘’“”〉》」』]")
    context_words = ("영화", "드라마", "예능", "노래", "웹툰", "책", "촬영지")
    for match in quoted.finditer(text):
        nearby = text[max(0, match.start() - 20) : min(len(text), match.end() + 20)]
        if not any(word in nearby for word in context_words):
            continue
        candidates.append(
            EntityCandidate(
                text=match.group(1),
                entity_type=EntityType.CONTENT_TITLE,
                confidence=0.78,
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


def _person_context_candidates(text: str) -> list[EntityCandidate]:
    pattern = re.compile(
        r"(?<![0-9A-Za-z가-힣])([가-힣]{2,10})(?:가|이|은|는)\s+"
        r"(?=[^.!?]{0,40}(?:콘서트|공연|팬미팅))"
    )
    return [
        EntityCandidate(
            text=match.group(1),
            entity_type=EntityType.PERSON,
            confidence=0.84,
            extractor="rule",
            start_char=match.start(1),
            end_char=match.end(1),
        )
        for match in pattern.finditer(text)
    ]

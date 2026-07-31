import re
from dataclasses import dataclass

from app.models.source_document import SourceDocument
from app.services.keyword_normalization_service import normalize_keyword


STOP_WORDS = {
    "오늘",
    "영상",
    "뉴스",
    "기사",
    "공개",
    "관련",
    "대한",
    "이번",
    "정말",
    "바로",
    "최근",
    "인기",
    "화제",
    "새로운",
    "합니다",
    "입니다",
    "그리고",
    "하지만",
    "통해",
    "모습",
    "내용",
}

MAX_KEYWORD_LENGTH = 30
MAX_CANDIDATES_PER_DOCUMENT = 100
HASHTAG_PATTERN = re.compile(r"#[0-9A-Za-z가-힣_-]+")
TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")


@dataclass(frozen=True)
class ExtractedKeyword:
    keyword: str
    normalized_keyword: str


def extract_keywords_from_document(document: SourceDocument) -> list[ExtractedKeyword]:
    return extract_keywords_from_text(f"{document.title}\n{document.text}")


def extract_keywords_from_text(text: str) -> list[ExtractedKeyword]:
    candidates: list[str] = []
    candidates.extend(match.group(0) for match in HASHTAG_PATTERN.finditer(text))

    tokens = [match.group(0) for match in TOKEN_PATTERN.finditer(text)]
    candidates.extend(token for token in tokens if 2 <= len(token) <= MAX_KEYWORD_LENGTH)

    for size in (2, 3):
        for index in range(0, max(len(tokens) - size + 1, 0)):
            phrase_tokens = tokens[index : index + size]
            if any(token.isdigit() for token in phrase_tokens):
                continue
            candidates.append(" ".join(phrase_tokens))
            if len(candidates) >= MAX_CANDIDATES_PER_DOCUMENT * 2:
                break
        if len(candidates) >= MAX_CANDIDATES_PER_DOCUMENT * 2:
            break

    extracted: list[ExtractedKeyword] = []
    seen_normalized: set[str] = set()
    for candidate in candidates:
        normalized = normalize_keyword(candidate)
        if not _is_valid_normalized_keyword(normalized):
            continue
        if len(normalized) > MAX_KEYWORD_LENGTH:
            continue
        if normalized in seen_normalized:
            continue
        extracted.append(ExtractedKeyword(keyword=candidate, normalized_keyword=normalized))
        seen_normalized.add(normalized)
        if len(extracted) >= MAX_CANDIDATES_PER_DOCUMENT:
            break

    return extracted


def _is_valid_normalized_keyword(normalized: str | None) -> bool:
    if normalized is None:
        return False
    if normalized in STOP_WORDS:
        return False
    if normalized.isdigit():
        return False
    return True

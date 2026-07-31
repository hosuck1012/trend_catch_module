import re


_VALID_KEYWORD_PATTERN = re.compile(r"[^0-9A-Za-z가-힣]+")


def normalize_keyword(keyword: str) -> str | None:
    normalized = keyword.strip().lower()
    normalized = normalized.replace("#", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("-", "")
    normalized = normalized.replace("_", "")
    normalized = _VALID_KEYWORD_PATTERN.sub("", normalized)

    if len(normalized) < 2:
        return None

    return normalized

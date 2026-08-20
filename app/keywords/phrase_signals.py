from functools import lru_cache
import json
from pathlib import Path

from app.keywords.keyword_normalizer import normalize_keyword


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "keyword_phrase_suffixes.json"


def phrase_suffix_categories() -> dict[str, tuple[str, ...]]:
    return _load_categories(DATA_PATH.stat().st_mtime_ns)


@lru_cache(maxsize=4)
def _load_categories(_modified_ns: int) -> dict[str, tuple[str, ...]]:
    payload = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    categories = payload.get("categories", {})
    return {
        str(category): tuple(str(value).strip() for value in values if str(value).strip())
        for category, values in categories.items()
    }


def phrase_suffixes() -> tuple[str, ...]:
    values = {
        normalize_keyword(value) or ""
        for suffixes in phrase_suffix_categories().values()
        for value in suffixes
    }
    return tuple(sorted((value for value in values if value), key=lambda value: (-len(value), value)))


def matching_phrase_suffix(value: str) -> str | None:
    normalized = normalize_keyword(value) or ""
    return next(
        (
            suffix
            for suffix in phrase_suffixes()
            if normalized.endswith(suffix)
            and not (suffix == "전시" and normalized.endswith("에이전시"))
        ),
        None,
    )


def is_standalone_phrase_suffix(value: str) -> bool:
    normalized = normalize_keyword(value) or ""
    return bool(normalized and normalized in phrase_suffixes())


def phrase_specificity_signal(candidate_type: str, value: str) -> int:
    if candidate_type == "protected_phrase":
        return 3
    if candidate_type != "specific_phrase":
        return 0
    normalized = normalize_keyword(value) or ""
    suffix = matching_phrase_suffix(value)
    if not suffix or normalized == suffix:
        return 0
    score = 2
    if " " in value:
        score += 1
    if any(char.isdigit() for char in value):
        score += 1
    if any(char.isascii() and char.isupper() for char in value):
        score += 1
    return score

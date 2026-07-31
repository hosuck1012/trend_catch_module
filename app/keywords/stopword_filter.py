from functools import lru_cache
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def load_word_set(filename: str) -> frozenset[str]:
    path = DATA_DIR / filename
    return _load_word_set(filename, path.stat().st_mtime_ns)


@lru_cache(maxsize=None)
def _load_word_set(filename: str, _modified_ns: int) -> frozenset[str]:
    path = DATA_DIR / filename
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def rejection_reason(candidate: str, normalized: str) -> str | None:
    lowered = candidate.strip().lower()
    if normalized in load_word_set("url_artifacts.txt") or lowered in load_word_set(
        "url_artifacts.txt"
    ):
        return "url_artifact"
    if lowered in load_word_set("stopwords_en.txt"):
        return "english_stopword"
    if normalized in load_word_set("stopwords_ko.txt"):
        return "korean_stopword"
    if normalized in load_word_set("generic_travel_words.txt"):
        return "generic_word"
    return None


def is_generic(candidate: str, normalized: str) -> bool:
    return rejection_reason(candidate, normalized) in {
        "korean_stopword",
        "english_stopword",
        "generic_word",
    }

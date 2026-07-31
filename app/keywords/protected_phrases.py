from functools import lru_cache
import json
from pathlib import Path

from app.keywords.keyword_normalizer import normalize_keyword


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "protected_phrases.json"
PROTECTABLE_ENTITY_TYPES = {"CONTENT_TITLE", "EVENT", "FOOD", "MEME"}


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

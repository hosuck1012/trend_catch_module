import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.ner.entity_labels import EntityCandidate, EntityType


PARTICLES = "은|는|이|가|을|를|에|에서|으로|로|와|과|도|의|만|까지|부터|에게|께서"
BOUNDARY_CHARS = r"\s,.!?;:'\"()\[\]{}<>/\\·-"


@dataclass(frozen=True)
class DictionaryEntry:
    canonical: str
    entity_type: EntityType
    aliases: tuple[str, ...]


@lru_cache(maxsize=1)
def load_location_dictionary() -> tuple[DictionaryEntry, ...]:
    path = Path(__file__).resolve().parents[2] / "data" / "korean_locations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: list[DictionaryEntry] = []
    for canonical, values in payload.items():
        aliases = tuple(dict.fromkeys((canonical, *values.get("aliases", []))))
        entries.append(
            DictionaryEntry(
                canonical=canonical,
                entity_type=EntityType(values.get("type", "LOCATION")),
                aliases=aliases,
            )
        )
    return tuple(entries)


def extract_dictionary_entities(text: str) -> list[EntityCandidate]:
    candidates: list[EntityCandidate] = []
    for entry in load_location_dictionary():
        for alias in sorted(entry.aliases, key=len, reverse=True):
            pattern = re.compile(
                rf"(?<![0-9A-Za-z가-힣]){re.escape(alias)}"
                rf"(?=$|[{BOUNDARY_CHARS}]|(?:{PARTICLES})(?=$|[{BOUNDARY_CHARS}]))"
            )
            for match in pattern.finditer(text):
                candidates.append(
                    EntityCandidate(
                        text=match.group(0),
                        canonical_text=entry.canonical,
                        entity_type=entry.entity_type,
                        confidence=0.95,
                        extractor="dictionary",
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )
    return candidates


def canonical_location_for(value: str) -> str | None:
    normalized = value.strip().lower()
    for entry in load_location_dictionary():
        if any(alias.lower() == normalized for alias in entry.aliases):
            return entry.canonical
    return None

from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    LOCATION = "LOCATION"
    PLACE = "PLACE"
    PERSON = "PERSON"
    CONTENT_TITLE = "CONTENT_TITLE"
    EVENT = "EVENT"
    FOOD = "FOOD"
    BRAND = "BRAND"
    MEME = "MEME"


NER_PIPELINE_VERSION = "v2"
NER_LABEL_VERSION = "travel-v2"


ENTITY_LABEL_DESCRIPTIONS: dict[EntityType, str] = {
    EntityType.LOCATION: "city, province, country, district or geographic region",
    EntityType.PLACE: (
        "tourist attraction, landmark, station, airport, beach, park or venue"
    ),
    EntityType.PERSON: "celebrity, actor, singer or public figure",
    EntityType.CONTENT_TITLE: (
        "movie, drama, television show, song, book, exhibition or program title"
    ),
    EntityType.EVENT: (
        "named festival, fair, expo, exhibition, concert, sports event or marathon"
    ),
    EntityType.FOOD: "specific food, dish, dessert, beverage or food trend",
    EntityType.BRAND: "company, organization, product or brand",
    EntityType.MEME: "online meme, viral phrase, social media challenge or trend phrase",
}

GLINER_LABEL_TO_ENTITY_TYPE = {
    description: entity_type
    for entity_type, description in ENTITY_LABEL_DESCRIPTIONS.items()
}

ENTITY_SPECIFICITY = {
    EntityType.LOCATION: 1,
    EntityType.PERSON: 2,
    EntityType.BRAND: 3,
    EntityType.PLACE: 4,
    EntityType.EVENT: 5,
    EntityType.FOOD: 6,
    EntityType.CONTENT_TITLE: 7,
    EntityType.MEME: 8,
}


@dataclass(frozen=True)
class EntityCandidate:
    text: str
    entity_type: EntityType
    confidence: float
    extractor: str
    start_char: int | None = None
    end_char: int | None = None
    canonical_text: str | None = None
    normalized_text: str | None = None

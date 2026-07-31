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


ENTITY_LABEL_DESCRIPTIONS: dict[EntityType, str] = {
    EntityType.LOCATION: "geographic location, country, province, city or district",
    EntityType.PLACE: (
        "tourist attraction, landmark, station, airport, beach, park, building or venue"
    ),
    EntityType.PERSON: "person, celebrity, actor, singer or public figure",
    EntityType.CONTENT_TITLE: "movie, drama, television show, song, webtoon or book title",
    EntityType.EVENT: "festival, concert, exhibition, sports event or public event",
    EntityType.FOOD: "food, dish, dessert, beverage or local specialty",
    EntityType.BRAND: "brand, company, product brand or organization",
    EntityType.MEME: "internet meme, viral phrase, challenge or online trend expression",
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

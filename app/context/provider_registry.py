from app.ner.entity_labels import EntityType


WIKIPEDIA_PROVIDER = "wikipedia_ko"
MANUAL_PROVIDERS = {"namuwiki_manual", "manual"}
ALLOWED_CONTEXT_PROVIDERS = {WIKIPEDIA_PROVIDER, *MANUAL_PROVIDERS}
ALLOWED_MATCH_STATUSES = {"matched", "ambiguous", "unmatched", "manual", "error"}

ENTITY_SEARCH_PRIORITY = {
    EntityType.LOCATION.value: 1,
    EntityType.PLACE.value: 2,
    EntityType.CONTENT_TITLE.value: 3,
    EntityType.EVENT.value: 4,
    EntityType.FOOD.value: 5,
    EntityType.PERSON.value: 6,
    EntityType.BRAND.value: 7,
    EntityType.MEME.value: 8,
}

CONTEXT_TRAVEL_PRIORITY = {
    EntityType.PLACE.value: 1,
    EntityType.LOCATION.value: 2,
    EntityType.EVENT.value: 3,
    EntityType.FOOD.value: 4,
    EntityType.CONTENT_TITLE.value: 5,
    EntityType.PERSON.value: 6,
    EntityType.BRAND.value: 7,
    EntityType.MEME.value: 8,
}

TRAVEL_SUITABILITY = {
    EntityType.PLACE.value: 100.0,
    EntityType.LOCATION.value: 100.0,
    EntityType.EVENT.value: 85.0,
    EntityType.FOOD.value: 80.0,
    EntityType.CONTENT_TITLE.value: 60.0,
    EntityType.PERSON.value: 35.0,
    EntityType.BRAND.value: 30.0,
    EntityType.MEME.value: 25.0,
}


def provider_source_score(provider: str, match_status: str) -> float:
    if provider == WIKIPEDIA_PROVIDER:
        return {
            "matched": 100.0,
            "ambiguous": 30.0,
            "unmatched": 0.0,
            "error": 0.0,
        }.get(match_status, 0.0)
    if provider == "namuwiki_manual":
        return 60.0
    if provider == "manual":
        return 50.0
    return 0.0

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DestinationSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    entity_type: Literal["LOCATION", "PLACE", "EVENT", "FOOD"]
    reason: str = Field(min_length=1, max_length=600)
    source_entity: str = Field(min_length=1, max_length=200)
    relation_score: float = Field(ge=0, le=100)
    context_available: bool


class ContentIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    format: Literal["article", "short_video", "social_post", "travel_course"]
    angle: str = Field(min_length=1, max_length=600)
    target_audience: str = Field(min_length=1, max_length=300)


class TrendExplanation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trend_summary: str = Field(min_length=1, max_length=1000)
    rising_reason: str = Field(min_length=1, max_length=1200)
    evidence_summary: list[Annotated[str, Field(min_length=1, max_length=600)]] = Field(max_length=5)
    travel_relevance_score: int = Field(ge=0, le=100)
    travel_relevance_level: Literal["high", "medium", "low", "none"]
    travel_relevance_reason: str = Field(min_length=1, max_length=1000)
    recommended_destinations: list[DestinationSuggestion] = Field(max_length=5)
    content_ideas: list[ContentIdea] = Field(max_length=5)
    cautions: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(max_length=5)
    evidence_refs: list[Annotated[str, Field(min_length=1, max_length=200)]] = Field(max_length=10)
    confidence_score: int = Field(ge=0, le=100)

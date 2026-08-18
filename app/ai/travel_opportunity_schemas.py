from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DestinationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    entity_type: Literal["LOCATION", "PLACE", "EVENT", "FOOD"]
    reason: str = Field(min_length=1, max_length=600)
    evidence_ref: str = Field(min_length=1, max_length=100)
    verified_from_input: bool


class TravelContentIdea(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    concept: str = Field(min_length=1, max_length=800)
    destination: str = Field(min_length=1, max_length=200)
    format: Literal["article", "short_video", "social_post", "travel_course"]
    target_audience: str = Field(min_length=1, max_length=300)
    why_it_works: str = Field(min_length=1, max_length=600)


class FinalTravelOpportunityAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(min_length=1, max_length=255)
    final_decision: Literal["accept", "review", "reject"]
    final_travel_score: int = Field(ge=0, le=100)
    trend_context_summary: str = Field(min_length=1, max_length=1000)
    why_now: str = Field(min_length=1, max_length=1000)
    travel_angle: str = Field(min_length=1, max_length=1000)
    destination_candidates: list[DestinationCandidate] = Field(max_length=3)
    content_ideas: list[TravelContentIdea] = Field(max_length=3)
    evidence_refs: list[Annotated[str, Field(min_length=1, max_length=100)]] = Field(
        max_length=20
    )
    needs_external_verification: bool
    verification_queries: list[
        Annotated[str, Field(min_length=1, max_length=300)]
    ] = Field(max_length=3)
    cautions: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        max_length=5
    )
    confidence_score: int = Field(ge=0, le=100)

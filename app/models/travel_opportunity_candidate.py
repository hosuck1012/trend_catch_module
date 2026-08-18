from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TravelOpportunityCandidate(Base):
    __tablename__ = "travel_opportunity_candidates"
    __table_args__ = (
        UniqueConstraint(
            "normalized_keyword",
            "week_start",
            "keyword_context_id",
            name="uq_travel_candidate_keyword_week_context",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    normalized_keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    keyword_context_id: Mapped[int] = mapped_column(
        ForeignKey("keyword_contexts.id", ondelete="CASCADE"),
        index=True,
    )
    primary_entity: Mapped[str | None] = mapped_column(String(500), nullable=True)
    primary_entity_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    travel_category: Mapped[str] = mapped_column(String(50), index=True)
    entity_prior_score: Mapped[float] = mapped_column(Float)
    positive_context_score: Mapped[float] = mapped_column(Float)
    negative_context_penalty: Mapped[float] = mapped_column(Float)
    trend_evidence_score: Mapped[float] = mapped_column(Float)
    source_diversity_score: Mapped[float] = mapped_column(Float)
    travel_pre_score: Mapped[float] = mapped_column(Float, index=True)
    prefilter_status: Mapped[str] = mapped_column(String(20), index=True)
    matched_positive_terms_json: Mapped[str] = mapped_column(Text, default="[]")
    matched_negative_terms_json: Mapped[str] = mapped_column(Text, default="[]")
    reasoning_codes_json: Mapped[str] = mapped_column(Text, default="[]")
    semantic_travel_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    semantic_status: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    semantic_positive_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    semantic_negative_category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    trend_strength_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    context_clarity_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    travel_convertibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_precision_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    evidence_gate: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    evidence_codes_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_document_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_source_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ranking_status: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    rank_in_week: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ranking_version: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    calculated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    cluster_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    cluster_representative: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    gemini_eligible: Mapped[bool | None] = mapped_column(Boolean, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    keyword_context: Mapped["KeywordContext"] = relationship()

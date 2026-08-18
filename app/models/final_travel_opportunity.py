from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class FinalTravelOpportunity(Base):
    __tablename__ = "final_travel_opportunities"
    __table_args__ = (
        UniqueConstraint(
            "normalized_keyword",
            "week_start",
            "gemini_model",
            "prompt_version",
            name="uq_final_travel_keyword_week_model_prompt",
        ),
        CheckConstraint(
            "final_decision IN ('accept', 'review', 'reject')",
            name="ck_final_travel_decision",
        ),
        CheckConstraint(
            "analysis_status IN ('completed', 'partial', 'cached', 'error')",
            name="ck_final_travel_analysis_status",
        ),
        CheckConstraint(
            "final_travel_score >= 0 AND final_travel_score <= 100",
            name="ck_final_travel_score",
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 100",
            name="ck_final_travel_confidence",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    normalized_keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    travel_opportunity_candidate_id: Mapped[int | None] = mapped_column(
        ForeignKey("travel_opportunity_candidates.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    gemini_model: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(100))
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    final_decision: Mapped[str] = mapped_column(String(20), index=True)
    final_travel_score: Mapped[float] = mapped_column(Float, index=True)
    trend_context_summary: Mapped[str] = mapped_column(Text)
    why_now: Mapped[str] = mapped_column(Text)
    travel_angle: Mapped[str] = mapped_column(Text)
    destinations_json: Mapped[str] = mapped_column(Text, default="[]")
    content_ideas_json: Mapped[str] = mapped_column(Text, default="[]")
    evidence_refs_json: Mapped[str] = mapped_column(Text, default="[]")
    verification_queries_json: Mapped[str] = mapped_column(Text, default="[]")
    cautions_json: Mapped[str] = mapped_column(Text, default="[]")
    needs_external_verification: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_score: Mapped[float] = mapped_column(Float)
    analysis_status: Mapped[str] = mapped_column(String(20), index=True)
    gemini_call_count: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit_count: Mapped[int] = mapped_column(Integer, default=0)
    input_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

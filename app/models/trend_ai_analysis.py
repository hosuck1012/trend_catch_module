from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TrendAIAnalysis(Base):
    __tablename__ = "trend_ai_analyses"
    __table_args__ = (
        UniqueConstraint(
            "normalized_keyword",
            "week_start",
            "model_name",
            "prompt_version",
            name="uq_trend_ai_keyword_week_model_prompt",
        ),
        CheckConstraint(
            "analysis_status IN ('pending', 'completed', 'partial', 'skipped', 'error')",
            name="ck_trend_ai_analysis_status",
        ),
        CheckConstraint(
            "travel_relevance_level IS NULL OR "
            "travel_relevance_level IN ('high', 'medium', 'low', 'none')",
            name="ck_trend_ai_travel_level",
        ),
        CheckConstraint(
            "travel_relevance_score IS NULL OR "
            "(travel_relevance_score >= 0 AND travel_relevance_score <= 100)",
            name="ck_trend_ai_travel_score",
        ),
        CheckConstraint(
            "confidence_score IS NULL OR "
            "(confidence_score >= 0 AND confidence_score <= 100)",
            name="ck_trend_ai_confidence_score",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    normalized_keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    model_name: Mapped[str] = mapped_column(String(255))
    prompt_version: Mapped[str] = mapped_column(String(100))
    analysis_status: Mapped[str] = mapped_column(String(30), index=True)
    trend_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    rising_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    travel_relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    travel_relevance_level: Mapped[str | None] = mapped_column(String(20), nullable=True)
    travel_relevance_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_destinations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_ideas_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    cautions_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_refs_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    raw_response_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

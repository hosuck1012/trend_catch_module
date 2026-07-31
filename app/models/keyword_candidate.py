from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class KeywordCandidate(Base):
    __tablename__ = "keyword_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"), index=True
    )
    candidate_text: Mapped[str] = mapped_column(String(255))
    normalized_candidate: Mapped[str] = mapped_column(String(255), index=True)
    candidate_type: Mapped[str] = mapped_column(String(30))
    extractor: Mapped[str] = mapped_column(String(30), index=True)
    quality_score: Mapped[float] = mapped_column(Float)
    accepted: Mapped[bool] = mapped_column(Boolean, index=True)
    rejection_reason: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)
    title_occurrence: Mapped[int] = mapped_column(Integer, default=0)
    body_occurrence: Mapped[int] = mapped_column(Integer, default=0)
    entity_type: Mapped[str | None] = mapped_column(String(50), nullable=True)
    entity_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    pipeline_version: Mapped[str] = mapped_column(String(20), index=True)

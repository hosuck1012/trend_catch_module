from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EntityExtractionState(Base):
    __tablename__ = "entity_extraction_states"

    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        primary_key=True,
    )
    input_hash: Mapped[str] = mapped_column(String(64), index=True)
    model_name: Mapped[str] = mapped_column(String(255))
    pipeline_version: Mapped[str] = mapped_column(String(50), index=True)
    label_version: Mapped[str] = mapped_column(String(50))
    model_succeeded: Mapped[bool] = mapped_column(Boolean)
    mention_count: Mapped[int] = mapped_column(Integer)
    processed_at: Mapped[datetime] = mapped_column(DateTime)

    document: Mapped["SourceDocument"] = relationship()

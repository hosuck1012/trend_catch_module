from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "normalized_text",
            "entity_type",
            "start_char",
            "end_char",
            name="uq_entity_mention_document_entity_span",
        ),
        CheckConstraint(
            "entity_type IN ('LOCATION', 'PLACE', 'PERSON', 'CONTENT_TITLE', "
            "'EVENT', 'FOOD', 'BRAND', 'MEME')",
            name="ck_entity_mention_type",
        ),
        CheckConstraint(
            "extractor IN ('gliner', 'dictionary', 'rule', 'merged')",
            name="ck_entity_mention_extractor",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_entity_mention_confidence",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        index=True,
    )
    text: Mapped[str] = mapped_column(String(500))
    normalized_text: Mapped[str] = mapped_column(String(500), index=True)
    entity_type: Mapped[str] = mapped_column(String(50), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    extractor: Mapped[str] = mapped_column(String(20))
    start_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_char: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source: Mapped[str] = mapped_column(String(50), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)

    document: Mapped["SourceDocument"] = relationship(back_populates="entity_mentions")

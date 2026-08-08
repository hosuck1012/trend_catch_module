from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class KeywordContext(Base):
    __tablename__ = "keyword_contexts"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "normalized_keyword",
            "context_hash",
            name="uq_keyword_context_document_keyword_hash",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
        index=True,
    )
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    normalized_keyword: Mapped[str] = mapped_column(String(255), index=True)
    previous_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    matched_sentence: Mapped[str] = mapped_column(Text)
    next_sentence: Mapped[str | None] = mapped_column(Text, nullable=True)
    combined_context: Mapped[str] = mapped_column(Text)
    occurrence_index: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(50), index=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    context_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    document: Mapped["SourceDocument"] = relationship()

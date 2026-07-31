from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class KeywordOccurrence(Base):
    __tablename__ = "keyword_occurrences"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "normalized_keyword",
            name="uq_keyword_occurrences_document_normalized",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("source_documents.id", ondelete="CASCADE"),
    )
    keyword: Mapped[str] = mapped_column(String(255))
    normalized_keyword: Mapped[str] = mapped_column(String(255), index=True)
    source: Mapped[str] = mapped_column(String(50), index=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime, index=True)

    document: Mapped["SourceDocument"] = relationship(back_populates="keyword_occurrences")

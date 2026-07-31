from datetime import datetime

from sqlalchemy import CheckConstraint, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EntityContext(Base):
    __tablename__ = "entity_contexts"
    __table_args__ = (
        UniqueConstraint(
            "normalized_entity",
            "entity_type",
            "provider",
            "page_url",
            name="uq_entity_context_entity_type_provider_url",
        ),
        CheckConstraint(
            "provider IN ('wikipedia_ko', 'namuwiki_manual', 'manual')",
            name="ck_entity_context_provider",
        ),
        CheckConstraint(
            "match_status IN ('matched', 'ambiguous', 'unmatched', 'manual', 'error')",
            name="ck_entity_context_match_status",
        ),
        CheckConstraint(
            "entity_type IN ('LOCATION', 'PLACE', 'PERSON', 'CONTENT_TITLE', "
            "'EVENT', 'FOOD', 'BRAND', 'MEME')",
            name="ck_entity_context_type",
        ),
        CheckConstraint(
            "match_score >= 0 AND match_score <= 1",
            name="ck_entity_context_match_score",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    normalized_entity: Mapped[str] = mapped_column(String(500), index=True)
    entity_text: Mapped[str] = mapped_column(String(500))
    entity_type: Mapped[str] = mapped_column(String(50), index=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    page_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    page_title: Mapped[str] = mapped_column(String(500))
    page_url: Mapped[str] = mapped_column(String(2000))
    summary: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    match_score: Mapped[float] = mapped_column(Float)
    match_status: Mapped[str] = mapped_column(String(30), index=True)
    source_language: Mapped[str] = mapped_column(String(20))
    license_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    attribution_text: Mapped[str | None] = mapped_column(String(500), nullable=True)
    revision_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    trend_links: Mapped[list["TrendContextLink"]] = relationship(
        back_populates="entity_context",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

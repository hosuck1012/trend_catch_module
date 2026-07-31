from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TrendContextLink(Base):
    __tablename__ = "trend_context_links"
    __table_args__ = (
        UniqueConstraint(
            "keyword",
            "week_start",
            "entity_context_id",
            name="uq_trend_context_keyword_week_context",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    entity_context_id: Mapped[int] = mapped_column(
        ForeignKey("entity_contexts.id", ondelete="CASCADE"),
        index=True,
    )
    normalized_entity: Mapped[str] = mapped_column(String(500), index=True)
    entity_type: Mapped[str] = mapped_column(String(50), index=True)
    context_score: Mapped[float] = mapped_column(Float)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    updated_at: Mapped[datetime] = mapped_column(DateTime)

    entity_context: Mapped["EntityContext"] = relationship(back_populates="trend_links")

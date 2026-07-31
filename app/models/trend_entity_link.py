from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TrendEntityLink(Base):
    __tablename__ = "trend_entity_links"
    __table_args__ = (
        UniqueConstraint(
            "keyword",
            "week_start",
            "normalized_entity",
            "entity_type",
            name="uq_trend_entity_keyword_week_entity_type",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    entity_text: Mapped[str] = mapped_column(String(500))
    normalized_entity: Mapped[str] = mapped_column(String(500), index=True)
    entity_type: Mapped[str] = mapped_column(String(50), index=True)
    mention_count: Mapped[int] = mapped_column(Integer)
    document_count: Mapped[int] = mapped_column(Integer)
    source_count: Mapped[int] = mapped_column(Integer)
    average_confidence: Mapped[float] = mapped_column(Float)
    relation_score: Mapped[float] = mapped_column(Float)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)
    calculated_at: Mapped[datetime] = mapped_column(DateTime)

from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SearchValidationResult(Base):
    __tablename__ = "search_validation_results"
    __table_args__ = (
        UniqueConstraint(
            "keyword",
            "week_start",
            name="uq_search_validation_keyword_week_start",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    google_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    naver_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    combined_score: Mapped[float] = mapped_column(Float)
    provider_count: Mapped[int] = mapped_column(Integer)
    coverage_score: Mapped[float] = mapped_column(Float)
    current_average: Mapped[float] = mapped_column(Float)
    previous_average: Mapped[float] = mapped_column(Float)
    growth_rate: Mapped[float] = mapped_column(Float)
    calculated_at: Mapped[datetime] = mapped_column(DateTime)

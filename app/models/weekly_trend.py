from datetime import date, datetime

from sqlalchemy import Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class WeeklyTrend(Base):
    __tablename__ = "weekly_trends"
    __table_args__ = (
        UniqueConstraint("keyword", "week_start", name="uq_weekly_trends_keyword_week_start"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    keyword: Mapped[str] = mapped_column(String(255), index=True)
    week_start: Mapped[date] = mapped_column(Date, index=True)
    week_end: Mapped[date] = mapped_column(Date)
    weekly_mentions: Mapped[int] = mapped_column(Integer)
    previous_weekly_mentions: Mapped[int] = mapped_column(Integer)
    active_days: Mapped[int] = mapped_column(Integer)
    source_count: Mapped[int] = mapped_column(Integer)
    growth_rate: Mapped[float] = mapped_column(Float)
    peak_day_share: Mapped[float] = mapped_column(Float)
    persistence_score: Mapped[float] = mapped_column(Float)
    diversity_score: Mapped[float] = mapped_column(Float)
    freshness_score: Mapped[float] = mapped_column(Float)
    volume_score: Mapped[float] = mapped_column(Float)
    growth_score: Mapped[float] = mapped_column(Float)
    search_interest_score: Mapped[float] = mapped_column(Float)
    one_day_spike_penalty: Mapped[float] = mapped_column(Float)
    spam_penalty: Mapped[float] = mapped_column(Float)
    final_score: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(50))
    calculated_at: Mapped[datetime] = mapped_column(DateTime)
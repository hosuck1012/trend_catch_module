from datetime import date, datetime

from sqlalchemy import CheckConstraint, Date, DateTime, Float, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class SearchInterestObservation(Base):
    __tablename__ = "search_interest_observations"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "normalized_keyword",
            "observed_date",
            "geo",
            name="uq_search_interest_provider_keyword_date_geo",
        ),
        CheckConstraint(
            "provider IN ('google_trends', 'naver_datalab', 'manual')",
            name="ck_search_interest_provider",
        ),
        CheckConstraint(
            "source_type IN ('csv', 'manual')",
            name="ck_search_interest_source_type",
        ),
        CheckConstraint(
            "interest_value >= 0 AND interest_value <= 100",
            name="ck_search_interest_value_range",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    provider: Mapped[str] = mapped_column(String(50), index=True)
    keyword: Mapped[str] = mapped_column(String(255))
    normalized_keyword: Mapped[str] = mapped_column(String(255), index=True)
    observed_date: Mapped[date] = mapped_column(Date, index=True)
    interest_value: Mapped[float] = mapped_column(Float)
    geo: Mapped[str] = mapped_column(String(50))
    source_type: Mapped[str] = mapped_column(String(20))
    imported_at: Mapped[datetime] = mapped_column(DateTime)

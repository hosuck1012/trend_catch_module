from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)


if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    import app.models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _ensure_sqlite_weekly_trend_columns()


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _ensure_sqlite_weekly_trend_columns() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "weekly_trends" not in inspector.get_table_names():
        return
    existing_columns = {column["name"] for column in inspector.get_columns("weekly_trends")}
    required_columns = {
        "freshness_score": "REAL NOT NULL DEFAULT 0",
        "volume_score": "REAL NOT NULL DEFAULT 0",
        "growth_score": "REAL NOT NULL DEFAULT 0",
        "search_interest_score": "REAL NOT NULL DEFAULT 50",
        "one_day_spike_penalty": "REAL NOT NULL DEFAULT 0",
        "spam_penalty": "REAL NOT NULL DEFAULT 0",
    }
    missing_columns = [
        (name, definition)
        for name, definition in required_columns.items()
        if name not in existing_columns
    ]
    if not missing_columns:
        return
    with engine.begin() as connection:
        for name, definition in missing_columns:
            connection.execute(text(f"ALTER TABLE weekly_trends ADD COLUMN {name} {definition}"))

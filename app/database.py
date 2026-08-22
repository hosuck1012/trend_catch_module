from collections.abc import Generator

from sqlalchemy import create_engine, event, inspect, select, text
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
    _ensure_sqlite_nullable_search_interest()
    _ensure_sqlite_keyword_occurrence_columns()
    _ensure_sqlite_travel_opportunity_columns()


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
        "trend_score": "REAL",
        "keyword_quality_score": "REAL",
        "search_interest_score": "REAL",
        "search_interest_available": "BOOLEAN NOT NULL DEFAULT 0",
        "search_provider_count": "INTEGER NOT NULL DEFAULT 0",
        "pipeline_version": "VARCHAR(20) NOT NULL DEFAULT 'legacy'",
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


def _ensure_sqlite_keyword_occurrence_columns() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "keyword_occurrences" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("keyword_occurrences")}
    required = {
        "keyword_quality_score": "REAL",
        "pipeline_version": "VARCHAR(20) NOT NULL DEFAULT 'legacy'",
    }
    with engine.begin() as connection:
        for name, definition in required.items():
            if name not in existing:
                connection.execute(
                    text(f"ALTER TABLE keyword_occurrences ADD COLUMN {name} {definition}")
                )


def _ensure_sqlite_nullable_search_interest() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    if "weekly_trends" not in inspector.get_table_names():
        return
    columns = inspector.get_columns("weekly_trends")
    search_column = next(
        (column for column in columns if column["name"] == "search_interest_score"),
        None,
    )
    if search_column is None or search_column.get("nullable", True):
        return

    from app.models.search_validation_result import SearchValidationResult
    from app.models.weekly_trend import WeeklyTrend

    column_names = [column["name"] for column in columns]
    quoted = ", ".join(f'"{name}"' for name in column_names)
    with engine.connect() as connection:
        rows = list(connection.execute(text(f"SELECT {quoted} FROM weekly_trends")).mappings())
        validations = {
            (row.keyword, str(row.week_start)): row.provider_count
            for row in connection.execute(
                select(
                    SearchValidationResult.keyword,
                    SearchValidationResult.week_start,
                    SearchValidationResult.provider_count,
                )
            )
        } if "search_validation_results" in inspector.get_table_names() else {}

    payloads = []
    model_columns = {column.name for column in WeeklyTrend.__table__.columns}
    for row in rows:
        values = {name: row[name] for name in column_names if name in model_columns}
        provider_count = validations.get((row["keyword"], str(row["week_start"])), 0)
        values["search_provider_count"] = provider_count
        values["search_interest_available"] = provider_count > 0
        if provider_count == 0:
            values["search_interest_score"] = None
        values.setdefault("trend_score", None)
        values.setdefault("keyword_quality_score", None)
        values.setdefault("pipeline_version", "legacy")
        payloads.append(values)

    with engine.begin() as connection:
        connection.execute(text("DROP TABLE weekly_trends"))
        WeeklyTrend.__table__.create(connection)
        if payloads:
            connection.execute(WeeklyTrend.__table__.insert(), payloads)


def _ensure_sqlite_travel_opportunity_columns() -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    inspector = inspect(engine)
    table_name = "travel_opportunity_candidates"
    if table_name not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns(table_name)}
    required = {
        "semantic_travel_score": "REAL",
        "rule_input_hash": "VARCHAR(64)",
        "rule_version": "VARCHAR(50)",
        "rule_calculated_at": "DATETIME",
        "semantic_status": "VARCHAR(30)",
        "embedding_model": "VARCHAR(255)",
        "semantic_positive_score": "REAL",
        "semantic_positive_category": "VARCHAR(50)",
        "semantic_negative_score": "REAL",
        "semantic_negative_category": "VARCHAR(50)",
        "embedding_input_hash": "VARCHAR(64)",
        "semantic_calculated_at": "DATETIME",
        "trend_strength_score": "REAL",
        "context_clarity_score": "REAL",
        "travel_convertibility_score": "REAL",
        "evidence_confidence_score": "REAL",
        "high_precision_score": "REAL",
        "evidence_gate": "VARCHAR(30)",
        "evidence_codes_json": "TEXT",
        "evidence_document_count": "INTEGER",
        "evidence_source_count": "INTEGER",
        "ranking_status": "VARCHAR(30)",
        "rank_in_week": "INTEGER",
        "ranking_version": "VARCHAR(50)",
        "calculated_at": "DATETIME",
        "cluster_id": "VARCHAR(64)",
        "cluster_representative": "BOOLEAN",
        "gemini_eligible": "BOOLEAN",
    }
    with engine.begin() as connection:
        for name, definition in required.items():
            if name not in existing:
                connection.execute(
                    text(f"ALTER TABLE {table_name} ADD COLUMN {name} {definition}")
                )

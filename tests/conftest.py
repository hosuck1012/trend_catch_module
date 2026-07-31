import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

TEST_DB_PATH = Path(__file__).parent / "test_trend_engine.sqlite3"
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH.resolve().as_posix()}"
os.environ["SCHEDULER_ENABLED"] = "false"
os.environ["RUN_COLLECTION_ON_STARTUP"] = "false"
os.environ["NER_ENABLED"] = "false"

from app.database import Base, SessionLocal, engine  # noqa: E402
from app.config import get_settings  # noqa: E402
import app.models  # noqa: E402,F401
from app.main import app  # noqa: E402


@pytest.fixture(autouse=True)
def clean_database() -> None:
    get_settings.cache_clear()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def db_session():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()

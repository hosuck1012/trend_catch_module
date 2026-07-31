import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    service_name: str = os.getenv("SERVICE_NAME", "trend-engine")
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./trend_engine.sqlite3")


@lru_cache
def get_settings() -> Settings:
    return Settings()

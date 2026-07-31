import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    service_name: str
    database_url: str
    youtube_api_key: str
    youtube_region_code: str
    youtube_max_results: int
    newsis_rss_feeds: str
    newsis_rss_timeout_seconds: int
    scheduler_enabled: bool
    collection_interval_hours: int
    weekly_calculation_day: str
    weekly_calculation_hour: int
    weekly_calculation_minute: int
    scheduler_timezone: str
    run_collection_on_startup: bool
    ner_enabled: bool
    ner_model_name: str
    ner_device: str
    ner_threshold: float
    ner_text_max_chars: int
    ner_batch_size: int
    ner_max_documents_per_run: int


@lru_cache
def get_settings() -> Settings:
    return Settings(
        service_name=os.getenv("SERVICE_NAME", "trend-engine"),
        database_url=os.getenv("DATABASE_URL", "sqlite:///./trend_engine.sqlite3"),
        youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
        youtube_region_code=os.getenv("YOUTUBE_REGION_CODE", "KR"),
        youtube_max_results=_int_from_env("YOUTUBE_MAX_RESULTS", 50),
        newsis_rss_feeds=os.getenv(
            "NEWSIS_RSS_FEEDS",
            "sokbo,culture,entertain,country,society,industry",
        ),
        newsis_rss_timeout_seconds=_int_from_env("NEWSIS_RSS_TIMEOUT_SECONDS", 15),
        scheduler_enabled=_bool_from_env("SCHEDULER_ENABLED", False),
        collection_interval_hours=_int_from_env("COLLECTION_INTERVAL_HOURS", 6),
        weekly_calculation_day=os.getenv("WEEKLY_CALCULATION_DAY", "mon").strip().lower(),
        weekly_calculation_hour=_int_from_env("WEEKLY_CALCULATION_HOUR", 8),
        weekly_calculation_minute=_int_from_env("WEEKLY_CALCULATION_MINUTE", 0),
        scheduler_timezone=os.getenv("SCHEDULER_TIMEZONE", "Asia/Seoul").strip(),
        run_collection_on_startup=_bool_from_env("RUN_COLLECTION_ON_STARTUP", False),
        ner_enabled=_bool_from_env("NER_ENABLED", True),
        ner_model_name=os.getenv("NER_MODEL_NAME", "urchade/gliner_multi-v2.1").strip(),
        ner_device=os.getenv("NER_DEVICE", "cpu").strip().lower(),
        ner_threshold=_float_from_env("NER_THRESHOLD", 0.45),
        ner_text_max_chars=_int_from_env("NER_TEXT_MAX_CHARS", 1500),
        ner_batch_size=_int_from_env("NER_BATCH_SIZE", 4),
        ner_max_documents_per_run=_int_from_env("NER_MAX_DOCUMENTS_PER_RUN", 100),
    )


def _int_from_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _bool_from_env(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _float_from_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        name = name.strip()
        if name and name not in os.environ:
            os.environ[name] = value.strip().strip('"').strip("'")


_load_env_file()

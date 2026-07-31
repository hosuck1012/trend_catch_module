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
    keyword_pipeline_version: str
    keyword_min_length_ko: int
    keyword_min_length_en: int
    keyword_max_length: int
    keyword_min_quality_score: float
    keyword_max_candidates_per_document: int
    keyword_title_weight: float
    keyword_ner_weight: float
    keyword_phrase_weight: float
    keyword_source_diversity_weight: float
    keyword_enable_kiwi: bool
    keyword_keep_legacy_results: bool
    wikipedia_enabled: bool
    wikipedia_language: str
    wikipedia_search_limit: int
    wikipedia_timeout_seconds: int
    wikipedia_summary_max_chars: int
    wikimedia_client_name: str
    wikimedia_client_version: str
    wikimedia_contact_url: str
    wikimedia_contact_email: str
    context_match_threshold: float
    context_max_entities_per_run: int
    gemini_enabled: bool
    gemini_api_key: str
    gemini_model: str
    gemini_timeout_seconds: int
    gemini_max_input_chars: int
    gemini_max_documents: int
    gemini_max_contexts: int
    gemini_temperature: float
    gemini_max_output_tokens: int
    gemini_analysis_cache_hours: int
    gemini_max_items_per_run: int


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
        ner_batch_size=_int_from_env("NER_BATCH_SIZE", 1),
        ner_max_documents_per_run=_int_from_env("NER_MAX_DOCUMENTS_PER_RUN", 100),
        keyword_pipeline_version=os.getenv("KEYWORD_PIPELINE_VERSION", "v2").strip(),
        keyword_min_length_ko=_int_from_env("KEYWORD_MIN_LENGTH_KO", 2),
        keyword_min_length_en=_int_from_env("KEYWORD_MIN_LENGTH_EN", 3),
        keyword_max_length=_int_from_env("KEYWORD_MAX_LENGTH", 40),
        keyword_min_quality_score=_float_from_env("KEYWORD_MIN_QUALITY_SCORE", 45),
        keyword_max_candidates_per_document=_int_from_env(
            "KEYWORD_MAX_CANDIDATES_PER_DOCUMENT", 30
        ),
        keyword_title_weight=_float_from_env("KEYWORD_TITLE_WEIGHT", 1.5),
        keyword_ner_weight=_float_from_env("KEYWORD_NER_WEIGHT", 1.8),
        keyword_phrase_weight=_float_from_env("KEYWORD_PHRASE_WEIGHT", 1.4),
        keyword_source_diversity_weight=_float_from_env(
            "KEYWORD_SOURCE_DIVERSITY_WEIGHT", 1.2
        ),
        keyword_enable_kiwi=_bool_from_env("KEYWORD_ENABLE_KIWI", True),
        keyword_keep_legacy_results=_bool_from_env(
            "KEYWORD_KEEP_LEGACY_RESULTS", False
        ),
        wikipedia_enabled=_bool_from_env("WIKIPEDIA_ENABLED", True),
        wikipedia_language=os.getenv("WIKIPEDIA_LANGUAGE", "ko").strip().lower(),
        wikipedia_search_limit=_int_from_env("WIKIPEDIA_SEARCH_LIMIT", 5),
        wikipedia_timeout_seconds=_int_from_env("WIKIPEDIA_TIMEOUT_SECONDS", 15),
        wikipedia_summary_max_chars=_int_from_env(
            "WIKIPEDIA_SUMMARY_MAX_CHARS", 1000
        ),
        wikimedia_client_name=os.getenv(
            "WIKIMEDIA_CLIENT_NAME", "TrendCatchModule"
        ).strip(),
        wikimedia_client_version=os.getenv(
            "WIKIMEDIA_CLIENT_VERSION", "0.1"
        ).strip(),
        wikimedia_contact_url=os.getenv("WIKIMEDIA_CONTACT_URL", "").strip(),
        wikimedia_contact_email=os.getenv("WIKIMEDIA_CONTACT_EMAIL", "").strip(),
        context_match_threshold=_float_from_env("CONTEXT_MATCH_THRESHOLD", 0.70),
        context_max_entities_per_run=_int_from_env(
            "CONTEXT_MAX_ENTITIES_PER_RUN", 30
        ),
        gemini_enabled=_bool_from_env("GEMINI_ENABLED", False),
        gemini_api_key=os.getenv("GEMINI_API_KEY", "").strip(),
        gemini_model=os.getenv("GEMINI_MODEL", "").strip(),
        gemini_timeout_seconds=_int_from_env("GEMINI_TIMEOUT_SECONDS", 30),
        gemini_max_input_chars=_int_from_env("GEMINI_MAX_INPUT_CHARS", 12000),
        gemini_max_documents=_int_from_env("GEMINI_MAX_DOCUMENTS", 8),
        gemini_max_contexts=_int_from_env("GEMINI_MAX_CONTEXTS", 5),
        gemini_temperature=_float_from_env("GEMINI_TEMPERATURE", 0.2),
        gemini_max_output_tokens=_int_from_env(
            "GEMINI_MAX_OUTPUT_TOKENS", 1800
        ),
        gemini_analysis_cache_hours=_int_from_env(
            "GEMINI_ANALYSIS_CACHE_HOURS", 24
        ),
        gemini_max_items_per_run=_int_from_env("GEMINI_MAX_ITEMS_PER_RUN", 10),
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

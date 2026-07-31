import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DashboardSettings:
    api_base_url: str
    request_timeout_seconds: int
    page_size: int


def get_dashboard_settings() -> DashboardSettings:
    _load_root_env()
    return DashboardSettings(
        api_base_url=os.getenv(
            "DASHBOARD_API_BASE_URL", "http://127.0.0.1:8000"
        ).rstrip("/"),
        request_timeout_seconds=_positive_int(
            os.getenv("DASHBOARD_REQUEST_TIMEOUT_SECONDS"), 20
        ),
        page_size=_positive_int(os.getenv("DASHBOARD_PAGE_SIZE"), 20),
    )


def _positive_int(raw: str | None, default: int) -> int:
    try:
        value = int(raw) if raw else default
    except ValueError:
        return default
    return value if value > 0 else default


def _load_root_env() -> None:
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

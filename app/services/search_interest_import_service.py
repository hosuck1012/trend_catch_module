import csv
import io
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone

from sqlalchemy.orm import Session

from app.repositories.search_interest_repository import ObservationValues, upsert_observation
from app.services.keyword_normalization_service import normalize_keyword


ALLOWED_SEARCH_PROVIDERS = {"google_trends", "naver_datalab", "manual"}
MAX_CSV_BYTES = 5 * 1024 * 1024
REQUIRED_CSV_COLUMNS = {"keyword", "date", "value"}


class SearchInterestImportError(ValueError):
    pass


@dataclass(frozen=True)
class ImportRowError:
    row: int
    reason: str


@dataclass(frozen=True)
class ManualObservationValue:
    observed_date: date
    interest_value: float


@dataclass(frozen=True)
class SearchInterestImportResult:
    provider: str
    received_rows: int
    inserted_rows: int
    updated_rows: int
    skipped_rows: int
    keywords: list[str]
    start_date: date | None
    end_date: date | None
    errors: list[ImportRowError]


def import_search_interest_csv(
    session: Session,
    *,
    provider: str,
    content: bytes,
    default_geo: str = "KR",
) -> SearchInterestImportResult:
    _validate_provider(provider)
    if len(content) > MAX_CSV_BYTES:
        raise SearchInterestImportError("CSV 파일은 5MB 이하여야 합니다.")
    try:
        decoded = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SearchInterestImportError("CSV 파일은 UTF-8 또는 UTF-8 BOM 형식이어야 합니다.") from exc

    reader = csv.DictReader(io.StringIO(decoded, newline=""))
    if reader.fieldnames is None:
        raise SearchInterestImportError("CSV 헤더가 없습니다.")
    normalized_headers = [header.strip().lower() for header in reader.fieldnames]
    missing_columns = REQUIRED_CSV_COLUMNS - set(normalized_headers)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise SearchInterestImportError(f"필수 CSV 컬럼이 없습니다: {missing}")
    reader.fieldnames = normalized_headers

    imported_at = _utc_now()
    received_rows = 0
    inserted_rows = 0
    updated_rows = 0
    skipped_rows = 0
    errors: list[ImportRowError] = []
    keywords: list[str] = []
    seen_keywords: set[str] = set()
    observed_dates: list[date] = []

    for row_number, row in enumerate(reader, start=2):
        received_rows += 1
        try:
            values = _csv_row_to_values(
                row,
                provider=provider,
                default_geo=default_geo,
                imported_at=imported_at,
            )
        except ValueError as exc:
            skipped_rows += 1
            errors.append(ImportRowError(row=row_number, reason=str(exc)))
            continue

        action = upsert_observation(session, values)
        if action == "inserted":
            inserted_rows += 1
        elif action == "updated":
            updated_rows += 1
        else:
            skipped_rows += 1
        if values.normalized_keyword not in seen_keywords:
            keywords.append(values.keyword)
            seen_keywords.add(values.normalized_keyword)
        observed_dates.append(values.observed_date)

    session.commit()
    return SearchInterestImportResult(
        provider=provider,
        received_rows=received_rows,
        inserted_rows=inserted_rows,
        updated_rows=updated_rows,
        skipped_rows=skipped_rows,
        keywords=keywords,
        start_date=min(observed_dates) if observed_dates else None,
        end_date=max(observed_dates) if observed_dates else None,
        errors=errors,
    )


def import_manual_observations(
    session: Session,
    *,
    provider: str,
    keyword: str,
    geo: str,
    observations: list[ManualObservationValue],
) -> SearchInterestImportResult:
    _validate_provider(provider)
    normalized_keyword = normalize_keyword(keyword)
    if normalized_keyword is None:
        raise SearchInterestImportError("유효한 keyword가 필요합니다.")
    clean_keyword = keyword.strip()
    try:
        clean_geo = _normalize_geo(geo, "KR")
    except ValueError as exc:
        raise SearchInterestImportError(str(exc)) from exc
    imported_at = _utc_now()
    inserted_rows = 0
    updated_rows = 0
    skipped_rows = 0
    dates: list[date] = []

    for observation in observations:
        interest_value = _validate_interest_value(observation.interest_value)
        action = upsert_observation(
            session,
            ObservationValues(
                provider=provider,
                keyword=clean_keyword,
                normalized_keyword=normalized_keyword,
                observed_date=observation.observed_date,
                interest_value=interest_value,
                geo=clean_geo,
                source_type="manual",
                imported_at=imported_at,
            ),
        )
        if action == "inserted":
            inserted_rows += 1
        elif action == "updated":
            updated_rows += 1
        else:
            skipped_rows += 1
        dates.append(observation.observed_date)

    session.commit()
    return SearchInterestImportResult(
        provider=provider,
        received_rows=len(observations),
        inserted_rows=inserted_rows,
        updated_rows=updated_rows,
        skipped_rows=skipped_rows,
        keywords=[clean_keyword],
        start_date=min(dates) if dates else None,
        end_date=max(dates) if dates else None,
        errors=[],
    )


def _csv_row_to_values(
    row: dict[str, str | None],
    *,
    provider: str,
    default_geo: str,
    imported_at: datetime,
) -> ObservationValues:
    keyword = (row.get("keyword") or "").strip()
    normalized_keyword = normalize_keyword(keyword)
    if normalized_keyword is None:
        raise ValueError("keyword가 비어 있거나 유효하지 않습니다.")
    raw_date = (row.get("date") or "").strip()
    try:
        observed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("date는 YYYY-MM-DD 형식이어야 합니다.") from exc
    raw_value = (row.get("value") or "").strip()
    try:
        interest_value = _validate_interest_value(float(raw_value))
    except ValueError as exc:
        raise ValueError("value는 0 이상 100 이하의 숫자여야 합니다.") from exc
    return ObservationValues(
        provider=provider,
        keyword=keyword,
        normalized_keyword=normalized_keyword,
        observed_date=observed_date,
        interest_value=interest_value,
        geo=_normalize_geo(row.get("geo"), default_geo),
        source_type="csv",
        imported_at=imported_at,
    )


def _validate_interest_value(value: float) -> float:
    if not math.isfinite(value) or value < 0 or value > 100:
        raise ValueError("value는 0 이상 100 이하의 숫자여야 합니다.")
    return float(value)


def _normalize_geo(value: str | None, default_geo: str) -> str:
    geo = (value or default_geo or "KR").strip().upper()
    if not geo or len(geo) > 50:
        raise ValueError("geo가 유효하지 않습니다.")
    return geo


def _validate_provider(provider: str) -> None:
    if provider not in ALLOWED_SEARCH_PROVIDERS:
        raise SearchInterestImportError(f"지원하지 않는 provider입니다: {provider}")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

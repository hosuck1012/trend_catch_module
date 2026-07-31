import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
from urllib.parse import quote, urlparse

import httpx

from app.config import get_settings
from app.context.context_normalizer import clean_plain_text


WIKIMEDIA_CONTACT_REQUIRED_MESSAGE = (
    "Wikimedia API 호출을 위해 프로젝트 URL 또는 연락 이메일이 필요합니다."
)
WIKIMEDIA_POLICY_REJECTION_MESSAGE = (
    "Wikimedia User-Agent 정책에 의해 요청이 거부되었습니다. "
    "WIKIMEDIA_CONTACT_URL 또는 WIKIMEDIA_CONTACT_EMAIL 설정을 확인하세요."
)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_MAX_ATTEMPTS = 3


class WikipediaClientError(RuntimeError):
    pass


class WikipediaRateLimitError(WikipediaClientError):
    pass


class WikimediaConfigurationError(WikipediaClientError):
    pass


class WikimediaPolicyError(WikipediaClientError):
    pass


def build_wikimedia_user_agent(
    *,
    client_name: str,
    client_version: str,
    contact_url: str = "",
    contact_email: str = "",
) -> str:
    name = _validate_user_agent_component(client_name, "WIKIMEDIA_CLIENT_NAME")
    version = _validate_user_agent_component(
        client_version,
        "WIKIMEDIA_CLIENT_VERSION",
    )
    url = contact_url.strip()
    email = contact_email.strip()
    if url:
        parsed = urlparse(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or _contains_control_characters(url)
            or any(character.isspace() for character in url)
        ):
            raise WikimediaConfigurationError(
                "WIKIMEDIA_CONTACT_URL은 유효한 http 또는 https URL이어야 합니다."
            )
        contact = url
    elif email:
        if not _EMAIL_PATTERN.fullmatch(email) or _contains_control_characters(email):
            raise WikimediaConfigurationError(
                "WIKIMEDIA_CONTACT_EMAIL 형식이 올바르지 않습니다."
            )
        contact = email
    else:
        raise WikimediaConfigurationError(WIKIMEDIA_CONTACT_REQUIRED_MESSAGE)
    return f"{name}/{version} ({contact})"


@dataclass(frozen=True)
class WikipediaSearchResult:
    page_id: str | None
    title: str
    page_url: str
    snippet: str
    redirect_title: str | None


@dataclass(frozen=True)
class WikipediaPageSummary:
    page_id: str | None
    title: str
    page_url: str
    extract: str
    description: str | None
    revision_id: str | None
    redirect_from: str | None


class WikipediaClient:
    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        settings = get_settings()
        self.language = settings.wikipedia_language or "ko"
        self.endpoint = f"https://{self.language}.wikipedia.org/w/api.php"
        self.search_limit = min(max(settings.wikipedia_search_limit, 1), 5)
        self.summary_max_chars = max(settings.wikipedia_summary_max_chars, 1)
        self._configuration_error: WikimediaConfigurationError | None = None
        try:
            self.user_agent = build_wikimedia_user_agent(
                client_name=settings.wikimedia_client_name,
                client_version=settings.wikimedia_client_version,
                contact_url=settings.wikimedia_contact_url,
                contact_email=settings.wikimedia_contact_email,
            )
        except WikimediaConfigurationError as exc:
            self.user_agent = None
            self._configuration_error = exc

        headers = self._request_headers() if self.user_agent else None
        self._client = client
        self._owns_client = client is None
        if self._client is None and headers is not None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(settings.wikipedia_timeout_seconds),
                headers=headers,
            )
        elif self._client is not None and headers is not None:
            self._client.headers.update(headers)
        self._semaphore = asyncio.Semaphore(3)

    async def __aenter__(self) -> "WikipediaClient":
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    def ensure_configured(self) -> None:
        if self._configuration_error is not None:
            raise self._configuration_error
        if self._client is None or self.user_agent is None:
            raise WikimediaConfigurationError(WIKIMEDIA_CONTACT_REQUIRED_MESSAGE)

    async def search(self, query: str) -> list[WikipediaSearchResult]:
        payload = await self._get(
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": self.search_limit,
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
            }
        )
        rows = payload.get("query", {}).get("search", [])
        results: list[WikipediaSearchResult] = []
        for row in rows:
            title = str(row.get("title", "")).strip()
            if not title:
                continue
            results.append(
                WikipediaSearchResult(
                    page_id=_optional_string(row.get("pageid")),
                    title=title,
                    page_url=self.page_url(title),
                    snippet=clean_plain_text(str(row.get("snippet", ""))),
                    redirect_title=(
                        str(row["redirecttitle"]).strip()
                        if row.get("redirecttitle")
                        else None
                    ),
                )
            )
        return results

    async def get_page_summary(self, title: str) -> WikipediaPageSummary | None:
        payload = await self._get(
            {
                "action": "query",
                "prop": "extracts|info|pageprops",
                "titles": title,
                "redirects": 1,
                "exintro": 1,
                "explaintext": 1,
                "inprop": "url",
                "format": "json",
                "formatversion": 2,
                "utf8": 1,
            }
        )
        query = payload.get("query", {})
        pages = query.get("pages", [])
        page = next((item for item in pages if not item.get("missing")), None)
        if page is None:
            return None
        page_title = str(page.get("title") or title).strip()
        pageprops = page.get("pageprops") or {}
        redirects = query.get("redirects") or []
        redirect_from = (
            str(redirects[0].get("from")).strip()
            if redirects and redirects[0].get("from")
            else None
        )
        description = pageprops.get("wikibase-shortdesc") or pageprops.get("description")
        return WikipediaPageSummary(
            page_id=_optional_string(page.get("pageid")),
            title=page_title,
            page_url=str(page.get("fullurl") or self.page_url(page_title)),
            extract=clean_plain_text(
                str(page.get("extract", "")),
                max_chars=self.summary_max_chars,
            ),
            description=(clean_plain_text(str(description)) if description else None),
            revision_id=_optional_string(page.get("lastrevid")),
            redirect_from=redirect_from,
        )

    def page_url(self, title: str) -> str:
        encoded_title = quote(title.replace(" ", "_"), safe="()")
        return f"https://{self.language}.wikipedia.org/wiki/{encoded_title}"

    def search_url(self, query: str) -> str:
        return f"https://{self.language}.wikipedia.org/w/index.php?search={quote(query)}"

    async def _get(self, params: dict[str, object]) -> dict[str, object]:
        self.ensure_configured()
        assert self._client is not None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                async with self._semaphore:
                    response = await self._client.get(self.endpoint, params=params)
            except httpx.TimeoutException as exc:
                if attempt == _MAX_ATTEMPTS - 1:
                    raise WikipediaClientError(
                        "Wikipedia API 요청 시간이 초과되었습니다."
                    ) from exc
                await asyncio.sleep(_retry_delay(attempt=attempt, retry_after=None))
                continue
            except httpx.HTTPError as exc:
                raise WikipediaClientError("Wikipedia API 연결에 실패했습니다.") from exc

            if response.status_code in _RETRYABLE_STATUS_CODES:
                if attempt == _MAX_ATTEMPTS - 1:
                    if response.status_code == 429:
                        raise WikipediaRateLimitError(
                            "Wikipedia API 요청 한도를 초과했습니다."
                        )
                    raise WikipediaClientError(
                        "Wikipedia API 응답을 처리할 수 없습니다: "
                        f"HTTP {response.status_code}"
                    )
                await asyncio.sleep(
                    _retry_delay(
                        attempt=attempt,
                        retry_after=response.headers.get("Retry-After"),
                    )
                )
                continue
            if response.status_code == 403:
                if _is_user_agent_policy_response(response.text):
                    raise WikimediaPolicyError(WIKIMEDIA_POLICY_REJECTION_MESSAGE)
                raise WikipediaClientError(
                    "Wikipedia API 응답을 처리할 수 없습니다: HTTP 403"
                )
            try:
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                raise WikipediaClientError(
                    f"Wikipedia API 응답을 처리할 수 없습니다: HTTP {response.status_code}"
                ) from exc
            if not isinstance(payload, dict):
                raise WikipediaClientError("Wikipedia API 응답 형식이 올바르지 않습니다.")
            if payload.get("error"):
                raise WikipediaClientError("Wikipedia API가 오류를 반환했습니다.")
            return payload
        raise WikipediaRateLimitError("Wikipedia API 요청 한도를 초과했습니다.")

    def _request_headers(self) -> dict[str, str]:
        assert self.user_agent is not None
        return {
            "User-Agent": self.user_agent,
            "Api-User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
        }


def _optional_string(value) -> str | None:
    return str(value) if value is not None else None


def _validate_user_agent_component(value: str, setting_name: str) -> str:
    cleaned = value.strip()
    if (
        not cleaned
        or _contains_control_characters(cleaned)
        or any(character.isspace() for character in cleaned)
        or any(character in cleaned for character in "/()")
    ):
        raise WikimediaConfigurationError(f"{setting_name} 설정이 올바르지 않습니다.")
    return cleaned


def _contains_control_characters(value: str) -> bool:
    return any(ord(character) < 32 or ord(character) == 127 for character in value)


def _is_user_agent_policy_response(response_text: str) -> bool:
    normalized = response_text.casefold()
    return any(
        marker in normalized
        for marker in ("user-agent", "robot policy", "contact information")
    )


def _retry_delay(*, attempt: int, retry_after: str | None) -> float:
    if retry_after:
        stripped = retry_after.strip()
        try:
            return min(max(float(stripped), 0.0), 60.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(stripped)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
                return min(max(seconds, 0.0), 60.0)
            except (TypeError, ValueError, OverflowError):
                pass
    return 0.1 * (2**attempt)

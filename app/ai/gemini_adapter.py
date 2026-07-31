import asyncio
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
import json

from pydantic import ValidationError

from app.ai.gemini_prompt import SYSTEM_INSTRUCTION
from app.ai.gemini_schemas import TrendExplanation
from app.config import get_settings


RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3


class GeminiAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        status_code: int | None = None,
        retries: int = 0,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.retries = retries


class GeminiConfigurationError(GeminiAdapterError):
    pass


class GeminiSchemaError(GeminiAdapterError):
    pass


class GeminiAdapter:
    def __init__(self, client_factory: Callable[[], object] | None = None) -> None:
        settings = get_settings()
        self.enabled = settings.gemini_enabled
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model
        self.timeout_seconds = max(settings.gemini_timeout_seconds, 1)
        self.temperature = min(max(settings.gemini_temperature, 0.0), 2.0)
        self.max_output_tokens = max(settings.gemini_max_output_tokens, 1)
        self.max_requests = max(settings.gemini_max_items_per_run, 1) * MAX_ATTEMPTS
        self._client_factory = client_factory
        self._client = None
        self.request_count = 0

    @property
    def client_loaded(self) -> bool:
        return self._client is not None

    def ensure_configured(self) -> None:
        if not self.enabled:
            raise GeminiConfigurationError(
                "Gemini 분석이 비활성화되어 있습니다.",
                code="disabled",
            )
        if not self.api_key:
            raise GeminiConfigurationError(
                "GEMINI_API_KEY 설정이 필요합니다.",
                code="api_key_missing",
            )
        if not self.model_name:
            raise GeminiConfigurationError(
                "GEMINI_MODEL 설정이 필요합니다.",
                code="model_missing",
            )

    async def generate(self, *, user_prompt: str) -> TrendExplanation:
        self.ensure_configured()
        client = self._get_client()
        for attempt in range(MAX_ATTEMPTS):
            if self.request_count >= self.max_requests:
                raise GeminiAdapterError(
                    "Gemini 요청 횟수 제한을 초과했습니다.",
                    code="request_limit",
                    retries=attempt,
                )
            self.request_count += 1
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt,
                        config={
                            "system_instruction": SYSTEM_INSTRUCTION,
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_output_tokens,
                            "response_mime_type": "application/json",
                            "response_json_schema": TrendExplanation.model_json_schema(),
                        },
                    ),
                    timeout=self.timeout_seconds,
                )
                return _validated_response(response)
            except GeminiSchemaError:
                raise
            except (asyncio.TimeoutError, TimeoutError) as exc:
                if attempt == MAX_ATTEMPTS - 1:
                    raise GeminiAdapterError(
                        "Gemini API 요청 시간이 초과되었습니다.",
                        code="timeout",
                        retries=attempt,
                    ) from exc
                await asyncio.sleep(_retry_delay(attempt=attempt, exc=exc))
            except Exception as exc:
                error = _convert_sdk_error(exc, retries=attempt)
                if error.status_code not in RETRYABLE_STATUS_CODES:
                    raise error from exc
                if attempt == MAX_ATTEMPTS - 1:
                    raise error from exc
                await asyncio.sleep(_retry_delay(attempt=attempt, exc=exc))
        raise GeminiAdapterError(
            "Gemini API 요청에 실패했습니다.",
            code="request_failed",
            retries=2,
        )

    async def close(self) -> None:
        if self._client is None:
            return
        aio_client = getattr(self._client, "aio", None)
        aclose = getattr(aio_client, "aclose", None)
        if callable(aclose):
            await aclose()
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise GeminiConfigurationError(
                "google-genai 패키지가 설치되어 있지 않습니다.",
                code="sdk_missing",
            ) from exc
        self._client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                timeout=self.timeout_seconds * 1000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
        return self._client


def _validated_response(response) -> TrendExplanation:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, TrendExplanation):
        return parsed
    if parsed is not None:
        try:
            return TrendExplanation.model_validate(parsed)
        except ValidationError as exc:
            raise GeminiSchemaError(
                "Gemini 구조화 응답이 스키마와 일치하지 않습니다.",
                code="schema_invalid",
            ) from exc
    text = getattr(response, "text", None)
    if text:
        try:
            return TrendExplanation.model_validate(json.loads(text))
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            raise GeminiSchemaError(
                "Gemini JSON 응답이 스키마와 일치하지 않습니다.",
                code="schema_invalid",
            ) from exc
    finish_reason = _finish_reason(response)
    if "SAFETY" in finish_reason or "BLOCK" in finish_reason:
        raise GeminiAdapterError(
            "Gemini 안전 정책으로 응답이 거부되었습니다.",
            code="safety_blocked",
        )
    raise GeminiSchemaError(
        "선택한 Gemini 모델이 요청한 구조화 출력을 반환하지 않았습니다.",
        code="structured_output_unsupported",
    )


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return ""
    return str(getattr(candidates[0], "finish_reason", "")).upper()


def _convert_sdk_error(exc: Exception, *, retries: int) -> GeminiAdapterError:
    status_code = _status_code(exc)
    message = str(exc).casefold()
    if status_code in {401, 403} or "api key" in message or "api_key" in message:
        return GeminiAdapterError(
            "Gemini API 인증에 실패했습니다.",
            code="authentication_error",
            status_code=status_code,
            retries=retries,
        )
    if status_code == 404 or "model" in message and "not found" in message:
        return GeminiAdapterError(
            "설정한 Gemini 모델을 사용할 수 없습니다.",
            code="model_error",
            status_code=status_code,
            retries=retries,
        )
    if "safety" in message or "blocked" in message:
        return GeminiAdapterError(
            "Gemini 안전 정책으로 요청이 거부되었습니다.",
            code="safety_blocked",
            status_code=status_code,
            retries=retries,
        )
    if status_code == 400 and ("schema" in message or "response" in message):
        return GeminiSchemaError(
            "선택한 Gemini 모델이 구조화 출력 스키마를 지원하지 않습니다.",
            code="structured_output_unsupported",
            status_code=status_code,
            retries=retries,
        )
    return GeminiAdapterError(
        "Gemini API 요청에 실패했습니다.",
        code="rate_limited" if status_code == 429 else "api_error",
        status_code=status_code,
        retries=retries,
    )


def _status_code(exc: Exception) -> int | None:
    for value in (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _retry_delay(*, attempt: int, exc: Exception) -> float:
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        raw = str(retry_after).strip()
        try:
            return min(max(float(raw), 0.0), 60.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(raw)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return min(
                    max((retry_at - datetime.now(timezone.utc)).total_seconds(), 0.0),
                    60.0,
                )
            except (TypeError, ValueError, OverflowError):
                pass
    return 0.25 * (2**attempt)

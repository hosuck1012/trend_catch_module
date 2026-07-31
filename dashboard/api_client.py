from urllib.parse import quote

import httpx


CONNECTION_ERROR_MESSAGE = (
    "분석 서버에 연결할 수 없습니다.\nFastAPI 서버가 실행 중인지 확인하세요."
)


class DashboardAPIError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class DashboardConnectionError(DashboardAPIError):
    pass


class DashboardTimeoutError(DashboardAPIError):
    pass


class DashboardInvalidResponseError(DashboardAPIError):
    pass


class DashboardAPIClient:
    def __init__(
        self,
        *,
        base_url: str,
        timeout_seconds: int,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._client = client

    def get(self, path: str, *, params: dict[str, object] | None = None) -> dict:
        return self._request("GET", path, params=params)

    def post(self, path: str, *, params: dict[str, object] | None = None) -> dict:
        return self._request("POST", path, params=params)

    def get_overview(self, *, week_start: str | None = None) -> dict:
        return self.get("/api/dashboard/overview", params=_compact({"week_start": week_start}))

    def get_trends(self, **params) -> dict:
        return self.get("/api/dashboard/trends", params=_compact(params))

    def get_trend_detail(self, keyword: str, *, week_start: str | None = None) -> dict:
        encoded = quote(keyword, safe="")
        return self.get(
            f"/api/dashboard/trends/{encoded}",
            params=_compact({"week_start": week_start}),
        )

    def get_ai_analyses(self, **params) -> dict:
        return self.get("/api/ai-analysis", params=_compact(params))

    def get_ai_analysis(self, keyword: str, *, week_start: str | None = None) -> dict:
        encoded = quote(keyword, safe="")
        return self.get(
            f"/api/ai-analysis/by-keyword/{encoded}",
            params=_compact({"week_start": week_start}),
        )

    def get_ai_status(self) -> dict:
        return self.get("/api/ai-analysis/status")

    def run_ai_analysis(
        self,
        *,
        keyword: str,
        force: bool = False,
        week_start: str | None = None,
    ) -> dict:
        return self.post(
            "/api/ai-analysis/generate",
            params=_compact(
                {
                    "keyword": keyword,
                    "limit": 1,
                    "force": force,
                    "week_start": week_start,
                }
            ),
        )

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, object] | None,
    ) -> dict:
        owns_client = self._client is None
        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        try:
            response = client.request(
                method,
                f"{self.base_url}{path}",
                params=params,
            )
        except httpx.TimeoutException as exc:
            raise DashboardTimeoutError(
                "분석 서버 응답 시간이 초과되었습니다.",
            ) from exc
        except httpx.RequestError as exc:
            raise DashboardConnectionError(CONNECTION_ERROR_MESSAGE) from exc
        finally:
            if owns_client:
                client.close()
        try:
            payload = response.json()
        except ValueError as exc:
            raise DashboardInvalidResponseError(
                "분석 서버가 올바른 JSON 응답을 반환하지 않았습니다.",
                status_code=response.status_code,
            ) from exc
        if response.is_error:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise DashboardAPIError(
                str(detail or f"분석 서버 요청이 실패했습니다. HTTP {response.status_code}"),
                status_code=response.status_code,
            )
        if not isinstance(payload, dict):
            raise DashboardInvalidResponseError(
                "분석 서버 응답 형식이 올바르지 않습니다.",
                status_code=response.status_code,
            )
        return payload


def _compact(values: dict[str, object | None]) -> dict[str, object]:
    return {key: value for key, value in values.items() if value is not None and value != ""}

import httpx
import pytest

from dashboard.api_client import (
    CONNECTION_ERROR_MESSAGE,
    DashboardAPIClient,
    DashboardConnectionError,
    DashboardInvalidResponseError,
    DashboardTimeoutError,
)
from dashboard.formatters import format_number, format_text, trend_dataframe
from dashboard.state import DashboardFilters, build_trend_query


def test_dashboard_api_client_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["week_start"] == "2026-07-27"
        return httpx.Response(200, json={"selected_week": "2026-07-27"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = DashboardAPIClient(
            base_url="http://testserver",
            timeout_seconds=20,
            client=http_client,
        )
        result = client.get_overview(week_start="2026-07-27")

    assert result["selected_week"] == "2026-07-27"


def test_dashboard_api_client_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        client = _client(http_client)
        with pytest.raises(DashboardConnectionError) as exc_info:
            client.get_overview()

    assert str(exc_info.value) == CONNECTION_ERROR_MESSAGE


def test_dashboard_api_client_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        with pytest.raises(DashboardTimeoutError):
            _client(http_client).get_overview()


def test_dashboard_api_client_invalid_json() -> None:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, text="not-json"))
    with httpx.Client(transport=transport) as http_client:
        with pytest.raises(DashboardInvalidResponseError):
            _client(http_client).get_overview()


def test_gemini_button_client_posts_one_keyword() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"status": "ok", "completed": 1})

    with httpx.Client(transport=httpx.MockTransport(handler)) as http_client:
        result = _client(http_client).run_ai_analysis(
            keyword="거제야호",
            force=False,
            week_start="2026-07-27",
        )

    assert result["completed"] == 1
    assert captured["method"] == "POST"
    assert captured["params"]["keyword"] == "거제야호"
    assert captured["params"]["limit"] == "1"
    assert captured["params"]["force"] == "false"


def test_nullable_dashboard_formatters() -> None:
    frame = trend_dataframe(
        [
            {
                "rank": 1,
                "keyword": "거제야호",
                "final_score": 80,
                "trend_score": None,
                "primary_entity": None,
                "watchlist": False,
            }
        ]
    )

    assert format_number(None) == "-"
    assert format_text(None) == "정보 없음"
    assert frame.iloc[0]["trend_score"] == "-"
    assert frame.iloc[0]["주요 객체"] == "정보 없음"
    assert frame.iloc[0]["Watchlist"] == "아니요"


def test_zero_score_filters_are_not_sent() -> None:
    params = build_trend_query(
        DashboardFilters(
            week_start="2026-07-27",
            source=None,
            min_final_score=0,
            min_travel_score=0,
            travel_level=None,
            ai_status=None,
            watchlist_only=False,
            query="",
        ),
        page_size=20,
    )

    assert params["min_final_score"] is None
    assert params["min_travel_score"] is None
    assert params["limit"] == 20


def _client(http_client: httpx.Client) -> DashboardAPIClient:
    return DashboardAPIClient(
        base_url="http://testserver",
        timeout_seconds=20,
        client=http_client,
    )

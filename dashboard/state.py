from dataclasses import dataclass

import streamlit as st

from dashboard.api_client import DashboardAPIClient
from dashboard.config import get_dashboard_settings


@dataclass(frozen=True)
class DashboardFilters:
    week_start: str | None
    source: str | None
    min_final_score: float
    min_travel_score: float
    travel_level: str | None
    ai_status: str | None
    watchlist_only: bool
    query: str
    include_low_quality: bool = True


def _client(base_url: str, timeout_seconds: int) -> DashboardAPIClient:
    return DashboardAPIClient(base_url=base_url, timeout_seconds=timeout_seconds)


@st.cache_data(ttl=60, show_spinner=False)
def load_overview(base_url: str, timeout_seconds: int, week_start: str | None) -> dict:
    return _client(base_url, timeout_seconds).get_overview(week_start=week_start)


@st.cache_data(ttl=60, show_spinner=False)
def load_trends(base_url: str, timeout_seconds: int, params: tuple) -> dict:
    return _client(base_url, timeout_seconds).get_trends(**dict(params))


@st.cache_data(ttl=120, show_spinner=False)
def load_trend_detail(
    base_url: str,
    timeout_seconds: int,
    keyword: str,
    week_start: str | None,
) -> dict:
    return _client(base_url, timeout_seconds).get_trend_detail(
        keyword,
        week_start=week_start,
    )


@st.cache_data(ttl=60, show_spinner=False)
def load_ai_analyses(base_url: str, timeout_seconds: int, params: tuple) -> dict:
    return _client(base_url, timeout_seconds).get_ai_analyses(**dict(params))


@st.cache_data(ttl=120, show_spinner=False)
def load_ai_analysis(
    base_url: str,
    timeout_seconds: int,
    keyword: str,
    week_start: str | None,
) -> dict:
    return _client(base_url, timeout_seconds).get_ai_analysis(
        keyword,
        week_start=week_start,
    )


@st.cache_data(ttl=30, show_spinner=False)
def load_ai_status(base_url: str, timeout_seconds: int) -> dict:
    return _client(base_url, timeout_seconds).get_ai_status()


def run_ai_analysis(*, keyword: str, force: bool, week_start: str | None) -> dict:
    settings = get_dashboard_settings()
    return _client(settings.api_base_url, settings.request_timeout_seconds).run_ai_analysis(
        keyword=keyword,
        force=force,
        week_start=week_start,
    )


def clear_dashboard_cache() -> None:
    st.cache_data.clear()


def build_trend_query(filters: DashboardFilters, *, page_size: int) -> dict[str, object]:
    return {
        "week_start": filters.week_start,
        "query": filters.query,
        "source": filters.source,
        "watchlist_only": filters.watchlist_only,
        "include_low_quality": filters.include_low_quality,
        "min_final_score": filters.min_final_score if filters.min_final_score > 0 else None,
        "min_travel_score": filters.min_travel_score if filters.min_travel_score > 0 else None,
        "travel_level": filters.travel_level,
        "ai_status": filters.ai_status,
        "limit": page_size,
        "offset": 0,
    }


def render_sidebar(overview: dict) -> DashboardFilters:
    weeks = [str(value) for value in overview.get("available_weeks", [])]
    selected = str(overview.get("selected_week")) if overview.get("selected_week") else None
    with st.sidebar:
        st.subheader("분석 주차")
        week_start = st.selectbox(
            "주차 선택",
            options=weeks,
            index=weeks.index(selected) if selected in weeks else 0,
            disabled=not weeks,
        ) if weeks else None
        st.subheader("필터")
        sources = [item.get("name") for item in overview.get("source_distribution", [])]
        source_choice = st.selectbox("데이터 출처", ["전체", *sources])
        min_final_score = st.slider("최소 final_score", 0.0, 100.0, 0.0, 1.0)
        min_travel_score = st.slider("최소 여행 연관성 점수", 0.0, 100.0, 0.0, 1.0)
        travel_choice = st.selectbox("여행 연관성 등급", ["전체", "high", "medium", "low", "none"])
        ai_choice = st.selectbox("AI 분석 상태", ["전체", "completed", "partial", "error", "not_analyzed"])
        watchlist_only = st.checkbox("Watchlist만 표시")
        include_low_quality = st.checkbox("저품질 키워드 포함", value=True)
        query = st.text_input("검색어 입력", placeholder="키워드 검색")
        if st.button("데이터 새로고침", use_container_width=True):
            clear_dashboard_cache()
            st.rerun()
    return DashboardFilters(
        week_start=week_start,
        source=None if source_choice == "전체" else source_choice,
        min_final_score=min_final_score,
        min_travel_score=min_travel_score,
        travel_level=None if travel_choice == "전체" else travel_choice,
        ai_status=None if ai_choice == "전체" else ai_choice,
        watchlist_only=watchlist_only,
        query=query,
        include_low_quality=include_low_quality,
    )

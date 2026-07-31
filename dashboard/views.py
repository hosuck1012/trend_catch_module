import pandas as pd
import streamlit as st

from dashboard.api_client import DashboardAPIError
from dashboard.components.charts import render_charts
from dashboard.components.header import render_header
from dashboard.components.metric_cards import render_metric_cards
from dashboard.components.pipeline_status import render_pipeline_status
from dashboard.components.trend_detail import render_ai_analysis, render_trend_detail
from dashboard.components.trend_table import render_trend_table
from dashboard.config import get_dashboard_settings
from dashboard.formatters import (
    STATUS_LABELS,
    TRAVEL_LEVEL_LABELS,
    format_datetime,
    format_number,
)
from dashboard.state import (
    build_trend_query,
    clear_dashboard_cache,
    load_ai_analyses,
    load_ai_analysis,
    load_ai_status,
    load_overview,
    load_trend_detail,
    load_trends,
    render_sidebar,
    run_ai_analysis,
)


EMPTY_DATA_MESSAGE = (
    "분석 데이터가 없습니다.\n"
    "먼저 데이터 수집과 주간 트렌드 계산을 실행하세요."
)


def render_trend_dashboard() -> None:
    render_header("트렌드 대시보드")
    context = _load_dashboard_context()
    if context is None:
        return
    overview, trends = context
    st.caption(
        f"키워드 파이프라인: {overview.get('keyword_pipeline_version') or '정보 없음'}"
    )
    _render_week_fallback_notice(overview)
    render_metric_cards(overview.get("metric_cards") or [])
    st.divider()
    render_charts(
        trends,
        source_distribution=overview.get("source_distribution") or [],
        entity_distribution=overview.get("entity_distribution") or [],
    )
    render_trend_table(trends)


def render_ai_analysis_page() -> None:
    render_header("AI 분석")
    context = _load_dashboard_context()
    if context is None:
        return
    overview, trends = context
    filters = st.session_state.get("dashboard_filters")
    week_start = filters.week_start if filters else overview.get("selected_week")

    try:
        ai_status = load_ai_status(*_connection_args())
    except DashboardAPIError as exc:
        st.error(str(exc))
        ai_status = {}
    _render_ai_status(ai_status)
    st.subheader("수동 분석 실행")
    ai_candidates = _select_ai_candidates(trends)
    keywords = [item.get("normalized_keyword") for item in ai_candidates]
    if keywords:
        selected_keyword = st.selectbox("키워드", keywords, key="ai_run_keyword")
        force = st.checkbox("force 재분석", value=False, key="ai_force")
        running = bool(st.session_state.get("ai_analysis_running", False))
        if st.button(
            "선택 키워드 AI 분석 실행",
            type="primary",
            disabled=running,
            use_container_width=False,
        ):
            st.session_state["ai_analysis_running"] = True
            try:
                with st.spinner("AI 분석을 실행하고 있습니다..."):
                    result = run_ai_analysis(
                        keyword=selected_keyword,
                        force=force,
                        week_start=week_start,
                    )
                _render_ai_run_result(result)
                clear_dashboard_cache()
            except DashboardAPIError as exc:
                st.error(str(exc))
            finally:
                st.session_state["ai_analysis_running"] = False
    else:
        st.info(EMPTY_DATA_MESSAGE)

    st.divider()
    try:
        payload = load_ai_analyses(
            *_connection_args(),
            tuple(sorted({"week_start": week_start, "limit": 100}.items())),
        )
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    analyses = payload.get("items") or []
    if not analyses:
        st.info(
            "AI 분석 결과가 없습니다.\n"
            "키워드를 선택하고 AI 분석 실행 버튼을 눌러주세요."
        )
        return
    _render_ai_analysis_list(analyses)
    selected = st.selectbox(
        "상세 분석 키워드",
        [item.get("normalized_keyword") or item.get("keyword") for item in analyses],
        key="ai_detail_keyword",
    )
    try:
        analysis = load_ai_analysis(*_connection_args(), selected, week_start)
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    matching_trend = next(
        (item for item in trends if item.get("normalized_keyword") == selected),
        None,
    )
    render_ai_analysis(analysis, trend=matching_trend)


def render_keyword_detail_page() -> None:
    render_header("키워드 상세")
    context = _load_dashboard_context()
    if context is None:
        return
    overview, trends = context
    if not trends:
        st.info(EMPTY_DATA_MESSAGE)
        return
    filters = st.session_state.get("dashboard_filters")
    week_start = filters.week_start if filters else overview.get("selected_week")
    labels = {
        item.get("normalized_keyword"): item.get("keyword")
        for item in trends
        if item.get("normalized_keyword")
    }
    selected = st.selectbox(
        "키워드 선택",
        list(labels),
        format_func=lambda value: labels.get(value, value),
    )
    try:
        detail = load_trend_detail(*_connection_args(), selected, week_start)
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    render_trend_detail(detail)


def render_pipeline_status_page() -> None:
    render_header("파이프라인 상태")
    try:
        initial = load_overview(*_connection_args(), None)
        filters = render_sidebar(initial)
        overview = load_overview(*_connection_args(), filters.week_start)
    except DashboardAPIError as exc:
        st.error(str(exc))
        return
    st.caption(f'기준 주차: {overview.get("selected_week") or "정보 없음"}')
    _render_week_fallback_notice(overview)
    render_pipeline_status(overview.get("pipeline_status") or [])


def _load_dashboard_context() -> tuple[dict, list[dict]] | None:
    try:
        overview = load_overview(*_connection_args(), None)
        filters = render_sidebar(overview)
        st.session_state["dashboard_filters"] = filters
        if filters.week_start != overview.get("selected_week"):
            overview = load_overview(*_connection_args(), filters.week_start)
        params = build_trend_query(
            filters,
            page_size=get_dashboard_settings().page_size,
        )
        payload = load_trends(*_connection_args(), tuple(sorted(params.items())))
    except DashboardAPIError as exc:
        st.error(str(exc))
        return None
    if not overview.get("selected_week"):
        st.info(EMPTY_DATA_MESSAGE)
        return None
    return overview, payload.get("items") or []


def _connection_args() -> tuple[str, int]:
    settings = get_dashboard_settings()
    return settings.api_base_url, settings.request_timeout_seconds


def _render_ai_status(status: dict) -> None:
    columns = st.columns(4)
    columns[0].metric("Gemini", "활성" if status.get("gemini_enabled") else "비활성")
    columns[1].metric("모델", status.get("configured_model") or "미설정")
    columns[2].metric("완료", format_number(status.get("completed_count"), digits=0))
    columns[3].metric("부분/오류", f'{status.get("partial_count", 0)} / {status.get("error_count", 0)}')
    st.caption(
        f'API 키 설정: {"예" if status.get("api_key_configured") else "아니요"} · '
        f'최근 생성: {format_datetime(status.get("last_generated_at"))}'
    )


def _render_ai_run_result(result: dict) -> None:
    completed = int(result.get("completed") or 0)
    partial = int(result.get("partial") or 0)
    skipped = int(result.get("skipped") or 0)
    errors = int(result.get("errors") or 0)
    message = f"완료 {completed}, 부분 성공 {partial}, 캐시 사용 {skipped}, 오류 {errors}"
    if errors:
        st.error(message)
    elif partial:
        st.warning(message)
    elif skipped:
        st.info(message)
    else:
        st.success(message)


def _render_ai_analysis_list(analyses: list[dict]) -> None:
    st.subheader("분석 결과")
    rows = []
    for analysis in analyses:
        rows.append(
            {
                "키워드": analysis.get("keyword"),
                "상태": STATUS_LABELS.get(analysis.get("analysis_status"), analysis.get("analysis_status")),
                "여행 점수": format_number(analysis.get("travel_relevance_score")),
                "여행 등급": TRAVEL_LEVEL_LABELS.get(analysis.get("travel_relevance_level"), "정보 없음"),
                "신뢰도": format_number(analysis.get("confidence_score")),
                "모델": analysis.get("model_name"),
                "생성 시각": format_datetime(analysis.get("generated_at")),
                "요약": analysis.get("trend_summary") or "정보 없음",
                "상승 원인": analysis.get("rising_reason") or "정보 없음",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _select_ai_candidates(trends: list[dict], *, limit: int = 5) -> list[dict]:
    eligible = [
        item
        for item in trends
        if item.get("normalized_keyword")
        and item.get("final_score") is not None
        and item.get("keyword_quality_score") is not None
        and int(item.get("document_count") or 0) > 0
        and not item.get("suspicious", False)
    ]
    return sorted(
        eligible,
        key=lambda item: (
            -(float(item.get("final_score") or 0)),
            -(float(item.get("keyword_quality_score") or 0)),
            -(int(item.get("source_count") or 0)),
            -(int(item.get("document_count") or 0)),
            str(item.get("normalized_keyword") or ""),
        ),
    )[:limit]


def _render_week_fallback_notice(overview: dict) -> None:
    if overview.get("week_fallback_used"):
        st.warning(
            f'요청한 주차 {overview.get("requested_week") or "정보 없음"}에 데이터가 없어 '
            f'최신 데이터 주차 {overview.get("selected_week") or "정보 없음"}를 표시합니다.'
        )

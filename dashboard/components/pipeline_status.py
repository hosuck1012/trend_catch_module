import streamlit as st

from dashboard.formatters import STATUS_LABELS, format_datetime, format_number


STATUS_COLORS = {
    "healthy": "#dff2e9",
    "partial": "#fff0d6",
    "configuration_required": "#fff0d6",
    "error": "#fde3df",
    "disabled": "#e9edf1",
    "no_data": "#eef2f5",
}


def render_pipeline_status(items: list[dict]) -> None:
    if not items:
        st.info("파이프라인 상태 데이터가 없습니다.")
        return
    columns = st.columns(4)
    for index, item in enumerate(items):
        with columns[index % 4]:
            with st.container(border=True):
                status = item.get("status", "no_data")
                st.markdown(f'**{item.get("label") or "모듈"}**')
                st.markdown(
                    f'<span class="tc-status" style="background:{STATUS_COLORS.get(status, "#eef2f5")}">{STATUS_LABELS.get(status, status)}</span>',
                    unsafe_allow_html=True,
                )
                st.metric("데이터", format_number(item.get("count"), digits=0))
                details = item.get("details") or {}
                if item.get("key") == "wikipedia":
                    st.caption(
                        f'일치 {details.get("matched", 0)} · 수동 {details.get("manual", 0)} · '
                        f'모호 {details.get("ambiguous", 0)} · 미일치 {details.get("unmatched", 0)} · '
                        f'오류 {details.get("error", details.get("errors", 0))}'
                    )
                if item.get("key") == "gemini":
                    st.caption(
                        f'활성: {"예" if details.get("enabled") else "아니요"} · '
                        f'모델: {details.get("configured_model") or "미설정"}\n\n'
                        f'완료 {details.get("completed_count", 0)} · 부분 {details.get("partial_count", 0)} · '
                        f'캐시 {details.get("cached_count", 0)} · 오류 {details.get("error_count", 0)}'
                    )
                    if details.get("last_generated_at"):
                        st.caption(f'최근 생성: {format_datetime(details["last_generated_at"])}')
                if item.get("key") == "keyword_quality":
                    st.caption(
                        f'버전 {details.get("pipeline_version") or "정보 없음"} · '
                        f'활성 {details.get("active_keywords", 0)} · 거절 {details.get("rejected_candidates", 0)} · '
                        f'의심 {details.get("suspicious_keywords", 0)} · '
                        f'평균 {format_number(details.get("average_quality_score"))}'
                    )

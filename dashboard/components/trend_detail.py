import pandas as pd
import streamlit as st

from dashboard.formatters import (
    STATUS_LABELS,
    format_datetime,
    format_number,
    format_search_interest,
    format_text,
)
from dashboard.components.source_panel import render_contexts, render_documents


def render_trend_detail(detail: dict) -> None:
    trend = detail.get("trend") or {}
    st.subheader(format_text(trend.get("keyword")))
    columns = st.columns(7)
    metrics = [
        ("trend_score", trend.get("trend_score")),
        ("final_score", trend.get("final_score")),
        ("검색 관심도", trend.get("search_interest_score")),
        ("growth_rate", trend.get("growth_rate")),
        ("acceleration", trend.get("acceleration")),
        ("출처 수", trend.get("source_count")),
        ("문서 수", trend.get("document_count")),
    ]
    for column, (label, value) in zip(columns, metrics):
        display = format_search_interest(value) if label == "검색 관심도" else format_number(value)
        column.metric(label, display)

    st.subheader("관련 객체")
    entities = detail.get("entities") or []
    if entities:
        st.dataframe(pd.DataFrame(entities), use_container_width=True, hide_index=True)
    else:
        st.info("연결된 객체가 없습니다.")
    render_contexts(detail.get("contexts") or [])
    render_ai_analysis(detail.get("ai_analysis"), trend=trend)
    render_documents(detail.get("documents") or [])


def render_ai_analysis(analysis: dict | None, *, trend: dict | None = None) -> None:
    st.subheader("AI 분석")
    if not analysis:
        st.info("AI 분석 결과가 없습니다.\n키워드를 선택하고 AI 분석 실행 버튼을 눌러주세요.")
        return
    analysis_status = analysis.get("analysis_status")
    if analysis_status == "error":
        st.error(
            analysis.get("error_message")
            or "AI 분석 중 오류가 발생했습니다. 설정과 모델 응답 형식을 확인하세요."
        )
    else:
        st.caption(f'분석 상태: {STATUS_LABELS.get(analysis_status, format_text(analysis_status))}')
    trend = trend or {}
    columns = st.columns(5)
    columns[0].metric("final_score", format_number(trend.get("final_score")))
    columns[1].metric("키워드 품질", format_number(trend.get("keyword_quality_score")))
    columns[2].metric("여행 연관성", format_number(analysis.get("travel_relevance_score")))
    columns[3].metric("여행 등급", format_text(analysis.get("travel_relevance_level")))
    columns[4].metric("신뢰도", format_number(analysis.get("confidence_score")))
    st.caption(f'생성 모델: {format_text(analysis.get("model_name"))}')
    st.markdown("#### 트렌드 요약")
    st.write(analysis.get("trend_summary") or "정보 없음")
    st.markdown("#### 상승 원인")
    st.write(analysis.get("rising_reason") or "정보 없음")
    st.markdown("#### 여행 연관성 설명")
    st.write(analysis.get("travel_relevance_reason") or "정보 없음")
    st.markdown("#### 핵심 근거")
    for item in analysis.get("evidence_summary") or []:
        st.markdown(f"- {item}")
    destinations = analysis.get("recommended_destinations") or []
    st.markdown("#### 추천 여행지")
    if destinations:
        st.dataframe(pd.DataFrame(destinations), use_container_width=True, hide_index=True)
    else:
        st.caption("추천 가능한 여행지가 없습니다.")
    st.markdown("#### 콘텐츠 아이디어")
    for idea in analysis.get("content_ideas") or []:
        with st.container(border=True):
            st.markdown(f'**{idea.get("title") or "정보 없음"}** · {idea.get("format") or "정보 없음"}')
            st.write(idea.get("angle") or "정보 없음")
            st.caption(f'대상: {idea.get("target_audience") or "정보 없음"}')
    cautions = analysis.get("cautions") or []
    if cautions:
        st.warning("\n".join(f"- {item}" for item in cautions))
    refs = analysis.get("evidence_refs") or []
    if refs:
        st.markdown("#### 근거 참조")
        st.code(" · ".join(refs), language=None)
    st.caption(f'생성 시각: {format_datetime(analysis.get("generated_at"))}')

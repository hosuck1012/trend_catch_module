import pandas as pd
import plotly.express as px
import streamlit as st


TRAVEL_COLORS = {
    "high": "#2f7d65",
    "medium": "#3579a8",
    "low": "#d99a35",
    "none": "#8c98a4",
    "미분석": "#c5cbd1",
}


def render_charts(
    trends: list[dict],
    *,
    source_distribution: list[dict],
    entity_distribution: list[dict],
) -> None:
    left, right = st.columns(2)
    with left:
        _top_scores(trends)
    with right:
        _search_scatter(trends)
    left, right = st.columns(2)
    with left:
        _travel_scores(trends)
    with right:
        _source_distribution(source_distribution)
    if entity_distribution:
        _entity_distribution(entity_distribution)


def _top_scores(trends: list[dict]) -> None:
    data = sorted(trends, key=lambda item: item.get("final_score") or 0, reverse=True)[:10]
    st.subheader("상위 트렌드 점수")
    if not data:
        st.info("표시할 트렌드가 없습니다.")
        return
    frame = pd.DataFrame(data).sort_values("final_score")
    figure = px.bar(frame, x="final_score", y="keyword", orientation="h", color_discrete_sequence=["#2f7d65"])
    figure.update_layout(xaxis_title="final_score", yaxis_title=None, showlegend=False, height=380)
    st.plotly_chart(figure, use_container_width=True)


def _search_scatter(trends: list[dict]) -> None:
    data = [item for item in trends if item.get("search_interest_score") is not None]
    st.subheader("검색 관심도와 최종 점수")
    if not data:
        st.info("검색 관심도 데이터가 없습니다.")
        return
    figure = px.scatter(
        pd.DataFrame(data),
        x="search_interest_score",
        y="final_score",
        hover_name="keyword",
        color_discrete_sequence=["#d46a4c"],
    )
    figure.update_layout(xaxis_title="검색 관심도", yaxis_title="final_score", height=380)
    st.plotly_chart(figure, use_container_width=True)


def _travel_scores(trends: list[dict]) -> None:
    st.subheader("여행 연관성")
    data = []
    for item in trends[:15]:
        level = item.get("travel_relevance_level") or "미분석"
        data.append({**item, "travel_level_display": level, "travel_score_display": item.get("travel_relevance_score") or 0})
    if not data:
        st.info("여행 연관성 데이터가 없습니다.")
        return
    figure = px.bar(
        pd.DataFrame(data),
        x="keyword",
        y="travel_score_display",
        color="travel_level_display",
        color_discrete_map=TRAVEL_COLORS,
    )
    figure.update_layout(xaxis_title=None, yaxis_title="여행 연관성 점수", height=380, legend_title=None)
    st.plotly_chart(figure, use_container_width=True)


def _source_distribution(distribution: list[dict]) -> None:
    st.subheader("출처 분포")
    if not distribution:
        st.info("수집 문서 출처 데이터가 없습니다.")
        return
    figure = px.pie(
        pd.DataFrame(distribution),
        names="name",
        values="count",
        hole=0.45,
        color_discrete_sequence=["#32715f", "#3579a8", "#d99a35", "#8f5d8f", "#d46a4c"],
    )
    figure.update_layout(height=380, legend_title=None)
    st.plotly_chart(figure, use_container_width=True)


def _entity_distribution(distribution: list[dict]) -> None:
    st.subheader("객체 유형 분포")
    figure = px.bar(
        pd.DataFrame(distribution),
        x="name",
        y="count",
        color="name",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    figure.update_layout(xaxis_title=None, yaxis_title="객체 수", showlegend=False, height=340)
    st.plotly_chart(figure, use_container_width=True)

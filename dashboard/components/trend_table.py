import streamlit as st

from dashboard.formatters import trend_dataframe


def render_trend_table(items: list[dict]) -> None:
    st.subheader("트렌드 목록")
    if not items:
        st.info("필터 조건에 맞는 트렌드가 없습니다.")
        return
    st.dataframe(
        trend_dataframe(items),
        use_container_width=True,
        hide_index=True,
        height=min(680, 42 + len(items) * 35),
    )

import streamlit as st


def configure_page() -> None:
    st.set_page_config(
        page_title="Trend Catch",
        page_icon=":material/monitoring:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .stApp { background: #f7f9fb; color: #17212b; }
        .block-container { max-width: 1480px; padding-top: 1.5rem; padding-bottom: 3rem; }
        h1, h2, h3 { letter-spacing: 0; color: #17212b; }
        div[data-testid="stMetric"] { background: #ffffff; border: 1px solid #dfe5eb; border-radius: 6px; padding: 14px 16px; min-height: 116px; }
        div[data-testid="stMetricValue"] { font-size: 1.65rem; }
        div[data-testid="stDataFrame"] { border: 1px solid #dfe5eb; border-radius: 6px; overflow: hidden; }
        .tc-eyebrow { color: #32715f; font-size: 0.78rem; font-weight: 700; text-transform: uppercase; }
        .tc-subtitle { color: #536273; max-width: 760px; margin-top: -0.55rem; }
        .tc-status { display: inline-block; border-radius: 4px; padding: 2px 7px; font-size: 0.78rem; font-weight: 650; }
        @media (max-width: 760px) { .block-container { padding-left: 0.8rem; padding-right: 0.8rem; } div[data-testid="stMetricValue"] { font-size: 1.3rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(section: str | None = None) -> None:
    st.markdown('<div class="tc-eyebrow">Trend intelligence</div>', unsafe_allow_html=True)
    st.title("Trend Catch")
    st.markdown("### 여행 트렌드 인텔리전스 대시보드")
    st.markdown(
        '<p class="tc-subtitle">뉴스, YouTube, 검색 관심도와 AI 분석을 결합해 여행 트렌드를 탐색합니다.</p>',
        unsafe_allow_html=True,
    )
    if section:
        st.divider()
        st.subheader(section)

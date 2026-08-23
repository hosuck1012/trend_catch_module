import streamlit as st

from dashboard.components.header import configure_page


configure_page()

navigation = st.navigation(
    [
        st.Page("pages/1_트렌드_대시보드.py", title="트렌드 대시보드", icon=":material/monitoring:"),
        st.Page("pages/2_AI_분석.py", title="AI 분석", icon=":material/psychology:"),
        st.Page("pages/3_키워드_상세.py", title="키워드 상세", icon=":material/search:"),
        st.Page("pages/4_파이프라인_상태.py", title="파이프라인 상태", icon=":material/account_tree:"),
        st.Page(
            "pages/5_여행_기회_V2.py",
            title="여행 기회 V2",
            icon=":material/travel_explore:",
        ),
    ]
)
navigation.run()

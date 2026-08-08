import streamlit as st

from dashboard.config import get_dashboard_settings
from dashboard.api_client import DashboardAPIClient, DashboardAPIError
from dashboard.components.metric_cards import render_metric_cards


st.set_page_config(page_title="여행 기회 V2", layout="wide")
st.title("여행 기회 V2")

settings = get_dashboard_settings()
client = DashboardAPIClient(
    base_url=settings.api_base_url,
    timeout_seconds=settings.request_timeout_seconds,
)

try:
    summary = client.get_travel_opportunity_summary()
    candidates = client.get_travel_opportunities(status="strong", limit=50)
    review_candidates = client.get_travel_opportunities(status="review", limit=50)
except DashboardAPIError as exc:
    st.error(str(exc))
    st.stop()

cards = [
    {"label": "전체 트렌드 후보", "value": summary.get("raw_keyword_count", 0)},
    {"label": "Context 생성 후보", "value": summary.get("context_candidate_count", 0)},
    {"label": "Travel Review 후보", "value": summary.get("travel_prefilter_count", 0)},
    {"label": "Strong 후보", "value": summary.get("strong_candidate_count", 0)},
    {"label": "LLM 예상 호출 수", "value": summary.get("estimated_gemini_calls", 0)},
]
render_metric_cards(cards)

st.caption(f"LLM 호출 감소율: {summary.get('llm_reduction_rate', 0)}%")

items = candidates.get("items", []) + review_candidates.get("items", [])
items = sorted(items, key=lambda item: item.get("score", 0), reverse=True)
if not items:
    st.info("저장된 REVIEW / STRONG 여행 후보가 없습니다. dry-run 결과는 API 응답에서 확인하세요.")
    st.stop()

for item in items:
    with st.container(border=True):
        st.subheader(item["keyword"])
        left, middle, right = st.columns(3)
        left.metric("Travel Pre Score", item.get("score", 0))
        middle.metric("분류", item.get("category", "OTHER"))
        right.metric("상태", item.get("status", "-"))
        st.write("근거", ", ".join(item.get("reasoning_codes", [])) or "-")
        contexts = item.get("contexts", [])
        if contexts:
            context = contexts[0]
            st.markdown("**문맥**")
            st.write("이전:", context.get("previous_sentence") or "-")
            st.write("현재:", context.get("matched_sentence") or "-")
            st.write("다음:", context.get("next_sentence") or "-")
        entity = item.get("primary_entity")
        entity_type = item.get("primary_entity_type")
        st.write("관련 Entity:", f"{entity_type}: {entity}" if entity else "-")

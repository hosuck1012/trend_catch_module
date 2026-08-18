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
    report = client.get_travel_calibration_report()
except DashboardAPIError as exc:
    st.error(str(exc))
    st.stop()

funnel = report.get("funnel", {})
cards = [
    {"label": "Raw", "value": funnel.get("raw_keyword", 0)},
    {"label": "Quality", "value": funnel.get("keyword_quality_passed", 0)},
    {"label": "Rule", "value": funnel.get("rule_candidate", 0)},
    {"label": "Semantic", "value": funnel.get("semantic_candidate", 0)},
    {"label": "High Precision", "value": funnel.get("high_precision_candidate", 0)},
    {"label": "Gemini 예정", "value": funnel.get("gemini_eligible", 0)},
]
render_metric_cards(cards)

st.caption(
    f"LLM 호출 감소율: {funnel.get('llm_reduction_rate', 0)}% | "
    f"연율화 후보 추정: {report.get('annualized_candidate_estimate', 0)}"
)

items = report.get("top_20_candidates", [])
if not items:
    st.info("저장된 High Precision 여행 후보가 없습니다.")
    st.stop()

for item in items:
    with st.container(border=True):
        st.subheader(item["keyword"])
        st.metric("High Precision Score", item.get("high_precision_score", 0))
        score_columns = st.columns(4)
        score_columns[0].metric("Trend Strength", item.get("trend_strength_score", 0))
        score_columns[1].metric("Context Clarity", item.get("context_clarity_score", 0))
        score_columns[2].metric("Travel Convertibility", item.get("travel_convertibility_score", 0))
        score_columns[3].metric("Evidence Confidence", item.get("evidence_confidence_score", 0))
        left, middle, right = st.columns(3)
        left.metric("Evidence Gate", item.get("evidence_gate", "-"))
        middle.metric("Ranking Status", item.get("ranking_status", "-"))
        right.metric("Gemini Eligible", "YES" if item.get("gemini_eligible") else "NO")
        st.write("Travel Category:", item.get("travel_category", "OTHER"))
        st.write("Semantic Category:", item.get("semantic_category") or "-")
        st.write("Evidence Codes:", ", ".join(item.get("evidence_codes", [])) or "-")
        st.write(
            "Cluster:",
            item.get("cluster_id") or "-",
            "(대표)" if item.get("cluster_representative") else "",
        )
        contexts = item.get("contexts", [])
        if contexts:
            st.markdown("**문맥**")
            for context in contexts[:3]:
                st.write(context)

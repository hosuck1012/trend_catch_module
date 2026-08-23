import streamlit as st

from dashboard.config import get_dashboard_settings
from dashboard.api_client import DashboardAPIClient, DashboardAPIError
from dashboard.components.metric_cards import render_metric_cards
from dashboard.travel_opportunity_formatter import (
    final_destination_names,
    final_funnel_cards,
    split_final_opportunities,
)


st.title("여행 기회 V2")

settings = get_dashboard_settings()
client = DashboardAPIClient(
    base_url=settings.api_base_url,
    timeout_seconds=settings.request_timeout_seconds,
)

try:
    report = client.get_travel_calibration_report()
    cost_report = client.get_travel_cost_report()
    final_results = client.get_final_travel_opportunities(limit=100)
except DashboardAPIError as exc:
    st.error(str(exc))
    st.stop()

render_metric_cards(final_funnel_cards(cost_report))

st.caption(
    f"LLM 호출 감소율: {cost_report.get('overall_llm_reduction_rate', 0)}% | "
    f"연율화 후보 추정: {report.get('annualized_candidate_estimate', 0)}"
)

items = report.get("top_20_candidates", [])
eligible_items = [item for item in items if item.get("gemini_eligible")]
if eligible_items:
    labels = {
        item["normalized_keyword"]: item.get("keyword", item["normalized_keyword"])
        for item in eligible_items
    }
    selected_keyword = st.selectbox(
        "최종 분석 후보",
        list(labels),
        format_func=lambda value: labels.get(value, value),
    )
    if st.button("선택 후보 최종 AI 분석", type="primary"):
        try:
            result = client.finalize_travel_opportunities(
                keyword=selected_keyword,
                limit=1,
                force=False,
                dry_run=False,
            )
            if result.get("errors"):
                st.error(result.get("items", [{}])[0].get("error_message", "분석 오류"))
            else:
                st.success("최종 분석이 완료되었습니다.")
                st.rerun()
        except DashboardAPIError as exc:
            st.error(str(exc))
else:
    st.info("현재 Gemini Eligible 후보가 없습니다.")

accepted, review_results = split_final_opportunities(final_results.get("items", []))

st.header("최종 여행 콘텐츠 기회")
if not accepted:
    st.info("확정된 최종 여행 콘텐츠 기회가 없습니다.")
for item in accepted:
    with st.container(border=True):
        st.subheader(item["keyword"])
        score, decision = st.columns(2)
        score.metric("최종 점수", item.get("final_travel_score", 0))
        decision.metric("판정", "ACCEPT")
        st.markdown("**왜 지금 뜨는가**")
        st.write(item.get("why_now") or "-")
        st.markdown("**여행 연결**")
        st.write(item.get("travel_angle") or "-")
        st.write("추천 지역:", ", ".join(final_destination_names(item)) or "-")
        st.markdown("**콘텐츠 아이디어**")
        for idea in item.get("content_ideas", []):
            st.write(f"- {idea.get('title')}: {idea.get('concept')}")
        st.write("근거:", ", ".join(item.get("evidence_refs", [])) or "-")

st.header("추가 근거 필요")
if not review_results:
    st.info("추가 검증 대기 후보가 없습니다.")
for item in review_results:
    with st.container(border=True):
        st.subheader(item["keyword"])
        score, confidence, decision = st.columns(3)
        score.metric("최종 점수", item.get("final_travel_score", 0))
        confidence.metric("신뢰도", item.get("confidence_score", 0))
        decision.metric("판정", "REVIEW")
        st.write("여행 가능성:", item.get("travel_angle") or "-")
        verified_destinations = final_destination_names(item)
        st.write(
            "입력 근거로 확인된 지역:",
            ", ".join(verified_destinations) or "-",
        )
        st.write("근거:", ", ".join(item.get("evidence_refs", [])) or "-")
        if item.get("needs_external_verification"):
            st.warning(
                "상세 촬영 장소는 외부 검증이 필요합니다. "
                "입력 근거에 없는 장소는 확정 정보로 표시하지 않습니다."
            )
        st.write("확인 필요:", ", ".join(item.get("cautions", [])) or "-")
        st.write("검색 제안:", ", ".join(item.get("verification_queries", [])) or "-")

st.header("High Precision 후보")
if not items:
    st.info("저장된 High Precision 여행 후보가 없습니다.")
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

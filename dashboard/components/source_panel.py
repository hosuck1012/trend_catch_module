import streamlit as st

from dashboard.formatters import format_datetime, is_safe_http_url


def render_contexts(contexts: list[dict]) -> None:
    st.subheader("맥락 정보")
    if not contexts:
        st.info("연결된 맥락 정보가 없습니다.")
        return
    for context in contexts:
        with st.expander(f'{context.get("page_title") or "정보 없음"} · {context.get("provider") or "정보 없음"}'):
            st.write(context.get("summary") or "정보 없음")
            st.caption(f'상태: {context.get("match_status") or "정보 없음"} · 점수: {context.get("context_score") if context.get("context_score") is not None else "-"}')
            if is_safe_http_url(context.get("page_url")):
                st.link_button("출처 페이지 열기", context["page_url"])


def render_documents(documents: list[dict]) -> None:
    st.subheader("근거 문서")
    if not documents:
        st.info("연결된 근거 문서가 없습니다.")
        return
    for document in documents:
        st.markdown(f'**{document.get("title") or "정보 없음"}**')
        st.caption(f'{document.get("source") or "정보 없음"} · {format_datetime(document.get("published_at"))}')
        st.write(document.get("snippet") or "정보 없음")
        if is_safe_http_url(document.get("url")):
            st.link_button("원문 URL", document["url"])
        st.divider()

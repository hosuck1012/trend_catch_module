import streamlit as st

from dashboard.formatters import format_number


def render_metric_cards(cards: list[dict]) -> None:
    if not cards:
        return
    columns = st.columns(min(len(cards), 6))
    for column, card in zip(columns, cards):
        delta = card.get("delta")
        column.metric(
            card.get("label", "지표"),
            format_number(card.get("value")),
            format_number(delta) if delta is not None else None,
        )

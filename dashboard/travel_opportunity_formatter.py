def final_destination_names(item: dict) -> list[str]:
    return [
        str(destination.get("name"))
        for destination in item.get("destination_candidates", [])
        if destination.get("verified_from_input") and destination.get("name")
    ]


def split_final_opportunities(items: list[dict]) -> tuple[list[dict], list[dict]]:
    accepted = [item for item in items if item.get("final_decision") == "accept"]
    review = [item for item in items if item.get("final_decision") == "review"]
    return accepted, review


def final_funnel_cards(cost_report: dict) -> list[dict]:
    return [
        {"label": "Raw", "value": cost_report.get("raw_keyword_count", 0)},
        {"label": "Quality", "value": cost_report.get("quality_keyword_count", 0)},
        {"label": "Rule", "value": cost_report.get("rule_candidate_count", 0)},
        {"label": "Semantic", "value": cost_report.get("semantic_candidate_count", 0)},
        {
            "label": "High Precision",
            "value": cost_report.get("high_precision_candidate_count", 0),
        },
        {
            "label": "Gemini Eligible",
            "value": cost_report.get("gemini_eligible_count", 0),
        },
        {
            "label": "Final Accept",
            "value": cost_report.get("final_accept_count", 0),
        },
    ]

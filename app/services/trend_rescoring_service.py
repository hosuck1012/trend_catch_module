from app.models.weekly_trend import WeeklyTrend
from app.services.trend_scoring_service import calculate_final_score


def rescore_weekly_trend(
    trend: WeeklyTrend,
    search_interest_score: float | None,
    *,
    provider_count: int = 0,
) -> None:
    trend.search_interest_score = (
        round(max(0.0, min(100.0, search_interest_score)), 2)
        if search_interest_score is not None
        else None
    )
    trend.search_interest_available = search_interest_score is not None and provider_count > 0
    trend.search_provider_count = provider_count
    trend.final_score = round(
        calculate_final_score(
            volume_score=trend.volume_score,
            growth_score=trend.growth_score,
            persistence_score=trend.persistence_score,
            diversity_score=trend.diversity_score,
            search_interest_score=trend.search_interest_score,
            freshness_score=trend.freshness_score,
            one_day_spike_penalty=trend.one_day_spike_penalty,
            spam_penalty=trend.spam_penalty,
        ),
        2,
    )

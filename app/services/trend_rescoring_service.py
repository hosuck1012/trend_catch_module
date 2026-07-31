from app.models.weekly_trend import WeeklyTrend
from app.services.trend_scoring_service import calculate_final_score


def rescore_weekly_trend(trend: WeeklyTrend, search_interest_score: float) -> None:
    trend.search_interest_score = round(max(0.0, min(100.0, search_interest_score)), 2)
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

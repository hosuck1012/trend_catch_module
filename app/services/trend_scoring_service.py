from dataclasses import dataclass


KNOWN_SOURCE_COUNT = 2


@dataclass(frozen=True)
class KeywordMetricsInput:
    keyword: str
    weekly_mentions: int
    previous_weekly_mentions: int
    active_days: int
    source_count: int
    peak_day_mentions: int
    recent_mentions: int
    keyword_quality_score: float = 0.0


@dataclass(frozen=True)
class KeywordTrendScore:
    keyword: str
    weekly_mentions: int
    previous_weekly_mentions: int
    active_days: int
    source_count: int
    growth_rate: float
    peak_day_share: float
    persistence_score: float
    diversity_score: float
    freshness_score: float
    volume_score: float
    growth_score: float
    trend_score: float
    keyword_quality_score: float
    search_interest_score: float | None
    search_interest_available: bool
    search_provider_count: int
    one_day_spike_penalty: float
    spam_penalty: float
    final_score: float
    status: str


def score_keyword(metrics: KeywordMetricsInput, max_weekly_mentions: int) -> KeywordTrendScore:
    growth_rate = calculate_growth_rate(
        metrics.weekly_mentions,
        metrics.previous_weekly_mentions,
    )
    peak_day_share = (
        metrics.peak_day_mentions / metrics.weekly_mentions
        if metrics.weekly_mentions > 0
        else 0.0
    )
    persistence_score = clamp(metrics.active_days / 7 * 100)
    diversity_score = clamp(metrics.source_count / KNOWN_SOURCE_COUNT * 100)
    freshness_score = (
        clamp(metrics.recent_mentions / metrics.weekly_mentions * 100)
        if metrics.weekly_mentions > 0
        else 0.0
    )
    volume_score = (
        clamp(metrics.weekly_mentions / max_weekly_mentions * 100)
        if max_weekly_mentions > 0
        else 0.0
    )
    growth_score = growth_rate_to_score(growth_rate)
    search_interest_score = None
    one_day_spike_penalty = calculate_one_day_spike_penalty(peak_day_share)
    spam_penalty = calculate_spam_penalty(metrics.keyword)
    final_score = calculate_final_score(
        volume_score=volume_score,
        growth_score=growth_score,
        persistence_score=persistence_score,
        diversity_score=diversity_score,
        search_interest_score=search_interest_score,
        freshness_score=freshness_score,
        one_day_spike_penalty=one_day_spike_penalty,
        spam_penalty=spam_penalty,
    )
    trend_score = final_score
    status = classify_status(
        weekly_mentions=metrics.weekly_mentions,
        previous_weekly_mentions=metrics.previous_weekly_mentions,
        active_days=metrics.active_days,
        source_count=metrics.source_count,
        growth_rate=growth_rate,
        peak_day_share=peak_day_share,
    )

    return KeywordTrendScore(
        keyword=metrics.keyword,
        weekly_mentions=metrics.weekly_mentions,
        previous_weekly_mentions=metrics.previous_weekly_mentions,
        active_days=metrics.active_days,
        source_count=metrics.source_count,
        growth_rate=round(growth_rate, 4),
        peak_day_share=round(peak_day_share, 4),
        persistence_score=round(persistence_score, 2),
        diversity_score=round(diversity_score, 2),
        freshness_score=round(freshness_score, 2),
        volume_score=round(volume_score, 2),
        growth_score=round(growth_score, 2),
        trend_score=round(trend_score, 2),
        keyword_quality_score=round(metrics.keyword_quality_score, 2),
        search_interest_score=None,
        search_interest_available=False,
        search_provider_count=0,
        one_day_spike_penalty=round(one_day_spike_penalty, 2),
        spam_penalty=round(spam_penalty, 2),
        final_score=round(final_score, 2),
        status=status,
    )


def calculate_growth_rate(weekly_mentions: int, previous_weekly_mentions: int) -> float:
    if previous_weekly_mentions == 0 and weekly_mentions > 0:
        return 1.0
    if previous_weekly_mentions == 0 and weekly_mentions == 0:
        return 0.0
    return (weekly_mentions - previous_weekly_mentions) / previous_weekly_mentions


def growth_rate_to_score(growth_rate: float) -> float:
    if growth_rate <= 0:
        return 0.0
    if growth_rate >= 1.0:
        return 100.0
    return clamp(growth_rate * 100)


def calculate_one_day_spike_penalty(peak_day_share: float) -> float:
    return 20.0 if peak_day_share > 0.7 else 0.0


def calculate_spam_penalty(keyword: str) -> float:
    return 0.0


def calculate_final_score(
    *,
    volume_score: float,
    growth_score: float,
    persistence_score: float,
    diversity_score: float,
    search_interest_score: float | None,
    freshness_score: float,
    one_day_spike_penalty: float,
    spam_penalty: float,
) -> float:
    components = [
        (volume_score, 0.25),
        (growth_score, 0.25),
        (persistence_score, 0.20),
        (diversity_score, 0.15),
        (freshness_score, 0.05),
    ]
    if search_interest_score is not None:
        components.append((search_interest_score, 0.10))
    weight_sum = sum(weight for _, weight in components)
    weighted = sum(score * weight for score, weight in components) / weight_sum
    return clamp(
        weighted
        - one_day_spike_penalty
        - spam_penalty
    )


def classify_status(
    *,
    weekly_mentions: int,
    previous_weekly_mentions: int,
    active_days: int,
    source_count: int,
    growth_rate: float,
    peak_day_share: float,
) -> str:
    if weekly_mentions >= 3 and (peak_day_share > 0.7 or active_days < 3):
        return "watchlist"
    if (
        weekly_mentions >= 3
        and active_days >= 3
        and source_count >= 2
        and peak_day_share <= 0.7
        and weekly_mentions > previous_weekly_mentions
        and growth_rate > 0
    ):
        return "weekly_trend"
    if weekly_mentions >= 3 and active_days >= 3 and growth_rate <= 0:
        return "stable"
    return "insufficient_data"


def clamp(value: float, minimum: float = 0.0, maximum: float = 100.0) -> float:
    return max(minimum, min(maximum, value))

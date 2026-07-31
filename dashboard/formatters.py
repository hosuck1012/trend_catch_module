from datetime import datetime
from urllib.parse import urlparse

import pandas as pd


STATUS_LABELS = {
    "healthy": "정상",
    "partial": "일부 성공",
    "configuration_required": "설정 필요",
    "error": "오류",
    "disabled": "비활성",
    "no_data": "데이터 없음",
    "completed": "완료",
    "skipped": "캐시 사용",
    "not_analyzed": "미분석",
    "pending": "실행 중",
}

TRAVEL_LEVEL_LABELS = {
    "high": "높음",
    "medium": "보통",
    "low": "낮음",
    "none": "없음",
}


def format_number(value, *, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "예" if value else "아니요"
    if isinstance(value, int):
        return f"{value:,}"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def format_search_interest(value) -> str:
    if value is None:
        return "미검증"
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "미검증"


def format_text(value) -> str:
    if value is None or str(value).strip() == "":
        return "정보 없음"
    return str(value)


def format_datetime(value) -> str:
    if not value:
        return "정보 없음"
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime(
            "%Y-%m-%d %H:%M"
        )
    except ValueError:
        return str(value)


def is_safe_http_url(value: str | None) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def trend_dataframe(items: list[dict]) -> pd.DataFrame:
    rows = []
    for item in sorted(items, key=lambda row: row.get("final_score") or 0, reverse=True):
        rows.append(
            {
                "순위": item.get("rank", "-"),
                "키워드": (
                    f"⚠ {format_text(item.get('keyword'))}"
                    if item.get("suspicious")
                    else format_text(item.get("keyword"))
                ),
                "final_score": format_number(item.get("final_score")),
                "trend_score": format_number(item.get("trend_score")),
                "keyword_quality_score": format_number(item.get("keyword_quality_score")),
                "검색 관심도": format_search_interest(item.get("search_interest_score")),
                "증가율": format_number(item.get("growth_rate")),
                "출처 수": format_number(item.get("source_count"), digits=0),
                "문서 수": format_number(item.get("document_count"), digits=0),
                "주요 객체": format_text(item.get("primary_entity")),
                "주요 맥락": format_text(item.get("primary_context_title")),
                "AI 상태": STATUS_LABELS.get(item.get("ai_status"), format_text(item.get("ai_status"))),
                "여행 점수": format_number(item.get("travel_relevance_score")),
                "여행 등급": TRAVEL_LEVEL_LABELS.get(
                    item.get("travel_relevance_level"),
                    "미분석" if not item.get("ai_analysis_available") else "정보 없음",
                ),
                "Watchlist": "예" if item.get("watchlist") else "아니요",
                "파이프라인": format_text(item.get("pipeline_version")),
            }
        )
    return pd.DataFrame(rows)

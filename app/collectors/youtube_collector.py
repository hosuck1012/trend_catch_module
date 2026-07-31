from typing import Any

import httpx


YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


class YouTubeCollectorError(Exception):
    pass


class YouTubeApiAuthError(YouTubeCollectorError):
    pass


class YouTubeApiError(YouTubeCollectorError):
    pass


class YouTubeNetworkError(YouTubeCollectorError):
    pass


async def fetch_most_popular_videos(
    *,
    api_key: str,
    region_code: str,
    max_results: int,
) -> dict[str, Any]:
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": region_code,
        "maxResults": max_results,
        "key": api_key,
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(YOUTUBE_VIDEOS_URL, params=params)
    except httpx.TimeoutException as exc:
        raise YouTubeNetworkError("YouTube API 요청 시간이 초과되었습니다.") from exc
    except httpx.RequestError as exc:
        raise YouTubeNetworkError("YouTube API에 연결할 수 없습니다.") from exc

    if response.status_code in {400, 403}:
        raise YouTubeApiAuthError("YouTube API 키, API 활성화 상태 또는 quota를 확인하세요.")
    if response.status_code >= 400:
        raise YouTubeApiError("YouTube API 요청에 실패했습니다.")

    try:
        payload = response.json()
    except ValueError as exc:
        raise YouTubeApiError("YouTube API 응답 형식이 올바르지 않습니다.") from exc

    if not isinstance(payload, dict):
        raise YouTubeApiError("YouTube API 응답 형식이 올바르지 않습니다.")
    return payload

from enum import StrEnum


class TravelCategory(StrEnum):
    FILM_LOCATION = "FILM_LOCATION"
    DRAMA_LOCATION = "DRAMA_LOCATION"
    SHOW_LOCATION = "SHOW_LOCATION"
    FESTIVAL = "FESTIVAL"
    CONCERT = "CONCERT"
    EXHIBITION = "EXHIBITION"
    POPUP = "POPUP"
    FOOD = "FOOD"
    SPORTS_EVENT = "SPORTS_EVENT"
    LOCAL_CULTURE = "LOCAL_CULTURE"
    NATURE = "NATURE"
    LANDMARK = "LANDMARK"
    REGIONAL_MEME = "REGIONAL_MEME"
    OTHER = "OTHER"


PREFILTER_REJECTED = "rejected"
PREFILTER_WEAK = "weak"
PREFILTER_REVIEW = "review"
PREFILTER_STRONG = "strong"
REVIEWABLE_STATUSES = {PREFILTER_REVIEW, PREFILTER_STRONG}


ENTITY_PRIORS = {
    "PLACE": 35,
    "LOCATION": 30,
    "EVENT": 30,
    "FOOD": 25,
    "CONTENT_TITLE": 20,
    "MEME": 15,
    "BRAND": 5,
    "PERSON": 5,
}


FILM_TERMS = {"영화", "개봉", "감독", "작품", "촬영", "촬영지", "로케이션"}
DRAMA_TERMS = {"드라마", "촬영", "촬영지", "배경", "로케이션"}
SHOW_TERMS = {"예능", "방송", "촬영", "촬영지"}
FESTIVAL_TERMS = {"축제", "페스티벌", "개최", "열리는"}
CONCERT_TERMS = {"콘서트", "공연"}
EXHIBITION_TERMS = {"전시", "전시회"}
POPUP_TERMS = {"팝업", "팝업스토어"}
SPORTS_TERMS = {"마라톤", "경기"}
FOOD_TERMS = {"맛집", "음식", "디저트", "카페", "시장"}
NATURE_TERMS = {"해변", "섬", "산", "공원"}
LANDMARK_TERMS = {"명소", "랜드마크", "성지", "순례"}
LOCAL_CULTURE_TERMS = {"지역", "도시", "마을", "관광", "여행", "방문", "체험"}
FINANCE_TERMS = {"주가", "영업이익", "실적", "증권", "투자", "금리", "환율"}
LEGAL_TERMS = {"재판", "소송", "범죄", "분쟁"}
ACCIDENT_TERMS = {"사망", "사고", "부상"}

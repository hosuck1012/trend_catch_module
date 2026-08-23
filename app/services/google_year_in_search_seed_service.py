from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.weekly_trend import WeeklyTrend
from app.repositories.trend_repository import get_latest_occurrence_date
from app.services.keyword_normalization_service import normalize_keyword


GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE = "google_yis_2025_kr"
GOOGLE_YEAR_IN_SEARCH_2025_KR_URL = "https://trends.withgoogle.com/year-in-search/2025/kr/"
SEED_DAYS_PER_KEYWORD = 3


@dataclass(frozen=True)
class YearInSearchKeyword:
    category: str
    rank: int
    keyword: str


@dataclass(frozen=True)
class GoogleYearInSearchSeedResult:
    year: int
    geo: str
    categories: int
    received_keywords: int
    inserted_documents: int
    inserted_occurrences: int
    skipped_keywords: int
    week_start: date
    week_end: date


YEAR_IN_SEARCH_2025_KR_KEYWORDS: tuple[YearInSearchKeyword, ...] = (
    YearInSearchKeyword("뉴스", 1, "2025년 대한민국 대통령 선거"),
    YearInSearchKeyword("뉴스", 2, "상생페이백"),
    YearInSearchKeyword("뉴스", 3, "민생회복 소비쿠폰"),
    YearInSearchKeyword("뉴스", 4, "산불"),
    YearInSearchKeyword("뉴스", 5, "찰리 커크 피살 사건"),
    YearInSearchKeyword("뉴스", 6, "캄보디아 납치"),
    YearInSearchKeyword("뉴스", 7, "노란봉투법"),
    YearInSearchKeyword("뉴스", 8, "사전투표"),
    YearInSearchKeyword("뉴스", 9, "2025년 경주 APEC 정상회의"),
    YearInSearchKeyword("뉴스", 10, "유심보호서비스"),
    YearInSearchKeyword("인물", 1, "이재명"),
    YearInSearchKeyword("인물", 2, "김문수"),
    YearInSearchKeyword("인물", 3, "이준석"),
    YearInSearchKeyword("인물", 4, "한덕수"),
    YearInSearchKeyword("인물", 5, "김민석"),
    YearInSearchKeyword("인물", 6, "젠슨 황"),
    YearInSearchKeyword("인물", 7, "강선우"),
    YearInSearchKeyword("인물", 8, "권영국"),
    YearInSearchKeyword("인물", 9, "백종원"),
    YearInSearchKeyword("인물", 10, "홍민택"),
    YearInSearchKeyword("뜻 검색", 1, "파기환송"),
    YearInSearchKeyword("뜻 검색", 2, "파면"),
    YearInSearchKeyword("뜻 검색", 3, "각하"),
    YearInSearchKeyword("뜻 검색", 4, "기각"),
    YearInSearchKeyword("뜻 검색", 5, "Gnarly"),
    YearInSearchKeyword("뜻 검색", 6, "느좋"),
    YearInSearchKeyword("뜻 검색", 7, "에겐남"),
    YearInSearchKeyword("뜻 검색", 8, "아자스"),
    YearInSearchKeyword("뜻 검색", 9, "선종"),
    YearInSearchKeyword("뜻 검색", 10, "테토남"),
    YearInSearchKeyword("방법", 1, "민생회복 소비쿠폰 신청 방법"),
    YearInSearchKeyword("방법", 2, "상생페이백 사용 방법"),
    YearInSearchKeyword("방법", 3, "사전투표 방법"),
    YearInSearchKeyword("방법", 4, "챗GPT 지브리풍 이미지 생성 방법"),
    YearInSearchKeyword("방법", 5, "차상위계층 확인 방법"),
    YearInSearchKeyword("방법", 6, "유심 교체 방법"),
    YearInSearchKeyword("방법", 7, "소상공인 부담 경감 크레딧 사용 방법"),
    YearInSearchKeyword("방법", 8, "출구조사 방법"),
    YearInSearchKeyword("방법", 9, "KT 소액 결제 차단 방법"),
    YearInSearchKeyword("방법", 10, "기후 변화 대응 방법"),
    YearInSearchKeyword("AI Tools", 1, "챗GPT"),
    YearInSearchKeyword("AI Tools", 2, "제미나이"),
    YearInSearchKeyword("AI Tools", 3, "제타"),
    YearInSearchKeyword("AI Tools", 4, "퍼플렉시티"),
    YearInSearchKeyword("AI Tools", 5, "딥시크"),
    YearInSearchKeyword("AI Tools", 6, "그록"),
    YearInSearchKeyword("AI Tools", 7, "나노바나나"),
    YearInSearchKeyword("AI Tools", 8, "구글 AI 스튜디오"),
    YearInSearchKeyword("AI Tools", 9, "크랙"),
    YearInSearchKeyword("AI Tools", 10, "LM아레나"),
    YearInSearchKeyword("영화", 1, "케이팝 데몬 헌터스"),
    YearInSearchKeyword("영화", 2, "미키 17"),
    YearInSearchKeyword("영화", 3, "극장판 체인소 맨: 레제편"),
    YearInSearchKeyword("영화", 4, "좀비딸"),
    YearInSearchKeyword("영화", 5, "히든페이스"),
    YearInSearchKeyword("영화", 6, "노이즈"),
    YearInSearchKeyword("영화", 7, "극장판 귀멸의 칼날: 무한성편"),
    YearInSearchKeyword("영화", 8, "어쩔수가없다"),
    YearInSearchKeyword("영화", 9, "F1 더 무비"),
    YearInSearchKeyword("영화", 10, "서브스턴스"),
    YearInSearchKeyword("드라마/시리즈", 1, "폭싹 속았수다"),
    YearInSearchKeyword("드라마/시리즈", 2, "오징어 게임 (시즌 2)"),
    YearInSearchKeyword("드라마/시리즈", 3, "오징어 게임 (시즌 3)"),
    YearInSearchKeyword("드라마/시리즈", 4, "중증외상센터"),
    YearInSearchKeyword("드라마/시리즈", 5, "폭군의 셰프"),
    YearInSearchKeyword("드라마/시리즈", 6, "환승연애4"),
    YearInSearchKeyword("드라마/시리즈", 7, "신병 (시즌 3)"),
    YearInSearchKeyword("드라마/시리즈", 8, "다 이루어질지니"),
    YearInSearchKeyword("드라마/시리즈", 9, "모태 솔로지만 연애는 하고 싶어"),
    YearInSearchKeyword("드라마/시리즈", 10, "데블스 플랜: 데스룸"),
    YearInSearchKeyword("K-POP 노래", 1, "Golden (헌트릭스)"),
    YearInSearchKeyword("K-POP 노래", 2, "Soda Pop (사자보이즈)"),
    YearInSearchKeyword("K-POP 노래", 3, "너에게 닿기를 (10CM)"),
    YearInSearchKeyword("K-POP 노래", 4, "Your Idol (사자보이즈)"),
    YearInSearchKeyword("K-POP 노래", 5, "APT. (로제, 브루노 마스)"),
    YearInSearchKeyword("K-POP 노래", 6, "시작의 아이 (마크툽)"),
    YearInSearchKeyword("K-POP 노래", 7, "돌림판 (머쉬베놈)"),
    YearInSearchKeyword("K-POP 노래", 8, "FAMOUS (올데이 프로젝트)"),
    YearInSearchKeyword("K-POP 노래", 9, "나는 반딧불 (황가람)"),
    YearInSearchKeyword("K-POP 노래", 10, "오늘만 I LOVE YOU (보이넥스트도어)"),
    YearInSearchKeyword("K-POP 댄스", 1, "Soda Pop (사자보이즈)"),
    YearInSearchKeyword("K-POP 댄스", 2, "Golden (헌트릭스)"),
    YearInSearchKeyword("K-POP 댄스", 3, "like JENNY (제니)"),
    YearInSearchKeyword("K-POP 댄스", 4, "REBEL HEART (아이브)"),
    YearInSearchKeyword("K-POP 댄스", 5, "Rich Man (에스파)"),
    YearInSearchKeyword("K-POP 댄스", 6, "BEEP (이즈나)"),
    YearInSearchKeyword("K-POP 댄스", 7, "Whiplash (에스파)"),
    YearInSearchKeyword("K-POP 댄스", 8, "Your Idol (사자보이즈)"),
    YearInSearchKeyword("K-POP 댄스", 9, "첫 만남은 계획대로 되지 않아 (투어스)"),
    YearInSearchKeyword("K-POP 댄스", 10, "BANG BANG BANG (빅뱅)"),
    YearInSearchKeyword("스포츠 매치", 1, "토트넘 vs"),
    YearInSearchKeyword("스포츠 매치", 2, "로스앤젤레스 FC vs"),
    YearInSearchKeyword("스포츠 매치", 3, "대한민국 축구 국가대표팀 vs"),
    YearInSearchKeyword("스포츠 매치", 4, "로스앤젤레스 다저스 vs"),
    YearInSearchKeyword("스포츠 매치", 5, "한화 이글스 vs"),
    YearInSearchKeyword("스포츠 매치", 6, "샌프란시스코 자이언츠 vs"),
    YearInSearchKeyword("스포츠 매치", 7, "미드 시즌 인비테이셔널"),
    YearInSearchKeyword("스포츠 매치", 8, "FIFA 클럽 월드컵"),
    YearInSearchKeyword("스포츠 매치", 9, "파리 생제르맹 FC vs"),
    YearInSearchKeyword("스포츠 매치", 10, "일본 축구 국가대표팀 vs 브라질 축구 국가대표팀"),
    YearInSearchKeyword("게임", 1, "사과게임 (フルーツボックス)"),
    YearInSearchKeyword("게임", 2, "마비노기 모바일"),
    YearInSearchKeyword("게임", 3, "아이온2"),
    YearInSearchKeyword("게임", 4, "카오스 제로 나이트메어"),
    YearInSearchKeyword("게임", 5, "패스 오브 엑자일 2"),
    YearInSearchKeyword("게임", 6, "듀엣 나이트 어비스"),
    YearInSearchKeyword("게임", 7, "스텔라 소라"),
    YearInSearchKeyword("게임", 8, "이스케이프 프롬 덕코프"),
    YearInSearchKeyword("게임", 9, "아크 레이더스"),
    YearInSearchKeyword("게임", 10, "세븐나이츠 리버스"),
    YearInSearchKeyword("레시피", 1, "LA 갈비"),
    YearInSearchKeyword("레시피", 2, "쫀득쿠키"),
    YearInSearchKeyword("레시피", 3, "소금빵"),
    YearInSearchKeyword("레시피", 4, "연어 깍두기"),
    YearInSearchKeyword("레시피", 5, "오코노미야키"),
    YearInSearchKeyword("레시피", 6, "휘낭시에"),
    YearInSearchKeyword("레시피", 7, "짜장면"),
    YearInSearchKeyword("레시피", 8, "육회"),
    YearInSearchKeyword("레시피", 9, "규동"),
    YearInSearchKeyword("레시피", 10, "라죽"),
    YearInSearchKeyword("여행지", 1, "상하이"),
    YearInSearchKeyword("여행지", 2, "호치민"),
    YearInSearchKeyword("여행지", 3, "나고야"),
    YearInSearchKeyword("여행지", 4, "마쓰야마"),
    YearInSearchKeyword("여행지", 5, "시드니"),
    YearInSearchKeyword("여행지", 6, "두바이"),
    YearInSearchKeyword("여행지", 7, "미야코지마"),
    YearInSearchKeyword("여행지", 8, "로스앤젤레스"),
    YearInSearchKeyword("여행지", 9, "하와이"),
    YearInSearchKeyword("여행지", 10, "푸켓"),
    YearInSearchKeyword("패션", 1, "영포티룩"),
    YearInSearchKeyword("패션", 2, "페미닌룩"),
    YearInSearchKeyword("패션", 3, "드뮤어룩"),
    YearInSearchKeyword("패션", 4, "놈코어룩"),
    YearInSearchKeyword("패션", 5, "클리비지룩"),
    YearInSearchKeyword("패션", 6, "커플 시밀러룩"),
    YearInSearchKeyword("패션", 7, "러시아 일진룩"),
    YearInSearchKeyword("패션", 8, "동탄 미시룩"),
    YearInSearchKeyword("패션", 9, "모던룩"),
    YearInSearchKeyword("패션", 10, "보헤미안룩"),
    YearInSearchKeyword("밈", 1, "칠 가이"),
    YearInSearchKeyword("밈", 2, "기가 차드"),
    YearInSearchKeyword("밈", 3, "골반이 안 멈추는데 어떡해"),
    YearInSearchKeyword("밈", 4, "이건 첫번째 레슨"),
    YearInSearchKeyword("밈", 5, "햄부기"),
    YearInSearchKeyword("밈", 6, "이탈리안 브레인롯"),
    YearInSearchKeyword("밈", 7, "67"),
    YearInSearchKeyword("밈", 8, "누가 범인일까"),
    YearInSearchKeyword("밈", 9, "서열정리"),
    YearInSearchKeyword("밈", 10, "내가 그걸 모를까"),
    YearInSearchKeyword("챌린지", 1, "아이스크림 챌린지"),
    YearInSearchKeyword("챌린지", 2, "터미널 챌린지"),
    YearInSearchKeyword("챌린지", 3, "이안 챌린지"),
    YearInSearchKeyword("챌린지", 4, "이라이라 챌린지"),
    YearInSearchKeyword("챌린지", 5, "영어 발음 챌린지"),
    YearInSearchKeyword("챌린지", 6, "Soda Pop 챌린지"),
    YearInSearchKeyword("챌린지", 7, "Wait 챌린지"),
    YearInSearchKeyword("챌린지", 8, "고양이 그림자 챌린지"),
    YearInSearchKeyword("챌린지", 9, "바라밤 챌린지"),
    YearInSearchKeyword("챌린지", 10, "도레미 챌린지"),
    YearInSearchKeyword("바이럴 간식", 1, "크보빵"),
    YearInSearchKeyword("바이럴 간식", 2, "삼양1963"),
    YearInSearchKeyword("바이럴 간식", 3, "메롱바"),
    YearInSearchKeyword("바이럴 간식", 4, "칸쵸 '내 이름을 찾아라'"),
    YearInSearchKeyword("바이럴 간식", 5, "수건 케이크"),
    YearInSearchKeyword("바이럴 간식", 6, "두바이 초콜릿"),
    YearInSearchKeyword("바이럴 간식", 7, "초코송이 제주말차케이크맛"),
    YearInSearchKeyword("바이럴 간식", 8, "대롱대롱"),
    YearInSearchKeyword("바이럴 간식", 9, "카이막"),
    YearInSearchKeyword("바이럴 간식", 10, "공차 아이스크림"),
)


def seed_google_year_in_search_2025_kr(
    session: Session,
    *,
    week_end: date | None = None,
) -> GoogleYearInSearchSeedResult:
    resolved_week_end = week_end or get_latest_occurrence_date(session) or date.today()
    week_start = resolved_week_end - timedelta(days=6)
    collected_at = datetime.combine(resolved_week_end, time(hour=12))
    inserted_documents = 0
    inserted_occurrences = 0
    skipped_keywords = 0

    for item in YEAR_IN_SEARCH_2025_KR_KEYWORDS:
        normalized_keyword = normalize_keyword(item.keyword)
        if normalized_keyword is None:
            skipped_keywords += 1
            continue
        for day_offset in (0, 2, 4):
            occurred_date = resolved_week_end - timedelta(days=day_offset)
            occurred_at = datetime.combine(occurred_date, time(hour=12))
            source_id = _source_id(item, occurred_date)
            document = session.scalar(
                select(SourceDocument).where(
                    SourceDocument.source == GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE,
                    SourceDocument.source_id == source_id,
                )
            )
            if document is None:
                document = SourceDocument(
                    source=GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE,
                    source_id=source_id,
                    title=f"Google Year in Search 2025 Korea: {item.keyword}",
                    text=(
                        f"{item.keyword}\n"
                        f"Google Year in Search 2025 대한민국 {item.category} "
                        f"{item.rank}위 검색어 seed."
                    ),
                    published_at=occurred_at,
                    collected_at=collected_at,
                    views=None,
                    likes=None,
                    comments=None,
                    url=GOOGLE_YEAR_IN_SEARCH_2025_KR_URL,
                )
                session.add(document)
                session.flush()
                inserted_documents += 1
            occurrence = session.scalar(
                select(KeywordOccurrence).where(
                    KeywordOccurrence.document_id == document.id,
                    KeywordOccurrence.normalized_keyword == normalized_keyword,
                )
            )
            if occurrence is None:
                session.add(
                    KeywordOccurrence(
                        document_id=document.id,
                        keyword=item.keyword,
                        normalized_keyword=normalized_keyword,
                        source=GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE,
                        occurred_at=occurred_at,
                        keyword_quality_score=90.0,
                        pipeline_version="v2",
                    )
                )
                inserted_occurrences += 1

    session.commit()
    return GoogleYearInSearchSeedResult(
        year=2025,
        geo="KR",
        categories=len({item.category for item in YEAR_IN_SEARCH_2025_KR_KEYWORDS}),
        received_keywords=len(YEAR_IN_SEARCH_2025_KR_KEYWORDS),
        inserted_documents=inserted_documents,
        inserted_occurrences=inserted_occurrences,
        skipped_keywords=skipped_keywords,
        week_start=week_start,
        week_end=resolved_week_end,
    )


def apply_google_year_in_search_watchlist_overrides(
    session: Session,
    *,
    week_start: date,
    week_end: date,
) -> int:
    official_keywords = set(
        session.scalars(
            select(KeywordOccurrence.normalized_keyword)
            .where(
                KeywordOccurrence.source == GOOGLE_YEAR_IN_SEARCH_2025_KR_SOURCE,
                KeywordOccurrence.occurred_at >= datetime.combine(week_start, time.min),
                KeywordOccurrence.occurred_at <= datetime.combine(week_end, time.max),
            )
            .distinct()
        ).all()
    )
    if not official_keywords:
        return 0

    updated = 0
    trends = session.scalars(
        select(WeeklyTrend).where(
            WeeklyTrend.week_start == week_start,
            WeeklyTrend.keyword.in_(official_keywords),
            WeeklyTrend.status == "insufficient_data",
        )
    ).all()
    for trend in trends:
        trend.status = "watchlist"
        trend.pipeline_version = "v2"
        updated += 1
    if updated:
        session.flush()
    return updated


def _source_id(item: YearInSearchKeyword, occurred_date: date) -> str:
    category = item.category.replace(" ", "_").replace("/", "_")
    return f"2025-kr:{category}:{item.rank}:{occurred_date.isoformat()}"

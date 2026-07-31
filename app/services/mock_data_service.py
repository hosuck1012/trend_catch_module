from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from random import Random

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.source_document import SourceDocument


MOCK_BASE_DATE = date(2026, 7, 28)
MOCK_SOURCES = ("youtube", "naver_news")


@dataclass(frozen=True)
class MockCollectionResult:
    inserted_documents: int
    skipped_documents: int
    start_date: date
    end_date: date


@dataclass(frozen=True)
class KeywordPattern:
    keyword: str
    daily_counts: tuple[int, ...]
    title_template: str
    text_template: str


MOCK_PATTERNS = (
    KeywordPattern(
        keyword="거제야호",
        daily_counts=(1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4),
        title_template="{keyword} {day_label}",
        text_template="{keyword}",
    ),
    KeywordPattern(
        keyword="두바이초콜릿챌린지",
        daily_counts=(0, 0, 1, 0, 0, 0, 0, 0, 18, 0, 0, 1, 0, 0),
        title_template="{keyword} {day_label}",
        text_template="{keyword}",
    ),
    KeywordPattern(
        keyword="제주도여행",
        daily_counts=(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3),
        title_template="{keyword} {day_label}",
        text_template="{keyword}",
    ),
    KeywordPattern(
        keyword="폭싹속았수다촬영지",
        daily_counts=(0, 0, 1, 0, 1, 0, 1, 2, 2, 3, 4, 4, 5, 5),
        title_template="{keyword} {day_label}",
        text_template="{keyword}",
    ),
)


def collect_mock_data(session: Session) -> MockCollectionResult:
    start_date = MOCK_BASE_DATE - timedelta(days=13)
    inserted_documents = 0
    skipped_documents = 0
    rng = Random(20260728)

    for pattern in MOCK_PATTERNS:
        keyword_sequence = 0
        for day_offset, daily_count in enumerate(pattern.daily_counts):
            published_date = start_date + timedelta(days=day_offset)
            for daily_sequence in range(daily_count):
                source = MOCK_SOURCES[keyword_sequence % len(MOCK_SOURCES)]
                source_id = _source_id(pattern.keyword, source, published_date, daily_sequence)

                existing_id = session.scalar(
                    select(SourceDocument.id).where(
                        SourceDocument.source == source,
                        SourceDocument.source_id == source_id,
                    )
                )
                if existing_id is not None:
                    skipped_documents += 1
                    keyword_sequence += 1
                    continue

                published_at = datetime.combine(
                    published_date,
                    time(hour=9 + (daily_sequence % 10), minute=(keyword_sequence * 7) % 60),
                )
                unique_marker = f"mocktopic{day_offset:02d}{daily_sequence:02d}{keyword_sequence:03d}"
                document = SourceDocument(
                    source=source,
                    source_id=source_id,
                    title=pattern.title_template.format(
                        keyword=pattern.keyword,
                        day_label=published_date.isoformat(),
                    ),
                    text=f"{pattern.text_template.format(keyword=pattern.keyword)} {unique_marker}",
                    published_at=published_at,
                    collected_at=published_at + timedelta(hours=2),
                    views=_metric(rng, source, 1_000, 35_000),
                    likes=_metric(rng, source, 30, 3_200),
                    comments=_metric(rng, source, 3, 450),
                    url=f"https://example.com/{source}/{source_id}",
                )
                session.add(document)
                inserted_documents += 1
                keyword_sequence += 1

    session.commit()
    return MockCollectionResult(
        inserted_documents=inserted_documents,
        skipped_documents=skipped_documents,
        start_date=start_date,
        end_date=MOCK_BASE_DATE,
    )


def _source_id(keyword: str, source: str, published_date: date, sequence: int) -> str:
    return f"mock:{keyword}:{published_date.isoformat()}:{source}:{sequence}"


def _metric(rng: Random, source: str, low: int, high: int) -> int:
    value = rng.randint(low, high)
    if source == "youtube":
        return int(value * 1.2)
    return value
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import distinct, func, select
from sqlalchemy.orm import Session

from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.services.keyword_extraction_service import ExtractedKeyword, extract_keywords_from_document
from app.services.keyword_normalization_service import normalize_keyword
from app.config import get_settings
from app.services.keyword_extraction_v2_service import analyze_documents, persist_new_occurrences


@dataclass(frozen=True)
class KeywordExtractionResult:
    processed_documents: int
    skipped_documents: int
    inserted_occurrences: int


@dataclass(frozen=True)
class TopKeywordItem:
    keyword: str
    normalized_keyword: str
    mentions: int
    active_days: int
    source_count: int
    sources: list[str]


@dataclass(frozen=True)
class TopKeywordResult:
    total_occurrences: int
    unique_keywords: int
    items: list[TopKeywordItem]


@dataclass(frozen=True)
class KeywordDetail:
    keyword: str
    mentions: int
    active_days: int
    source_count: int
    sources: list[str]
    first_seen: datetime
    last_seen: datetime


def extract_keywords_for_documents(session: Session) -> KeywordExtractionResult:
    documents = session.scalars(select(SourceDocument).order_by(SourceDocument.id)).all()
    if get_settings().keyword_pipeline_version == "v2":
        analysis = analyze_documents(list(documents))
        processed, skipped, inserted = persist_new_occurrences(
            session, list(documents), analysis
        )
        return KeywordExtractionResult(processed, skipped, inserted)
    processed_documents = 0
    skipped_documents = 0
    inserted_occurrences = 0

    for document in documents:
        extracted_keywords = extract_keywords_from_document(document)
        existing_normalized = set(
            session.scalars(
                select(KeywordOccurrence.normalized_keyword).where(
                    KeywordOccurrence.document_id == document.id
                )
            ).all()
        )
        new_keywords = [
            keyword
            for keyword in extracted_keywords
            if keyword.normalized_keyword not in existing_normalized
        ]

        if not new_keywords:
            skipped_documents += 1
            continue

        processed_documents += 1
        for keyword in new_keywords:
            session.add(_to_occurrence(document, keyword))
            inserted_occurrences += 1

    session.commit()
    return KeywordExtractionResult(
        processed_documents=processed_documents,
        skipped_documents=skipped_documents,
        inserted_occurrences=inserted_occurrences,
    )


def get_top_keywords(session: Session, limit: int) -> TopKeywordResult:
    total_occurrences = session.scalar(select(func.count(KeywordOccurrence.id))) or 0
    unique_keywords = (
        session.scalar(select(func.count(distinct(KeywordOccurrence.normalized_keyword)))) or 0
    )

    rows = session.execute(
        select(
            func.min(KeywordOccurrence.keyword).label("keyword"),
            KeywordOccurrence.normalized_keyword,
            func.count(KeywordOccurrence.id).label("mentions"),
            func.count(distinct(func.date(KeywordOccurrence.occurred_at))).label("active_days"),
            func.count(distinct(KeywordOccurrence.source)).label("source_count"),
        )
        .group_by(KeywordOccurrence.normalized_keyword)
        .order_by(
            func.count(KeywordOccurrence.id).desc(),
            func.count(distinct(func.date(KeywordOccurrence.occurred_at))).desc(),
            KeywordOccurrence.normalized_keyword.asc(),
        )
        .limit(limit)
    ).all()

    items = [
        TopKeywordItem(
            keyword=row.keyword,
            normalized_keyword=row.normalized_keyword,
            mentions=row.mentions,
            active_days=row.active_days,
            source_count=row.source_count,
            sources=_get_sources(session, row.normalized_keyword),
        )
        for row in rows
    ]
    return TopKeywordResult(
        total_occurrences=total_occurrences,
        unique_keywords=unique_keywords,
        items=items,
    )


def get_keyword_detail(session: Session, normalized_keyword: str) -> KeywordDetail | None:
    normalized = normalize_keyword(normalized_keyword)
    if normalized is None:
        return None

    row = session.execute(
        select(
            func.min(KeywordOccurrence.keyword).label("keyword"),
            func.count(KeywordOccurrence.id).label("mentions"),
            func.count(distinct(func.date(KeywordOccurrence.occurred_at))).label("active_days"),
            func.count(distinct(KeywordOccurrence.source)).label("source_count"),
            func.min(KeywordOccurrence.occurred_at).label("first_seen"),
            func.max(KeywordOccurrence.occurred_at).label("last_seen"),
        ).where(KeywordOccurrence.normalized_keyword == normalized)
    ).one()

    if row.mentions == 0:
        return None

    return KeywordDetail(
        keyword=row.keyword,
        mentions=row.mentions,
        active_days=row.active_days,
        source_count=row.source_count,
        sources=_get_sources(session, normalized),
        first_seen=row.first_seen,
        last_seen=row.last_seen,
    )


def _to_occurrence(document: SourceDocument, keyword: ExtractedKeyword) -> KeywordOccurrence:
    return KeywordOccurrence(
        document_id=document.id,
        keyword=keyword.keyword,
        normalized_keyword=keyword.normalized_keyword,
        source=document.source,
        occurred_at=document.published_at,
    )


def _get_sources(session: Session, normalized_keyword: str) -> list[str]:
    return list(
        session.scalars(
            select(KeywordOccurrence.source)
            .where(KeywordOccurrence.normalized_keyword == normalized_keyword)
            .distinct()
            .order_by(KeywordOccurrence.source)
        ).all()
    )

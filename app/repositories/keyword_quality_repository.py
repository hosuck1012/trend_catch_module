from datetime import date, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.models.keyword_candidate import KeywordCandidate
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument


def get_documents_for_quality(
    session: Session,
    *,
    since_days: int,
    limit: int,
    source: str | None = None,
    week_start: date | None = None,
) -> list[SourceDocument]:
    latest = session.scalar(select(func.max(SourceDocument.published_at)))
    if latest is None:
        return []
    if week_start:
        end_at = datetime.combine(week_start + timedelta(days=6), datetime.max.time())
    else:
        end_at = datetime.combine(latest.date(), datetime.max.time())
    since = datetime.combine(
        (end_at.date() - timedelta(days=max(since_days - 1, 0))),
        datetime.min.time(),
    )
    query = (
        select(SourceDocument)
        .options(selectinload(SourceDocument.entity_mentions))
        .where(SourceDocument.published_at >= since, SourceDocument.published_at <= end_at)
        .order_by(SourceDocument.published_at.desc(), SourceDocument.id.desc())
    )
    if source:
        query = query.where(SourceDocument.source == source).limit(limit)
        return list(session.scalars(query).all())
    documents = list(session.scalars(query.limit(min(limit * 10, 5000))).all())
    by_source: dict[str, list[SourceDocument]] = {}
    for document in documents:
        by_source.setdefault(document.source, []).append(document)
    selected: list[SourceDocument] = []
    source_names = sorted(by_source)
    index = 0
    while len(selected) < limit:
        added = False
        for source_name in source_names:
            rows = by_source[source_name]
            if index < len(rows):
                selected.append(rows[index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        index += 1
    return selected


def replace_candidate_audit(
    session: Session,
    *,
    document_ids: list[int],
    rows: list[dict[str, object]],
) -> None:
    if document_ids:
        session.execute(
            delete(KeywordCandidate).where(KeywordCandidate.document_id.in_(document_ids))
        )
    session.add_all(KeywordCandidate(**row) for row in rows)


def delete_occurrences_for_documents(session: Session, document_ids: list[int]) -> None:
    if document_ids:
        session.execute(
            delete(KeywordOccurrence).where(KeywordOccurrence.document_id.in_(document_ids))
        )

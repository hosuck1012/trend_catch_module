from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.keywords.candidate_extractor import (
    CandidateEvidence,
    EntityEvidence,
    extract_candidates,
)
from app.keywords.keyword_normalizer import canonical_display
from app.keywords.keyword_quality import QualityContext, evaluate_candidate
from app.keywords.tokenizer import (
    KiwiTokenizerAdapter,
    RegexFallbackTokenizer,
    ResilientTokenizer,
    Tokenizer,
)
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument


@dataclass(frozen=True)
class AuditedCandidate:
    document_id: int
    candidate_text: str
    normalized_candidate: str
    candidate_type: str
    extractor: str
    quality_score: float
    accepted: bool
    rejection_reason: str | None
    title_occurrence: int
    body_occurrence: int
    entity_type: str | None
    entity_confidence: float | None


@dataclass(frozen=True)
class KeywordAggregate:
    keyword: str
    normalized_keyword: str
    quality_score: float
    document_count: int
    source_count: int
    extraction_reasons: list[str]


@dataclass(frozen=True)
class QualityAnalysis:
    processed_documents: int
    candidates: list[AuditedCandidate]
    accepted: list[KeywordAggregate]
    rejected_counts: Counter[str]
    extractor_counts: Counter[str]


def build_tokenizer(settings: Settings, tokenizer: Tokenizer | None = None) -> Tokenizer:
    if tokenizer is not None:
        return tokenizer
    if not settings.keyword_enable_kiwi:
        return RegexFallbackTokenizer()
    return ResilientTokenizer(KiwiTokenizerAdapter())


def analyze_documents(
    documents: list[SourceDocument],
    *,
    tokenizer: Tokenizer | None = None,
    settings: Settings | None = None,
    now: datetime | None = None,
) -> QualityAnalysis:
    settings = settings or get_settings()
    tokenizer = build_tokenizer(settings, tokenizer)
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    raw_by_document: dict[int, list[CandidateEvidence]] = {}
    documents_by_id = {document.id: document for document in documents}
    for document in documents:
        entities = [
            EntityEvidence(mention.text, mention.entity_type, mention.confidence)
            for mention in getattr(document, "entity_mentions", [])
        ]
        raw_by_document[document.id] = extract_candidates(
            title=document.title or "",
            body=document.text or "",
            tokenizer=tokenizer,
            entities=entities,
        )

    document_ids: dict[str, set[int]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    latest: dict[str, datetime] = {}
    for document_id, candidates in raw_by_document.items():
        document = documents_by_id[document_id]
        for candidate in candidates:
            key = candidate.normalized_candidate
            document_ids[key].add(document_id)
            sources[key].add(document.source)
            latest[key] = max(latest.get(key, document.published_at), document.published_at)

    audited: list[AuditedCandidate] = []
    for document_id, candidates in raw_by_document.items():
        ranked: list[AuditedCandidate] = []
        for candidate in candidates:
            key = candidate.normalized_candidate
            decision = evaluate_candidate(
                candidate,
                QualityContext(len(document_ids[key]), len(sources[key]), latest[key], now),
                settings,
            )
            ranked.append(
                AuditedCandidate(
                    document_id=document_id,
                    candidate_text=candidate.candidate_text,
                    normalized_candidate=key,
                    candidate_type=candidate.candidate_type,
                    extractor=candidate.extractor,
                    quality_score=decision.quality_score,
                    accepted=decision.accepted,
                    rejection_reason=decision.rejection_reason,
                    title_occurrence=candidate.title_occurrence,
                    body_occurrence=candidate.body_occurrence,
                    entity_type=candidate.entity_type,
                    entity_confidence=candidate.entity_confidence,
                )
            )
        ranked.sort(key=lambda row: (row.accepted, row.quality_score, row.title_occurrence), reverse=True)
        audited.extend(ranked[: settings.keyword_max_candidates_per_document])

    accepted_rows = [row for row in audited if row.accepted]
    accepted_by_keyword: dict[str, list[AuditedCandidate]] = defaultdict(list)
    for row in accepted_rows:
        accepted_by_keyword[row.normalized_candidate].append(row)
    aggregates = [
        KeywordAggregate(
            keyword=canonical_display([row.candidate_text for row in rows]),
            normalized_keyword=normalized,
            quality_score=round(sum(row.quality_score for row in rows) / len(rows), 2),
            document_count=len({row.document_id for row in rows}),
            source_count=len({documents_by_id[row.document_id].source for row in rows}),
            extraction_reasons=sorted({row.extractor for row in rows}),
        )
        for normalized, rows in accepted_by_keyword.items()
    ]
    aggregates.sort(
        key=lambda row: (
            -row.quality_score,
            -row.document_count,
            -row.source_count,
            row.normalized_keyword,
        )
    )
    return QualityAnalysis(
        processed_documents=len(documents),
        candidates=audited,
        accepted=aggregates,
        rejected_counts=Counter(
            row.rejection_reason or "unknown" for row in audited if not row.accepted
        ),
        extractor_counts=Counter(row.extractor for row in audited),
    )


def persist_new_occurrences(
    session: Session,
    documents: list[SourceDocument],
    analysis: QualityAnalysis,
) -> tuple[int, int, int]:
    settings = get_settings()
    documents_by_id = {document.id: document for document in documents}
    accepted = [row for row in analysis.candidates if row.accepted]
    existing = set(
        session.execute(
            select(KeywordOccurrence.document_id, KeywordOccurrence.normalized_keyword).where(
                KeywordOccurrence.document_id.in_(documents_by_id)
            )
        ).all()
    ) if documents_by_id else set()
    inserted = 0
    processed_ids: set[int] = set()
    for row in accepted:
        key = (row.document_id, row.normalized_candidate)
        if key in existing:
            continue
        document = documents_by_id[row.document_id]
        session.add(
            KeywordOccurrence(
                document_id=document.id,
                keyword=row.candidate_text,
                normalized_keyword=row.normalized_candidate,
                source=document.source,
                occurred_at=document.published_at,
                keyword_quality_score=row.quality_score,
                pipeline_version=settings.keyword_pipeline_version,
            )
        )
        existing.add(key)
        processed_ids.add(document.id)
        inserted += 1
    session.commit()
    return len(processed_ids), len(documents) - len(processed_ids), inserted

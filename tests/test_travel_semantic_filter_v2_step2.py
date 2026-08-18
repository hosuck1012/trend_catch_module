from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app.api.travel_opportunities import (
    _cached_semantic_scorer,
    get_travel_semantic_scorer,
)
from app.config import get_settings
from app.context_v2.embedding_adapter import (
    EmbeddingDisabledError,
    FakeEmbeddingAdapter,
    SentenceTransformerEmbeddingAdapter,
    clear_embedding_adapter_cache,
    get_embedding_adapter,
)
from app.context_v2.semantic_scorer import (
    SemanticScorer,
    build_candidate_embedding_text,
    classify_semantic_score,
    load_semantic_anchors,
    semantic_score_from_similarities,
)
from app.main import app
from app.models.keyword_context import KeywordContext
from app.models.source_document import SourceDocument
from app.models.travel_opportunity_candidate import TravelOpportunityCandidate
from app.repositories.travel_ranking_repository import get_semantic_candidates
from app.repositories.final_travel_opportunity_repository import get_eligible_candidates
from app.services.travel_semantic_filter_service import (
    semantic_filter_travel_opportunities,
    semantic_input_hash,
)
from app.services.travel_ranking_service import score_travel_convertibility


WEEK_START = date(2026, 8, 10)
WEEK_END = date(2026, 8, 16)
NOW = datetime(2026, 8, 12, 9, 0, 0)


def test_anchor_file_has_version_and_all_categories() -> None:
    anchors = load_semantic_anchors()

    assert anchors.version == "v1"
    assert len(anchors.positive) == 14
    assert sum(map(len, anchors.positive.values())) == 42
    assert len(anchors.negative) == 5
    assert sum(map(len, anchors.negative.values())) == 17
    assert "OTHER" in anchors.positive


def test_anchor_version_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="version mismatch"):
        load_semantic_anchors(expected_version="v2")


def test_e5_formatting_fake_adapter_injection_and_normalization() -> None:
    adapter = FakeEmbeddingAdapter(lambda text: [3.0, 4.0])

    assert adapter.encode_queries(["anchor"]) == [[0.6, 0.8]]
    assert adapter.encode_passages(["context"]) == [[0.6, 0.8]]
    assert adapter.encoded_queries == ["query: anchor"]
    assert adapter.encoded_passages == ["passage: context"]


def test_sentence_transformer_adapter_is_lazy_and_reuses_one_model() -> None:
    calls: list[tuple[str, str]] = []
    encoded: list[list[str]] = []

    class ArrayResult:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeModel:
        def encode(self, texts, **_kwargs):
            encoded.append(texts)
            return ArrayResult([[1.0, 0.0] for _ in texts])

    def factory(model_name: str, device: str):
        calls.append((model_name, device))
        return FakeModel()

    adapter = SentenceTransformerEmbeddingAdapter(
        model_name="fake/model",
        device="cpu",
        batch_size=16,
        model_factory=factory,
    )
    assert adapter.is_loaded is False

    adapter.encode_queries(["첫 anchor"])
    adapter.encode_passages(["첫 context"])

    assert calls == [("fake/model", "cpu")]
    assert encoded == [["query: 첫 anchor"], ["passage: 첫 context"]]
    assert adapter.is_loaded is True


def test_default_adapter_factory_reuses_singleton_without_loading_model() -> None:
    clear_embedding_adapter_cache()
    first = get_embedding_adapter(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=16,
        enabled=True,
    )
    second = get_embedding_adapter(
        model_name="intfloat/multilingual-e5-small",
        device="cpu",
        batch_size=16,
        enabled=True,
    )
    assert first is second
    assert first.is_loaded is False


def test_default_adapter_factory_is_singleton_during_concurrent_first_access() -> None:
    clear_embedding_adapter_cache()

    def get_one():
        return get_embedding_adapter(
            model_name="intfloat/multilingual-e5-small",
            device="cpu",
            batch_size=16,
            enabled=True,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        adapters = list(executor.map(lambda _index: get_one(), range(32)))

    assert len({id(adapter) for adapter in adapters}) == 1
    assert adapters[0].is_loaded is False


def test_disabled_adapter_does_not_load_or_encode() -> None:
    adapter = SentenceTransformerEmbeddingAdapter(
        model_name="fake/model",
        device="cpu",
        batch_size=16,
        enabled=False,
        model_factory=lambda *_args: pytest.fail("model must not load"),
    )
    with pytest.raises(EmbeddingDisabledError):
        adapter.encode_passages(["context"])
    assert adapter.is_loaded is False


def test_candidate_embedding_text_contains_context_fields_and_is_bounded() -> None:
    context = SimpleNamespace(
        previous_sentence="이전 문장",
        matched_sentence="일치 문장",
        next_sentence="다음 문장",
        combined_context="결합 문장 " * 100,
    )
    candidate = SimpleNamespace(
        keyword="8월의 크리스마스",
        normalized_keyword="8월의크리스마스",
        primary_entity="8월의 크리스마스",
        primary_entity_type="CONTENT_TITLE",
        travel_category="FILM_LOCATION",
        keyword_context=context,
    )

    text = build_candidate_embedding_text(candidate, max_chars=240)

    assert len(text) == 240
    assert "keyword: 8월의 크리스마스" in text
    assert "primary_entity_type: CONTENT_TITLE" in text
    assert "previous_sentence: 이전 문장" in text
    assert "matched_sentence: 일치 문장" in text


@pytest.mark.parametrize(
    ("text", "positive_category", "negative_category", "rejected"),
    [
        (
            "1998년 개봉한 허진호 감독의 영화 '8월의 크리스마스'가 다시 소개되고 있다.",
            "FILM_LOCATION",
            "FINANCE",
            False,
        ),
        ("부산 광안리에서 개최되는 부산불꽃축제가 올해도 열린다.", "FESTIVAL", "FINANCE", False),
        ("두바이 초콜릿 디저트가 성수동 카페를 중심으로 유행하고 있다.", "FOOD", "FINANCE", False),
        ("삼성전자 영업이익과 주가 전망이 발표됐다.", "FILM_LOCATION", "FINANCE", True),
        ("해당 인물의 법적 분쟁과 재판이 진행됐다.", "FILM_LOCATION", "LEGAL", True),
    ],
)
def test_semantic_meaning_directions(
    text: str,
    positive_category: str,
    negative_category: str,
    rejected: bool,
) -> None:
    scorer = _semantic_scorer()

    result = scorer.evaluate([text])[0]

    assert result.positive_category == positive_category
    assert result.negative_category == negative_category
    assert result.best_positive_similarity >= 0
    assert result.best_negative_similarity >= 0
    assert 0 <= result.semantic_travel_score <= 100
    assert (result.semantic_status == "semantic_rejected") is rejected


def test_semantic_status_uses_configured_boundaries() -> None:
    thresholds = {
        "reject_threshold": 0.35,
        "review_threshold": 0.55,
        "strong_threshold": 0.70,
    }
    assert classify_semantic_score(0.34, **thresholds) == "semantic_rejected"
    assert classify_semantic_score(0.35, **thresholds) == "semantic_weak"
    assert classify_semantic_score(0.55, **thresholds) == "semantic_review"
    assert classify_semantic_score(0.70, **thresholds) == "semantic_strong"


def test_realistic_close_e5_margins_cover_all_statuses() -> None:
    thresholds = {
        "reject_threshold": 0.35,
        "review_threshold": 0.55,
        "strong_threshold": 0.70,
    }
    cases = [
        (0.785, 0.80, "GENERAL_NON_TRAVEL", "semantic_rejected"),
        (0.798, 0.80, "GENERAL_NON_TRAVEL", "semantic_weak"),
        (0.804, 0.80, "GENERAL_NON_TRAVEL", "semantic_review"),
        (0.82, 0.80, "GENERAL_NON_TRAVEL", "semantic_strong"),
        (0.805, 0.81, "FINANCE", "semantic_rejected"),
    ]
    for positive, negative, category, expected in cases:
        score = semantic_score_from_similarities(
            positive,
            negative,
            negative_category=category,
        )
        assert classify_semantic_score(score, **thresholds) == expected


def test_input_hash_changes_for_model_context_anchor_or_text() -> None:
    base = semantic_input_hash(
        model_name="model-a",
        context_hash="context-a",
        anchor_version="v1",
        candidate_text="text-a",
        scorer_signature="scorer-a",
    )
    variants = {
        semantic_input_hash(
            model_name="model-b",
            context_hash="context-a",
            anchor_version="v1",
            candidate_text="text-a",
            scorer_signature="scorer-a",
        ),
        semantic_input_hash(
            model_name="model-a",
            context_hash="context-b",
            anchor_version="v1",
            candidate_text="text-a",
            scorer_signature="scorer-a",
        ),
        semantic_input_hash(
            model_name="model-a",
            context_hash="context-a",
            anchor_version="v2",
            candidate_text="text-a",
            scorer_signature="scorer-a",
        ),
        semantic_input_hash(
            model_name="model-a",
            context_hash="context-a",
            anchor_version="v1",
            candidate_text="text-b",
            scorer_signature="scorer-a",
        ),
        semantic_input_hash(
            model_name="model-a",
            context_hash="context-a",
            anchor_version="v1",
            candidate_text="text-a",
            scorer_signature="scorer-b",
        ),
    }
    assert len(base) == 64
    assert base not in variants
    assert len(variants) == 5


def test_service_includes_weak_excludes_rejected_persists_and_caches(db_session) -> None:
    candidates = _seed_candidates(db_session)
    candidates["film"].ranking_status = "priority_candidate"
    candidates["film"].ranking_version = "stale-ranking"
    candidates["film"].cluster_representative = True
    candidates["film"].gemini_eligible = True
    db_session.commit()
    scorer = _semantic_scorer()

    dry_run = semantic_filter_travel_opportunities(
        db_session,
        scorer=scorer,
        week_start=WEEK_START,
        dry_run=True,
        force=False,
        limit=100,
    )
    db_session.expire_all()
    assert dry_run.processed == 4
    assert dry_run.semantic_strong == 3
    assert dry_run.semantic_rejected == 1
    assert candidates["film"].semantic_status is None
    assert candidates["excluded"].semantic_status is None

    stored = semantic_filter_travel_opportunities(
        db_session,
        scorer=scorer,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
    )
    db_session.expire_all()
    assert stored.processed == 4
    assert candidates["film"].semantic_status == "semantic_strong"
    assert candidates["finance"].semantic_status == "semantic_rejected"
    assert candidates["excluded"].semantic_status is None
    assert candidates["film"].embedding_model == "fake-e5"
    assert candidates["film"].semantic_positive_score is not None
    assert candidates["film"].semantic_negative_score is not None
    assert candidates["film"].embedding_input_hash is not None
    assert candidates["film"].semantic_calculated_at is not None
    assert candidates["film"].ranking_status is None
    assert candidates["film"].ranking_version is None
    assert candidates["film"].cluster_representative is None
    assert candidates["film"].gemini_eligible is None
    assert get_eligible_candidates(
        db_session,
        week_start=WEEK_START,
        normalized_keyword="8월의크리스마스",
        limit=10,
    ) == []

    passage_calls = len(scorer.adapter.encoded_passages)
    cached = semantic_filter_travel_opportunities(
        db_session,
        scorer=scorer,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
    )
    assert cached.cache_hits == 4
    assert len(scorer.adapter.encoded_passages) == passage_calls


def test_disabled_service_returns_without_db_changes(monkeypatch, db_session) -> None:
    candidates = _seed_candidates(db_session)
    monkeypatch.setenv("TRAVEL_EMBEDDING_ENABLED", "false")
    get_settings.cache_clear()
    scorer = _semantic_scorer()

    result = semantic_filter_travel_opportunities(
        db_session,
        scorer=scorer,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
    )

    assert result.status == "disabled"
    assert result.processed == 0
    assert scorer.adapter.encoded_passages == []
    assert candidates["film"].semantic_status is None


def test_disabled_dependency_does_not_load_anchor_file(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_EMBEDDING_ENABLED", "false")
    get_settings.cache_clear()
    _cached_semantic_scorer.cache_clear()
    monkeypatch.setattr(
        "app.api.travel_opportunities.load_semantic_anchors",
        lambda **_kwargs: pytest.fail("disabled mode must not load anchors"),
    )

    scorer = get_travel_semantic_scorer()

    assert scorer.adapter.enabled is False


def test_invalid_anchor_version_is_sanitized_as_service_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("TRAVEL_EMBEDDING_ENABLED", "true")
    monkeypatch.setenv("TRAVEL_SEMANTIC_ANCHOR_VERSION", "missing-version")
    get_settings.cache_clear()
    _cached_semantic_scorer.cache_clear()

    with pytest.raises(HTTPException) as exc_info:
        get_travel_semantic_scorer()

    assert exc_info.value.status_code == 503
    assert "ValueError" in str(exc_info.value.detail)


def test_semantic_api_dry_run_and_gemini_not_called(monkeypatch, client, db_session) -> None:
    candidates = _seed_candidates(db_session)
    scorer = _semantic_scorer()
    app.dependency_overrides[get_travel_semantic_scorer] = lambda: scorer

    async def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("Gemini must not run during local semantic filtering")

    monkeypatch.setattr("app.ai.gemini_adapter.GeminiAdapter.generate", forbidden_generate)
    try:
        response = client.post(
            f"/api/travel-opportunities/semantic-filter?week_start={WEEK_START.isoformat()}"
        )
    finally:
        app.dependency_overrides.pop(get_travel_semantic_scorer, None)

    db_session.expire_all()
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "dry_run"
    assert payload["dry_run"] is True
    assert payload["processed"] == 4
    assert payload["semantic_strong"] == 3
    assert payload["semantic_rejected"] == 1
    assert payload["estimated_gemini_candidates"] == 3
    assert payload["model_name"] == "fake-e5"
    assert candidates["film"].semantic_status is None


def test_ranking_repository_requires_semantic_review_or_strong(db_session) -> None:
    candidates = _seed_candidates(db_session)
    scorer = _semantic_scorer()
    semantic_filter_travel_opportunities(
        db_session,
        scorer=scorer,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
    )

    rows = get_semantic_candidates(db_session, week_start=WEEK_START, limit=100)
    keywords = {row.normalized_keyword for row in rows}

    assert keywords == {"8월의크리스마스", "부산불꽃축제", "두바이초콜릿"}
    assert candidates["finance"].normalized_keyword not in keywords
    assert candidates["excluded"].normalized_keyword not in keywords


def test_estimated_gemini_candidates_are_distinct_keywords(db_session) -> None:
    candidates = _seed_candidates(db_session)
    film = candidates["film"]
    original_context = film.keyword_context
    duplicate_context = KeywordContext(
        document_id=original_context.document_id,
        keyword=film.keyword,
        normalized_keyword=film.normalized_keyword,
        previous_sentence="다른 이전 문장이다.",
        matched_sentence="영화의 다른 촬영 장소도 소개된다.",
        next_sentence="다른 다음 문장이다.",
        combined_context="영화의 다른 촬영 장소도 소개된다.",
        occurrence_index=1,
        source="test",
        published_at=NOW,
        context_hash="semantic-context-film-duplicate",
        created_at=NOW,
        updated_at=NOW,
    )
    db_session.add(duplicate_context)
    db_session.flush()
    db_session.add(
        TravelOpportunityCandidate(
            keyword=film.keyword,
            normalized_keyword=film.normalized_keyword,
            week_start=WEEK_START,
            week_end=WEEK_END,
            keyword_context_id=duplicate_context.id,
            primary_entity=film.primary_entity,
            primary_entity_type=film.primary_entity_type,
            travel_category=film.travel_category,
            entity_prior_score=film.entity_prior_score,
            positive_context_score=film.positive_context_score,
            negative_context_penalty=film.negative_context_penalty,
            trend_evidence_score=film.trend_evidence_score,
            source_diversity_score=film.source_diversity_score,
            travel_pre_score=film.travel_pre_score,
            prefilter_status="weak",
            matched_positive_terms_json="[]",
            matched_negative_terms_json="[]",
            reasoning_codes_json="[]",
            created_at=NOW,
            updated_at=NOW,
        )
    )
    db_session.commit()

    result = semantic_filter_travel_opportunities(
        db_session,
        scorer=_semantic_scorer(),
        week_start=WEEK_START,
        dry_run=True,
        force=False,
        limit=100,
    )

    assert result.processed == 5
    assert result.semantic_strong == 4
    assert result.estimated_gemini_candidates == 3
    assert len({item.normalized_keyword for item in result.top_candidates}) == len(
        result.top_candidates
    )


def test_ranking_negative_penalty_requires_negative_similarity_to_dominate(db_session) -> None:
    candidates = _seed_candidates(db_session)
    row = candidates["festival"]
    row.semantic_travel_score = 80
    row.semantic_positive_category = "FESTIVAL"
    row.semantic_positive_score = 0.82
    row.semantic_negative_category = "FINANCE"
    row.semantic_negative_score = 0.78

    positive_dominant = score_travel_convertibility(rows=[row], entity_types={"EVENT"})
    row.semantic_negative_score = 0.84
    negative_dominant = score_travel_convertibility(rows=[row], entity_types={"EVENT"})

    assert positive_dominant - negative_dominant == 45


def _semantic_scorer() -> SemanticScorer:
    return SemanticScorer(
        adapter=FakeEmbeddingAdapter(_semantic_vector),
        anchors=load_semantic_anchors(),
        reject_threshold=0.35,
        review_threshold=0.55,
        strong_threshold=0.70,
    )


def _semantic_vector(text: str) -> list[float]:
    lowered = text.lower()
    if any(term in lowered for term in ("주가", "영업이익", "금융시장", "투자 전망")):
        return [0, 0, 0, 1, 0, 0]
    if any(term in lowered for term in ("법적 분쟁", "재판", "소송", "법적 처벌")):
        return [0, 0, 0, 0, 1, 0]
    if any(term in lowered for term in ("영화", "로케이션")):
        return [1, 0, 0, 0, 0, 0]
    if any(term in lowered for term in ("축제", "행사 개최")):
        return [0, 1, 0, 0, 0, 0]
    if any(term in lowered for term in ("음식", "디저트", "맛집")):
        return [0, 0, 1, 0, 0, 0]
    return [0, 0, 0, 0, 0, 1]


def _seed_candidates(db_session) -> dict[str, TravelOpportunityCandidate]:
    definitions = [
        (
            "film",
            "8월의 크리스마스",
            "8월의크리스마스",
            "1998년 개봉한 허진호 감독의 영화 '8월의 크리스마스'가 다시 소개되고 있다.",
            "CONTENT_TITLE",
            "FILM_LOCATION",
            "weak",
        ),
        (
            "festival",
            "부산불꽃축제",
            "부산불꽃축제",
            "부산 광안리에서 개최되는 부산불꽃축제가 올해도 열린다.",
            "EVENT",
            "FESTIVAL",
            "review",
        ),
        (
            "food",
            "두바이 초콜릿",
            "두바이초콜릿",
            "두바이 초콜릿 디저트가 성수동 카페를 중심으로 유행하고 있다.",
            "FOOD",
            "FOOD",
            "strong",
        ),
        (
            "finance",
            "삼성전자",
            "삼성전자",
            "삼성전자 영업이익과 주가 전망이 발표됐다.",
            "BRAND",
            "OTHER",
            "weak",
        ),
        (
            "excluded",
            "법적분쟁인물",
            "법적분쟁인물",
            "해당 인물의 법적 분쟁과 재판이 진행됐다.",
            "PERSON",
            "OTHER",
            "rejected",
        ),
    ]
    result: dict[str, TravelOpportunityCandidate] = {}
    for index, (
        key,
        keyword,
        normalized,
        sentence,
        entity_type,
        category,
        prefilter_status,
    ) in enumerate(definitions):
        document = SourceDocument(
            source="test",
            source_id=f"semantic-{index}",
            title=keyword,
            text=sentence,
            published_at=NOW,
            collected_at=NOW,
            views=None,
            likes=None,
            comments=None,
            url=None,
        )
        db_session.add(document)
        db_session.flush()
        context = KeywordContext(
            document_id=document.id,
            keyword=keyword,
            normalized_keyword=normalized,
            previous_sentence="관련 배경을 설명한다.",
            matched_sentence=sentence,
            next_sentence="추가 소식을 전한다.",
            combined_context=f"관련 배경을 설명한다. {sentence} 추가 소식을 전한다.",
            occurrence_index=0,
            source="test",
            published_at=NOW,
            context_hash=f"semantic-context-{index}",
            created_at=NOW,
            updated_at=NOW,
        )
        db_session.add(context)
        db_session.flush()
        candidate = TravelOpportunityCandidate(
            keyword=keyword,
            normalized_keyword=normalized,
            week_start=WEEK_START,
            week_end=WEEK_END,
            keyword_context_id=context.id,
            primary_entity=keyword,
            primary_entity_type=entity_type,
            travel_category=category,
            entity_prior_score=20,
            positive_context_score=10,
            negative_context_penalty=0,
            trend_evidence_score=10,
            source_diversity_score=5,
            travel_pre_score=50,
            prefilter_status=prefilter_status,
            matched_positive_terms_json="[]",
            matched_negative_terms_json="[]",
            reasoning_codes_json="[]",
            created_at=NOW,
            updated_at=NOW,
        )
        db_session.add(candidate)
        db_session.flush()
        result[key] = candidate
    db_session.commit()
    return result

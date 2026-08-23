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
from app.context_v2.semantic_precision import (
    build_semantic_precision_evidence,
    calibrate_semantic_score,
)
from app.keywords.tokenizer import RegexFallbackTokenizer
from app.main import app
from app.models.entity_mention import EntityMention
from app.models.keyword_candidate import KeywordCandidate
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
        (0.8005, 0.80, "GENERAL_NON_TRAVEL", "semantic_weak"),
        (0.812, 0.80, "GENERAL_NON_TRAVEL", "semantic_review"),
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
    defaults = {
        "model_name": "model-a",
        "context_hash": "context-a",
        "anchor_version": "v1",
        "scoring_version": "v2",
        "candidate_text": "text-a",
        "scorer_signature": "scorer-a",
        "precision_signature": "precision-a",
        "rule_input_hash": "rule-hash-a",
        "rule_version": "rule-v1",
        "rule_status": "weak",
        "rule_score": 50.0,
        "rule_category": "FESTIVAL",
    }
    base = semantic_input_hash(**defaults)
    variants = {
        semantic_input_hash(**(defaults | {"model_name": "model-b"})),
        semantic_input_hash(**(defaults | {"context_hash": "context-b"})),
        semantic_input_hash(**(defaults | {"anchor_version": "v2"})),
        semantic_input_hash(**(defaults | {"scoring_version": "v3"})),
        semantic_input_hash(**(defaults | {"candidate_text": "text-b"})),
        semantic_input_hash(**(defaults | {"scorer_signature": "scorer-b"})),
        semantic_input_hash(**(defaults | {"precision_signature": "precision-b"})),
        semantic_input_hash(**(defaults | {"rule_input_hash": "rule-hash-b"})),
        semantic_input_hash(**(defaults | {"rule_version": "rule-v2"})),
        semantic_input_hash(**(defaults | {"rule_status": "review"})),
        semantic_input_hash(**(defaults | {"rule_score": 55.0})),
        semantic_input_hash(**(defaults | {"rule_category": "CONCERT"})),
    }
    assert len(base) == 64
    assert base not in variants
    assert len(variants) == 12


@pytest.mark.parametrize("keyword", ["가격", "친절", "적용", "도전", "주역"])
def test_generic_single_topic_is_rejected_despite_travel_context(keyword: str) -> None:
    candidate = _precision_candidate(
        keyword=keyword,
        entity_type=None,
        travel_category="OTHER",
        context="보령머드축제 행사에서 관광객을 위한 프로그램을 진행한다.",
    )
    evidence = build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )

    result = _calibrate(
        evidence,
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status == "semantic_rejected"
    assert result.semantic_travel_score == 0
    assert "GENERIC_TOPIC" in result.reasoning_codes
    assert "TOPIC_SPECIFICITY_FAIL" in result.reasoning_codes
    assert "CATEGORY_LOCAL_EVIDENCE_FAIL" in result.reasoning_codes


def test_single_high_value_entity_and_multi_token_topic_are_exceptions() -> None:
    location = _precision_candidate(
        keyword="제주",
        entity_type="LOCATION",
        travel_category="LANDMARK",
        context="제주의 관광 명소를 소개한다.",
    )
    location_evidence = build_semantic_precision_evidence(
        location,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )
    phrase = _precision_candidate(
        keyword="성수 팝업",
        entity_type=None,
        travel_category="POPUP",
        context="성수에서 새로운 팝업스토어가 열린다.",
    )
    phrase_evidence = build_semantic_precision_evidence(
        phrase,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )

    location_result = _calibrate(
        location_evidence,
        positive_category="LANDMARK",
        positive=0.81,
        negative=0.80,
    )
    phrase_result = _calibrate(
        phrase_evidence,
        positive_category="POPUP",
        positive=0.81,
        negative=0.80,
    )

    assert location_result.semantic_status == "semantic_review"
    assert "HIGH_VALUE_ENTITY" in location_result.reasoning_codes
    assert phrase_result.semantic_status == "semantic_review"
    assert "MULTI_TOKEN_TOPIC" in phrase_result.reasoning_codes


def test_category_coherence_failure_caps_specific_topic_at_weak() -> None:
    candidate = _precision_candidate(
        keyword="두바이 초콜릿",
        entity_type=None,
        travel_category="FOOD",
        context="두바이 초콜릿이 성수동 카페에서 판매된다.",
    )
    evidence = build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )

    result = _calibrate(
        evidence,
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status == "semantic_weak"
    assert result.semantic_travel_score == 54.99
    assert "SEMANTIC_CATEGORY_UNSUPPORTED" in result.reasoning_codes


def test_multi_token_generic_phrase_does_not_pass_specificity() -> None:
    candidate = _precision_candidate(
        keyword="가격 인상",
        entity_type=None,
        travel_category="OTHER",
        context="축제 입장권 가격 인상 기준이 적용됐다.",
    )
    evidence = build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )

    result = _calibrate(
        evidence,
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status == "semantic_rejected"
    assert "GENERIC_TOPIC" in result.reasoning_codes


def test_context_entity_exception_requires_keyword_match() -> None:
    candidate = _precision_candidate(
        keyword="제주",
        entity_type=None,
        travel_category="LANDMARK",
        context="제주 관광 명소를 소개한다.",
    )
    matching = SimpleNamespace(
        text="제주",
        normalized_text="제주",
        entity_type="LOCATION",
        confidence=0.9,
    )
    unrelated = SimpleNamespace(
        text="부산불꽃축제",
        normalized_text="부산불꽃축제",
        entity_type="EVENT",
        confidence=0.9,
    )
    location_evidence = build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=[matching],
        tokenizer=RegexFallbackTokenizer(),
    )
    generic = _precision_candidate(
        keyword="프로그램",
        entity_type=None,
        travel_category="FESTIVAL",
        context="부산불꽃축제 프로그램을 안내한다.",
    )
    generic_evidence = build_semantic_precision_evidence(
        generic,
        quality_signal=None,
        context_entities=[unrelated],
        tokenizer=RegexFallbackTokenizer(),
    )

    assert location_evidence.topic_specificity_pass is True
    assert "HIGH_VALUE_ENTITY" in location_evidence.topic_codes
    assert generic_evidence.topic_specificity_pass is False


def test_safety_caps_hold_with_non_default_thresholds() -> None:
    candidate = _precision_candidate(
        keyword="두바이 초콜릿",
        entity_type="FOOD",
        travel_category="FOOD",
        context="두바이 초콜릿이 카페에서 판매된다.",
    )
    evidence = build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )
    unsupported = calibrate_semantic_score(
        positive_similarity=0.88,
        positive_category="FESTIVAL",
        negative_similarity=0.84,
        negative_category="GENERAL_NON_TRAVEL",
        evidence=evidence,
        reject_threshold=0.0,
        review_threshold=0.50,
        strong_threshold=0.54,
    )
    generic = _precision_candidate(
        keyword="가격",
        entity_type=None,
        travel_category="OTHER",
        context="여행 가격 정보다.",
    )
    generic_evidence = build_semantic_precision_evidence(
        generic,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )
    rejected = calibrate_semantic_score(
        positive_similarity=0.88,
        positive_category="FESTIVAL",
        negative_similarity=0.84,
        negative_category="GENERAL_NON_TRAVEL",
        evidence=generic_evidence,
        reject_threshold=0.0,
        review_threshold=0.50,
        strong_threshold=0.54,
    )

    assert unsupported.semantic_status == "semantic_weak"
    assert rejected.semantic_status == "semantic_rejected"


def test_margin_gate_rejects_negative_and_caps_low_positive_margin() -> None:
    candidate = _precision_candidate(
        keyword="부산불꽃축제",
        entity_type="EVENT",
        travel_category="FESTIVAL",
        context="부산 광안리에서 부산불꽃축제가 개최된다.",
    )
    evidence = build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=[],
        tokenizer=RegexFallbackTokenizer(),
    )

    negative = _calibrate(
        evidence,
        positive_category="FESTIVAL",
        positive=0.84,
        negative=0.85,
    )
    low = _calibrate(
        evidence,
        positive_category="FESTIVAL",
        positive=0.8505,
        negative=0.85,
    )

    assert negative.semantic_status == "semantic_rejected"
    assert "NEGATIVE_SEMANTIC_DOMINANT" in negative.reasoning_codes
    assert low.semantic_status == "semantic_weak"
    assert "LOW_SEMANTIC_MARGIN" in low.reasoning_codes


def test_entity_alignment_and_locality_are_candidate_scoped() -> None:
    exact = _localized_precision_candidate(
        keyword="제주",
        entity_type="LOCATION",
        travel_category="LANDMARK",
        matched="제주의 관광 명소를 방문한다.",
    )
    partial = _localized_precision_candidate(
        keyword="폭싹 속았수다 촬영지",
        entity_type="CONTENT_TITLE",
        travel_category="DRAMA_LOCATION",
        matched="드라마 폭싹 속았수다 촬영지를 여행객이 방문한다.",
        entity_text="폭싹 속았수다",
    )
    unrelated = _localized_precision_candidate(
        keyword="국가",
        entity_type=None,
        travel_category="LANDMARK",
        matched="근대 주권국가를 설명한다.",
        entity_text="서울",
        context_entity_type="LOCATION",
    )

    exact_evidence = _build_precision_evidence(exact)
    partial_evidence = _build_precision_evidence(partial)
    unrelated_evidence = _build_precision_evidence(unrelated)

    assert exact_evidence.entity_alignment == "EXACT"
    assert exact_evidence.entity_locality == "MATCHED_SENTENCE"
    assert "ENTITY_KEYWORD_EXACT_MATCH" in exact_evidence.topic_codes
    assert partial_evidence.entity_alignment == "PARTIAL"
    assert "ENTITY_KEYWORD_PARTIAL_MATCH" in partial_evidence.topic_codes
    assert unrelated_evidence.entity_alignment == "UNALIGNED"
    assert unrelated_evidence.topic_specificity_pass is False
    assert "ENTITY_KEYWORD_UNALIGNED" in unrelated_evidence.topic_codes


def test_adjacent_entity_is_local_but_document_only_entity_is_not() -> None:
    adjacent = _localized_precision_candidate(
        keyword="부산불꽃축제",
        entity_type="EVENT",
        travel_category="FESTIVAL",
        matched="이 행사는 올해도 열린다.",
        previous="부산불꽃축제를 찾는 관광객이 늘었다.",
    )
    document_only = _localized_precision_candidate(
        keyword="부산불꽃축제",
        entity_type="EVENT",
        travel_category="FESTIVAL",
        matched="관련 정책을 설명한다.",
        entity_in_local_context=False,
    )

    adjacent_evidence = _build_precision_evidence(adjacent)
    document_evidence = _build_precision_evidence(document_only)

    assert adjacent_evidence.entity_locality == "ADJACENT_SENTENCE"
    assert "ENTITY_IN_ADJACENT_CONTEXT" in adjacent_evidence.topic_codes
    assert document_evidence.entity_locality == "DOCUMENT_ONLY"
    assert "DOCUMENT_EVIDENCE_ONLY" in document_evidence.topic_codes
    assert _calibrate(
        document_evidence,
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    ).semantic_status not in {"semantic_review", "semantic_strong"}


@pytest.mark.parametrize(
    ("keyword", "entity_type", "category", "context"),
    [
        ("부산불꽃축제", "EVENT", "FESTIVAL", "부산불꽃축제가 광안리에서 개최된다."),
        ("서울 냉면", "FOOD", "FOOD", "서울 냉면 맛집을 찾는 관광객이 늘었다."),
        ("경복궁", "PLACE", "LANDMARK", "경복궁을 찾는 관광객과 방문객이 늘었다."),
        (
            "폭싹 속았수다 촬영지",
            "CONTENT_TITLE",
            "DRAMA_LOCATION",
            "드라마 폭싹 속았수다 촬영지를 여행객이 방문한다.",
        ),
    ],
)
def test_category_matrix_requires_aligned_local_evidence(
    keyword: str,
    entity_type: str,
    category: str,
    context: str,
) -> None:
    candidate = _localized_precision_candidate(
        keyword=keyword,
        entity_type=entity_type,
        travel_category=category,
        matched=context,
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category=category,
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status in {"semantic_review", "semantic_strong"}
    assert "CATEGORY_LOCAL_EVIDENCE_PASS" in result.reasoning_codes


@pytest.mark.parametrize(
    ("keyword", "entity_type", "category", "context", "expected_reviewable"),
    [
        ("울산", "LOCATION", "FESTIVAL", "울산 여름 축제를 찾는 관광객이 늘었다.", True),
        ("울산", "LOCATION", "LANDMARK", "울산 기업의 영업이익이 발표됐다.", False),
        ("냉면", "FOOD", "FOOD", "서울 냉면 맛집을 찾는 관광객이 늘었다.", True),
        ("냉면", "FOOD", "FOOD", "냉면 가격이 올랐다.", False),
        ("뮌헨", "LOCATION", "FESTIVAL", "뮌헨 옥토버페스트를 찾는 여행객이 늘었다.", True),
        ("뮌헨", "LOCATION", "SPORTS_EVENT", "뮌헨 축구팀 경기 결과가 발표됐다.", False),
        ("송파구", "LOCATION", "LOCAL_CULTURE", "송파구 석촌호수 축제가 열린다.", False),
    ],
)
def test_single_token_policy_depends_on_local_travel_evidence(
    keyword: str,
    entity_type: str,
    category: str,
    context: str,
    expected_reviewable: bool,
) -> None:
    candidate = _localized_precision_candidate(
        keyword=keyword,
        entity_type=entity_type,
        travel_category=category,
        matched=context,
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category=category,
        positive=0.88,
        negative=0.84,
    )

    assert (result.semantic_status in {"semantic_review", "semantic_strong"}) is expected_reviewable
    expected_code = (
        "SINGLE_TOKEN_WITH_TRAVEL_EVIDENCE"
        if expected_reviewable
        else "SINGLE_TOKEN_INSUFFICIENT_EVIDENCE"
    )
    assert expected_code in result.reasoning_codes


def test_malformed_joined_particle_fragment_is_rejected() -> None:
    candidate = _localized_precision_candidate(
        keyword="서울 서",
        entity_type="LOCATION",
        travel_category="SPORTS_EVENT",
        matched="세계보치아선수권대회가 다음달 서울서 개막한다.",
        entity_text="서울서",
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category="SPORTS_EVENT",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status == "semantic_rejected"
    assert "MALFORMED_TOPIC" in result.reasoning_codes


def test_compact_korean_compound_is_not_treated_as_malformed() -> None:
    candidate = _localized_precision_candidate(
        keyword="서울 숲",
        entity_type="PLACE",
        travel_category="NATURE",
        matched="서울숲을 찾는 관광객이 늘었다.",
        entity_text="서울숲",
    )
    evidence = _build_precision_evidence(candidate)
    result = _calibrate(
        evidence,
        positive_category="NATURE",
        positive=0.88,
        negative=0.84,
    )

    assert evidence.keyword_locality == "MATCHED_SENTENCE"
    assert evidence.malformed_topic is False
    assert result.semantic_status in {"semantic_review", "semantic_strong"}


def test_unrelated_document_entity_does_not_suppress_local_primary() -> None:
    candidate = _localized_precision_candidate(
        keyword="울산",
        entity_type="LOCATION",
        travel_category="LANDMARK",
        matched="울산을 찾는 관광객이 늘었다.",
        entity_text="부산",
        context_entity_type="LOCATION",
        entity_in_local_context=False,
    )
    candidate.primary_entity = "울산"
    evidence = _build_precision_evidence(candidate)
    result = _calibrate(
        evidence,
        positive_category="LANDMARK",
        positive=0.88,
        negative=0.84,
    )

    assert evidence.entity_locality == "MATCHED_SENTENCE"
    assert result.semantic_status in {"semantic_review", "semantic_strong"}


def test_generic_visit_alone_does_not_manufacture_festival_evidence() -> None:
    candidate = _localized_precision_candidate(
        keyword="서울",
        entity_type="LOCATION",
        travel_category="FESTIVAL",
        matched="서울을 방문했다.",
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status not in {"semantic_review", "semantic_strong"}
    assert "CATEGORY_LOCAL_EVIDENCE_FAIL" in result.reasoning_codes


def test_other_category_requires_explicit_travel_evidence() -> None:
    candidate = _localized_precision_candidate(
        keyword="한강공원",
        entity_type="PLACE",
        travel_category="OTHER",
        matched="스타벅스가 한강공원을 찾아 커피를 제공하고 캠페인을 진행했다.",
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category="OTHER",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status not in {"semantic_review", "semantic_strong"}
    assert "CATEGORY_LOCAL_EVIDENCE_FAIL" in result.reasoning_codes


def test_regional_meme_requires_regional_context() -> None:
    candidate = _localized_precision_candidate(
        keyword="신조어밈",
        entity_type="MEME",
        travel_category="REGIONAL_MEME",
        matched="신조어밈이 온라인에서 유행한다.",
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category="REGIONAL_MEME",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status not in {"semantic_review", "semantic_strong"}


@pytest.mark.parametrize(
    ("keyword", "entity_type", "category", "context"),
    [
        ("한라산", "PLACE", "NATURE", "한라산을 찾는 관광객이 늘었다."),
        ("서울마라톤", "EVENT", "SPORTS_EVENT", "서울마라톤이 잠실경기장에서 개최된다."),
    ],
)
def test_nature_and_hosted_sports_events_remain_reviewable(
    keyword: str,
    entity_type: str,
    category: str,
    context: str,
) -> None:
    candidate = _localized_precision_candidate(
        keyword=keyword,
        entity_type=entity_type,
        travel_category=category,
        matched=context,
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category=category,
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status in {"semantic_review", "semantic_strong"}


@pytest.mark.parametrize("context", ["서울엔 관광객이 늘었다.", "서울이다."])
def test_contracted_particle_and_copula_preserve_locality(context: str) -> None:
    candidate = _localized_precision_candidate(
        keyword="서울",
        entity_type="LOCATION",
        travel_category="LANDMARK",
        matched=context,
    )

    assert _build_precision_evidence(candidate).keyword_locality == "MATCHED_SENTENCE"


def test_local_category_context_changes_precision_cache_signature() -> None:
    first = _localized_precision_candidate(
        keyword="울산",
        entity_type="LOCATION",
        travel_category="LANDMARK",
        matched="울산을 찾는 관광객이 늘었다.",
    )
    second = _localized_precision_candidate(
        keyword="울산",
        entity_type="LOCATION",
        travel_category="LANDMARK",
        matched="울산 기업의 영업이익이 발표됐다.",
    )

    assert (
        _build_precision_evidence(first).cache_signature
        != _build_precision_evidence(second).cache_signature
    )


def test_news_dateline_location_is_not_local_topic_evidence() -> None:
    candidate = _localized_precision_candidate(
        keyword="서울",
        entity_type="LOCATION",
        travel_category="FESTIVAL",
        matched="[서울=뉴시스] 인천 펜타포트 락 페스티벌에 관광객이 방문한다.",
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status not in {"semantic_review", "semantic_strong"}
    assert "DOCUMENT_EVIDENCE_ONLY" in result.reasoning_codes


def test_rule_category_does_not_circularly_satisfy_semantic_category() -> None:
    candidate = _localized_precision_candidate(
        keyword="국가",
        entity_type=None,
        travel_category="FESTIVAL",
        matched="근대 주권국가의 제도를 설명하는 전시다.",
        entity_text="서울",
        context_entity_type="LOCATION",
        entity_in_local_context=False,
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category="FESTIVAL",
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status == "semantic_rejected"
    assert "CATEGORY_LOCAL_EVIDENCE_FAIL" in result.reasoning_codes


@pytest.mark.parametrize(
    ("keyword", "entity_type", "category", "context"),
    [
        ("폭싹 속았수다", "CONTENT_TITLE", "DRAMA_LOCATION", "드라마 폭싹 속았수다 촬영지를 관광객이 방문한다."),
        ("폭싹 속았수다 촬영지", "CONTENT_TITLE", "DRAMA_LOCATION", "드라마 폭싹 속았수다 촬영지를 관광객이 방문한다."),
        ("설봉산 별빛축제", "EVENT", "FESTIVAL", "설봉산 별빛축제가 이천에서 개최된다."),
        ("홍천강 별빛음악 맥주축제", "EVENT", "FESTIVAL", "홍천강 별빛음악 맥주축제가 개최된다."),
        ("펜타포트 락 페스티벌", "EVENT", "FESTIVAL", "펜타포트 락 페스티벌이 인천에서 열린다."),
        ("경남고성공룡세계엑스포", "EVENT", "EXHIBITION", "경남고성공룡세계엑스포 전시를 관람한다."),
        ("부산국제불교박람회", "EVENT", "LOCAL_CULTURE", "부산국제불교박람회에서 불교 문화 체험을 진행한다."),
        ("2026 국악관현악축제", "EVENT", "FESTIVAL", "2026 국악관현악축제가 개최된다."),
    ],
)
def test_gold_topics_remain_semantic_reviewable(
    keyword: str,
    entity_type: str,
    category: str,
    context: str,
) -> None:
    candidate = _localized_precision_candidate(
        keyword=keyword,
        entity_type=entity_type,
        travel_category=category,
        matched=context,
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category=category,
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status in {"semantic_review", "semantic_strong"}


@pytest.mark.parametrize(
    ("keyword", "entity_type", "category", "context"),
    [
        ("국가", "LOCATION", "LANDMARK", "근대 주권국가의 제도를 설명한다."),
        ("식음료", "FOOD", "FOOD", "식음료 시장 진입과 기업 실적을 발표했다."),
    ],
)
def test_existing_generic_false_positive_fixtures_are_not_reviewable(
    keyword: str,
    entity_type: str,
    category: str,
    context: str,
) -> None:
    candidate = _localized_precision_candidate(
        keyword=keyword,
        entity_type=entity_type,
        travel_category=category,
        matched=context,
    )
    result = _calibrate(
        _build_precision_evidence(candidate),
        positive_category=category,
        positive=0.88,
        negative=0.84,
    )

    assert result.semantic_status not in {"semantic_review", "semantic_strong"}


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


def test_service_rejects_generic_topics_with_fake_adapter(db_session) -> None:
    for index, keyword in enumerate(("가격", "친절", "적용", "도전", "주역")):
        _add_candidate(
            db_session,
            key=f"generic-{index}",
            keyword=keyword,
            normalized=keyword,
            context="보령머드축제 행사에서 관광객을 위한 프로그램을 진행한다.",
            entity_type=None,
            category="OTHER",
        )

    result = semantic_filter_travel_opportunities(
        db_session,
        scorer=_semantic_scorer(),
        week_start=WEEK_START,
        dry_run=True,
        force=False,
        limit=100,
        topic_tokenizer=RegexFallbackTokenizer(),
    )

    assert result.processed == 5
    assert result.semantic_rejected == 5
    assert result.semantic_review == 0
    assert result.semantic_strong == 0
    assert all("GENERIC_TOPIC" in row.reasoning_codes for row in result.top_candidates)


def test_brand_sports_topic_requires_event_location_or_visit_evidence(db_session) -> None:
    plain = _add_candidate(
        db_session,
        key="brand-plain",
        keyword="두산",
        normalized="두산",
        context="두산은 리그 순위에서 4위로 올라섰고 승차를 3경기로 좁혔다.",
        entity_type="BRAND",
        category="SPORTS_EVENT",
        quality_entity_type="BRAND",
        context_entity_types=(("서울", "LOCATION"),),
    )
    visit = _add_candidate(
        db_session,
        key="brand-visit",
        keyword="두산",
        normalized="두산",
        context="서울 잠실야구장에서 열리는 두산 홈 경기를 관람하려고 팬들이 방문한다.",
        entity_type="BRAND",
        category="SPORTS_EVENT",
        quality_entity_type="BRAND",
        context_entity_types=(("서울", "LOCATION"), ("두산 홈 경기", "EVENT")),
    )

    semantic_filter_travel_opportunities(
        db_session,
        scorer=_semantic_scorer(),
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
        topic_tokenizer=RegexFallbackTokenizer(),
    )
    db_session.expire_all()

    assert plain.semantic_status not in {"semantic_review", "semantic_strong"}
    assert visit.semantic_status in {"semantic_review", "semantic_strong"}


def test_scoring_version_change_invalidates_force_false_cache(db_session) -> None:
    candidate = _seed_candidates(db_session)["festival"]
    first = _semantic_scorer(scoring_version="v1")
    semantic_filter_travel_opportunities(
        db_session,
        scorer=first,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
        topic_tokenizer=RegexFallbackTokenizer(),
    )
    first_hash = candidate.embedding_input_hash
    second = _semantic_scorer(scoring_version="v2")

    result = semantic_filter_travel_opportunities(
        db_session,
        scorer=second,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
        topic_tokenizer=RegexFallbackTokenizer(),
    )
    db_session.expire_all()

    assert result.cache_hits == 0
    assert candidate.embedding_input_hash != first_hash
    assert second.adapter.encoded_passages


def test_rule_hash_and_version_change_invalidates_force_false_cache(db_session) -> None:
    candidate = _seed_candidates(db_session)["festival"]
    candidate.rule_input_hash = "rule-hash-a"
    candidate.rule_version = "rule-v1"
    db_session.commit()
    first = _semantic_scorer()
    semantic_filter_travel_opportunities(
        db_session,
        scorer=first,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
        topic_tokenizer=RegexFallbackTokenizer(),
    )
    first_hash = candidate.embedding_input_hash
    candidate.rule_input_hash = "rule-hash-b"
    candidate.rule_version = "rule-v2"
    db_session.commit()
    second = _semantic_scorer()

    result = semantic_filter_travel_opportunities(
        db_session,
        scorer=second,
        week_start=WEEK_START,
        dry_run=False,
        force=False,
        limit=100,
        topic_tokenizer=RegexFallbackTokenizer(),
    )
    db_session.expire_all()

    assert result.cache_hits == 3
    assert candidate.embedding_input_hash != first_hash
    assert len(second.adapter.encoded_passages) == 1


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
    assert payload["scoring_version"] == "v2"
    assert payload["top_candidates"][0]["reasoning_codes"]
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
    assert result.semantic_strong == 3
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


def _semantic_scorer(*, scoring_version: str = "v2") -> SemanticScorer:
    return SemanticScorer(
        adapter=FakeEmbeddingAdapter(_semantic_vector),
        anchors=load_semantic_anchors(),
        reject_threshold=0.35,
        review_threshold=0.55,
        strong_threshold=0.70,
        scoring_version=scoring_version,
    )


def _semantic_vector(text: str) -> list[float]:
    lowered = text.lower()
    if any(term in lowered for term in ("주가", "영업이익", "금융시장", "투자 전망")):
        return [0, 0, 0, 0, 1, 0, 0]
    if any(term in lowered for term in ("법적 분쟁", "재판", "소송", "법적 처벌")):
        return [0, 0, 0, 0, 0, 1, 0]
    if any(term in lowered for term in ("영화", "로케이션")):
        return [1, 0, 0, 0, 0, 0, 0]
    if any(term in lowered for term in ("축제", "행사 개최")):
        return [0, 1, 0, 0, 0, 0, 0]
    if any(term in lowered for term in ("음식", "디저트", "맛집")):
        return [0, 0, 1, 0, 0, 0, 0]
    if any(term in lowered for term in ("스포츠", "경기", "마라톤", "야구")):
        return [0, 0, 0, 1, 0, 0, 0]
    return [0, 0, 0, 0, 0, 0, 1]


def _precision_candidate(
    *,
    keyword: str,
    entity_type: str | None,
    travel_category: str,
    context: str,
):
    return SimpleNamespace(
        keyword=keyword,
        normalized_keyword=keyword.replace(" ", ""),
        primary_entity=keyword if entity_type else None,
        primary_entity_type=entity_type,
        travel_category=travel_category,
        matched_positive_terms_json="[]",
        keyword_context=SimpleNamespace(combined_context=context),
    )


def _localized_precision_candidate(
    *,
    keyword: str,
    entity_type: str | None,
    travel_category: str,
    matched: str,
    previous: str | None = None,
    next_sentence: str | None = None,
    entity_text: str | None = None,
    context_entity_type: str | None = None,
    entity_in_local_context: bool = True,
):
    resolved_entity_text = entity_text or keyword
    resolved_entity_type = context_entity_type or entity_type
    combined = " ".join(
        value for value in (previous, matched, next_sentence) if value
    )
    context_entities = []
    if resolved_entity_type:
        mention_text = resolved_entity_text
        if not entity_in_local_context and mention_text in combined:
            mention_text = f"문서전용 {mention_text}"
        context_entities.append(
            SimpleNamespace(
                text=mention_text,
                normalized_text=mention_text.replace(" ", ""),
                entity_type=resolved_entity_type,
                confidence=0.9,
            )
        )
    return SimpleNamespace(
        keyword=keyword,
        normalized_keyword=keyword.replace(" ", ""),
        primary_entity=resolved_entity_text if entity_type else None,
        primary_entity_type=entity_type,
        travel_category=travel_category,
        matched_positive_terms_json="[]",
        keyword_context=SimpleNamespace(
            previous_sentence=previous,
            matched_sentence=matched,
            next_sentence=next_sentence,
            combined_context=combined,
        ),
        context_entities=context_entities,
    )


def _build_precision_evidence(candidate):
    return build_semantic_precision_evidence(
        candidate,
        quality_signal=None,
        context_entities=candidate.context_entities,
        tokenizer=RegexFallbackTokenizer(),
    )


def _calibrate(
    evidence,
    *,
    positive_category: str,
    positive: float,
    negative: float,
):
    return calibrate_semantic_score(
        positive_similarity=positive,
        positive_category=positive_category,
        negative_similarity=negative,
        negative_category="GENERAL_NON_TRAVEL",
        evidence=evidence,
        reject_threshold=0.35,
        review_threshold=0.55,
        strong_threshold=0.70,
    )


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


def _add_candidate(
    db_session,
    *,
    key: str,
    keyword: str,
    normalized: str,
    context: str,
    entity_type: str | None,
    category: str,
    quality_entity_type: str | None = None,
    context_entity_types: tuple[tuple[str, str], ...] = (),
) -> TravelOpportunityCandidate:
    document = SourceDocument(
        source="test",
        source_id=f"precision-{key}",
        title=keyword,
        text=context,
        published_at=NOW,
        collected_at=NOW,
        views=None,
        likes=None,
        comments=None,
        url=None,
    )
    db_session.add(document)
    db_session.flush()
    keyword_context = KeywordContext(
        document_id=document.id,
        keyword=keyword,
        normalized_keyword=normalized,
        previous_sentence=None,
        matched_sentence=context,
        next_sentence=None,
        combined_context=context,
        occurrence_index=0,
        source="test",
        published_at=NOW,
        context_hash=f"precision-context-{key}",
        created_at=NOW,
        updated_at=NOW,
    )
    db_session.add(keyword_context)
    db_session.flush()
    if quality_entity_type:
        db_session.add(
            KeywordCandidate(
                document_id=document.id,
                candidate_text=keyword,
                normalized_candidate=normalized,
                candidate_type="entity",
                extractor="ner",
                quality_score=70,
                accepted=True,
                rejection_reason=None,
                title_occurrence=1,
                body_occurrence=1,
                entity_type=quality_entity_type,
                entity_confidence=0.9,
                created_at=NOW,
                pipeline_version="v2",
            )
        )
    for index, (text, mention_type) in enumerate(context_entity_types):
        db_session.add(
            EntityMention(
                document_id=document.id,
                text=text,
                normalized_text=text.replace(" ", ""),
                entity_type=mention_type,
                confidence=0.9,
                extractor="merged",
                start_char=index,
                end_char=index + len(text),
                source="test",
                occurred_at=NOW,
                created_at=NOW,
            )
        )
    candidate = TravelOpportunityCandidate(
        keyword=keyword,
        normalized_keyword=normalized,
        week_start=WEEK_START,
        week_end=WEEK_END,
        keyword_context_id=keyword_context.id,
        primary_entity=keyword if entity_type else None,
        primary_entity_type=entity_type,
        travel_category=category,
        entity_prior_score=5,
        positive_context_score=10,
        negative_context_penalty=0,
        trend_evidence_score=10,
        source_diversity_score=5,
        travel_pre_score=50,
        prefilter_status="weak",
        matched_positive_terms_json="[]",
        matched_negative_terms_json="[]",
        reasoning_codes_json="[]",
        created_at=NOW,
        updated_at=NOW,
    )
    db_session.add(candidate)
    db_session.commit()
    return candidate

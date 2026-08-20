from datetime import datetime
from types import SimpleNamespace

import pytest

from app.keywords.candidate_extractor import EntityEvidence, extract_candidates
from app.keywords.keyword_normalizer import normalize_keyword
from app.keywords.phrase_signals import phrase_suffix_categories, phrase_suffixes
from app.keywords.tokenizer import RegexFallbackTokenizer
from app.services.keyword_extraction_v2_service import analyze_documents


NOW = datetime(2026, 8, 1, 9, 0, 0)
GOLD_PHRASES = (
    "거제야호",
    "두바이 초콜릿",
    "두바이 초콜릿 챌린지",
    "폭싹 속았수다",
    "폭싹 속았수다 촬영지",
    "설봉산 별빛축제",
    "홍천강 별빛음악 맥주축제",
    "펜타포트 락 페스티벌",
    "한강 밤핑",
    "경남고성공룡세계엑스포",
    "부산국제불교박람회",
    "2026 국악관현악축제",
    "제주도 여행",
    "뱅크시: 스틸 히어",
    "RESTOPIA",
)
NEGATIVE_PHRASES = (
    "가격",
    "친절",
    "적용",
    "증가",
    "방문",
    "최근",
    "있습니다",
    "com",
    "by",
    "the",
    "축제",
    "전시",
    "공연",
    "여행",
)


def _document(document_id: int, title: str, body: str = ""):
    return SimpleNamespace(
        id=document_id,
        title=title,
        text=body,
        source="test",
        published_at=NOW,
        entity_mentions=[],
    )


def _candidate(title: str, expected: str):
    result = analyze_documents(
        [_document(1, title)],
        tokenizer=RegexFallbackTokenizer(),
        now=NOW,
    )
    normalized = normalize_keyword(expected)
    return next(row for row in result.candidates if row.normalized_candidate == normalized)


@pytest.mark.parametrize("phrase", GOLD_PHRASES)
def test_gold_phrases_are_generated_and_accepted(phrase: str) -> None:
    candidate = _candidate(phrase, phrase)

    assert candidate.accepted is True
    assert candidate.quality_score >= 45


@pytest.mark.parametrize("phrase", NEGATIVE_PHRASES)
def test_standalone_generic_and_garbage_remain_rejected(phrase: str) -> None:
    candidate = _candidate(phrase, phrase)

    assert candidate.accepted is False


@pytest.mark.parametrize(
    "phrase",
    (
        "설봉산 별빛축제",
        "홍천강 별빛음악 맥주축제",
        "펜타포트 락 페스티벌",
        "한강 밤핑",
    ),
)
def test_noun_and_proper_topic_runs_use_event_suffix(phrase: str) -> None:
    candidate = _candidate(phrase, phrase)

    assert candidate.candidate_type == "specific_phrase"
    assert candidate.extractor == "phrase_pattern"


@pytest.mark.parametrize(
    "phrase",
    (
        "폭싹 속았수다 촬영지",
        "두바이 초콜릿 챌린지",
        "RESTOPIA 팝업",
    ),
)
def test_title_food_and_english_proper_modifiers_are_preserved(phrase: str) -> None:
    assert _candidate(phrase, phrase).accepted is True


def test_quoted_and_colon_titles_are_protected() -> None:
    quoted = _candidate("신작 '바람이 분다' 촬영지 공개", "바람이 분다")
    colon = _candidate("뱅크시: 스틸 히어", "뱅크시 스틸 히어")

    assert quoted.candidate_type == "protected_phrase"
    assert colon.candidate_type == "protected_phrase"
    assert quoted.accepted is True
    assert colon.accepted is True


def test_unrelated_news_quote_is_not_protected() -> None:
    candidates = extract_candidates(
        title='정부는 "가격 안정"을 강조했다',
        body="",
        tokenizer=RegexFallbackTokenizer(),
    )
    normalized = normalize_keyword("가격 안정")
    match = next(row for row in candidates if row.normalized_candidate == normalized)

    assert match.candidate_type != "protected_phrase"


@pytest.mark.parametrize(
    "title, quoted_value",
    (
        ('BYD 전시장 오픈…"수도권 공략 박차"', "수도권 공략 박차"),
        ('"별빛 아래 신선한 맥주 한 잔" 홍천 맥주축제 참여', "별빛 아래 신선한 맥주 한 잔"),
        (
            '팝업 "앞으로도 다양한 현장을 찾아 지역사회와 함께하겠다"',
            "앞으로도 다양한 현장을 찾아 지역사회와 함께하겠다",
        ),
    ),
)
def test_headline_slogans_near_topic_words_are_not_protected(
    title: str,
    quoted_value: str,
) -> None:
    candidates = extract_candidates(
        title=title,
        body="",
        tokenizer=RegexFallbackTokenizer(),
    )
    normalized = normalize_keyword(quoted_value)

    assert not any(
        row.normalized_candidate == normalized and row.candidate_type == "protected_phrase"
        for row in candidates
    )


def test_numeric_event_and_stopword_inside_specific_phrase_are_allowed() -> None:
    numeric = _candidate("2026 국악관현악축제", "2026 국악관현악축제")
    stopword = _candidate("제주도 여행", "제주도 여행")

    assert numeric.accepted is True
    assert stopword.accepted is True
    assert stopword.rejection_reason is None


def test_duplicate_normalized_phrase_is_merged_to_specific_candidate() -> None:
    candidates = extract_candidates(
        title="설봉산 별빛축제",
        body="설봉산 별빛축제가 열린다.",
        tokenizer=RegexFallbackTokenizer(),
    )
    normalized = normalize_keyword("설봉산 별빛축제")
    matches = [row for row in candidates if row.normalized_candidate == normalized]

    assert len(matches) == 1
    assert matches[0].candidate_type == "specific_phrase"


def test_suffix_dictionary_categories_and_count() -> None:
    categories = phrase_suffix_categories()

    assert set(categories) == {"EVENT", "TRAVEL", "TREND_MODIFIER"}
    assert len(phrase_suffixes()) == 24
    assert {"축제", "촬영지", "여행", "챌린지"} <= set(phrase_suffixes())


def test_agency_word_does_not_collide_with_exhibition_suffix() -> None:
    candidates = extract_candidates(
        title="금융전문 채용업체 카본에이전시",
        body="",
        tokenizer=RegexFallbackTokenizer(),
    )

    assert not any(row.extractor == "phrase_pattern" for row in candidates)


@pytest.mark.parametrize(
    "phrase",
    ("정부 지원 축제", "지역 방문 관광"),
)
def test_generic_components_inside_suffix_phrase_remain_rejected(phrase: str) -> None:
    assert _candidate(phrase, phrase).accepted is False


def test_quoted_generic_suffix_and_news_colon_are_not_structurally_protected() -> None:
    candidates = extract_candidates(
        title="정부 발표: 가격 안정 대책",
        body='관계자는 "지역 경제 회복 관광"을 강조했다.',
        tokenizer=RegexFallbackTokenizer(),
    )

    assert not any(row.extractor == "structural_phrase" for row in candidates)


def test_structural_phrase_does_not_discard_matching_ner_evidence() -> None:
    candidates = extract_candidates(
        title="신작 '서울역' 공개",
        body="",
        tokenizer=RegexFallbackTokenizer(),
        entities=[EntityEvidence("서울역", "LOCATION", 0.91)],
    )
    match = next(row for row in candidates if row.normalized_candidate == "서울역")

    assert match.extractor == "ner"
    assert match.entity_type == "LOCATION"


def test_noisy_leading_title_keeps_shorter_event_anchor() -> None:
    candidate = _candidate("이천시 설봉산 별빛축제 개최", "설봉산 별빛축제")

    assert candidate.accepted is True
    assert candidate.extractor == "phrase_pattern"

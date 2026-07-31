from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.keywords.candidate_extractor import clean_analysis_text
from app.keywords.keyword_normalizer import canonical_display, normalize_keyword
from app.keywords.tokenizer import RegexFallbackTokenizer, Token
from app.models.entity_context import EntityContext
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.search_interest_observation import SearchInterestObservation
from app.models.search_validation_result import SearchValidationResult
from app.models.source_document import SourceDocument
from app.models.weekly_trend import WeeklyTrend
from app.services.keyword_extraction_v2_service import analyze_documents
from app.services.keyword_rebuild_service import rebuild_keywords
from app.services.trend_scoring_service import calculate_final_score
from dashboard.formatters import format_search_interest, trend_dataframe


NOW = datetime(2026, 8, 1, 9, 0, 0)


class FakeTokenizer:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.calls: list[str] = []

    def tokenize(self, text: str) -> list[Token]:
        self.calls.append(text)
        return self.tokens if text else []


def _document(
    document_id: int,
    text: str,
    *,
    source: str = "youtube",
    entities: list | None = None,
    title: str | None = None,
):
    return SimpleNamespace(
        id=document_id,
        title=title if title is not None else text,
        text="" if title is None else text,
        source=source,
        published_at=NOW,
        entity_mentions=entities or [],
    )


@pytest.mark.parametrize(
    "raw, absent",
    [
        ("링크 https://example.com/a?utm_source=x 이후", ["https", "example", "com", "utm"]),
        ("<style>.x{}</style><b>부산</b><script>alert(1)</script>", ["style", "script", "alert"]),
        ("[제주](https://example.com) test@example.com", ["example", "com"]),
    ],
)
def test_analysis_text_removes_url_html_email_and_markdown(raw, absent) -> None:
    cleaned = clean_analysis_text(raw).lower()
    assert all(value not in cleaned for value in absent)


@pytest.mark.parametrize(
    "text, rejected",
    [
        ("있습니다 동시에 증가 방문 찾는", {"있습니다", "동시에", "증가", "방문", "찾는"}),
        ("The video on official music channel by company", {"the", "video", "on", "official", "music", "channel", "by", "company"}),
        ("http https www com net org co kr html php amp utm source medium campaign", {"http", "https", "www", "com"}),
        ("2026 7 가", {"2026", "7", "가"}),
    ],
)
def test_stopword_artifact_numeric_and_short_rejections(text, rejected) -> None:
    result = analyze_documents(
        [_document(1, text)], tokenizer=RegexFallbackTokenizer(), now=NOW
    )
    accepted = {row.normalized_keyword for row in result.accepted}
    assert not (accepted & {normalize_keyword(value) for value in rejected})


def test_fake_tokenizer_is_injected() -> None:
    fake = FakeTokenizer([Token("고유명사", "NNP", 0, 4)])
    result = analyze_documents([_document(1, "고유명사")], tokenizer=fake, now=NOW)

    assert fake.calls
    assert "고유명사" in {row.normalized_keyword for row in result.accepted}


def test_kiwi_disabled_uses_fallback(monkeypatch) -> None:
    monkeypatch.setenv("KEYWORD_ENABLE_KIWI", "false")
    get_settings.cache_clear()
    try:
        result = analyze_documents([_document(1, "부산불꽃축제")], now=NOW)
    finally:
        get_settings.cache_clear()
    assert "부산불꽃축제" in {row.normalized_keyword for row in result.accepted}


@pytest.mark.parametrize(
    "sentence, expected, excluded",
    [
        (
            "폭싹 속았수다 촬영지로 알려진 제주 금오름에 방문객이 증가하고 있습니다.",
            {"폭싹속았수다", "폭싹속았수다촬영지", "제주금오름", "금오름"},
            {"알려진", "방문객", "증가", "있습니다"},
        ),
        (
            "부산불꽃축제가 광안리해수욕장에서 열리며 관련 영상이 YouTube.com에 공개됐다.",
            {"부산불꽃축제", "광안리해수욕장"},
            {"com", "관련", "영상", "공개됐다"},
        ),
        (
            "두바이 초콜릿 챌린지가 성수동 카페를 중심으로 유행하고 있다.",
            {"두바이초콜릿", "두바이초콜릿챌린지", "성수동"},
            {"중심", "유행", "있다"},
        ),
        (
            "The video was uploaded on the official music channel by the company.",
            set(),
            {"the", "video", "on", "official", "music", "channel", "by", "company"},
        ),
        (
            "거제야호 밈으로 거제 여행에 대한 관심이 높아지고 있다.",
            {"거제야호", "거제"},
            {"밈", "여행", "대한", "관심", "높아지고", "있다"},
        ),
    ],
)
def test_required_quality_samples(sentence, expected, excluded) -> None:
    result = analyze_documents(
        [_document(1, sentence)], tokenizer=RegexFallbackTokenizer(), now=NOW
    )
    accepted = {row.normalized_keyword for row in result.accepted}
    assert expected <= accepted
    assert not (accepted & excluded)


@pytest.mark.parametrize("entity_type", ["CONTENT_TITLE", "EVENT", "FOOD", "MEME"])
def test_high_priority_entity_types_are_candidates(entity_type) -> None:
    entity = SimpleNamespace(text="테스트 객체", entity_type=entity_type, confidence=0.95)
    result = analyze_documents(
        [
            _document(1, "테스트 객체", entities=[entity]),
            _document(2, "테스트 객체", source="newsis_rss", entities=[entity]),
        ],
        tokenizer=RegexFallbackTokenizer(),
        now=NOW,
    )
    candidate = next(row for row in result.candidates if row.entity_type == entity_type)
    assert candidate.extractor == "ner"
    assert candidate.accepted is True


def test_ner_match_adds_quality_weight() -> None:
    entity = SimpleNamespace(text="콜드플레이", entity_type="CONTENT_TITLE", confidence=0.95)
    plain = analyze_documents(
        [_document(1, "콜드플레이"), _document(2, "콜드플레이")],
        tokenizer=FakeTokenizer([Token("콜드플레이", "NNP", 0, 5)]),
        now=NOW,
    )
    with_ner = analyze_documents(
        [
            _document(3, "콜드플레이", entities=[entity]),
            _document(4, "콜드플레이", entities=[entity]),
        ],
        tokenizer=FakeTokenizer([Token("콜드플레이", "NNP", 0, 5)]),
        now=NOW,
    )
    assert with_ner.accepted[0].quality_score > plain.accepted[0].quality_score


def test_wrong_single_body_ner_is_suppressed() -> None:
    entity = SimpleNamespace(text="메이드카페", entity_type="MEME", confidence=0.6)
    result = analyze_documents(
        [_document(1, "메이드카페", title="일반 제목", entities=[entity])],
        tokenizer=RegexFallbackTokenizer(),
        now=NOW,
    )
    candidate = next(row for row in result.candidates if row.entity_type == "MEME")
    assert candidate.accepted is False


def test_title_ner_document_and_source_weights_raise_quality() -> None:
    body_only = _document(1, "콜드플레이", title="일반 제목")
    title_one = _document(2, "콜드플레이")
    title_many = [
        _document(3, "콜드플레이", source="youtube"),
        _document(4, "콜드플레이", source="newsis_rss"),
    ]
    tokenizer = FakeTokenizer([Token("콜드플레이", "NNP", 0, 5)])
    one = analyze_documents([body_only], tokenizer=tokenizer, now=NOW)
    titled = analyze_documents([title_one], tokenizer=tokenizer, now=NOW)
    many = analyze_documents(title_many, tokenizer=tokenizer, now=NOW)
    one_score = max((row.quality_score for row in one.candidates if row.normalized_candidate == "콜드플레이"), default=0)
    title_score = max(row.quality_score for row in titled.candidates if row.normalized_candidate == "콜드플레이")
    many_score = max(row.quality_score for row in many.candidates if row.normalized_candidate == "콜드플레이")
    assert title_score > one_score
    assert many_score > title_score


def test_canonical_normalization_and_space_deduplication() -> None:
    assert normalize_keyword("부산 불꽃 축제") == normalize_keyword("부산불꽃축제")
    assert normalize_keyword("부산  불꽃축제") == "부산불꽃축제"
    assert canonical_display(["부산불꽃축제", "부산 불꽃 축제", "부산 불꽃 축제"]) == "부산 불꽃 축제"


def test_quality_preview_and_rebuild_dry_run_do_not_change_database(client, db_session) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")
    client.post("/api/trends/recalculate")
    before = _counts(db_session)

    preview = client.post("/api/keywords/quality-preview?limit=300&since_days=14")
    rebuild = client.post("/api/keywords/rebuild?dry_run=true&since_days=14")
    db_session.expire_all()

    assert preview.status_code == 200
    assert preview.json()["processed_documents"] == 120
    assert rebuild.status_code == 200
    assert rebuild.json()["dry_run"] is True
    assert _counts(db_session) == before


def test_rebuild_requires_force_when_not_dry_run(client) -> None:
    client.post("/api/collect/mock")
    response = client.post("/api/keywords/rebuild?dry_run=false&force=false")
    assert response.status_code == 400


def test_rebuild_transaction_rolls_back(monkeypatch, client, db_session) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")
    before = _counts(db_session)

    def fail(*_args, **_kwargs):
        raise RuntimeError("forced rollback")

    monkeypatch.setattr("app.services.keyword_rebuild_service._reconnect_search", fail)
    with pytest.raises(RuntimeError):
        rebuild_keywords(
            db_session,
            week_start=date(2026, 7, 22),
            since_days=14,
            dry_run=False,
            force=True,
            limit=500,
        )
    assert _counts(db_session) == before


def test_rebuild_is_week_scoped_and_preserves_source_and_manual_context(
    monkeypatch, client, db_session
) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")
    client.post("/api/trends/recalculate")
    current = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == "거제야호"))
    assert current is not None
    old_values = {
        column.name: getattr(current, column.name)
        for column in WeeklyTrend.__table__.columns
        if column.name != "id"
    }
    old_values.update(
        keyword="이전주보존",
        week_start=date(2026, 7, 15),
        week_end=date(2026, 7, 21),
    )
    db_session.add(WeeklyTrend(**old_values))
    db_session.add(
        EntityContext(
            normalized_entity="수동맥락",
            entity_text="수동맥락",
            entity_type="PLACE",
            provider="manual",
            page_id=None,
            page_title="수동맥락",
            page_url="https://example.test/manual-context",
            summary="사용자가 입력한 수동 맥락",
            description=None,
            match_score=1.0,
            match_status="manual",
            source_language="ko",
            license_name=None,
            attribution_text=None,
            revision_id=None,
            retrieved_at=NOW,
            updated_at=NOW,
        )
    )
    db_session.commit()
    source_count = db_session.scalar(select(func.count(SourceDocument.id)))

    async def forbidden_generate(*_args, **_kwargs):
        raise AssertionError("Gemini must not run during rebuild")

    monkeypatch.setattr(
        "app.ai.gemini_adapter.GeminiAdapter.generate", forbidden_generate
    )
    result = rebuild_keywords(
        db_session,
        week_start=date(2026, 7, 22),
        since_days=14,
        dry_run=False,
        force=True,
        limit=500,
    )

    assert result.status == "ok"
    assert db_session.scalar(
        select(WeeklyTrend).where(
            WeeklyTrend.keyword == "이전주보존",
            WeeklyTrend.week_start == date(2026, 7, 15),
        )
    ) is not None
    assert db_session.scalar(select(func.count(SourceDocument.id))) == source_count
    assert db_session.scalar(
        select(func.count(EntityContext.id)).where(EntityContext.match_status == "manual")
    ) == 1
    assert (
        db_session.scalar(
            select(func.count(SearchValidationResult.id)).where(
                SearchValidationResult.provider_count == 0
            )
        )
        == 0
    )


def test_missing_search_score_is_renormalized_and_real_fifty_is_distinct() -> None:
    values = dict(
        volume_score=80,
        growth_score=70,
        persistence_score=60,
        diversity_score=50,
        freshness_score=40,
        one_day_spike_penalty=0,
        spam_penalty=0,
    )
    missing = calculate_final_score(**values, search_interest_score=None)
    actual_fifty = calculate_final_score(**values, search_interest_score=50)
    assert missing != actual_fifty
    assert format_search_interest(None) == "미검증"
    assert format_search_interest(50) == "50.00"


def test_trend_score_mapping_and_dashboard_formatter(client, db_session) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")
    client.post("/api/trends/recalculate")
    response = client.get("/api/dashboard/trends?limit=100")
    item = next(row for row in response.json()["items"] if row["keyword"] == "거제야호")
    frame = trend_dataframe([item])
    assert item["trend_score"] is not None
    assert item["keyword_quality_score"] >= 45
    assert frame.iloc[0]["trend_score"] != "-"
    assert frame.iloc[0]["검색 관심도"] == "미검증"


def test_incomplete_legacy_final_score_is_not_presented_as_valid(client, db_session) -> None:
    client.post("/api/collect/mock")
    client.post("/api/keywords/extract")
    client.post("/api/trends/recalculate")
    trend = db_session.scalar(select(WeeklyTrend).where(WeeklyTrend.keyword == "거제야호"))
    assert trend is not None
    trend.trend_score = None
    trend.pipeline_version = "legacy"
    db_session.commit()

    default_items = client.get("/api/dashboard/trends?limit=100").json()["items"]
    assert all(row["keyword"] != "거제야호" for row in default_items)

    item = next(
        row
        for row in client.get(
            "/api/dashboard/trends?limit=100&include_low_quality=true"
        ).json()["items"]
        if row["keyword"] == "거제야호"
    )
    assert item["trend_score"] is None
    assert item["final_score"] is None


def _counts(session) -> tuple[int, int, int, int]:
    return (
        session.scalar(select(func.count(SourceDocument.id))) or 0,
        session.scalar(select(func.count(KeywordOccurrence.id))) or 0,
        session.scalar(select(func.count(WeeklyTrend.id))) or 0,
        session.scalar(select(func.count(SearchInterestObservation.id))) or 0,
    )

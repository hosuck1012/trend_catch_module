import asyncio
from datetime import datetime

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.models.entity_extraction_state import EntityExtractionState
from app.models.entity_mention import EntityMention
from app.models.source_document import SourceDocument
from app.ner.entity_dictionary import extract_dictionary_entities
from app.ner.entity_labels import EntityType
from app.ner.entity_resolver import resolve_entities
from app.ner.entity_rules import extract_rule_entities
from app.ner.gliner_adapter import GlinerAdapter
from app.ner.text_chunking import MODEL_SAFE_MAX_CHARS, build_ner_chunks
from app.services.entity_extraction_service import (
    _shift_candidate,
    extract_entities,
    ner_input_hash,
)


GOLD_CASES = (
    (
        "경남고성공룡세계엑스포가 41일간 고성에서 열린다.",
        (("경남고성공룡세계엑스포", EntityType.EVENT), ("고성", EntityType.LOCATION)),
    ),
    (
        "설봉산 별빛축제가 설봉공원에서 열린다.",
        (("설봉산 별빛축제", EntityType.EVENT), ("설봉공원", EntityType.PLACE)),
    ),
    (
        "펜타포트 락 페스티벌이 송도에서 개최된다.",
        (("펜타포트 락 페스티벌", EntityType.EVENT), ("송도", EntityType.LOCATION)),
    ),
    (
        "드라마 '폭싹 속았수다' 촬영지를 찾는 여행객이 늘었다.",
        (("폭싹 속았수다", EntityType.CONTENT_TITLE),),
    ),
    (
        "두바이 초콜릿 챌린지가 SNS에서 유행하고 있다.",
        (("두바이 초콜릿", EntityType.FOOD), ("두바이 초콜릿 챌린지", EntityType.MEME)),
    ),
    (
        "한강 밤핑 행사가 여의도에서 열린다.",
        (("한강 밤핑", EntityType.MEME), ("여의도", EntityType.LOCATION)),
    ),
    (
        "2026 한국 대만 청소년 교류 음악회가 열린다.",
        (("2026 한국 대만 청소년 교류 음악회", EntityType.EVENT),),
    ),
    (
        "경주 관광특구와 강천산힐링스파를 방문한다.",
        (("경주 관광특구", EntityType.PLACE), ("강천산힐링스파", EntityType.PLACE)),
    ),
)


@pytest.mark.parametrize(("text", "expected"), GOLD_CASES)
def test_gold_entity_recall_rules_and_dictionary(text, expected) -> None:
    candidates = resolve_entities(
        [*extract_dictionary_entities(text), *extract_rule_entities(text)]
    )

    for expected_text, expected_type in expected:
        assert any(
            row.text == expected_text and row.entity_type == expected_type
            for row in candidates
        )


@pytest.mark.parametrize(
    "text",
    (
        "가격 상승",
        "친절한 직원",
        "정부 지원",
        "실적 발표",
        "주가 전망",
        "법적 분쟁",
        "대통령 발언",
        "사고 발생",
        "최근 증가",
        "작품이 박물관에 전시되어 있다",
        "금융전문 채용업체 카본에이전시",
        "10개 공연",
        "2026년 상반기 공연",
        "국내 미술관 전시",
    ),
)
def test_negative_text_does_not_create_topic_entities(text: str) -> None:
    candidates = extract_rule_entities(text)

    assert not any(
        row.entity_type in {EntityType.EVENT, EntityType.CONTENT_TITLE, EntityType.MEME}
        for row in candidates
    )


def test_reporting_quotes_are_not_paired_as_content_titles() -> None:
    text = (
        '\uc791\ud488\uc740 \uc9c8\ubb38\uc5d0\uc11c \ucd9c\ubc1c\ud55c \uc791\ud488"\uc774\ub77c\uba70 '
        '"\uc544\uc774\ub4e4\uc5d0\uac8c \uc0c1\uc0c1\ub825\uc744 \uc804\ud558\uace0\uc790 \ud588\ub2e4"\uace0 \uc804\ud588\ub2e4.'
    )

    candidates = extract_rule_entities(text)

    assert not any(row.entity_type == EntityType.CONTENT_TITLE for row in candidates)


@pytest.mark.parametrize(
    ("text", "expected", "entity_type"),
    (
        ("\ub450\ubc14\uc774\ucd08\ucf5c\ub9bf\ucc4c\ub9b0\uc9c0\uac00 \ud655\uc0b0\ub410\ub2e4.", "\ub450\ubc14\uc774\ucd08\ucf5c\ub9bf\ucc4c\ub9b0\uc9c0", EntityType.MEME),
        ("\ud3ed\uc2f9\uc18d\uc558\uc218\ub2e4\ucd2c\uc601\uc9c0 \ucf54\uc2a4\ub2e4.", "\ud3ed\uc2f9\uc18d\uc558\uc218\ub2e4\ucd2c\uc601\uc9c0", EntityType.CONTENT_TITLE),
        ("\ud2b9\ubcc4\uc804 '\ubc45\ud06c\uc2dc: \uc2a4\ud2f8 \ud788\uc5b4'\uac00 \uc5f4\ub9b0\ub2e4.", "\ubc45\ud06c\uc2dc: \uc2a4\ud2f8 \ud788\uc5b4", EntityType.CONTENT_TITLE),
    ),
)
def test_protected_entities_allow_compact_spacing_and_title_separator(
    text: str,
    expected: str,
    entity_type: EntityType,
) -> None:
    candidates = extract_rule_entities(text)

    assert any(row.text == expected and row.entity_type == entity_type for row in candidates)


@pytest.mark.parametrize("text", ("\ub300\uc911\uc74c\uc545\uc774 \uacf5\uc5f0\uc744 \uc774\ub04c\uc5c8\ub2e4.", "\uc9c0\uc5ed\uc774 \uacf5\uc5f0 \uc2dc\uc7a5\uc744 \ubd84\uc11d\ud588\ub2e4."))
def test_common_nouns_are_not_person_context_entities(text: str) -> None:
    candidates = extract_rule_entities(text)

    assert not any(row.entity_type == EntityType.PERSON for row in candidates)


@pytest.mark.parametrize("text", ("지역", "서울시장 발언", "부산"))
def test_short_place_suffix_does_not_match_generic_or_person_title(text: str) -> None:
    candidates = extract_rule_entities(text)

    assert not any(row.entity_type == EntityType.PLACE for row in candidates)


def test_title_and_sentence_chunks_preserve_offsets_and_merge_overlap() -> None:
    sentence = "설봉산 별빛축제가 설봉공원에서 열린다."
    body = f"{'A' * 55}. {sentence} {'B' * 55}."
    chunks = build_ner_chunks("여행 기사 제목", body, max_chars=100)
    occurrences = []
    for chunk in chunks:
        for candidate in extract_rule_entities(chunk.text):
            if candidate.entity_type == EntityType.EVENT:
                occurrences.append(_shift_candidate(candidate, chunk.start_char))

    matching = [row for row in occurrences if row.text == "설봉산 별빛축제"]
    assert chunks[0].kind == "title"
    assert len(matching) == 2
    assert matching[0].start_char == matching[1].start_char
    assert len(resolve_entities(matching)) == 1


def test_chunking_respects_gliner_safe_ceiling() -> None:
    body = " ".join(["\uae34\uc774\uad00\uad11\ubb38\uc7a5"] * 400)

    chunks = build_ner_chunks("\uc81c\ubaa9", body, max_chars=1500)

    assert chunks
    assert max(len(chunk.text) for chunk in chunks) <= MODEL_SAFE_MAX_CHARS


def test_adapter_uses_descriptive_labels_threshold_and_single_load(monkeypatch) -> None:
    loads = 0

    class FakeModel:
        def batch_predict_entities(self, texts, labels, threshold):
            assert len(texts) == 2
            assert any("named festival" in label for label in labels)
            assert threshold == 0.45
            return [[], []]

    def loader(_name, _device):
        nonlocal loads
        loads += 1
        return FakeModel()

    monkeypatch.setenv("NER_ENABLED", "true")
    get_settings.cache_clear()
    adapter = GlinerAdapter(loader)

    asyncio.run(adapter.predict(["one", "two"]))
    asyncio.run(adapter.predict(["one", "two"]))

    assert loads == 1
    assert adapter.get_status().model_load_count == 1


def test_input_hash_includes_pipeline_model_labels_and_document(monkeypatch) -> None:
    values = dict(
        title="title",
        body="body",
        model_name="model-a",
        model_enabled=True,
        threshold=0.45,
        max_chars=1500,
    )
    first = ner_input_hash(**values)

    assert first != ner_input_hash(**(values | {"model_name": "model-b"}))
    assert first != ner_input_hash(**(values | {"body": "changed"}))
    monkeypatch.setattr("app.services.entity_extraction_service.NER_PIPELINE_VERSION", "v3")
    assert first != ner_input_hash(**values)


def test_process_all_cursor_is_restart_safe_and_force_false(monkeypatch, db_session) -> None:
    monkeypatch.setenv("NER_ENABLED", "false")
    get_settings.cache_clear()
    for index in range(5):
        _add_document(db_session, index, f"제{index}회 서울 여름축제")

    first = asyncio.run(
        extract_entities(
            limit=1,
            force=False,
            source=None,
            since_days=30,
            dry_run=False,
            batch_size=2,
            process_all=True,
        )
    )
    state_times = dict(
        db_session.execute(
            select(
                EntityExtractionState.document_id,
                EntityExtractionState.processed_at,
            )
        ).all()
    )
    entity_count = db_session.scalar(select(func.count()).select_from(EntityMention))
    second = asyncio.run(
        extract_entities(
            limit=1,
            force=False,
            source=None,
            since_days=30,
            dry_run=False,
            batch_size=2,
            process_all=True,
        )
    )

    assert first.processed_documents == 5
    assert first.batches == 3
    assert second.processed_documents == 0
    assert second.skipped_documents == 5
    assert second.inserted_entities == 0
    assert db_session.scalar(select(func.count()).select_from(EntityMention)) == entity_count
    assert dict(
        db_session.execute(
            select(
                EntityExtractionState.document_id,
                EntityExtractionState.processed_at,
            )
        ).all()
    ) == state_times


def test_dry_run_leaves_mentions_and_state_unchanged(monkeypatch, db_session) -> None:
    monkeypatch.setenv("NER_ENABLED", "false")
    get_settings.cache_clear()
    _add_document(db_session, 1, "서울 여름축제")

    result = asyncio.run(
        extract_entities(
            limit=10,
            force=False,
            source=None,
            since_days=30,
            dry_run=True,
            batch_size=2,
            process_all=True,
        )
    )

    assert result.dry_run is True
    assert result.mentions_detected > 0
    assert result.inserted_entities == 0
    assert db_session.scalar(select(func.count()).select_from(EntityMention)) == 0
    assert db_session.scalar(select(func.count()).select_from(EntityExtractionState)) == 0


def test_extract_api_supports_dry_run_pagination(client, db_session) -> None:
    for index in range(3):
        _add_document(db_session, index, f"제{index}회 서울 여름축제")

    response = client.post(
        "/api/entities/extract?dry_run=true&process_all=true&batch_size=2&since_days=30"
    )
    payload = response.json()

    assert response.status_code == 200
    assert payload["processed_documents"] == 3
    assert payload["batches"] == 2
    assert payload["inserted_entities"] == 0
    assert db_session.scalar(select(func.count()).select_from(EntityMention)) == 0


def _add_document(session, index: int, title: str) -> SourceDocument:
    now = datetime.now()
    document = SourceDocument(
        source="newsis_rss",
        source_id=f"ner-v2-{index}",
        title=title,
        text="행사 준비 소식입니다.",
        published_at=now,
        collected_at=now,
        views=None,
        likes=None,
        comments=None,
        url=None,
    )
    session.add(document)
    session.commit()
    return document

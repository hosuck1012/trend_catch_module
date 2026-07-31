import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.config import get_settings
from app.models.entity_mention import EntityMention
from app.models.keyword_occurrence import KeywordOccurrence
from app.models.source_document import SourceDocument
from app.models.trend_entity_link import TrendEntityLink
from app.models.weekly_trend import WeeklyTrend
from app.ner.entity_dictionary import extract_dictionary_entities
from app.ner.entity_labels import (
    ENTITY_LABEL_DESCRIPTIONS,
    GLINER_LABEL_TO_ENTITY_TYPE,
    EntityCandidate,
    EntityType,
)
from app.ner.entity_resolver import resolve_entities
from app.ner.entity_rules import extract_rule_entities
from app.ner.gliner_adapter import GlinerAdapter
from app.services.entity_extraction_service import extract_entities
from app.services.trend_entity_link_service import (
    _calculate_links,
    link_trends_to_entities,
)


SAMPLES = (
    "폭싹 속았수다 촬영지로 제주 금오름이 화제다.",
    "부산불꽃축제가 광안리해수욕장에서 열린다.",
    "성수동에서 두바이 초콜릿 챌린지가 유행하고 있다.",
    "아이유가 서울월드컵경기장에서 콘서트를 연다.",
    "거제야호 밈 때문에 거제 여행이 주목받고 있다.",
)


def test_gliner_label_mapping_is_complete() -> None:
    assert set(ENTITY_LABEL_DESCRIPTIONS) == set(EntityType)
    assert set(GLINER_LABEL_TO_ENTITY_TYPE.values()) == set(EntityType)
    assert GLINER_LABEL_TO_ENTITY_TYPE[
        "city, province, country, district or geographic region"
    ] == EntityType.LOCATION


def test_ner_batch_size_default_is_one(monkeypatch) -> None:
    monkeypatch.delenv("NER_BATCH_SIZE", raising=False)
    get_settings.cache_clear()

    assert get_settings().ner_batch_size == 1


def test_ner_disabled_does_not_load_model(monkeypatch) -> None:
    called = 0

    def loader(_name, _device):
        nonlocal called
        called += 1
        raise AssertionError("model loader must not run")

    monkeypatch.setenv("NER_ENABLED", "false")
    get_settings.cache_clear()
    adapter = GlinerAdapter(loader)

    result = asyncio.run(adapter.predict([SAMPLES[0]]))

    assert result == [[]]
    assert called == 0
    assert adapter.get_status().status == "disabled"


def test_model_lazy_loading_and_singleton_reuse(monkeypatch) -> None:
    calls = 0

    class FakeModel:
        def predict_entities(self, text, labels, threshold):
            return [
                {
                    "text": "아이유",
                    "label": ENTITY_LABEL_DESCRIPTIONS[EntityType.PERSON],
                    "score": 0.91,
                    "start": 0,
                    "end": 3,
                }
            ]

    def loader(_name, device):
        nonlocal calls
        calls += 1
        assert device == "cpu"
        return FakeModel()

    monkeypatch.setenv("NER_ENABLED", "true")
    get_settings.cache_clear()
    adapter = GlinerAdapter(loader)
    assert adapter.get_status().status == "not_loaded"

    first = asyncio.run(adapter.predict([SAMPLES[3]]))
    second = asyncio.run(adapter.predict([SAMPLES[3]]))

    assert calls == 1
    assert first[0][0].entity_type == EntityType.PERSON
    assert second[0][0].text == "아이유"
    assert adapter.get_status().status == "ready"


def test_gliner_adapter_removes_korean_particle_from_span() -> None:
    converted = GlinerAdapter._convert_predictions(
        [
            {
                "text": "광안리해수욕장에서",
                "label": ENTITY_LABEL_DESCRIPTIONS[EntityType.PLACE],
                "score": 0.9,
                "start": 3,
                "end": 12,
            }
        ]
    )

    assert converted[0].text == "광안리해수욕장"
    assert converted[0].start_char == 3
    assert converted[0].end_char == 10


def test_model_loading_failure_keeps_rule_extraction(monkeypatch, db_session) -> None:
    _add_document(db_session, title=SAMPLES[1])
    monkeypatch.setenv("NER_ENABLED", "true")
    get_settings.cache_clear()
    adapter = GlinerAdapter(lambda _name, _device: (_ for _ in ()).throw(RuntimeError("load failed")))

    result = asyncio.run(
        extract_entities(
            limit=5,
            force=False,
            source=None,
            since_days=14,
            adapter=adapter,
        )
    )

    assert result.status == "partial_success"
    assert result.model_status == "error"
    assert result.inserted_entities >= 2
    assert result.errors == [result.model_error]
    assert {row.entity_type for row in db_session.scalars(select(EntityMention))} >= {
        "EVENT",
        "PLACE",
    }


def test_location_dictionary_and_canonical_normalization() -> None:
    candidates = resolve_entities(extract_dictionary_entities("제주도에서 서울시로 이동한다."))
    normalized = {item.normalized_text for item in candidates}

    assert "제주특별자치도" in normalized
    assert "서울특별시" in normalized
    assert all(item.confidence == 0.95 for item in candidates)


@pytest.mark.parametrize(
    ("text", "expected_text", "expected_type"),
    [
        (SAMPLES[0], "금오름", EntityType.PLACE),
        (SAMPLES[1], "부산불꽃축제", EntityType.EVENT),
        (SAMPLES[2], "두바이 초콜릿 챌린지", EntityType.MEME),
        (SAMPLES[3], "아이유", EntityType.PERSON),
        (SAMPLES[3], "서울월드컵경기장", EntityType.PLACE),
        (SAMPLES[0], "폭싹 속았수다", EntityType.CONTENT_TITLE),
    ],
)
def test_rule_extraction(text, expected_text, expected_type) -> None:
    candidates = extract_rule_entities(text)
    assert any(
        item.text == expected_text and item.entity_type == expected_type
        for item in candidates
    )


def test_duplicate_entity_merge_and_overlap_confidence_priority() -> None:
    candidates = [
        EntityCandidate("제주", EntityType.LOCATION, 0.80, "gliner", 0, 2),
        EntityCandidate("제주", EntityType.LOCATION, 0.95, "dictionary", 0, 2),
        EntityCandidate("제주", EntityType.PLACE, 0.70, "rule", 0, 2),
    ]

    resolved = resolve_entities(candidates)

    assert len(resolved) == 1
    assert resolved[0].entity_type == EntityType.LOCATION
    assert resolved[0].confidence == 0.95
    assert resolved[0].extractor == "merged"


def test_same_span_with_canonical_alias_keeps_higher_confidence() -> None:
    candidates = [
        EntityCandidate(
            "제주",
            EntityType.LOCATION,
            0.95,
            "dictionary",
            0,
            2,
            canonical_text="제주특별자치도",
        ),
        EntityCandidate("제주", EntityType.PLACE, 0.70, "gliner", 0, 2),
    ]

    resolved = resolve_entities(candidates)

    assert len(resolved) == 1
    assert resolved[0].normalized_text == "제주특별자치도"


def test_force_false_skips_and_force_true_reprocesses(monkeypatch, db_session) -> None:
    _add_document(db_session, title=SAMPLES[4])
    monkeypatch.setenv("NER_ENABLED", "false")
    get_settings.cache_clear()

    first = asyncio.run(extract_entities(limit=5, force=False, source=None, since_days=14))
    second = asyncio.run(extract_entities(limit=5, force=False, source=None, since_days=14))
    third = asyncio.run(extract_entities(limit=5, force=True, source=None, since_days=14))

    count = db_session.scalar(select(func.count(EntityMention.id)))
    assert first.inserted_entities > 0
    assert second.inserted_entities == 0
    assert second.skipped_documents == 1
    assert third.inserted_entities == first.inserted_entities
    assert count == first.inserted_entities


def test_extract_model_status_and_summary_apis(client, db_session) -> None:
    _add_document(db_session, title=SAMPLES[0], source="newsis_rss")

    before = client.get("/api/entities/model-status")
    extracted = client.post("/api/entities/extract?limit=5")
    summary = client.get("/api/entities/summary?entity_type=LOCATION")

    assert before.status_code == 200
    assert before.json()["status"] == "disabled"
    assert extracted.status_code == 200
    assert extracted.json()["model_status"] == "disabled"
    assert extracted.json()["inserted_entities"] >= 3
    assert extracted.json()["errors"] == []
    assert summary.status_code == 200
    assert summary.json()["total_entities"] >= 1
    assert summary.json()["items"][0]["canonical_text"] == "제주특별자치도"


def test_entity_mention_duplicate_save_is_prevented(client, db_session) -> None:
    _add_document(db_session, title=SAMPLES[1])

    client.post("/api/entities/extract?limit=5")
    first_count = db_session.scalar(select(func.count(EntityMention.id)))
    client.post("/api/entities/extract?limit=5")
    second_count = db_session.scalar(select(func.count(EntityMention.id)))

    assert first_count == second_count


def test_trend_entity_link_calculation_weight_and_primary(db_session) -> None:
    _seed_linkable_trend(db_session)

    result = link_trends_to_entities(db_session)
    links = list(db_session.scalars(select(TrendEntityLink)))
    primary = next(link for link in links if link.is_primary)

    assert result.status == "ok"
    assert result.inserted_links == 2
    assert primary.entity_type == "LOCATION"
    assert primary.normalized_entity == "제주특별자치도"
    assert primary.relation_score > next(
        link.relation_score for link in links if link.entity_type == "PERSON"
    )


@pytest.mark.parametrize("travel_type", ["LOCATION", "PLACE"])
def test_location_and_place_receive_travel_weight(travel_type) -> None:
    occurred_at = datetime(2026, 7, 30, 12)
    groups = {
        ("여행객체", travel_type): [
            SimpleNamespace(
                text="여행객체",
                document_id=1,
                source="youtube",
                confidence=0.5,
            )
        ],
        ("아이유", "PERSON"): [
            SimpleNamespace(
                text="아이유",
                document_id=1,
                source="youtube",
                confidence=0.5,
            )
        ],
    }

    links = _calculate_links(
        keyword="테스트",
        week_start=date(2026, 7, 25),
        week_end=date(2026, 7, 31),
        document_ids=[1],
        total_source_count=1,
        groups=groups,
        calculated_at=occurred_at,
    )
    by_type = {link["entity_type"]: link for link in links}

    assert by_type[travel_type]["relation_score"] == 99.0
    assert by_type["PERSON"]["relation_score"] == 90.0
    assert by_type[travel_type]["is_primary"] is True


def test_link_and_by_keyword_apis_and_weekly_extension(client, db_session) -> None:
    _seed_linkable_trend(db_session)

    linked = client.post("/api/entities/link-trends")
    detail = client.get("/api/entities/by-keyword/거제야호")
    weekly = client.get("/api/trends/weekly")

    assert linked.status_code == 200
    assert linked.json()["primary_links"] == 1
    assert detail.status_code == 200
    assert detail.json()["primary_entity"]["entity_type"] == "LOCATION"
    item = weekly.json()["items"][0]
    assert item["primary_entity_type"] == "LOCATION"
    assert item["travel_entity_count"] == 1


def test_by_keyword_404_and_unlinked_message(client, db_session) -> None:
    assert client.get("/api/entities/by-keyword/없는키워드").status_code == 404
    _add_weekly_trend(db_session, keyword="거제야호")

    response = client.get("/api/entities/by-keyword/거제야호")

    assert response.status_code == 200
    assert response.json()["entities"] == []
    assert "link-trends" in response.json()["message"]


def _add_document(
    session,
    *,
    title: str,
    text: str = "",
    source: str = "youtube",
) -> SourceDocument:
    now = datetime.now()
    document = SourceDocument(
        source=source,
        source_id=f"doc-{session.scalar(select(func.count(SourceDocument.id))) or 0}",
        title=title,
        text=text,
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


def _add_weekly_trend(session, *, keyword: str) -> WeeklyTrend:
    trend = WeeklyTrend(
        keyword=keyword,
        week_start=date(2026, 7, 25),
        week_end=date(2026, 7, 31),
        weekly_mentions=2,
        previous_weekly_mentions=1,
        active_days=2,
        source_count=2,
        growth_rate=1.0,
        peak_day_share=0.5,
        persistence_score=30,
        diversity_score=50,
        freshness_score=100,
        volume_score=40,
        growth_score=100,
        search_interest_score=50,
        one_day_spike_penalty=0,
        spam_penalty=0,
        final_score=60,
        status="weekly_trend",
        calculated_at=datetime.now(),
    )
    session.add(trend)
    session.commit()
    return trend


def _seed_linkable_trend(session) -> None:
    trend = _add_weekly_trend(session, keyword="거제야호")
    documents = [
        _add_document(session, title="거제야호 제주 여행", source="youtube"),
        _add_document(session, title="거제야호 제주 화제", source="newsis_rss"),
    ]
    for document in documents:
        document.published_at = datetime(2026, 7, 30, 12)
        session.add(
            KeywordOccurrence(
                document_id=document.id,
                keyword="거제야호",
                normalized_keyword="거제야호",
                source=document.source,
                occurred_at=datetime(2026, 7, 30, 12),
            )
        )
        session.add(
            EntityMention(
                document_id=document.id,
                text="제주",
                normalized_text="제주특별자치도",
                entity_type="LOCATION",
                confidence=0.95,
                extractor="dictionary",
                start_char=5,
                end_char=7,
                source=document.source,
                occurred_at=datetime(2026, 7, 30, 12),
                created_at=datetime.now(),
            )
        )
    session.add(
        EntityMention(
            document_id=documents[0].id,
            text="아이유",
            normalized_text="아이유",
            entity_type="PERSON",
            confidence=0.9,
            extractor="gliner",
            start_char=8,
            end_char=11,
            source="youtube",
            occurred_at=datetime(2026, 7, 30, 12),
            created_at=datetime.now(),
        )
    )
    session.commit()
    assert trend.status == "weekly_trend"

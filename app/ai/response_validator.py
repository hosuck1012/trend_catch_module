from dataclasses import dataclass

from app.ai.evidence_builder import EvidencePackage
from app.ai.gemini_schemas import DestinationSuggestion, TrendExplanation
from app.ner.entity_resolver import normalize_entity_text


@dataclass(frozen=True)
class ValidatedExplanation:
    explanation: TrendExplanation
    analysis_status: str
    removed_evidence_refs: tuple[str, ...]
    removed_destinations: tuple[str, ...]
    corrections: tuple[str, ...]


def validate_explanation(
    explanation: TrendExplanation,
    evidence: EvidencePackage,
) -> ValidatedExplanation:
    changed = False
    corrections: list[str] = []
    removed_refs = tuple(
        reference
        for reference in explanation.evidence_refs
        if reference not in evidence.valid_refs
    )
    valid_refs = list(
        dict.fromkeys(
            reference
            for reference in explanation.evidence_refs
            if reference in evidence.valid_refs
        )
    )
    if removed_refs:
        changed = True
        corrections.append("존재하지 않는 evidence reference를 제거했습니다.")

    entity_index = {
        normalize_entity_text(str(entity["entity_text"])): entity
        for entity in evidence.travel_entities
        if normalize_entity_text(str(entity["entity_text"]))
    }
    for entity in evidence.travel_entities:
        normalized = normalize_entity_text(str(entity["normalized_entity"]))
        if normalized:
            entity_index.setdefault(normalized, entity)
    context_titles: dict[str, set[str]] = {}
    for context in evidence.payload["contexts"]:
        normalized = normalize_entity_text(str(context["normalized_entity"]))
        title = normalize_entity_text(str(context["page_title"]))
        if normalized and title:
            context_titles.setdefault(normalized, set()).add(title)

    validated_destinations: list[DestinationSuggestion] = []
    removed_destinations: list[str] = []
    for destination in explanation.recommended_destinations:
        source_key = normalize_entity_text(destination.source_entity)
        entity = entity_index.get(source_key or "")
        if entity is None or destination.entity_type != entity["entity_type"]:
            removed_destinations.append(destination.name)
            continue
        name_key = normalize_entity_text(destination.name)
        entity_name_keys = {
            normalize_entity_text(str(entity["entity_text"])),
            normalize_entity_text(str(entity["normalized_entity"])),
        }
        normalized_entity = normalize_entity_text(str(entity["normalized_entity"])) or ""
        allowed_names = {item for item in entity_name_keys if item}
        allowed_names.update(context_titles.get(normalized_entity, set()))
        if name_key not in allowed_names:
            removed_destinations.append(destination.name)
            continue
        validated_destinations.append(
            destination.model_copy(
                update={
                    "relation_score": float(entity["relation_score"]),
                    "context_available": bool(context_titles.get(normalized_entity)),
                }
            )
        )
    if removed_destinations:
        changed = True
        corrections.append("입력 근거에 없는 추천 장소를 제거했습니다.")

    score = explanation.travel_relevance_score
    documents = evidence.payload.get("documents") or []
    if not documents or not valid_refs:
        if score > 40 or validated_destinations:
            changed = True
            corrections.append("유효한 문서 근거가 없어 여행 점수와 추천을 제한했습니다.")
        score = min(score, 40)
        validated_destinations = []
    if not evidence.travel_entities:
        if score > 40 or validated_destinations:
            changed = True
            corrections.append("여행 객체가 없어 여행 점수와 추천을 제한했습니다.")
        score = min(score, 40)
        validated_destinations = []
    high_travel_evidence = any(
        entity["entity_type"] in {"LOCATION", "PLACE"}
        and float(entity["relation_score"]) >= 70
        for entity in evidence.travel_entities
    )
    if score >= 75 and not high_travel_evidence:
        score = 74
        changed = True
        corrections.append("높은 여행 연관성 판정을 뒷받침할 장소 근거가 부족해 점수를 보정했습니다.")
    level = _level_for_score(score)
    if level != explanation.travel_relevance_level:
        changed = True
        corrections.append("여행 연관성 점수와 등급을 일치시켰습니다.")

    confidence = explanation.confidence_score
    if not valid_refs and confidence > 40:
        confidence = 40
        changed = True
        corrections.append("유효한 evidence reference가 없어 신뢰도 상한을 적용했습니다.")

    cautions = list(explanation.cautions)
    if evidence.input_truncated:
        changed = True
        _append_caution(cautions, "입력 길이 제한으로 일부 보조 근거가 제외되었습니다.")
    for correction in corrections:
        _append_caution(cautions, correction)

    corrected = explanation.model_copy(
        update={
            "evidence_refs": valid_refs,
            "recommended_destinations": validated_destinations,
            "travel_relevance_score": int(score),
            "travel_relevance_level": level,
            "confidence_score": int(confidence),
            "cautions": cautions[:5],
        }
    )
    return ValidatedExplanation(
        explanation=corrected,
        analysis_status="partial" if changed else "completed",
        removed_evidence_refs=removed_refs,
        removed_destinations=tuple(removed_destinations),
        corrections=tuple(corrections),
    )


def _level_for_score(score: int | float) -> str:
    if score >= 75:
        return "high"
    if score >= 50:
        return "medium"
    if score >= 20:
        return "low"
    return "none"


def _append_caution(cautions: list[str], value: str) -> None:
    if value not in cautions and len(cautions) < 5:
        cautions.append(value)

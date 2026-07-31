from dataclasses import dataclass, replace

from app.context.context_normalizer import context_tokens, normalize_context_text


TYPE_CONTEXT_TERMS = {
    "LOCATION": {"도시", "시", "군", "구", "지역", "도", "국가", "대한민국", "지리"},
    "PLACE": {"관광", "명소", "역", "공항", "해변", "공원", "시설", "장소", "건축"},
    "PERSON": {"배우", "가수", "인물", "연예인", "사람", "방송인"},
    "CONTENT_TITLE": {"영화", "드라마", "방송", "노래", "웹툰", "책", "작품"},
    "EVENT": {"축제", "콘서트", "전시", "마라톤", "행사", "공연"},
    "FOOD": {"음식", "요리", "디저트", "음료", "특산물"},
    "BRAND": {"기업", "회사", "제품", "브랜드", "단체", "조직"},
    "MEME": {"인터넷", "밈", "유행어", "챌린지", "온라인"},
}


@dataclass(frozen=True)
class ScoredCandidate:
    page_id: str | None
    page_title: str
    page_url: str
    snippet: str
    redirect_title: str | None
    match_score: float
    match_status: str


def generate_search_queries(
    *,
    entity_text: str,
    normalized_entity: str,
    entity_type: str,
    related_locations: list[str] | None = None,
) -> list[str]:
    base = entity_text.strip() or normalized_entity.strip()
    canonical = normalized_entity.strip()
    candidates = [canonical, base]
    location = next((item.strip() for item in related_locations or [] if item.strip()), "")
    suffix = {
        "PLACE": "관광지",
        "CONTENT_TITLE": "드라마",
        "EVENT": "행사",
        "FOOD": "음식",
        "BRAND": "브랜드",
        "MEME": "밈",
    }.get(entity_type)
    if entity_type == "PLACE" and location:
        candidates.append(f"{location} {base}")
    elif suffix:
        candidates.append(f"{base} {suffix}")

    queries: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = " ".join(candidate.split()).strip()
        key = normalize_context_text(cleaned)
        if not cleaned or not key or key in seen:
            continue
        seen.add(key)
        queries.append(cleaned)
        if len(queries) == 3:
            break
    return queries


def score_candidate(
    *,
    entity_text: str,
    normalized_entity: str,
    entity_type: str,
    context_text: str,
    page_id: str | None,
    page_title: str,
    page_url: str,
    snippet: str,
    redirect_title: str | None,
) -> ScoredCandidate:
    entity_raw = " ".join(entity_text.split()).strip().lower()
    canonical_raw = " ".join(normalized_entity.split()).strip().lower()
    title_raw = " ".join(page_title.split()).strip().lower()
    exact_match = title_raw in {entity_raw, canonical_raw}

    normalized_title = normalize_context_text(page_title, strip_parenthetical=True)
    normalized_values = {
        normalize_context_text(entity_text, strip_parenthetical=True),
        normalize_context_text(normalized_entity, strip_parenthetical=True),
    }
    normalized_match = bool(normalized_title and normalized_title in normalized_values)

    candidate_tokens = context_tokens(page_title, snippet)
    type_terms = TYPE_CONTEXT_TERMS.get(entity_type, set())
    type_match = bool(candidate_tokens & type_terms)
    source_tokens = context_tokens(entity_text, normalized_entity, context_text)
    overlap_ratio = (
        len(candidate_tokens & source_tokens) / max(len(source_tokens), 1)
        if source_tokens
        else 0.0
    )
    redirect_match = bool(
        redirect_title
        and normalize_context_text(redirect_title, strip_parenthetical=True)
        in normalized_values
    )

    score = (
        (0.40 if exact_match else 0.0)
        + (0.20 if normalized_match else 0.0)
        + (0.15 if type_match else 0.0)
        + min(overlap_ratio, 1.0) * 0.15
        + (0.10 if redirect_match else 0.0)
    )
    score = round(min(max(score, 0.0), 1.0), 4)
    return ScoredCandidate(
        page_id=page_id,
        page_title=page_title,
        page_url=page_url,
        snippet=snippet,
        redirect_title=redirect_title,
        match_score=score,
        match_status=_status_for_score(score, 0.70),
    )


def rank_candidates(
    candidates: list[ScoredCandidate], *, matched_threshold: float
) -> list[ScoredCandidate]:
    ranked = sorted(candidates, key=lambda item: item.match_score, reverse=True)
    ranked = [
        replace(item, match_status=_status_for_score(item.match_score, matched_threshold))
        for item in ranked
    ]
    if (
        len(ranked) >= 2
        and ranked[0].match_status == "matched"
        and ranked[0].match_score - ranked[1].match_score < 0.05
    ):
        ranked[0] = replace(ranked[0], match_status="ambiguous")
    return ranked


def _status_for_score(score: float, matched_threshold: float) -> str:
    if score >= matched_threshold:
        return "matched"
    if score >= 0.50:
        return "ambiguous"
    return "unmatched"

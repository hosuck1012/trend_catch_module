from app.keywords.tokenizer import Token
from app.keywords.keyword_normalizer import normalize_display, normalize_keyword
from app.keywords.phrase_signals import matching_phrase_suffix
from app.keywords.stopword_filter import is_generic


NOUN_TAG_PREFIXES = ("NNG", "NNP", "NNB", "NR", "SL", "SH")


def is_noun(token: Token) -> bool:
    return token.tag.startswith(NOUN_TAG_PREFIXES)


def noun_phrases(tokens: list[Token], *, max_terms: int = 4) -> list[str]:
    phrases: list[str] = []
    run: list[Token] = []
    for token in [*tokens, Token("", "BOUNDARY", 0, 0)]:
        if is_noun(token):
            run.append(token)
            continue
        phrases.extend(_phrases_from_run(run, max_terms=max_terms))
        run = []
    return phrases


def specific_phrases(text: str, tokens: list[Token], *, max_terms: int = 6) -> list[str]:
    """Return only anchored token runs ending in a configured topic suffix."""
    phrases: dict[str, str] = {}
    run: list[Token] = []
    for index, token in enumerate([*tokens, Token("", "BOUNDARY", 0, 0)]):
        if index < len(tokens) and _is_specific_token(tokens, index):
            run.append(token)
            continue
        _add_specific_phrases(text, run, phrases, max_terms=max_terms)
        run = []
    return list(phrases.values())


def _add_specific_phrases(
    text: str,
    run: list[Token],
    phrases: dict[str, str],
    *,
    max_terms: int,
) -> None:
    for end_index in range(len(run)):
        full_start = max(0, end_index - max_terms + 1)
        full_window = run[full_start : end_index + 1]
        raw = text[full_window[0].start : full_window[-1].end]
        suffix = matching_phrase_suffix(raw)
        if not suffix:
            continue
        start_indexes = {
            full_start,
            max(full_start, end_index - 2),
            max(full_start, end_index - 1),
        }
        for candidate_start in sorted(start_indexes):
            window = full_window[candidate_start - full_start :]
            while len(window) > 1:
                first = window[0]
                first_normalized = normalize_keyword(first.text) or ""
                if not first_normalized or not is_generic(first.text, first_normalized):
                    break
                window = window[1:]
            raw = normalize_display(text[window[0].start : window[-1].end])
            normalized = normalize_keyword(raw) or ""
            if not raw or not normalized or normalized == suffix:
                continue
            phrases.setdefault(normalized, raw)


def _is_specific_token(tokens: list[Token], index: int) -> bool:
    token = tokens[index]
    if is_noun(token) or token.tag == "SN":
        return True
    if token.tag.startswith("MM") and index + 1 < len(tokens):
        next_token = tokens[index + 1]
        return token.end == next_token.start and is_noun(next_token)
    return False


def _phrases_from_run(run: list[Token], *, max_terms: int) -> list[str]:
    result: list[str] = []
    for size in range(2, min(max_terms, len(run)) + 1):
        for index in range(len(run) - size + 1):
            values = [token.text for token in run[index : index + size]]
            result.append(" ".join(values))
            if all(_is_korean(value) for value in values):
                result.append("".join(values))
    return result


def _is_korean(value: str) -> bool:
    return bool(value) and all("가" <= char <= "힣" for char in value)

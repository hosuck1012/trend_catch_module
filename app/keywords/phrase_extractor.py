from app.keywords.tokenizer import Token


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

import re


_INVALID = re.compile(r"[^0-9A-Za-z가-힣\s]+")
_SPACES = re.compile(r"\s+")


def normalize_display(value: str) -> str:
    value = _INVALID.sub(" ", value.replace("#", " "))
    return _SPACES.sub(" ", value).strip()


def normalize_keyword(value: str) -> str | None:
    display = normalize_display(value).lower()
    normalized = display.replace(" ", "").replace("-", "").replace("_", "")
    return normalized or None


def canonical_display(forms: list[str]) -> str:
    cleaned = [normalize_display(form) for form in forms if normalize_display(form)]
    if not cleaned:
        return ""
    counts = {form: cleaned.count(form) for form in set(cleaned)}
    natural_forms = {
        form
        for form in counts
        if not (
            " " in form
            and all(len(part) == 1 for part in form.split())
            and form.replace(" ", "") in counts
        )
    }
    return max(
        natural_forms or counts.keys(),
        key=lambda form: (
            counts[form],
            bool(" " in form),
            -len(form),
            form,
        ),
    )

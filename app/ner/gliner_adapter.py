import asyncio
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable

from app.config import get_settings
from app.ner.entity_labels import (
    ENTITY_LABEL_DESCRIPTIONS,
    GLINER_LABEL_TO_ENTITY_TYPE,
    EntityCandidate,
)


class GlinerAdapterError(RuntimeError):
    pass


@dataclass(frozen=True)
class GlinerStatus:
    enabled: bool
    status: str
    model_name: str
    device: str
    threshold: float
    last_error: str | None


class GlinerAdapter:
    def __init__(self, model_loader: Callable[[str, str], Any] | None = None) -> None:
        self._model_loader = model_loader or _default_model_loader
        self._model: Any | None = None
        self._load_lock = Lock()
        self._status = "not_loaded"
        self._last_error: str | None = None
        self._loaded_model_name: str | None = None
        self._actual_device: str | None = None

    def get_status(self) -> GlinerStatus:
        settings = get_settings()
        if not settings.ner_enabled:
            status = "disabled"
        else:
            status = self._status
        return GlinerStatus(
            enabled=settings.ner_enabled,
            status=status,
            model_name=settings.ner_model_name,
            device=self._actual_device or settings.ner_device,
            threshold=settings.ner_threshold,
            last_error=self._last_error,
        )

    async def predict(self, texts: list[str]) -> list[list[EntityCandidate]]:
        settings = get_settings()
        if not settings.ner_enabled:
            return [[] for _ in texts]
        if not texts:
            return []
        try:
            model = await asyncio.to_thread(
                self._ensure_model,
                settings.ner_model_name,
                settings.ner_device,
            )
            return await asyncio.to_thread(
                self._predict_sync,
                model,
                texts,
                settings.ner_threshold,
            )
        except GlinerAdapterError:
            raise
        except Exception as exc:
            self._status = "error"
            self._last_error = _safe_error(exc)
            raise GlinerAdapterError(self._last_error) from exc

    def reset(self) -> None:
        with self._load_lock:
            self._model = None
            self._status = "not_loaded"
            self._last_error = None
            self._loaded_model_name = None
            self._actual_device = None

    def _ensure_model(self, model_name: str, requested_device: str) -> Any:
        with self._load_lock:
            if self._model is not None and self._loaded_model_name == model_name:
                return self._model
            if self._status == "error" and self._last_error:
                raise GlinerAdapterError(self._last_error)
            self._status = "loading"
            actual_device = _resolve_device(requested_device)
            try:
                model = self._model_loader(model_name, actual_device)
            except Exception as exc:
                self._status = "error"
                self._last_error = _safe_error(exc)
                raise GlinerAdapterError(self._last_error) from exc
            self._model = model
            self._loaded_model_name = model_name
            self._actual_device = actual_device
            self._status = "ready"
            self._last_error = None
            return model

    def _predict_sync(
        self,
        model: Any,
        texts: list[str],
        threshold: float,
    ) -> list[list[EntityCandidate]]:
        labels = list(GLINER_LABEL_TO_ENTITY_TYPE)
        if hasattr(model, "batch_predict_entities"):
            predictions = model.batch_predict_entities(
                texts,
                labels,
                threshold=threshold,
            )
        else:
            predictions = [
                model.predict_entities(text, labels, threshold=threshold)
                for text in texts
            ]
        return [self._convert_predictions(items) for items in predictions]

    @staticmethod
    def _convert_predictions(items: list[dict[str, Any]]) -> list[EntityCandidate]:
        candidates: list[EntityCandidate] = []
        for item in items:
            entity_type = GLINER_LABEL_TO_ENTITY_TYPE.get(str(item.get("label", "")))
            raw_text = str(item.get("text", ""))
            leading_spaces = len(raw_text) - len(raw_text.lstrip())
            text = raw_text.strip()
            start = _safe_int(item.get("start"))
            end = _safe_int(item.get("end"))
            text, removed_chars = _strip_korean_particle(text)
            if entity_type is None or not text:
                continue
            if start is not None:
                start += leading_spaces
            if end is not None:
                end -= removed_chars + (len(raw_text) - len(raw_text.rstrip()))
            candidates.append(
                EntityCandidate(
                    text=text,
                    entity_type=entity_type,
                    confidence=float(item.get("score", 0.0)),
                    extractor="gliner",
                    start_char=start,
                    end_char=end,
                )
            )
        return candidates


def _default_model_loader(model_name: str, device: str) -> Any:
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_name)
    if hasattr(model, "to"):
        model.to(device)
    return model


def _resolve_device(requested_device: str) -> str:
    normalized = (requested_device or "cpu").strip().lower()
    if normalized.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return normalized
        except Exception:
            pass
        return "cpu"
    return "cpu" if normalized not in {"cpu", "mps"} else normalized


def _safe_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)}"[:1000]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


KOREAN_PARTICLES = (
    "에서는",
    "에게서",
    "으로는",
    "까지는",
    "부터는",
    "에서",
    "에게",
    "께서",
    "으로",
    "와는",
    "과는",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "로",
    "와",
    "과",
    "도",
    "의",
)


def _strip_korean_particle(text: str) -> tuple[str, int]:
    for particle in KOREAN_PARTICLES:
        if text.endswith(particle) and len(text) - len(particle) >= 2:
            return text[: -len(particle)].rstrip(), len(particle)
    return text, 0


gliner_adapter = GlinerAdapter()

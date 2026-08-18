from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from threading import Lock
from typing import Protocol, runtime_checkable


class EmbeddingDisabledError(RuntimeError):
    """Raised when local embeddings are explicitly disabled."""


@runtime_checkable
class EmbeddingAdapter(Protocol):
    model_name: str
    enabled: bool

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode semantic anchors using the E5 query convention."""

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        """Encode candidate contexts using the E5 passage convention."""


class SentenceTransformerEmbeddingAdapter:
    """Lazy local E5 adapter with all model-specific formatting encapsulated."""

    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        batch_size: int,
        enabled: bool = True,
        model_factory: Callable[[str, str], object] | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.enabled = enabled
        self._model_factory = model_factory
        self._model: object | None = None
        self._load_lock = Lock()
        self._encode_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, prefix="query: ")

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, prefix="passage: ")

    def _encode(self, texts: Sequence[str], *, prefix: str) -> list[list[float]]:
        if not self.enabled:
            raise EmbeddingDisabledError("Local travel embeddings are disabled")
        if not texts:
            return []
        model = self._load_model()
        formatted = [f"{prefix}{text.strip()}" for text in texts]
        with self._encode_lock:
            embeddings = model.encode(
                formatted,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        return embeddings.tolist()

    def _load_model(self) -> object:
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                factory = self._model_factory or _default_model_factory
                self._model = factory(self.model_name, self.device)
        return self._model


class FakeEmbeddingAdapter:
    """Deterministic injectable adapter for tests; it never loads a model."""

    def __init__(
        self,
        vectorizer: Callable[[str], Sequence[float]],
        *,
        model_name: str = "fake-e5",
        enabled: bool = True,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self._vectorizer = vectorizer
        self.encoded_queries: list[str] = []
        self.encoded_passages: list[str] = []

    def encode_queries(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [f"query: {text.strip()}" for text in texts]
        self.encoded_queries.extend(formatted)
        return [self._normalized_vector(text) for text in formatted]

    def encode_passages(self, texts: Sequence[str]) -> list[list[float]]:
        formatted = [f"passage: {text.strip()}" for text in texts]
        self.encoded_passages.extend(formatted)
        return [self._normalized_vector(text) for text in formatted]

    def _normalized_vector(self, text: str) -> list[float]:
        if not self.enabled:
            raise EmbeddingDisabledError("Local travel embeddings are disabled")
        vector = [float(value) for value in self._vectorizer(text)]
        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


def _default_model_factory(model_name: str, device: str) -> object:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


_ADAPTER_CACHE: dict[
    tuple[str, str, int, bool], SentenceTransformerEmbeddingAdapter
] = {}
_ADAPTER_CACHE_LOCK = Lock()


def get_embedding_adapter(
    *,
    model_name: str,
    device: str,
    batch_size: int,
    enabled: bool,
) -> SentenceTransformerEmbeddingAdapter:
    key = (model_name, device, batch_size, enabled)
    with _ADAPTER_CACHE_LOCK:
        adapter = _ADAPTER_CACHE.get(key)
        if adapter is None:
            adapter = SentenceTransformerEmbeddingAdapter(
                model_name=model_name,
                device=device,
                batch_size=batch_size,
                enabled=enabled,
            )
            _ADAPTER_CACHE[key] = adapter
        return adapter


def clear_embedding_adapter_cache() -> None:
    with _ADAPTER_CACHE_LOCK:
        _ADAPTER_CACHE.clear()

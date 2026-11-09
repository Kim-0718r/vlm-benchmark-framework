from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from vlm_eval.schemas import EvalSample, ModelOutput


class BaseModelAdapter(ABC):
    """Common lifecycle and inference contract for every VLM adapter."""

    def __init__(self, *, name: str, model_id: str) -> None:
        if not name.strip():
            raise ValueError("name must not be empty")
        if not model_id.strip():
            raise ValueError("model_id must not be empty")

        self.name = name
        self.model_id = model_id
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load model resources once; repeated calls are safe."""
        if self._loaded:
            return
        self._load()
        self._loaded = True

    def unload(self) -> None:
        """Release model resources once; repeated calls are safe."""
        if not self._loaded:
            return
        try:
            self._unload()
        finally:
            self._loaded = False

    def generate(self, sample: EvalSample, *, prompt: str | None = None) -> ModelOutput:
        """Generate one answer while recording latency and isolating model errors."""
        if not self._loaded:
            raise RuntimeError(f"Model adapter '{self.name}' is not loaded")

        effective_prompt = sample.question if prompt is None else prompt
        self._before_generate()
        started_at = perf_counter()

        try:
            text = self._generate_text(sample=sample, prompt=effective_prompt)
            if not isinstance(text, str):
                raise TypeError("_generate_text() must return str")
            peak_vram_mb = self._after_generate()
            return ModelOutput(
                text=text,
                latency_ms=(perf_counter() - started_at) * 1000,
                peak_vram_mb=peak_vram_mb,
            )
        except Exception as exc:
            peak_vram_mb = self._after_generate()
            return ModelOutput(
                latency_ms=(perf_counter() - started_at) * 1000,
                peak_vram_mb=peak_vram_mb,
                error=f"{type(exc).__name__}: {exc}",
            )

    def model_info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "adapter_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "loaded": self.loaded,
        }

    def __enter__(self) -> BaseModelAdapter:
        self.load()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.unload()

    @abstractmethod
    def _load(self) -> None:
        """Model-specific resource loading hook."""

    def _unload(self) -> None:
        """Optional model-specific resource cleanup hook."""

    def _before_generate(self) -> None:
        """Optional measurement hook called immediately before inference."""

    def _after_generate(self) -> float | None:
        """Optional measurement hook returning peak VRAM in MiB."""
        return None

    @abstractmethod
    def _generate_text(self, *, sample: EvalSample, prompt: str) -> str:
        """Model-specific single-sample generation hook."""

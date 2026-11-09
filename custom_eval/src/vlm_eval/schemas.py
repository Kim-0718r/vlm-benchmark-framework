from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class EvalSample:
    """A single benchmark sample passed to a model adapter."""

    benchmark: str
    split: str
    sample_id: str
    question: str
    answers: list[str] = field(default_factory=list)
    media: list[Any] = field(default_factory=list, repr=False)
    media_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id)
        if not self.benchmark.strip():
            raise ValueError("benchmark must not be empty")
        if not self.split.strip():
            raise ValueError("split must not be empty")
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")


@dataclass(slots=True)
class ModelOutput:
    """Raw output and runtime measurements for one model request."""

    text: str = ""
    latency_ms: float | None = None
    peak_vram_mb: float | None = None
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class PredictionRecord:
    """JSON-serializable, scored result for one benchmark sample."""

    run_id: str
    model_name: str
    benchmark: str
    split: str
    sample_id: str
    question: str
    references: list[str] = field(default_factory=list)
    media_refs: list[str] = field(default_factory=list)
    raw_prediction: str = ""
    normalized_prediction: str = ""
    score: float | None = None
    latency_ms: float | None = None
    peak_vram_mb: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_result(
        cls,
        *,
        run_id: str,
        model_name: str,
        sample: EvalSample,
        output: ModelOutput,
        normalized_prediction: str = "",
        score: float | None = None,
    ) -> PredictionRecord:
        return cls(
            run_id=run_id,
            model_name=model_name,
            benchmark=sample.benchmark,
            split=sample.split,
            sample_id=sample.sample_id,
            question=sample.question,
            references=list(sample.answers),
            media_refs=list(sample.media_refs),
            raw_prediction=output.text,
            normalized_prediction=normalized_prediction,
            score=score,
            latency_ms=output.latency_ms,
            peak_vram_mb=output.peak_vram_mb,
            error=output.error,
            metadata=dict(sample.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

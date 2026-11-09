from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from vlm_eval.schemas import EvalSample


@dataclass(frozen=True, slots=True)
class ScoreResult:
    normalized_prediction: str
    score: float


class Scorer(Protocol):
    """Common scoring contract used by EvaluationRunner."""

    def score(self, sample: EvalSample, prediction: str) -> ScoreResult: ...

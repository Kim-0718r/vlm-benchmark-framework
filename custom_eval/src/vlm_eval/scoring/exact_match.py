from __future__ import annotations

import re

from vlm_eval.schemas import EvalSample
from vlm_eval.scoring.base import ScoreResult


def normalize_text(text: str) -> str:
    """A small, predictable normalizer suitable for pipeline smoke tests."""
    return re.sub(r"\s+", " ", text.strip().casefold())


class ExactMatchScorer:
    """Scores 1 when a normalized prediction exactly matches any answer."""

    def score(self, sample: EvalSample, prediction: str) -> ScoreResult:
        normalized = normalize_text(prediction)
        references = {normalize_text(answer) for answer in sample.answers}
        return ScoreResult(
            normalized_prediction=normalized,
            score=float(bool(references) and normalized in references),
        )

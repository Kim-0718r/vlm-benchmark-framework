from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample
from vlm_eval.scoring.base import ScoreResult


def normalize_ocrbench_answer(text: str) -> str:
    return " ".join(text.strip().casefold().split())


class OCRBenchScorer:
    """OCRBench v1 matching: answer containment, with strict math matching."""

    def score(self, sample: EvalSample, prediction: str) -> ScoreResult:
        normalized = normalize_ocrbench_answer(prediction)
        question_type = str(sample.metadata.get("question_type", "")).casefold()
        is_math_expression = "mathematical expression" in question_type
        correct = False
        for answer in sample.answers:
            reference = normalize_ocrbench_answer(answer)
            if is_math_expression:
                correct = normalized.replace(" ", "") == reference.replace(" ", "")
            else:
                correct = bool(reference) and reference in normalized
            if correct:
                break
        return ScoreResult(normalized_prediction=normalized, score=float(correct))


def aggregate_ocrbench(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    type_scores: dict[str, list[float]] = defaultdict(list)
    dataset_scores: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for record in records:
        if record.get("error") is None and record.get("score") is not None:
            score = float(record["score"])
            metadata = record.get("metadata", {})
            type_scores[str(metadata.get("question_type", "unknown"))].append(score)
            dataset_scores[str(metadata.get("dataset", "unknown"))].append(score)
            all_scores.append(score)
    return {
        "points": sum(all_scores),
        "accuracy": sum(all_scores) / len(all_scores) if all_scores else None,
        "count": len(all_scores),
        "by_question_type": {name: sum(values) for name, values in sorted(type_scores.items())},
        "by_dataset": {name: sum(values) for name, values in sorted(dataset_scores.items())},
    }

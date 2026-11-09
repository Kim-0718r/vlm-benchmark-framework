from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample
from vlm_eval.scoring.base import ScoreResult


def normalize_chartqa_answer(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def _number(text: str) -> float | None:
    value = text.strip().replace(",", "")
    if value.endswith("%"):
        value = value[:-1]
    try:
        return float(value)
    except ValueError:
        return None


class ChartQAScorer:
    """ChartQA relaxed accuracy: exact text or numeric answer within 5%."""

    def score(self, sample: EvalSample, prediction: str) -> ScoreResult:
        normalized = normalize_chartqa_answer(prediction)
        correct = False
        for answer in sample.answers:
            reference = normalize_chartqa_answer(answer)
            if normalized == reference:
                correct = True
                break
            pred_number, ref_number = _number(normalized), _number(reference)
            if pred_number is not None and ref_number is not None:
                if ref_number == 0:
                    correct = pred_number == 0
                else:
                    correct = abs(pred_number - ref_number) / abs(ref_number) <= 0.05
                if correct:
                    break
        return ScoreResult(normalized_prediction=normalized, score=float(correct))


def aggregate_chartqa(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    all_scores: list[float] = []
    for record in records:
        if record.get("error") is None and record.get("score") is not None:
            score = float(record["score"])
            question_type = str(record.get("metadata", {}).get("question_type", "unknown"))
            groups[question_type].append(score)
            all_scores.append(score)
    return {
        "overall": sum(all_scores) / len(all_scores) if all_scores else None,
        "by_question_type": {
            name: sum(scores) / len(scores) for name, scores in sorted(groups.items())
        },
        "count": len(all_scores),
    }

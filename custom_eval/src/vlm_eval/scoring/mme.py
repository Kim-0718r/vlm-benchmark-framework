from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample
from vlm_eval.scoring.base import ScoreResult

PERCEPTION_CATEGORIES = {
    "existence", "count", "position", "color", "posters", "celebrity",
    "scene", "landmark", "artwork", "OCR",
}
COGNITION_CATEGORIES = {
    "commonsense_reasoning", "numerical_calculation", "text_translation", "code_reasoning",
}


def normalize_mme_answer(text: str) -> str:
    tokens = re.findall(r"[a-z]+", text.casefold())
    if "yes" in tokens and "no" not in tokens:
        return "yes"
    if "no" in tokens and "yes" not in tokens:
        return "no"
    return " ".join(tokens)


class MMEScorer:
    def score(self, sample: EvalSample, prediction: str) -> ScoreResult:
        normalized = normalize_mme_answer(prediction)
        references = {normalize_mme_answer(answer) for answer in sample.answers}
        return ScoreResult(normalized_prediction=normalized, score=float(normalized in references))


def aggregate_mme(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("error") is None and record.get("score") is not None:
            category = str(record.get("metadata", {}).get("category", "unknown"))
            by_category[category].append(record)

    category_scores: dict[str, dict[str, float | int]] = {}
    for category, items in by_category.items():
        accuracy = 100.0 * sum(float(item["score"]) for item in items) / len(items)
        groups: dict[str, list[float]] = defaultdict(list)
        for item in items:
            image_id = str(item.get("metadata", {}).get("image_id", item["sample_id"]))
            groups[image_id].append(float(item["score"]))
        complete_groups = [scores for scores in groups.values() if len(scores) == 2]
        acc_plus = (
            100.0 * sum(all(scores) for scores in complete_groups) / len(complete_groups)
            if complete_groups
            else 0.0
        )
        category_scores[category] = {
            "count": len(items), "complete_pairs": len(complete_groups),
            "accuracy": accuracy, "accuracy_plus": acc_plus,
            "mme_score": accuracy + acc_plus,
        }

    def total(categories: set[str]) -> float:
        return sum(float(category_scores[name]["mme_score"]) for name in categories if name in category_scores)

    return {
        "categories": category_scores,
        "perception_score": total(PERCEPTION_CATEGORIES),
        "cognition_score": total(COGNITION_CATEGORIES),
    }

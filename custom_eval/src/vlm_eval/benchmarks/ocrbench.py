from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample


class OCRBenchLoader:
    def from_rows(self, rows: Iterable[Mapping[str, Any]], *, limit: int | None = None) -> Iterator[EvalSample]:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            for key in ("question", "answer", "image"):
                if key not in row:
                    raise ValueError(f"OCRBench row is missing required field: {key}")
            answers = row["answer"] if isinstance(row["answer"], (list, tuple)) else [row["answer"]]
            yield EvalSample(
                benchmark="ocrbench",
                split="test",
                sample_id=str(index),
                question=str(row["question"]),
                answers=[str(answer) for answer in answers],
                media=[row["image"]],
                media_refs=[f"ocrbench:{index}"],
                metadata={
                    "dataset": str(row.get("dataset", "unknown")),
                    "question_type": str(row.get("question_type", "unknown")),
                },
            )

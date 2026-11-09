from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample


class ChartQALoader:
    def from_rows(self, rows: Iterable[Mapping[str, Any]], *, limit: int | None = None) -> Iterator[EvalSample]:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            for key in ("question", "answer", "image"):
                if key not in row:
                    raise ValueError(f"ChartQA row is missing required field: {key}")
            question_type = str(row.get("type", "unknown"))
            yield EvalSample(
                benchmark="chartqa",
                split="test",
                sample_id=f"{question_type}:{index}",
                question=str(row["question"]),
                answers=[str(row["answer"])],
                media=[row["image"]],
                media_refs=[f"chartqa:{index}"],
                metadata={"question_type": question_type},
            )

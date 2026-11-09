from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample


class MMELoader:
    def from_rows(self, rows: Iterable[Mapping[str, Any]], *, limit: int | None = None) -> Iterator[EvalSample]:
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            for key in ("question_id", "question", "answer", "image", "category"):
                if key not in row:
                    raise ValueError(f"MME row is missing required field: {key}")
            image_id = str(row["question_id"])
            category = str(row["category"])
            yield EvalSample(
                benchmark="mme",
                split="test",
                sample_id=f"{image_id}::{index}",
                question=str(row["question"]),
                answers=[str(row["answer"])],
                media=[row["image"]],
                media_refs=[image_id],
                metadata={"image_id": image_id, "category": category},
            )

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from vlm_eval.schemas import EvalSample


class TextVQALoader:
    """Convert official TextVQA annotations (or HF-style rows) to EvalSample."""

    def __init__(self, *, image_dir: str | Path | None = None) -> None:
        self.image_dir = Path(image_dir) if image_dir is not None else None

    def load_json(
        self, annotation_path: str | Path, *, limit: int | None = None
    ) -> Iterator[EvalSample]:
        path = Path(annotation_path)
        with path.open("r", encoding="utf-8") as file:
            document = json.load(file)
        if not isinstance(document, dict) or not isinstance(document.get("data"), list):
            raise ValueError("TextVQA annotation must contain a 'data' list")
        yield from self.from_rows(document["data"], limit=limit)

    def from_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
        *,
        limit: int | None = None,
    ) -> Iterator[EvalSample]:
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")

        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            yield self._convert_row(row)

    def _convert_row(self, row: Mapping[str, Any]) -> EvalSample:
        missing = [key for key in ("question_id", "question", "image_id") if key not in row]
        if missing:
            raise ValueError(f"TextVQA row is missing required fields: {', '.join(missing)}")

        raw_answers = row.get("answers", [])
        if not isinstance(raw_answers, (list, tuple)):
            raise ValueError("TextVQA 'answers' must be a list")

        image_id = str(row["image_id"])
        media: list[Any] = []
        media_refs: list[str] = []
        if "image" in row and row["image"] is not None:
            media.append(row["image"])
            media_refs.append(image_id)
        elif self.image_dir is not None:
            image_path = self._image_path(image_id)
            media.append(image_path)
            media_refs.append(str(image_path))

        split = str(row.get("set_name") or "unknown")
        metadata = {
            "image_id": image_id,
            "image_width": row.get("image_width"),
            "image_height": row.get("image_height"),
        }
        return EvalSample(
            benchmark="textvqa",
            split=split,
            sample_id=str(row["question_id"]),
            question=str(row["question"]),
            answers=[str(answer) for answer in raw_answers if str(answer).strip()],
            media=media,
            media_refs=media_refs,
            metadata=metadata,
        )

    def _image_path(self, image_id: str) -> Path:
        assert self.image_dir is not None
        for suffix in (".jpg", ".jpeg", ".png"):
            candidate = self.image_dir / f"{image_id}{suffix}"
            if candidate.exists():
                return candidate
        return self.image_dir / f"{image_id}.jpg"

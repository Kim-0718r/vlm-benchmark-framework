from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from vlm_eval.schemas import PredictionRecord


class JsonlStore:
    """Append-only storage for per-sample evaluation results.

    Each call to ``append`` writes and flushes one complete JSON line.  This
    keeps all earlier predictions usable if an evaluation is interrupted.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: PredictionRecord | Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = record.to_dict() if isinstance(record, PredictionRecord) else dict(record)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)

        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(line + "\n")
            file.flush()
            os.fsync(file.fileno())

    def iter_records(self, *, ignore_incomplete_last_line: bool = True) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return

        with self.path.open("r", encoding="utf-8") as file:
            lines = file.readlines()

        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                is_last = index == len(lines) - 1
                if ignore_incomplete_last_line and is_last:
                    continue
                raise ValueError(
                    f"Invalid JSONL at {self.path}, line {index + 1}"
                ) from None
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL record must be an object at {self.path}, line {index + 1}"
                )
            yield value

    def read_all(self) -> list[dict[str, Any]]:
        return list(self.iter_records())

    def __len__(self) -> int:
        return sum(1 for _ in self.iter_records())

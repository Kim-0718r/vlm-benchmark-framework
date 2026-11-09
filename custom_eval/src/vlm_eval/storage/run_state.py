from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from vlm_eval.schemas import EvalSample, PredictionRecord
from vlm_eval.storage.jsonl import JsonlStore

SampleKey = tuple[str, str, str]


def sample_key(benchmark: str, split: str, sample_id: str | int) -> SampleKey:
    return benchmark, split, str(sample_id)


class RunState:
    """Tracks completed samples so interrupted evaluations can resume."""

    def __init__(self, records: Iterable[Mapping[str, Any]], *, retry_errors: bool = True) -> None:
        self.retry_errors = retry_errors
        self._finished: set[SampleKey] = set()
        for record in records:
            if retry_errors and record.get("error"):
                continue
            self._finished.add(
                sample_key(
                    str(record["benchmark"]),
                    str(record["split"]),
                    record["sample_id"],
                )
            )

    @classmethod
    def from_store(cls, store: JsonlStore, *, retry_errors: bool = True) -> RunState:
        return cls(store.iter_records(), retry_errors=retry_errors)

    def is_completed(self, sample: EvalSample) -> bool:
        return sample_key(sample.benchmark, sample.split, sample.sample_id) in self._finished

    def pending(self, samples: Iterable[EvalSample]) -> Iterable[EvalSample]:
        return (sample for sample in samples if not self.is_completed(sample))

    def mark_completed(self, record: PredictionRecord | Mapping[str, Any]) -> None:
        value = record.to_dict() if isinstance(record, PredictionRecord) else record
        if self.retry_errors and value.get("error"):
            return
        self._finished.add(
            sample_key(str(value["benchmark"]), str(value["split"]), value["sample_id"])
        )

    @property
    def completed_count(self) -> int:
        return len(self._finished)

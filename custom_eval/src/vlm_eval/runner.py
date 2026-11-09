from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample, PredictionRecord
from vlm_eval.scoring.base import Scorer
from vlm_eval.storage import JsonlStore, RunState


@dataclass(frozen=True, slots=True)
class RunSummary:
    run_id: str
    model_name: str
    previously_completed: int
    processed: int
    succeeded: int
    failed: int
    scored: int
    score_sum: float

    @property
    def mean_score(self) -> float | None:
        return self.score_sum / self.scored if self.scored else None


ProgressCallback = Callable[[PredictionRecord, RunSummary], None]
PromptBuilder = Callable[[EvalSample], str]


class EvaluationRunner:
    """Runs samples sequentially and durably saves every completed result."""

    def __init__(
        self,
        *,
        run_id: str,
        model: BaseModelAdapter,
        scorer: Scorer,
        store: JsonlStore,
        retry_errors: bool = True,
        on_result: ProgressCallback | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be empty")
        self.run_id = run_id
        self.model = model
        self.scorer = scorer
        self.store = store
        self.retry_errors = retry_errors
        self.on_result = on_result
        self.prompt_builder = prompt_builder

    def run(self, samples: Iterable[EvalSample]) -> RunSummary:
        state = RunState.from_store(self.store, retry_errors=self.retry_errors)
        previously_completed = state.completed_count
        processed = succeeded = failed = scored = 0
        score_sum = 0.0
        model_was_loaded = self.model.loaded

        if not model_was_loaded:
            self.model.load()

        try:
            for sample in state.pending(samples):
                prompt = self.prompt_builder(sample) if self.prompt_builder else None
                output = self.model.generate(sample, prompt=prompt)
                normalized_prediction = ""
                score: float | None = None

                if output.succeeded:
                    try:
                        score_result = self.scorer.score(sample, output.text)
                        normalized_prediction = score_result.normalized_prediction
                        score = score_result.score
                    except Exception as exc:
                        output.error = f"ScoringError: {type(exc).__name__}: {exc}"

                record = PredictionRecord.from_result(
                    run_id=self.run_id,
                    model_name=self.model.name,
                    sample=sample,
                    output=output,
                    normalized_prediction=normalized_prediction,
                    score=score,
                )
                self.store.append(record)
                state.mark_completed(record)

                processed += 1
                if record.error is None:
                    succeeded += 1
                else:
                    failed += 1
                if record.score is not None:
                    scored += 1
                    score_sum += record.score

                if self.on_result is not None:
                    self.on_result(
                        record,
                        self._summary(
                            previously_completed,
                            processed,
                            succeeded,
                            failed,
                            scored,
                            score_sum,
                        ),
                    )
        finally:
            if not model_was_loaded:
                self.model.unload()

        return self._summary(
            previously_completed,
            processed,
            succeeded,
            failed,
            scored,
            score_sum,
        )

    def _summary(
        self,
        previously_completed: int,
        processed: int,
        succeeded: int,
        failed: int,
        scored: int,
        score_sum: float,
    ) -> RunSummary:
        return RunSummary(
            run_id=self.run_id,
            model_name=self.model.name,
            previously_completed=previously_completed,
            processed=processed,
            succeeded=succeeded,
            failed=failed,
            scored=scored,
            score_sum=score_sum,
        )

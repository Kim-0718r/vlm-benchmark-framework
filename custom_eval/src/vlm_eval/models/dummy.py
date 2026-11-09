from __future__ import annotations

from collections.abc import Mapping, Set

from vlm_eval.models.base import BaseModelAdapter
from vlm_eval.schemas import EvalSample


class DummyAdapter(BaseModelAdapter):
    """Deterministic adapter used to test the evaluation pipeline without a GPU."""

    def __init__(
        self,
        *,
        name: str = "dummy",
        default_response: str = "yes",
        responses: Mapping[str, str] | None = None,
        error_sample_ids: Set[str] | None = None,
        echo_prompt: bool = False,
    ) -> None:
        super().__init__(name=name, model_id="dummy/local")
        self.default_response = default_response
        self.responses = {str(key): value for key, value in (responses or {}).items()}
        self.error_sample_ids = {str(value) for value in (error_sample_ids or set())}
        self.echo_prompt = echo_prompt

    def _load(self) -> None:
        pass

    def _generate_text(self, *, sample: EvalSample, prompt: str) -> str:
        if sample.sample_id in self.error_sample_ids:
            raise RuntimeError(f"simulated failure for sample {sample.sample_id}")
        if self.echo_prompt:
            return prompt
        return self.responses.get(sample.sample_id, self.default_response)

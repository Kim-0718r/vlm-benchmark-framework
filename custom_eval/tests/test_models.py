import pytest

from vlm_eval import EvalSample
from vlm_eval.models import DummyAdapter


def make_sample(sample_id: str = "1") -> EvalSample:
    return EvalSample(
        benchmark="toy",
        split="test",
        sample_id=sample_id,
        question="Is there an image?",
        answers=["yes"],
    )


def test_generate_requires_loaded_adapter() -> None:
    adapter = DummyAdapter()

    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.generate(make_sample())


def test_dummy_returns_deterministic_response() -> None:
    adapter = DummyAdapter(default_response="yes")
    adapter.load()

    output = adapter.generate(make_sample())

    assert output.succeeded is True
    assert output.text == "yes"
    assert output.latency_ms is not None
    assert output.latency_ms >= 0


def test_dummy_supports_per_sample_responses() -> None:
    adapter = DummyAdapter(responses={"2": "Product B"})

    with adapter:
        output = adapter.generate(make_sample("2"))

    assert output.text == "Product B"
    assert adapter.loaded is False


def test_dummy_captures_model_errors() -> None:
    adapter = DummyAdapter(error_sample_ids={"3"})
    adapter.load()

    output = adapter.generate(make_sample("3"))

    assert output.succeeded is False
    assert output.text == ""
    assert output.error == "RuntimeError: simulated failure for sample 3"


def test_dummy_can_echo_an_overridden_prompt() -> None:
    adapter = DummyAdapter(echo_prompt=True)
    adapter.load()

    output = adapter.generate(make_sample(), prompt="Answer briefly.")

    assert output.text == "Answer briefly."

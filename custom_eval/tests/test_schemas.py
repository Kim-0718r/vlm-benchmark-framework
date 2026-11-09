import json

import pytest

from vlm_eval import EvalSample, ModelOutput, PredictionRecord


def test_prediction_record_can_be_serialized() -> None:
    sample = EvalSample(
        benchmark="chartqa",
        split="test",
        sample_id=123,
        question="Which product has the highest value?",
        answers=["Product B"],
        media=[object()],
        media_refs=["chartqa/test/123.png"],
        metadata={"question_type": "human"},
    )
    output = ModelOutput(text="Product B", latency_ms=1250.5, peak_vram_mb=6120.0)
    record = PredictionRecord.from_result(
        run_id="pilot-001",
        model_name="dummy-model",
        sample=sample,
        output=output,
        normalized_prediction="product b",
        score=1.0,
    )
    parsed = json.loads(record.to_json())
    assert parsed["sample_id"] == "123"
    assert parsed["raw_prediction"] == "Product B"
    assert parsed["normalized_prediction"] == "product b"
    assert parsed["score"] == 1.0
    assert parsed["media_refs"] == ["chartqa/test/123.png"]
    assert "media" not in parsed


def test_failed_model_output() -> None:
    assert ModelOutput(error="CUDA out of memory").succeeded is False


def test_empty_question_is_rejected() -> None:
    with pytest.raises(ValueError, match="question"):
        EvalSample(benchmark="mme", split="test", sample_id="1", question="")

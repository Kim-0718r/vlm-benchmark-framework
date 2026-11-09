import json

import pytest

from vlm_eval import EvalSample, JsonlStore, ModelOutput, PredictionRecord, RunState


def make_record(sample_id: str, *, error: str | None = None) -> PredictionRecord:
    sample = EvalSample(
        benchmark="ocrbench",
        split="test",
        sample_id=sample_id,
        question="Read the text.",
        answers=["hello"],
    )
    return PredictionRecord.from_result(
        run_id="run-001",
        model_name="dummy",
        sample=sample,
        output=ModelOutput(text="hello" if error is None else "", error=error),
        normalized_prediction="hello" if error is None else "",
        score=1.0 if error is None else None,
    )


def test_store_appends_unicode_records(tmp_path) -> None:
    store = JsonlStore(tmp_path / "outputs" / "predictions.jsonl")
    record = make_record("한글-1")

    store.append(record)

    assert len(store) == 1
    assert store.read_all()[0]["sample_id"] == "한글-1"


def test_store_ignores_interrupted_last_line(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    store = JsonlStore(path)
    store.append(make_record("1"))
    with path.open("a", encoding="utf-8") as file:
        file.write('{"sample_id": "unfinished"')

    assert [row["sample_id"] for row in store.iter_records()] == ["1"]


def test_store_rejects_corruption_before_last_line(tmp_path) -> None:
    path = tmp_path / "predictions.jsonl"
    path.write_text('{bad json}\n' + json.dumps({"sample_id": "2"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        list(JsonlStore(path).iter_records())


def test_run_state_skips_successful_samples_and_retries_errors(tmp_path) -> None:
    store = JsonlStore(tmp_path / "predictions.jsonl")
    store.append(make_record("done"))
    store.append(make_record("failed", error="CUDA out of memory"))
    state = RunState.from_store(store)
    samples = [
        EvalSample("ocrbench", "test", "done", "Question 1"),
        EvalSample("ocrbench", "test", "failed", "Question 2"),
        EvalSample("ocrbench", "test", "new", "Question 3"),
    ]

    assert state.completed_count == 1
    assert [sample.sample_id for sample in state.pending(samples)] == ["failed", "new"]


def test_run_state_can_treat_errors_as_completed(tmp_path) -> None:
    store = JsonlStore(tmp_path / "predictions.jsonl")
    failed = make_record("failed", error="bad image")
    store.append(failed)

    state = RunState.from_store(store, retry_errors=False)

    assert state.completed_count == 1
    assert state.is_completed(EvalSample("ocrbench", "test", "failed", "Question"))


def test_mark_completed_updates_state_without_reloading_file() -> None:
    record = make_record("42")
    state = RunState([])

    state.mark_completed(record)

    sample = EvalSample("ocrbench", "test", "42", "Question")
    assert state.is_completed(sample)

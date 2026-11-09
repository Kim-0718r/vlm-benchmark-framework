from vlm_eval import EvalSample, EvaluationRunner, ExactMatchScorer, JsonlStore
from vlm_eval.models import DummyAdapter


def samples() -> list[EvalSample]:
    return [
        EvalSample("demo", "test", "1", "First?", ["yes"]),
        EvalSample("demo", "test", "2", "Second?", ["no"]),
        EvalSample("demo", "test", "3", "Third?", ["yes"]),
    ]


def test_runner_generates_scores_and_saves_each_record(tmp_path) -> None:
    store = JsonlStore(tmp_path / "predictions.jsonl")
    model = DummyAdapter(responses={"1": " YES ", "2": "wrong", "3": "yes"})
    progress_ids: list[str] = []
    runner = EvaluationRunner(
        run_id="demo-001",
        model=model,
        scorer=ExactMatchScorer(),
        store=store,
        on_result=lambda record, summary: progress_ids.append(record.sample_id),
    )

    summary = runner.run(samples())

    assert summary.processed == 3
    assert summary.succeeded == 3
    assert summary.failed == 0
    assert summary.mean_score == 2 / 3
    assert [row["score"] for row in store.read_all()] == [1.0, 0.0, 1.0]
    assert progress_ids == ["1", "2", "3"]
    assert model.loaded is False


def test_runner_resumes_without_repeating_completed_samples(tmp_path) -> None:
    store = JsonlStore(tmp_path / "predictions.jsonl")
    first_model = DummyAdapter()
    EvaluationRunner(
        run_id="first",
        model=first_model,
        scorer=ExactMatchScorer(),
        store=store,
    ).run(samples()[:2])

    second_model = DummyAdapter(responses={"3": "yes"})
    summary = EvaluationRunner(
        run_id="resumed",
        model=second_model,
        scorer=ExactMatchScorer(),
        store=store,
    ).run(samples())

    assert summary.previously_completed == 2
    assert summary.processed == 1
    assert len(store) == 3


def test_runner_saves_failure_and_retries_it_on_next_run(tmp_path) -> None:
    store = JsonlStore(tmp_path / "predictions.jsonl")
    failed_model = DummyAdapter(error_sample_ids={"1"})
    first = EvaluationRunner(
        run_id="failed-run",
        model=failed_model,
        scorer=ExactMatchScorer(),
        store=store,
    ).run(samples()[:1])
    assert first.failed == 1

    recovered_model = DummyAdapter(default_response="yes")
    second = EvaluationRunner(
        run_id="retry-run",
        model=recovered_model,
        scorer=ExactMatchScorer(),
        store=store,
    ).run(samples()[:1])

    assert second.processed == 1
    assert second.succeeded == 1
    assert len(store) == 2


def test_runner_does_not_unload_model_loaded_by_caller(tmp_path) -> None:
    model = DummyAdapter()
    model.load()
    runner = EvaluationRunner(
        run_id="loaded-model",
        model=model,
        scorer=ExactMatchScorer(),
        store=JsonlStore(tmp_path / "predictions.jsonl"),
    )

    runner.run(samples()[:1])

    assert model.loaded is True


def test_runner_can_apply_a_benchmark_prompt(tmp_path) -> None:
    model = DummyAdapter(echo_prompt=True)
    runner = EvaluationRunner(
        run_id="prompt-test",
        model=model,
        scorer=ExactMatchScorer(),
        store=JsonlStore(tmp_path / "predictions.jsonl"),
        prompt_builder=lambda sample: f"{sample.question} Short answer only.",
    )

    runner.run(samples()[:1])

    record = runner.store.read_all()[0]
    assert record["raw_prediction"] == "First? Short answer only."

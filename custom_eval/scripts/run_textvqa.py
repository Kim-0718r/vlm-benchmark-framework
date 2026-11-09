from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

from vlm_eval import (
    EvaluationRunner,
    JsonlStore,
    PredictionRecord,
    RunSummary,
    TextVQALoader,
    TextVQAScorer,
)
from vlm_eval.models import DummyAdapter, Qwen2VLAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small TextVQA validation evaluation with the Dummy model."
    )
    parser.add_argument("--model", choices=["dummy", "qwen2vl_2b"], default="dummy")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--response", default="yes", help="Dummy model's fixed answer")
    parser.add_argument("--run-id", default="textvqa-dummy")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry samples whose previous record contains an error",
    )
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")
    return args


def show_progress(record: PredictionRecord, summary: RunSummary) -> None:
    status = "OK" if record.error is None else "ERROR"
    score = "N/A" if record.score is None else f"{record.score:.3f}"
    print(
        f"[{summary.processed}] id={record.sample_id} "
        f"status={status} score={score} answer={record.raw_prediction!r}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    output = args.output or Path(f"outputs/textvqa/{args.model}/predictions.jsonl")
    print("Loading TextVQA validation stream...", flush=True)
    dataset = load_dataset(
        "facebook/textvqa",
        "textvqa",
        split="validation",
        streaming=True,
        trust_remote_code=True,
    )
    samples = TextVQALoader().from_rows(dataset.take(args.limit))
    store = JsonlStore(output)
    if args.model == "dummy":
        model = DummyAdapter(default_response=args.response)
    else:
        model = Qwen2VLAdapter(max_new_tokens=args.max_new_tokens)
    runner = EvaluationRunner(
        run_id=args.run_id,
        model=model,
        scorer=TextVQAScorer(),
        store=store,
        retry_errors=args.retry_errors,
        on_result=show_progress,
        prompt_builder=lambda sample: (
            f"{sample.question}\nAnswer the question using a single word or phrase."
        ),
    )

    summary = runner.run(samples)
    mean_score = "N/A" if summary.mean_score is None else f"{summary.mean_score:.4f}"
    print("\nEvaluation complete")
    print(f"  output: {output.resolve()}")
    print(f"  previously completed: {summary.previously_completed}")
    print(f"  processed this run: {summary.processed}")
    print(f"  succeeded: {summary.succeeded}")
    print(f"  failed: {summary.failed}")
    print(f"  mean score this run: {mean_score}")


if __name__ == "__main__":
    main()

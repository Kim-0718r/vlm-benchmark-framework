from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from vlm_eval import (
    ChartQALoader, ChartQAScorer, EvaluationRunner, JsonlStore,
    MMELoader, MMEScorer, OCRBenchLoader, OCRBenchScorer,
    TextVQALoader, TextVQAScorer,
)
from vlm_eval.models import (
    DummyAdapter,
    InternVLAdapter,
    MiniCPMV2Adapter,
    OpenFlamingoAdapter,
    Qwen2VLAdapter,
    SmolVLM2Adapter,
)
from vlm_eval.scoring import aggregate_chartqa, aggregate_mme, aggregate_ocrbench


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a VLM on a supported benchmark")
    parser.add_argument("--benchmark", choices=["textvqa", "chartqa", "ocrbench", "mme"], required=True)
    parser.add_argument(
        "--model",
        choices=[
            "dummy",
            "qwen2vl_2b",
            "openflamingo_3b_langinstruct",
            "internvl2_5_2b",
            "minicpm_v_2",
            "smolvlm2_2_2b",
        ],
        default="dummy",
    )
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--profile",
        choices=["native", "controlled"],
        default="native",
        help=(
            "native keeps each model's current visual preprocessing; controlled "
            "uses one native-resolution image crop per model and common greedy/cache settings"
        ),
    )
    parser.add_argument("--dummy-response", default="yes")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.limit < 1 or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be at least 1")
    return args


def load_samples(benchmark: str, limit: int, *, offset: int = 0):
    if offset < 0 or offset > limit:
        raise ValueError("offset must satisfy 0 <= offset <= limit")

    def remaining_rows(dataset):
        return dataset.skip(offset).take(limit - offset)

    if benchmark == "textvqa":
        rows = remaining_rows(load_dataset(
            "facebook/textvqa", "textvqa", split="validation",
            streaming=True, trust_remote_code=True,
        ))
        return TextVQALoader().from_rows(rows), TextVQAScorer()
    if benchmark == "chartqa":
        rows = remaining_rows(load_dataset("lmms-lab/ChartQA", split="test", streaming=True))
        return ChartQALoader().from_rows(rows), ChartQAScorer()
    if benchmark == "ocrbench":
        rows = remaining_rows(load_dataset(
            "parquet",
            data_files={"test": "hf://datasets/echo840/OCRBench/data/test-00000-of-00001.parquet"},
            split="test", streaming=True,
        ))
        return OCRBenchLoader().from_rows(rows), OCRBenchScorer()
    rows = remaining_rows(load_dataset(
        "parquet",
        data_files={"test": "hf://datasets/lmms-lab/MME/data/test-*.parquet"},
        split="test", streaming=True,
    ))
    return MMELoader().from_rows(rows), MMEScorer()


def prompt_for(benchmark: str, question: str) -> str:
    if benchmark == "mme":
        return f"{question}\nAnswer only yes or no."
    return f"{question}\nAnswer the question using a single word or phrase."


def create_model(
    model_name: str,
    max_new_tokens: int,
    *,
    profile: str = "native",
    dummy_response: str = "yes",
):
    """Build an adapter under a documented comparison protocol.

    Raw pixels cannot be identical without moving some models away from their
    trained input size (OpenFlamingo 224, SmolVLM2 384, InternVL 448). The
    controlled profile therefore standardizes one crop, greedy decoding, KV
    cache, batch size one, and the output-token budget.
    """
    if profile not in {"native", "controlled"}:
        raise ValueError("profile must be 'native' or 'controlled'")
    if model_name == "dummy":
        return DummyAdapter(default_response=dummy_response)
    if model_name == "qwen2vl_2b":
        pixels = 256 * 28 * 28
        if profile == "controlled":
            return Qwen2VLAdapter(
                min_pixels=pixels,
                max_pixels=pixels,
                max_new_tokens=max_new_tokens,
            )
        return Qwen2VLAdapter(max_new_tokens=max_new_tokens)
    if model_name == "openflamingo_3b_langinstruct":
        return OpenFlamingoAdapter(
            max_new_tokens=max_new_tokens,
            num_beams=1,
            precision="bf16" if profile == "controlled" else "fp16",
        )
    if model_name == "internvl2_5_2b":
        return InternVLAdapter(
            max_num=1 if profile == "controlled" else 4,
            max_new_tokens=max_new_tokens,
        )
    if model_name == "minicpm_v_2":
        return MiniCPMV2Adapter(max_new_tokens=max_new_tokens)
    if model_name == "smolvlm2_2_2b":
        return SmolVLM2Adapter(
            max_new_tokens=max_new_tokens,
            single_crop=profile == "controlled",
        )
    raise ValueError(f"unsupported model: {model_name}")


def latest_records(store: JsonlStore) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in store.iter_records():
        latest[str(record["sample_id"])] = record
    return list(latest.values())


def aggregate(benchmark: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    if benchmark == "chartqa":
        return aggregate_chartqa(records)
    if benchmark == "ocrbench":
        return aggregate_ocrbench(records)
    if benchmark == "mme":
        return aggregate_mme(records)
    scores = [float(r["score"]) for r in records if r.get("error") is None and r.get("score") is not None]
    return {"count": len(scores), "accuracy": sum(scores) / len(scores) if scores else None}


def main() -> None:
    args = parse_args()
    default_root = Path("outputs") if args.profile == "native" else Path("outputs") / args.profile
    output = args.output or default_root / args.benchmark / args.model / "predictions.jsonl"
    run_id = args.run_id or f"{args.benchmark}-{args.model}"
    store = JsonlStore(output)
    resume_offset = 0 if args.retry_errors else min(len(latest_records(store)), args.limit)
    if resume_offset:
        print(f"Fast resume: skipping {resume_offset} completed dataset rows", flush=True)
    samples, scorer = load_samples(args.benchmark, args.limit, offset=resume_offset)
    model = create_model(
        args.model,
        args.max_new_tokens,
        profile=args.profile,
        dummy_response=args.dummy_response,
    )
    summary = EvaluationRunner(
        run_id=run_id,
        model=model,
        scorer=scorer,
        store=store,
        retry_errors=args.retry_errors,
        prompt_builder=lambda sample: prompt_for(args.benchmark, sample.question),
        on_result=lambda record, progress: print(
            f"[{progress.processed}] id={record.sample_id} score={record.score} "
            f"answer={record.raw_prediction!r} error={record.error}", flush=True,
        ),
    ).run(samples)

    print("\nRun summary")
    print(json.dumps({
        "previously_completed": summary.previously_completed,
        "processed": summary.processed,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "output": str(output.resolve()),
    }, ensure_ascii=False, indent=2))
    print("\nBenchmark aggregate (all saved samples)")
    print(json.dumps(aggregate(args.benchmark, latest_records(store)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

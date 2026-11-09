from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_benchmark import create_model, latest_records, load_samples, prompt_for
from vlm_eval import EvaluationRunner, JsonlStore
from vlm_eval.models import (
    DummyAdapter,
    InternVLAdapter,
    MiniCPMV2Adapter,
    OpenFlamingoAdapter,
    Qwen2VLAdapter,
    SmolVLM2Adapter,
)
from vlm_eval.reporting import collect_results, write_report

FULL_LIMITS = {
    "textvqa": 5000,
    "chartqa": 2500,
    "ocrbench": 1000,
    "mme": 2374,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all supported benchmarks sequentially and generate a comparison report"
    )
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
        default="qwen2vl_2b",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--profile",
        choices=["native", "controlled"],
        default="native",
        help="Use native preprocessing or the one-crop controlled comparison protocol",
    )
    parser.add_argument("--limit", type=int, default=None, help="Override every benchmark limit for a pilot run")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    parser.add_argument("--dummy-response", default="yes")
    parser.add_argument(
        "--retry-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retry previously saved error records (default: skip them)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_model(
        args.model,
        args.max_new_tokens,
        profile=args.profile,
        dummy_response=args.dummy_response,
    )
    outputs_root = args.outputs if args.profile == "native" else args.outputs / args.profile
    report_root = args.report_dir if args.profile == "native" else args.report_dir / args.profile
    model.load()
    try:
        for benchmark, full_limit in FULL_LIMITS.items():
            limit = args.limit if args.limit is not None else full_limit
            output = outputs_root / benchmark / args.model / "predictions.jsonl"
            print(f"\n=== {benchmark}: limit={limit} ===", flush=True)
            store = JsonlStore(output)
            resume_offset = 0 if args.retry_errors else min(len(latest_records(store)), limit)
            if resume_offset:
                print(f"Fast resume: skipping {resume_offset} completed dataset rows", flush=True)
            samples, scorer = load_samples(benchmark, limit, offset=resume_offset)
            summary = EvaluationRunner(
                run_id=f"{benchmark}-{args.model}",
                model=model,
                scorer=scorer,
                store=store,
                retry_errors=args.retry_errors,
                prompt_builder=lambda sample, name=benchmark: prompt_for(name, sample.question),
                on_result=lambda record, progress: print(
                    f"[{progress.processed}] id={record.sample_id} score={record.score} "
                    f"latency={record.latency_ms:.0f}ms vram={record.peak_vram_mb} "
                    f"error={record.error}", flush=True,
                ),
            ).run(samples)
            print(json.dumps({
                "previously_completed": summary.previously_completed,
                "processed": summary.processed,
                "succeeded": summary.succeeded,
                "failed": summary.failed,
            }, indent=2))
    finally:
        model.unload()

    report = collect_results(outputs_root)
    paths = write_report(report, report_root)
    print("\n=== Reports ===")
    for kind, path in paths.items():
        print(f"{kind}: {path.resolve()}")


if __name__ == "__main__":
    main()

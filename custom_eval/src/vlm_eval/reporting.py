from __future__ import annotations

import csv
import json
import statistics
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from vlm_eval.scoring import aggregate_chartqa, aggregate_mme, aggregate_ocrbench
from vlm_eval.storage import JsonlStore

BENCHMARKS = ("textvqa", "chartqa", "ocrbench", "mme")


def latest_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[str(record["sample_id"])] = dict(record)
    return list(latest.values())


def summarize_records(benchmark: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if r.get("error") is None]
    scored = [r for r in valid if r.get("score") is not None]
    # A model failure is still an attempted benchmark item. Treat it as an
    # incorrect answer for metrics so models with OOM/runtime errors do not get
    # an inflated score from a smaller denominator.
    metric_records: list[dict[str, Any]] = []
    for record in records:
        metric_record = dict(record)
        if metric_record.get("error") is not None:
            metric_record["error"] = None
            metric_record["score"] = 0.0
        metric_records.append(metric_record)
    latencies = [float(r["latency_ms"]) for r in valid if r.get("latency_ms") is not None]
    vrams = [float(r["peak_vram_mb"]) for r in valid if r.get("peak_vram_mb") is not None]
    common: dict[str, Any] = {
        "saved_samples": len(records),
        "scored_samples": len(scored),
        "errors": len(records) - len(valid),
        "average_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "max_peak_vram_mb": max(vrams) if vrams else None,
    }
    if benchmark == "chartqa":
        common.update(aggregate_chartqa(metric_records))
    elif benchmark == "ocrbench":
        common.update(aggregate_ocrbench(metric_records))
    elif benchmark == "mme":
        mme = aggregate_mme(metric_records)
        common.update(mme)
        common["total_score"] = mme["perception_score"] + mme["cognition_score"]
    else:
        scores = [float(r["score"]) for r in metric_records if r.get("score") is not None]
        common["accuracy"] = statistics.fmean(scores) if scores else None
    return common


def collect_results(outputs_dir: str | Path) -> dict[str, dict[str, Any]]:
    root = Path(outputs_dir)
    report: dict[str, dict[str, Any]] = {}
    for benchmark in BENCHMARKS:
        benchmark_dir = root / benchmark
        if not benchmark_dir.exists():
            continue
        for model_dir in benchmark_dir.iterdir():
            path = model_dir / "predictions.jsonl"
            if not path.is_file():
                continue
            records = latest_records(JsonlStore(path).iter_records())
            report.setdefault(model_dir.name, {})[benchmark] = summarize_records(benchmark, records)
    return report


def write_report(report: Mapping[str, Mapping[str, Any]], output_dir: str | Path) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "model_comparison.json"
    csv_path = destination / "model_comparison.csv"
    markdown_path = destination / "model_comparison.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for model, benchmarks in sorted(report.items()):
        for benchmark, metrics in sorted(benchmarks.items()):
            rows.append({
                "model": model,
                "benchmark": benchmark,
                "saved_samples": metrics.get("saved_samples"),
                "errors": metrics.get("errors"),
                "accuracy": metrics.get("accuracy", metrics.get("overall")),
                "points": metrics.get("points"),
                "perception_score": metrics.get("perception_score"),
                "cognition_score": metrics.get("cognition_score"),
                "total_score": metrics.get("total_score"),
                "average_latency_ms": metrics.get("average_latency_ms"),
                "max_peak_vram_mb": metrics.get("max_peak_vram_mb"),
            })
    fieldnames = list(rows[0]) if rows else ["model", "benchmark"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def pct(value: Any) -> str:
        return "—" if value is None else f"{float(value) * 100:.2f}%"

    def number(value: Any, digits: int = 2) -> str:
        return "—" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "# VLM Benchmark Comparison", "",
        "| Model | TextVQA | ChartQA | OCRBench | MME Perception | MME Cognition | MME Total | Avg latency | Peak VRAM | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, benchmarks in sorted(report.items()):
        textvqa = benchmarks.get("textvqa", {})
        chartqa = benchmarks.get("chartqa", {})
        ocrbench = benchmarks.get("ocrbench", {})
        mme = benchmarks.get("mme", {})
        all_metrics = list(benchmarks.values())
        latency_values = [m.get("average_latency_ms") for m in all_metrics if m.get("average_latency_ms") is not None]
        vram_values = [m.get("max_peak_vram_mb") for m in all_metrics if m.get("max_peak_vram_mb") is not None]
        errors = sum(int(m.get("errors", 0)) for m in all_metrics)
        lines.append(
            f"| {model} | {pct(textvqa.get('accuracy'))} | {pct(chartqa.get('overall'))} | "
            f"{number(ocrbench.get('points'), 0)} | {number(mme.get('perception_score'))} | "
            f"{number(mme.get('cognition_score'))} | {number(mme.get('total_score'))} | "
            f"{number(statistics.fmean(latency_values) if latency_values else None)} ms | "
            f"{number(max(vram_values) if vram_values else None)} MiB | {errors} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}

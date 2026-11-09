import json

from vlm_eval import JsonlStore
from vlm_eval.reporting import collect_results, write_report


def record(sample_id, score, latency=100.0, vram=2048.0):
    return {
        "sample_id": sample_id, "score": score, "error": None,
        "latency_ms": latency, "peak_vram_mb": vram, "metadata": {},
    }


def test_report_collects_models_and_writes_three_formats(tmp_path) -> None:
    store = JsonlStore(tmp_path / "outputs" / "textvqa" / "model-a" / "predictions.jsonl")
    store.append(record("1", 1.0))
    store.append(record("2", 0.0, latency=200.0, vram=2100.0))

    report = collect_results(tmp_path / "outputs")
    paths = write_report(report, tmp_path / "reports")

    assert report["model-a"]["textvqa"]["accuracy"] == 0.5
    assert report["model-a"]["textvqa"]["average_latency_ms"] == 150.0
    assert all(path.exists() for path in paths.values())
    assert "50.00%" in paths["markdown"].read_text(encoding="utf-8")
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["model-a"]

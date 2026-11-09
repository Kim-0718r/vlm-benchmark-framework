from __future__ import annotations

import argparse
from pathlib import Path

from vlm_eval.reporting import collect_results, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model comparison reports")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--report-dir", type=Path, default=Path("outputs/reports"))
    args = parser.parse_args()
    report = collect_results(args.outputs)
    paths = write_report(report, args.report_dir)
    print(f"Models found: {len(report)}")
    for kind, path in paths.items():
        print(f"{kind}: {path.resolve()}")

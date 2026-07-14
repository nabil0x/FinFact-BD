#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "src").is_dir():
            return parent
    raise RuntimeError("Could not locate project root")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a claim-centric human validation workbook.")
    parser.add_argument("--input", required=True, type=Path, help="CSV exported by the new rewrite pipeline")
    parser.add_argument("--output", required=True, type=Path, help="XLSX workbook path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    sys.path.insert(0, str(project_root()))

    from src.generation.exporter import HumanValidationWorkbookBuilder
    from src.generation.metadata import SampleRecord

    samples = []
    with open(args.input, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in ("selected_claim", "rewrite_plan", "verification_scores"):
                row[key] = json.loads(row[key]) if row.get(key) else {}
            row["claim_index"] = int(row["claim_index"])
            row["temperature"] = float(row["temperature"])
            row["seed"] = int(row["seed"])
            row["regeneration_attempts"] = int(row["regeneration_attempts"])
            samples.append(SampleRecord(**row))
            if args.limit is not None and len(samples) >= args.limit:
                break
    HumanValidationWorkbookBuilder(args.output).build(samples)


if __name__ == "__main__":
    main()

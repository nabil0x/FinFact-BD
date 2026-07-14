#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not locate project root")


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run planning-guided Bangla claim rewrite generation.")
    parser.add_argument("--config", default="configs/rewrite_pipeline.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    root = project_root()
    sys.path.insert(0, str(root))
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_yaml(config_path)
    if args.seed is not None:
        config.setdefault("pipeline", {})["seed"] = args.seed

    from src.generation.pipeline import PlanningGuidedRewritePipeline

    pipeline = PlanningGuidedRewritePipeline(config)
    result = pipeline.run(input_csv=args.input, output_dir=args.output_dir, num_samples=args.num_samples)
    logging.getLogger(__name__).info(
        "Finished accepted=%d failed=%d output=%s",
        len(result.samples),
        len(result.failures),
        args.output_dir or config.get("paths", {}).get("output_dir"),
    )


if __name__ == "__main__":
    main()

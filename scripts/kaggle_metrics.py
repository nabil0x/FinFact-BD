#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def summarize_metrics(output_dir: Path, log_path: Path | None = None) -> Dict[str, Any]:
    metadata = read_json(output_dir / "metadata.json")
    checkpoint = read_json(output_dir / "checkpoint.json")
    stats = metadata.get("stats", {})
    runtime = stats.get("runtime", {}) if isinstance(stats, dict) else {}
    samples = checkpoint.get("samples", []) if isinstance(checkpoint, dict) else []
    failures = checkpoint.get("failures", []) if isinstance(checkpoint, dict) else []
    input_articles = int(stats.get("input_articles") or len(samples) + len(failures) or 0)
    generation_seconds = float(runtime.get("seconds", {}).get("generation", 0.0))
    planning_seconds = float(runtime.get("seconds", {}).get("planning", 0.0))
    verification = runtime.get("verification", {})
    verification_seconds = sum(float(value) for value in verification.get("seconds", {}).values()) if isinstance(verification, dict) else 0.0
    total_seconds = planning_seconds + generation_seconds
    attempts = sample_attempts(samples)
    failure_attempts = [len(failure.get("attempts", [])) for failure in failures if isinstance(failure.get("attempts"), list)]
    all_attempts = attempts + failure_attempts
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path and log_path.exists() else ""
    summary = {
        "output_dir": str(output_dir),
        "input_articles": input_articles,
        "accepted": len(samples),
        "failed": len(failures),
        "acceptance_rate": safe_ratio(len(samples), input_articles),
        "throughput": {
            "articles_per_minute": per_min(input_articles, total_seconds),
            "accepted_per_minute": per_min(len(samples), total_seconds),
            "total_runtime_seconds": round(total_seconds, 4),
        },
        "retries": {
            "average_attempts_per_article": round(mean(all_attempts), 4) if all_attempts else 0.0,
            "average_attempts_accepted": round(mean(attempts), 4) if attempts else 0.0,
            "first_attempt_acceptance_rate": safe_ratio(sum(1 for value in attempts if value == 1), len(samples)),
            "accepted_after_regeneration": sum(1 for value in attempts if value > 1),
        },
        "pipeline_health": {
            "planning_seconds": round(planning_seconds, 4),
            "generation_seconds": round(generation_seconds, 4),
            "verification_seconds": round(verification_seconds, 4),
            "planning_percent": percent(planning_seconds, total_seconds),
            "generation_percent": percent(generation_seconds, total_seconds),
            "verification_percent_of_generation": percent(verification_seconds, generation_seconds),
            "planned_checkpoint_hits": runtime.get("counters", {}).get("planned_checkpoint_hits", 0),
            "planned_articles_written": runtime.get("counters", {}).get("planned_articles", 0),
        },
        "verification": {
            "mean_verifier_seconds_per_item": mean_verifier_seconds(verification),
            "verifier_seconds": verification.get("seconds", {}) if isinstance(verification, dict) else {},
            "verifier_calls": verification.get("calls", {}) if isinstance(verification, dict) else {},
            "verifier_items": verification.get("items", {}) if isinstance(verification, dict) else {},
        },
        "memory": memory_metrics(log_text),
        "oom_recoveries": log_text.count("CUDA OOM during verification batch"),
        "checkpoint": {
            "checkpoint_exists": (output_dir / "checkpoint.json").exists(),
            "planned_articles_exists": (output_dir / "planned_articles.jsonl").exists(),
            "planned_articles_rows": count_lines(output_dir / "planned_articles.jsonl"),
        },
    }
    return summary


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sample_attempts(samples: Iterable[Dict[str, Any]]) -> List[int]:
    attempts: List[int] = []
    for sample in samples:
        value = sample.get("regeneration_attempts")
        if isinstance(value, int):
            attempts.append(value)
        elif isinstance(value, str) and value.isdigit():
            attempts.append(int(value))
    return attempts


def memory_metrics(log_text: str) -> Dict[str, Any]:
    gpu_values = [int(value) for value in re.findall(r"gpu_memory_allocated_mb=(\d+)", log_text)]
    ram_values = [float(value) for value in re.findall(r"cpu_ram_used_gb=([0-9.]+)", log_text)]
    return {
        "peak_gpu_memory_mb": max(gpu_values) if gpu_values else None,
        "peak_cpu_ram_gb": max(ram_values) if ram_values else None,
        "note": "memory fields require pipeline logs with gpu_memory_allocated_mb/cpu_ram_used_gb entries",
    }


def mean_verifier_seconds(verification: Dict[str, Any]) -> Dict[str, float]:
    seconds = verification.get("seconds", {}) if isinstance(verification, dict) else {}
    items = verification.get("items", {}) if isinstance(verification, dict) else {}
    result: Dict[str, float] = {}
    for name, total in seconds.items():
        item_count = int(items.get(name, 0))
        result[name] = round(float(total) / item_count, 6) if item_count else 0.0
    return result


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def per_min(count: int, seconds: float) -> float:
    return round((count / seconds) * 60.0, 4) if seconds > 0 else 0.0


def percent(part: float, whole: float) -> float:
    return round((part / whole) * 100.0, 2) if whole > 0 else 0.0


def safe_ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def print_summary(summary: Dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize FinFact-BD run metrics.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--write", action="store_true", help="Write metrics_summary.json into output-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_metrics(args.output_dir, args.log)
    if args.write:
        path = args.output_dir / "metrics_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_summary(summary)


if __name__ == "__main__":
    main()

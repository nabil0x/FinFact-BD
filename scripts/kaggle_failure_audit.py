#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.utils import sentence_spans


def audit_failures(output_dir: Path, preview: int = 10) -> Dict[str, Any]:
    checkpoint = read_json(output_dir / "checkpoint.json")
    failures = checkpoint.get("failures", []) if isinstance(checkpoint, dict) else []
    reason_counts: Counter[str] = Counter()
    verifier_reason_counts: Counter[str] = Counter()
    audited: List[Dict[str, Any]] = []
    for failure in failures:
        if not isinstance(failure, dict):
            continue
        reason = str(failure.get("reason") or "unknown")
        reason_counts[reason] += 1
        item = summarize_failure(failure)
        for attempt in item.get("attempts", []):
            for verifier_reason in attempt.get("reasons", []):
                verifier_reason_counts[str(verifier_reason)] += 1
        if len(audited) < preview:
            audited.append(item)
    return {
        "output_dir": str(output_dir),
        "failure_count": len(failures),
        "failure_reasons": dict(reason_counts),
        "verifier_reasons": dict(verifier_reason_counts),
        "failures": audited,
    }


def summarize_failure(failure: Dict[str, Any]) -> Dict[str, Any]:
    selected = failure.get("selected_claim") if isinstance(failure.get("selected_claim"), dict) else {}
    plan = failure.get("rewrite_plan") if isinstance(failure.get("rewrite_plan"), dict) else {}
    attempts = failure.get("attempts") if isinstance(failure.get("attempts"), list) else []
    return {
        "article_id": failure.get("article_id"),
        "reason": failure.get("reason"),
        "error": failure.get("error"),
        "claim_type": selected.get("claim_type"),
        "claim_sentence": selected.get("sentence"),
        "family": plan.get("family"),
        "target_span": plan.get("target_span"),
        "replacement": plan.get("replacement"),
        "attempts": [summarize_attempt(selected, attempt) for attempt in attempts],
    }


def summarize_attempt(selected: Dict[str, Any], attempt: Dict[str, Any]) -> Dict[str, Any]:
    rewritten = str(attempt.get("rewritten_article") or "")
    claim_index = int(selected.get("sentence_index") or 0)
    verification = attempt.get("verification") if isinstance(attempt.get("verification"), dict) else {}
    rewritten_sentence = sentence_at(rewritten, claim_index) if rewritten else ""
    return {
        "attempt": attempt.get("attempt"),
        "temperature": attempt.get("temperature"),
        "error": attempt.get("error"),
        "reasons": verification.get("reasons", []),
        "scores": verification.get("scores", {}),
        "original_sentence": selected.get("sentence"),
        "rewritten_sentence": rewritten_sentence,
    }


def sentence_at(text: str, index: int) -> str:
    spans = sentence_spans(text)
    if 0 <= index < len(spans):
        return spans[index].text
    return ""


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing checkpoint file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit failed FinFact-BD rewrite attempts.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preview", type=int, default=10)
    parser.add_argument("--write", action="store_true", help="Write failure_audit.json into output-dir.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = audit_failures(args.output_dir, args.preview)
    if args.write:
        path = args.output_dir / "failure_audit.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

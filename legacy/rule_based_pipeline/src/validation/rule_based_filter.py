#!/usr/bin/env python3
"""
FinFact-BD Rule-Based Quality Filter
Filters perturbations based on deterministic rules before LLM judging.
"""

import csv
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_output_dir

# =============================================================================
# CONFIG
# =============================================================================

config = get_config()
OUTPUT_DIR = get_output_dir()

# Filter thresholds from config
FILTER_CONFIG = config['quality_filter']
MIN_TEXT_LENGTH = FILTER_CONFIG.get('min_text_length', 50)
MAX_LENGTH_RATIO = FILTER_CONFIG.get('max_length_ratio', 2.0)
DEDUP_THRESHOLD = FILTER_CONFIG.get('dedup_similarity_threshold', 0.9)

# Input/output paths
PERTURBED_PATH = OUTPUT_DIR / "finfact_bd_perturbed.csv"
FILTERED_PATH = OUTPUT_DIR / "finfact_bd_perturbed_rule_filtered.csv"
REPORT_PATH = OUTPUT_DIR / "rule_filter_report.json"


def calculate_length_ratio(original: str, perturbed: str) -> float:
    """Calculate length ratio between original and perturbed text."""
    if len(original) == 0:
        return float('inf')
    return len(perturbed) / len(original)


def check_bengali_fluency(text: str) -> Tuple[bool, str]:
    """Basic Bengali fluency checks."""
    # Check for excessive special characters
    special_chars = len(re.findall(r'[!@#$%^&*()_+=\[\]{}|\\:";\'<>?,./]', text))
    if special_chars > len(text) * 0.1:
        return False, "excessive_special_chars"
    
    # Check for repeated characters (e.g., "aaaaaaa")
    repeated = re.findall(r'(.)\1{3,}', text)
    if repeated:
        return False, "repeated_characters"
    
    # Check for minimum word count (Bengali words are space-separated)
    words = text.split()
    if len(words) < 5:
        return False, "too_few_words"
    
    return True, "ok"


def check_text_changed(original: str, perturbed: str) -> Tuple[bool, str]:
    """Check if text was actually changed."""
    if original == perturbed:
        return False, "identical_texts"
    
    # Check if only whitespace/punctuation changed
    original_clean = re.sub(r'\s+', ' ', original).strip()
    perturbed_clean = re.sub(r'\s+', ' ', perturbed).strip()
    
    if original_clean == perturbed_clean:
        return False, "only_whitespace_changed"
    
    return True, "ok"


def check_length_constraints(original: str, perturbed: str) -> Tuple[bool, str]:
    """Check length constraints."""
    # Check minimum length
    if len(perturbed) < MIN_TEXT_LENGTH:
        return False, f"too_short_{len(perturbed)}"
    
    # Check maximum length ratio
    ratio = calculate_length_ratio(original, perturbed)
    if ratio > MAX_LENGTH_RATIO:
        return False, f"too_long_ratio_{ratio:.2f}"
    if ratio < (1 / MAX_LENGTH_RATIO):
        return False, f"too_short_ratio_{ratio:.2f}"
    
    return True, "ok"


def filter_perturbations(perturbed_path: Path) -> Tuple[List[Dict], List[Dict]]:
    """Apply rule-based filters to perturbations."""
    print(f"Loading perturbations from {perturbed_path}...")
    
    perturbations = []
    with open(perturbed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            perturbations.append(row)
    
    print(f"Loaded {len(perturbations)} perturbations")
    
    kept = []
    discarded = []
    
    # Track discard reasons
    discard_reasons = Counter()
    
    for row in perturbations:
        original = row.get("original_text", "")
        perturbed = row.get("text", "")
        
        # Apply filters
        passed = True
        reason = None
        
        # 1. Check if text changed
        changed, reason = check_text_changed(original, perturbed)
        if not changed:
            passed = False
            discard_reasons[reason] += 1
        
        # 2. Check length constraints
        if passed:
            ok, reason = check_length_constraints(original, perturbed)
            if not ok:
                passed = False
                discard_reasons[reason] += 1
        
        # 3. Check Bengali fluency
        if passed:
            ok, reason = check_bengali_fluency(perturbed)
            if not ok:
                passed = False
                discard_reasons[reason] += 1
        
        if passed:
            kept.append(row)
        else:
            row["_discard_reason"] = reason
            discarded.append(row)
    
    print(f"Kept: {len(kept)}")
    print(f"Discarded: {len(discarded)}")
    print(f"\nDiscard reasons:")
    for reason, count in discard_reasons.most_common():
        print(f"  {reason}: {count}")
    
    return kept, discarded


def save_results(kept: List[Dict], discarded: List[Dict], output_path: Path, report_path: Path):
    """Save filtered results and generate report."""
    # Save kept perturbations
    if kept:
        fieldnames = list(kept[0].keys())
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        print(f"Saved {len(kept)} kept perturbations to {output_path}")
    
    # Generate and save report
    report = {
        "total_input": len(kept) + len(discarded),
        "kept": len(kept),
        "discarded": len(discarded),
        "keep_rate": len(kept) / (len(kept) + len(discarded)) if (len(kept) + len(discarded)) > 0 else 0,
        "discard_reasons": {}
    }
    
    # Count discard reasons
    for row in discarded:
        reason = row.get("_discard_reason", "unknown")
        report["discard_reasons"][reason] = report["discard_reasons"].get(reason, 0) + 1
    
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {report_path}")


def main():
    """Main function."""
    # Check if perturbed file exists
    if not PERTURBED_PATH.exists():
        print(f"ERROR: {PERTURBED_PATH} not found.")
        return
    
    # Apply filters
    kept, discarded = filter_perturbations(PERTURBED_PATH)
    
    # Save results
    save_results(kept, discarded, FILTERED_PATH, REPORT_PATH)
    
    # Per-type analysis
    print("\nPer-type statistics:")
    type_counts = Counter(row.get("perturbation_type", "unknown") for row in kept)
    for ptype, count in type_counts.most_common():
        print(f"  {ptype}: {count}")


if __name__ == "__main__":
    main()

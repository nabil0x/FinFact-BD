#!/usr/bin/env python3
"""
FinFact-BD Rebalance: Rebuild the 20K balanced dataset from XNLI-filtered perturbations.

Usage:
  Place finfact_bd_perturbed_filtered.csv in papers/data/finfact_bd/
  Then run: /usr/bin/python3 rebalance_dataset.py
"""

import csv
import json
from pathlib import Path
from collections import Counter

DATA_DIR = Path("/mnt/work/phd/phd-prep/financial nlp/papers/data/finfact_bd")


def main():
    # Load filtered perturbed
    filtered_path = DATA_DIR / "finfact_bd_perturbed_filtered.csv"
    if not filtered_path.exists():
        print(f"ERROR: {filtered_path} not found.")
        print("Download it from Kaggle first.")
        return

    perturbed = []
    with open(filtered_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            perturbed.append(row)

    print(f"Loaded {len(perturbed)} filtered perturbed samples")

    # Count per type
    type_counts = Counter(row["perturbation_type"] for row in perturbed)
    print("Filtered distribution:")
    for t, c in type_counts.most_common():
        print(f"  {t}: {c}")

    # Determine balanced target per type
    # Strategy: take min(count, TARGET_PER_TYPE) for each type
    # This ensures we don't exceed what's available
    TARGET_PER_TYPE = 2000
    
    balanced_perturbed = []
    for ptype in type_counts:
        type_rows = [r for r in perturbed if r["perturbation_type"] == ptype]
        take = min(TARGET_PER_TYPE, len(type_rows))
        balanced_perturbed.extend(type_rows[:take])

    print(f"\nBalanced perturbed: {len(balanced_perturbed)}")
    balanced_types = Counter(r["perturbation_type"] for r in balanced_perturbed)
    for t, c in balanced_types.most_common():
        print(f"  {t}: {c}")

    # Load originals and match count
    originals_path = DATA_DIR / "finfact_bd_originals.csv"
    originals = []
    with open(originals_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            originals.append(row)

    # Balance: equal originals and perturbed
    n = min(len(originals), len(balanced_perturbed))
    originals_balanced = originals[:n]
    perturbed_balanced = balanced_perturbed[:n]

    print(f"\nFinal balanced dataset:")
    print(f"  Originals: {len(originals_balanced)}")
    print(f"  Perturbed: {len(perturbed_balanced)}")
    print(f"  Total: {len(originals_balanced) + len(perturbed_balanced)}")

    # Save updated files
    # Perturbed CSV
    if perturbed_balanced:
        fieldnames = list(perturbed_balanced[0].keys())
        with open(DATA_DIR / "finfact_bd_perturbed.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(perturbed_balanced)

    # Originals CSV (unchanged but trimmed to match)
    if originals_balanced:
        fieldnames = list(originals_balanced[0].keys())
        with open(DATA_DIR / "finfact_bd_originals.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(originals_balanced)

    # Combined CSV
    # Need compatible columns — originals won't have xnli fields, so add defaults
    combined_fields = list(perturbed_balanced[0].keys())
    combined = []
    for row in originals_balanced:
        combined_row = {k: row.get(k, "") for k in combined_fields}
        combined.append(combined_row)
    for row in perturbed_balanced:
        combined.append(row)

    with open(DATA_DIR / "finfact_bd_combined.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=combined_fields)
        writer.writeheader()
        writer.writerows(combined)

    # Metadata
    metadata = {
        "total_samples": len(combined),
        "original_count": len(originals_balanced),
        "perturbed_count": len(perturbed_balanced),
        "perturbation_types": list(balanced_types.keys()),
        "perturbation_distribution": dict(balanced_types),
        "source": "BENI v2",
        "language": "bn",
        "domain": "financial",
        "label_schema": {"0": "original/real", "1": "perturbed/fake"},
        "quality_filtered": True,
        "filter_model": "joeddav/xlm-roberta-large-xnli",
        "contradiction_threshold": 0.4,
        "version": "1.1",
    }
    with open(DATA_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved:")
    print(f"  finfact_bd_combined.csv ({len(combined)} samples)")
    print(f"  finfact_bd_originals.csv ({len(originals_balanced)} samples)")
    print(f"  finfact_bd_perturbed.csv ({len(perturbed_balanced)} samples)")
    print(f"  metadata.json")


if __name__ == "__main__":
    main()

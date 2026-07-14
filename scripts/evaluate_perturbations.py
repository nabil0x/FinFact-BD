#!/usr/bin/env python3
"""
FinFact-BD Perturbation Evaluation
====================================
Analyze mDeBERTa scores vs actual perturbation quality.

Shows examples from different score ranges to understand:
- Are high-scoring samples actually better perturbations?
- Are low-scoring samples actually weak perturbations?
- Is mDeBERTa failing on Bengali financial text?

Usage:
    python scripts/evaluate_perturbations.py
"""

import json
import csv
import io
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import zstandard as zstd


# =====================================================================
# PATHS
# =====================================================================
OUTPUT_DIR = Path("/mnt/work/phd/phd-prep/financial nlp/papers/IDEA_3_FinFact_BD/outputs")
SCORES_FILE = OUTPUT_DIR / "mdeberta_scores.json"
FILTERED_FILE = OUTPUT_DIR / "finfact_bd_perturbed_filtered.csv"

# Also try to load original rule-filtered for full comparison
ORIGINAL_FILE = Path(
    "/mnt/work/phd/phd-prep/financial nlp/papers/IDEA_3_FinFact_BD/data/finfact_bd/"
    "finfact_bd_perturbed_rule_filtered.csv.zst"
)


def load_csv(path: Path):
    samples = []
    if path.suffix == ".zst":
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                for row in csv.DictReader(text_stream):
                    samples.append(row)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                samples.append(row)
    return samples


# =====================================================================
# LOAD DATA
# =====================================================================
print("Loading scores...")
with open(SCORES_FILE) as f:
    scores = json.load(f)

print(f"Loaded {len(scores)} scored samples")

# Build lookup by article_id
score_lookup = {s["article_id"]: s for s in scores}

# Load original perturbations if available
perturbations = {}
if ORIGINAL_FILE.exists():
    print(f"Loading original perturbations from {ORIGINAL_FILE.name}...")
    rows = load_csv(ORIGINAL_FILE)
    perturbations = {row["article_id"]: row for row in rows}
    print(f"Loaded {len(perturbations)} perturbations")
else:
    print(f"Original file not found: {ORIGINAL_FILE}")
    print("Using filtered CSV only")
    rows = load_csv(FILTERED_FILE)
    perturbations = {row["article_id"]: row for row in rows}


# =====================================================================
# ANALYSIS
# =====================================================================
print("\n" + "=" * 70)
print("PERTURBATION EVALUATION")
print("=" * 70)

# 1. Score distribution by type
print("\n1. SCORE DISTRIBUTION BY PERTURBATION TYPE")
print("-" * 70)

type_scores = defaultdict(list)
for s in scores:
    type_scores[s["perturbation_type"]].append(s["mdeberta_contra_prob"])

print(f"{'Type':<30} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'n':>6}")
print("-" * 70)
for ptype in sorted(type_scores.keys()):
    sc = type_scores[ptype]
    print(f"{ptype:<30} {np.mean(sc):>8.4f} {np.median(sc):>8.4f} {np.std(sc):>8.4f} {np.min(sc):>8.4f} {np.max(sc):>8.4f} {len(sc):>6}")


# 2. Show examples from different score ranges
print("\n\n2. EXAMPLES FROM DIFFERENT SCORE RANGES")
print("-" * 70)

# Define score ranges
ranges = [
    ("Very Low (0.0-0.1)", 0.0, 0.1),
    ("Low (0.1-0.2)", 0.1, 0.2),
    ("Medium (0.2-0.3)", 0.2, 0.3),
    ("Borderline (0.3-0.4)", 0.3, 0.4),
    ("Kept (0.4-0.6)", 0.4, 0.6),
    ("High (0.6-1.0)", 0.6, 1.0),
]

# Group scores by range
range_groups = defaultdict(list)
for s in scores:
    prob = s["mdeberta_contra_prob"]
    for name, low, high in ranges:
        if low <= prob < high:
            range_groups[name].append(s)
            break

# Show 2 examples from each range
for name, low, high in ranges:
    group = range_groups[name]
    print(f"\n{'=' * 70}")
    print(f"RANGE: {name} (n={len(group)})")
    print(f"{'=' * 70}")

    # Show 2 examples
    for i, s in enumerate(group[:2]):
        article_id = s["article_id"]
        if article_id in perturbations:
            row = perturbations[article_id]
            original = row.get("original_text", "N/A")
            perturbed = row.get("text", "N/A")

            # Truncate for display
            orig_display = original[:200] + "..." if len(original) > 200 else original
            pert_display = perturbed[:200] + "..." if len(perturbed) > 200 else perturbed

            print(f"\n  Example {i+1}: {article_id}")
            print(f"  Perturbation type: {s['perturbation_type']}")
            print(f"  mDeBERTa score: {s['mdeberta_contra_prob']:.4f}")
            print(f"  mDeBERTa label: {s['mdeberta_label']}")
            print(f"\n  ORIGINAL ({len(original)} chars):")
            print(f"    {orig_display}")
            print(f"\n  PERTURBED ({len(perturbed)} chars):")
            print(f"    {pert_display}")

            # Simple diff: show character-level changes
            if original != perturbed:
                # Find first difference
                min_len = min(len(original), len(perturbed))
                diff_pos = next((j for j in range(min_len) if original[j] != perturbed[j]), min_len)
                print(f"\n  First difference at char {diff_pos}:")
                print(f"    Original: ...{original[max(0,diff_pos-20):diff_pos+30]}...")
                print(f"    Perturbed: ...{perturbed[max(0,diff_pos-20):diff_pos+30]}...")
        else:
            print(f"\n  Example {i+1}: {article_id} (not found in perturbations)")


# 3. Quantitative analysis
print("\n\n3. QUANTITATIVE ANALYSIS")
print("-" * 70)

# Text length comparison
orig_lens = []
pert_lens = []
for s in scores:
    aid = s["article_id"]
    if aid in perturbations:
        row = perturbations[aid]
        orig_lens.append(len(row.get("original_text", "")))
        pert_lens.append(len(row.get("text", "")))

if orig_lens:
    print(f"\nText length comparison:")
    print(f"  Original: mean={np.mean(orig_lens):.0f}, median={np.median(orig_lens):.0f}")
    print(f"  Perturbed: mean={np.mean(pert_lens):.0f}, median={np.median(pert_lens):.0f}")
    print(f"  Difference: mean={np.mean(pert_lens)-np.mean(orig_lens):.0f} chars")

# Word overlap (Jaccard)
jaccard_scores = []
for s in scores:
    aid = s["article_id"]
    if aid in perturbations:
        row = perturbations[aid]
        orig_words = set(row.get("original_text", "").split())
        pert_words = set(row.get("text", "").split())
        if orig_words and pert_words:
            jaccard = len(orig_words & pert_words) / len(orig_words | pert_words)
            jaccard_scores.append((s["mdeberta_contra_prob"], jaccard))

if jaccard_scores:
    contra_probs, jaccards = zip(*jaccard_scores)
    print(f"\nWord overlap (Jaccard) vs mDeBERTa score:")
    print(f"  Overall: mean={np.mean(jaccards):.4f}")

    # Correlation
    correlation = np.corrcoef(contra_probs, jaccards)[0, 1]
    print(f"  Correlation with mDeBERTa score: {correlation:.4f}")

    # By score range
    print(f"\n  Jaccard by mDeBERTa score range:")
    for name, low, high in ranges:
        range_jaccards = [j for c, j in jaccard_scores if low <= c < high]
        if range_jaccards:
            print(f"    {name}: mean={np.mean(range_jaccards):.4f} (n={len(range_jaccards)})")


# 4. What's happening with numerical perturbations?
print("\n\n4. NUMERICAL PERTURBATION DEEP DIVE")
print("-" * 70)

numerical_scores = [s for s in scores if s["perturbation_type"] == "numerical_perturbation"]
print(f"Total numerical perturbations: {len(numerical_scores)}")
print(f"Mean contradiction score: {np.mean([s['mdeberta_contra_prob'] for s in numerical_scores]):.4f}")

# Show some numerical examples
print("\nSample numerical perturbations:")
for s in numerical_scores[:5]:
    aid = s["article_id"]
    if aid in perturbations:
        row = perturbations[aid]
        orig = row.get("original_text", "")
        pert = row.get("text", "")

        # Find numbers in both
        import re
        orig_nums = re.findall(r'[\d০-৯]+', orig)
        pert_nums = re.findall(r'[\d০-৯]+', pert)

        print(f"\n  {aid}: score={s['mdeberta_contra_prob']:.4f}")
        print(f"    Original nums: {orig_nums[:5]}")
        print(f"    Perturbed nums: {pert_nums[:5]}")
        print(f"    Text snippet: ...{orig[:100]}...")
        print(f"    Perturbed: ...{pert[:100]}...")


# 5. Threshold sensitivity
print("\n\n5. THRESHOLD SENSITIVITY ANALYSIS")
print("-" * 70)

all_contra = [s["mdeberta_contra_prob"] for s in scores]
thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]
print(f"{'Threshold':>10} {'Kept':>8} {'%':>8} {'Per Type (min)':>15}")
print("-" * 50)

for t in thresholds:
    kept = sum(1 for c in all_contra if c >= t)
    pct = 100 * kept / len(all_contra)

    # Min per type
    type_kept = defaultdict(int)
    type_total = defaultdict(int)
    for s in scores:
        type_total[s["perturbation_type"]] += 1
        if s["mdeberta_contra_prob"] >= t:
            type_kept[s["perturbation_type"]] += 1

    min_per_type = min(100 * type_kept[pt] / type_total[pt] for pt in type_total) if type_total else 0
    print(f"{t:>10.2f} {kept:>8} {pct:>7.1f}% {min_per_type:>14.1f}%")


print("\n" + "=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)

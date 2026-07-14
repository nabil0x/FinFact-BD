#!/usr/bin/env python3
"""
FinFact-BD mDeBERTa Quality Filter
====================================
Single-model NLI filtering for Bengali financial misinformation detection.

Pipeline:
   1. Load rule-filtered perturbations from zstd CSV
   2. Score each (original, perturbed) pair using mDeBERTa-v3 multilingual NLI
   3. Keep samples with contradiction probability >= threshold

Validation:
   - Filter validated against human judgment (300 samples × 3 annotators)
   - Report: Fleiss' κ, model-human agreement, examples of accepted/rejected
   - This validation is the evidence reviewers need, not additional model complexity

Why mDeBERTa alone:
   - One of the strongest multilingual NLI models (2M+ training examples)
   - Used as a filter, not as the benchmark model
   - Simplifies pipeline without sacrificing quality
   - Dual-model (mDeBERTa + BanglaBERT) only if >2-3% improvement shown

Outputs:
   - finfact_bd_perturbed_filtered.csv   (quality-filtered perturbations)
   - mdeberta_scores.json                (per-sample NLI scores)
   - mdeberta_report.json                (summary statistics)

Usage:
   python scripts/mdeberta_filter.py
"""

import csv
import io
import json
import time
import os
from pathlib import Path
from collections import Counter

import numpy as np
import torch

import zstandard as zstd

from transformers import AutoModelForSequenceClassification, AutoTokenizer


# =====================================================================
# PATHS
# =====================================================================

# Kaggle path (primary)
KAGGLE_INPUT = Path(
    "/kaggle/input/datasets/annnasernabil/"
    "finfact-bd-perturbed-rule-filtered/"
    "finfact_bd_perturbed_rule_filtered.csv.zst"
)

def find_input_file():
    """Find rule-filtered CSV (Kaggle or local fallback)."""
    if KAGGLE_INPUT.exists():
        return KAGGLE_INPUT
    # Local fallback
    local = Path(__file__).parent.parent / "data" / "finfact_bd"
    for f in local.iterdir():
        if "rule_filtered" in f.name and f.suffix in (".csv", ".zst"):
            return f
    raise FileNotFoundError(
        "Upload finfact_bd_perturbed_rule_filtered.csv.zst as a Kaggle Dataset, "
        "or place it in data/finfact_bd/ locally."
    )

INPUT_FILE = find_input_file()
OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle").exists() else Path(__file__).parent.parent / "data" / "finfact_bd"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# =====================================================================
# CONFIGURATION
# =====================================================================
MODEL_ID = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

MAX_LEN = 512
BATCH_SIZE = 32
CONTRADICTION_THRESHOLD = 0.4

LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

# =====================================================================
# LOAD RULE-FILTERED PERTURBATIONS
# =====================================================================


def load_csv(path: Path):
    samples = []
    if path.suffix == ".zst":
        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                reader_csv = csv.DictReader(text_stream)
                for row in reader_csv:
                    samples.append(row)
    else:
        with open(path, "r", encoding="utf-8") as f:
            reader_csv = csv.DictReader(f)
            for row in reader_csv:
                samples.append(row)
    return samples


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found:\n{INPUT_FILE}\n\n"
        "Check that the Kaggle dataset is attached correctly."
    )

perturbed = load_csv(INPUT_FILE)

print(f"Loaded {len(perturbed):,} rule-filtered perturbations")
print(f"Source: {INPUT_FILE}")
print(f"\nColumns: {list(perturbed[0].keys())}")
print("\nPerturbation distribution:")
type_counts = Counter(row["perturbation_type"] for row in perturbed)
for ptype, count in type_counts.most_common():
    print(f"  {ptype:<30} {count:,}")

# =====================================================================
# LOAD MODEL
# =====================================================================
print(f"\nLoading mDeBERTa-v3 NLI model...")
print(f"  Model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model = model.to(DEVICE)
model.eval()

if DEVICE == "cuda":
    print(f"  GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# =====================================================================
# SCORE ALL SAMPLES
# =====================================================================
print(f"\nScoring {len(perturbed):,} pairs with mDeBERTa ({BATCH_SIZE}/batch)...")


def score_batch(pairs):
    """Score a batch using mDeBERTa NLI model."""
    inputs = tokenizer(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

    probs_np = probs.cpu().numpy()

    results = []
    for i in range(len(pairs)):
        pred_idx = int(probs_np[i].argmax())
        results.append({
            "label": LABEL_MAP[pred_idx],
            "probs": {
                "contradiction": float(probs_np[i][0]),
                "entailment": float(probs_np[i][1]),
                "neutral": float(probs_np[i][2]),
            },
        })
    return results


pairs = [(row["original_text"], row["text"]) for row in perturbed]

all_results = []
t0 = time.time()
for i in range(0, len(pairs), BATCH_SIZE):
    batch = pairs[i : i + BATCH_SIZE]
    results = score_batch(batch)
    all_results.extend(results)

    if (i // BATCH_SIZE) % 10 == 0:
        elapsed = time.time() - t0
        pct = min(100, 100 * (i + len(batch)) / len(pairs))
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (len(pairs) - i - len(batch)) / rate if rate > 0 else 0
        print(
            f"  [{pct:.0f}%] {i + len(batch)}/{len(pairs)} "
            f"\u2014 {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        )

elapsed = time.time() - t0
print(f"\nScoring complete in {elapsed:.0f}s ({elapsed / len(pairs) * 1000:.1f}ms/sample)")

# =====================================================================
# ANALYSIS
# =====================================================================
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

label_counts = Counter(r["label"] for r in all_results)
print("Overall NLI distribution:")
for label, count in label_counts.most_common():
    print(f"  {label}: {count} ({100 * count / len(all_results):.1f}%)")

type_results = {}
for row, result in zip(perturbed, all_results):
    ptype = row["perturbation_type"]
    type_results.setdefault(ptype, Counter())[result["label"]] += 1

print("\nPer-type distribution:")
for ptype, counts in sorted(type_results.items()):
    total = sum(counts.values())
    contra = 100 * counts.get("contradiction", 0) / total
    entail = 100 * counts.get("entailment", 0) / total
    print(f"  {ptype}: contradiction={contra:.0f}% entailment={entail:.0f}% (n={total})")

# =====================================================================
# FILTER AND SAVE
# =====================================================================
print("\n" + "=" * 60)
print("FILTER AND SAVE")
print("=" * 60)

kept = []
discarded = 0
for row, result in zip(perturbed, all_results):
    contra_prob = result["probs"]["contradiction"]
    if contra_prob >= CONTRADICTION_THRESHOLD:
        row["contradiction_score"] = f"{contra_prob:.4f}"
        row["mdeberta_label"] = result["label"]
        kept.append(row)
    else:
        discarded += 1

print(f"Kept: {len(kept)}, Discarded: {discarded} ({100 * discarded / len(perturbed):.1f}%)")

kept_types = Counter(row["perturbation_type"] for row in kept)
print("\nKept per type:")
for ptype, count in kept_types.most_common():
    print(f"  {ptype}: {count}")

# Save filtered CSV
csv_fields = list(kept[0].keys())
out_path = OUTPUT_DIR / "finfact_bd_perturbed_filtered.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_fields)
    writer.writeheader()
    writer.writerows(kept)
print(f"\nSaved filtered CSV: {out_path}")

# Save scores
scores = []
for row, result in zip(perturbed, all_results):
    scores.append({
        "article_id": row["article_id"],
        "perturbation_type": row["perturbation_type"],
        "mdeberta_label": result["label"],
        "mdeberta_contra_prob": result["probs"]["contradiction"],
    })
scores_path = OUTPUT_DIR / "mdeberta_scores.json"
with open(scores_path, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2, ensure_ascii=False)
print(f"Saved scores: {scores_path}")

# Save report
contra_probs = [r["probs"]["contradiction"] for r in all_results]
report = {
    "total_samples": len(perturbed),
    "kept": len(kept),
    "discarded": discarded,
    "keep_rate": len(kept) / len(perturbed),
    "threshold": CONTRADICTION_THRESHOLD,
    "model": MODEL_ID,
    "contradiction_stats": {
        "mean": float(np.mean(contra_probs)),
        "median": float(np.median(contra_probs)),
        "min": float(np.min(contra_probs)),
        "max": float(np.max(contra_probs)),
        "std": float(np.std(contra_probs)),
    },
    "per_type_keep_rate": {
        ptype: counts.get("contradiction", 0) / sum(counts.values())
        for ptype, counts in type_results.items()
    },
}
report_path = OUTPUT_DIR / "mdeberta_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Saved report: {report_path}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("MDEBERTA QUALITY FILTER \u2014 COMPLETE")
print("=" * 60)
print(f"Input:        {len(perturbed)} rule-filtered perturbations")
print(f"Output:       {len(kept)} quality-filtered perturbations")
print(f"Discarded:    {discarded} ({100 * discarded / len(perturbed):.1f}%)")
print(f"Threshold:    {CONTRADICTION_THRESHOLD}")
print(f"Model:        {MODEL_ID}")
print()
print("Files written to /kaggle/working/:")
print("  1. finfact_bd_perturbed_filtered.csv")
print("  2. mdeberta_scores.json")
print("  3. mdeberta_report.json")
print("=" * 60)

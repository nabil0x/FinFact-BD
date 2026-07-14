#!/usr/bin/env python3
"""
FinFact-BD Quality Filter: Dual-model contradiction detection.

Uses two models to validate perturbation quality:
1. mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (multilingual, ready-to-use)
2. BanglaBERT (Bengali-native, fine-tuned on xnli_bn)

Scores each (original, perturbed) pair for contradiction.
- High contradiction score → good perturbation (meaning changed)
- Low contradiction score → bad perturbation (essentially the same)
"""

import csv
import json
import sys
import time
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_output_dir

# =============================================================================
# CONFIG
# =============================================================================

config = get_config()
DATA_DIR = get_output_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config['quality_filter'].get('batch_size', 32)
MAX_LEN = 512

# Model configs from config file
MODEL_CONFIGS = config['quality_filter']['models']
CONTRADICTION_THRESHOLD = config['quality_filter']['contradiction_threshold']

# =============================================================================
# LOAD MODEL
# =============================================================================

print(f"Loading model {MODEL_NAME} on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()
print("Model loaded.")

# Label mapping for this model: 0=contradiction, 1=neutral, 2=entailment
LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}


def score_batch(pairs):
    """Score a batch of (premise, hypothesis) pairs. Returns list of dicts."""
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

    results = []
    for i in range(len(pairs)):
        prob_dict = {
            LABEL_MAP[j]: probs[i][j].item()
            for j in range(3)
        }
        predicted_label = LABEL_MAP[probs[i].argmax().item()]
        results.append({
            "label": predicted_label,
            "probs": prob_dict,
        })
    return results


def load_perturbed():
    """Load perturbed samples."""
    path = DATA_DIR / "finfact_bd_perturbed.csv"
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


def load_originals():
    """Load original samples."""
    path = DATA_DIR / "finfact_bd_originals.csv"
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    return samples


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading perturbed samples...")
    perturbed = load_perturbed()
    print(f"Loaded {len(perturbed)} perturbed samples.")

    # Build pairs: (original_text, perturbed_text)
    pairs = [(row["original_text"], row["text"]) for row in perturbed]

    # Score in batches
    print(f"Scoring {len(pairs)} pairs in batches of {BATCH_SIZE}...")
    all_results = []
    t0 = time.time()

    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        results = score_batch(batch)
        all_results.extend(results)

        if (i // BATCH_SIZE) % 20 == 0:
            elapsed = time.time() - t0
            pct = min(100, 100 * (i + len(batch)) / len(pairs))
            print(f"  [{pct:.0f}%] {i + len(batch)}/{len(pairs)} — {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"Scoring complete in {elapsed:.1f}s")

    # Analysis
    label_counts = Counter(r["label"] for r in all_results)
    print(f"\nOverall distribution:")
    for label, count in label_counts.most_common():
        print(f"  {label}: {count} ({100*count/len(all_results):.1f}%)")

    # Per-type analysis
    type_results = {}
    for row, result in zip(perturbed, all_results):
        ptype = row["perturbation_type"]
        if ptype not in type_results:
            type_results[ptype] = Counter()
        type_results[ptype][result["label"]] += 1

    print(f"\nPer-type distribution:")
    for ptype, counts in sorted(type_results.items()):
        total = sum(counts.values())
        contra_pct = 100 * counts.get("contradiction", 0) / total
        entail_pct = 100 * counts.get("entailment", 0) / total
        print(f"  {ptype}: contradiction={contra_pct:.0f}% entailment={entail_pct:.0f}% (n={total})")

    # Filter: keep samples with contradiction probability >= threshold
    print(f"\nFiltering with contradiction threshold >= {CONTRADICTION_THRESHOLD}...")
    kept = []
    discarded = 0
    for row, result in zip(perturbed, all_results):
        contra_prob = result["probs"]["contradiction"]
        if contra_prob >= CONTRADICTION_THRESHOLD:
            row["xnli_contradiction_prob"] = f"{contra_prob:.4f}"
            row["xnli_label"] = result["label"]
            kept.append(row)
        else:
            discarded += 1

    print(f"Kept: {len(kept)}, Discarded: {discarded} ({100*discarded/len(perturbed):.1f}%)")

    # Per-type kept counts
    kept_types = Counter(row["perturbation_type"] for row in kept)
    print(f"\nKept per type:")
    for ptype, count in kept_types.most_common():
        print(f"  {ptype}: {count}")

    # Save filtered perturbed
    if kept:
        out_path = DATA_DIR / "finfact_bd_perturbed_filtered.csv"
        fieldnames = list(kept[0].keys())
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        print(f"\nSaved filtered perturbed to {out_path}")

    # Save full results for analysis
    results_path = DATA_DIR / "xnli_scores.json"
    scores = []
    for row, result in zip(perturbed, all_results):
        scores.append({
            "article_id": row["article_id"],
            "perturbation_type": row["perturbation_type"],
            "xnli_label": result["label"],
            "contradiction_prob": result["probs"]["contradiction"],
            "neutral_prob": result["probs"]["neutral"],
            "entailment_prob": result["probs"]["entailment"],
        })
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Saved XNLI scores to {results_path}")

    # Summary stats
    contra_probs = [r["probs"]["contradiction"] for r in all_results]
    print(f"\nContradiction probability stats:")
    print(f"  Mean: {sum(contra_probs)/len(contra_probs):.3f}")
    print(f"  Median: {sorted(contra_probs)[len(contra_probs)//2]:.3f}")
    print(f"  Min: {min(contra_probs):.3f}")
    print(f"  Max: {max(contra_probs):.3f}")


if __name__ == "__main__":
    main()

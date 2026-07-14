#!/usr/bin/env python3
"""
FinFact-BD Dual-Model Contradiction Judge
Uses two models to validate perturbation quality:
1. mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (multilingual, ready-to-use)
2. BanglaBERT (Bengali-native, fine-tuned on xnli_bn) - optional

Scores each (original, perturbed) pair for contradiction.
"""

import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_output_dir

# =============================================================================
# CONFIG
# =============================================================================

config = get_config()
OUTPUT_DIR = get_output_dir()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model configs
MODEL_CONFIGS = config['quality_filter']['models']
CONTRADICTION_THRESHOLD = config['quality_filter']['contradiction_threshold']
BATCH_SIZE = 32
MAX_LEN = 512

# Input/output paths
RULE_FILTERED_PATH = OUTPUT_DIR / "finfact_bd_perturbed_rule_filtered.csv"
FILTERED_PATH = OUTPUT_DIR / "finfact_bd_perturbed_filtered.csv"
SCORES_PATH = OUTPUT_DIR / "dual_model_scores.json"
REPORT_PATH = OUTPUT_DIR / "dual_model_report.json"


class ModelEnsemble:
    """Ensemble of NLI models for contradiction detection."""
    
    def __init__(self, model_configs: List[Dict]):
        self.models = []
        self.tokenizers = []
        self.weights = []
        self.model_names = []
        
        for config in model_configs:
            model_id = config['model_id']
            weight = config.get('weight', 1.0)
            name = config.get('name', model_id)
            
            print(f"Loading {name} ({model_id}) on {DEVICE}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForSequenceClassification.from_pretrained(model_id)
                model = model.to(DEVICE)
                model.eval()
                
                self.models.append(model)
                self.tokenizers.append(tokenizer)
                self.weights.append(weight)
                self.model_names.append(name)
                
                print(f"  ✓ {name} loaded successfully")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")
    
    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict]:
        """Score a batch of (premise, hypothesis) pairs using ensemble."""
        if not self.models:
            raise ValueError("No models loaded")
        
        # Collect predictions from all models
        all_probs = []
        
        for model, tokenizer in zip(self.models, self.tokenizers):
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
            
            all_probs.append(probs.cpu().numpy())
        
        # Weighted ensemble
        # Label mapping: 0=contradiction, 1=neutral, 2=entailment
        ensemble_probs = None
        total_weight = sum(self.weights)
        
        for probs, weight in zip(all_probs, self.weights):
            if ensemble_probs is None:
                ensemble_probs = probs * (weight / total_weight)
            else:
                ensemble_probs += probs * (weight / total_weight)
        
        # Convert to results
        results = []
        for i in range(len(pairs)):
            probs_dict = {
                "contradiction": float(ensemble_probs[i][0]),
                "neutral": float(ensemble_probs[i][1]),
                "entailment": float(ensemble_probs[i][2]),
            }
            
            # Get predicted label
            pred_idx = ensemble_probs[i].argmax()
            label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
            predicted_label = label_map[pred_idx]
            
            results.append({
                "label": predicted_label,
                "probs": probs_dict,
                "model_predictions": {}
            })
            
            # Add individual model predictions
            for j, (model_name, probs) in enumerate(zip(self.model_names, all_probs)):
                results[i]["model_predictions"][model_name] = {
                    "label": label_map[probs[i].argmax()],
                    "contradiction_prob": float(probs[i][0])
                }
        
        return results


def load_rule_filtered() -> List[Dict]:
    """Load rule-filtered perturbations."""
    print(f"Loading rule-filtered perturbations from {RULE_FILTERED_PATH}...")
    
    samples = []
    with open(RULE_FILTERED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(row)
    
    print(f"Loaded {len(samples)} samples")
    return samples


def main():
    """Main function."""
    # Check if rule-filtered file exists
    if not RULE_FILTERED_PATH.exists():
        print(f"ERROR: {RULE_FILTERED_PATH} not found.")
        print("Run rule_based_filter.py first.")
        return
    
    # Load samples
    samples = load_rule_filtered()
    
    # Initialize ensemble
    print("\nInitializing model ensemble...")
    ensemble = ModelEnsemble(MODEL_CONFIGS)
    
    if not ensemble.models:
        print("ERROR: No models loaded successfully.")
        return
    
    # Build pairs
    pairs = [(row["original_text"], row["text"]) for row in samples]
    
    # Score in batches
    print(f"\nScoring {len(pairs)} pairs in batches of {BATCH_SIZE}...")
    all_results = []
    t0 = time.time()
    
    for i in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[i : i + BATCH_SIZE]
        results = ensemble.score_batch(batch)
        all_results.extend(results)
        
        if (i // BATCH_SIZE) % 10 == 0:
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
    for row, result in zip(samples, all_results):
        ptype = row["perturbation_type"]
        if ptype not in type_results:
            type_results[ptype] = Counter()
        type_results[ptype][result["label"]] += 1
    
    print(f"\nPer-type distribution:")
    for ptype, counts in sorted(type_results.items()):
        total = sum(counts.values())
        contra_pct = 100 * counts.get("contradiction", 0) / total
        print(f"  {ptype}: contradiction={contra_pct:.0f}% (n={total})")
    
    # Filter: keep samples with contradiction probability >= threshold
    print(f"\nFiltering with contradiction threshold >= {CONTRADICTION_THRESHOLD}...")
    kept = []
    discarded = 0
    
    for row, result in zip(samples, all_results):
        contra_prob = result["probs"]["contradiction"]
        if contra_prob >= CONTRADICTION_THRESHOLD:
            row["contradiction_score"] = f"{contra_prob:.4f}"
            row["ensemble_label"] = result["label"]
            row["model_predictions"] = json.dumps(result["model_predictions"])
            kept.append(row)
        else:
            discarded += 1
    
    print(f"Kept: {len(kept)}, Discarded: {discarded} ({100*discarded/len(samples):.1f}%)")
    
    # Per-type kept counts
    kept_types = Counter(row["perturbation_type"] for row in kept)
    print(f"\nKept per type:")
    for ptype, count in kept_types.most_common():
        print(f"  {ptype}: {count}")
    
    # Save filtered perturbed
    if kept:
        fieldnames = list(kept[0].keys())
        # Remove model_predictions from CSV (it's JSON)
        csv_fieldnames = [f for f in fieldnames if f != "model_predictions"]
        with open(FILTERED_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
            writer.writeheader()
            for row in kept:
                csv_row = {k: v for k, v in row.items() if k != "model_predictions"}
                writer.writerow(csv_row)
        print(f"\nSaved filtered perturbed to {FILTERED_PATH}")
    
    # Save full results for analysis
    scores = []
    for row, result in zip(samples, all_results):
        scores.append({
            "article_id": row["article_id"],
            "perturbation_type": row["perturbation_type"],
            "ensemble_label": result["label"],
            "contradiction_prob": result["probs"]["contradiction"],
            "neutral_prob": result["probs"]["neutral"],
            "entailment_prob": result["probs"]["entailment"],
            "model_predictions": result["model_predictions"],
        })
    
    with open(SCORES_PATH, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)
    print(f"Saved scores to {SCORES_PATH}")
    
    # Summary stats
    contra_probs = [r["probs"]["contradiction"] for r in all_results]
    print(f"\nContradiction probability stats:")
    print(f"  Mean: {sum(contra_probs)/len(contra_probs):.3f}")
    print(f"  Median: {sorted(contra_probs)[len(contra_probs)//2]:.3f}")
    print(f"  Min: {min(contra_probs):.3f}")
    print(f"  Max: {max(contra_probs):.3f}")
    
    # Save report
    report = {
        "total_samples": len(samples),
        "kept": len(kept),
        "discarded": discarded,
        "keep_rate": len(kept) / len(samples) if samples else 0,
        "threshold": CONTRADICTION_THRESHOLD,
        "models_used": ensemble.model_names,
        "contradiction_stats": {
            "mean": sum(contra_probs)/len(contra_probs),
            "median": sorted(contra_probs)[len(contra_probs)//2],
            "min": min(contra_probs),
            "max": max(contra_probs),
        },
        "per_type_keep_rate": {
            ptype: counts.get("contradiction", 0) / sum(counts.values())
            for ptype, counts in type_results.items()
        }
    }
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()

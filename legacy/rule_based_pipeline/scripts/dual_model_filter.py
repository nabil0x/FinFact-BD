#!/usr/bin/env python3
"""
FinFact-BD Dual-Model Quality Filter
=====================================
Two-stage NLI filtering for Bengali financial misinformation detection.

Pipeline:
  1. Fine-tune BanglaBERT on csebuetnlp/xnli_bn (~1.5 hrs on T4)
  2. Score all rule-filtered perturbations using ensemble:
     - mDeBERTa-v3 multilingual NLI (ready-to-use, weight=0.6)
     - BanglaBERT fine-tuned (weight=0.4)
  3. Keep samples with ensemble contradiction probability >= 0.4

Outputs:
  - finfact_bd_perturbed_filtered.csv   (quality-filtered perturbations)
  - dual_model_scores.json              (per-sample NLI scores)
  - dual_model_report.json              (summary statistics)
  - banglabert_nli_finetuned/           (fine-tuned model checkpoint)
  - banglabert_nli_metrics.json         (training metrics)

Usage (Kaggle T4):
  1. Upload finfact_bd_perturbed_rule_filtered.csv.zst as a Kaggle Dataset
  2. Add that dataset to this notebook's input
  3. python dual_model_filter.py
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

from huggingface_hub import hf_hub_download
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score

# =====================================================================
# PATHS
# =====================================================================
INPUT_FILE = Path(
    "/kaggle/input/datasets/annnasernabil/"
    "finfact-bd-perturbed-rule-filtered/"
    "finfact_bd_perturbed_rule_filtered.csv.zst"
)

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
# STAGE 1: FINE-TUNE BANGLABERT ON BENGALI NLI
# =====================================================================
print("\n" + "=" * 60)
print("STAGE 1: Fine-tune BanglaBERT on Bengali NLI")
print("=" * 60)

NLI_MODEL = "csebuetnlp/banglabert"
NLI_DATASET = "csebuetnlp/xnli_bn"
FINETUNED_DIR = OUTPUT_DIR / "banglabert_nli_finetuned"
FINETUNED_DIR.mkdir(exist_ok=True)

# --- Load Bengali NLI dataset via parquet download (bypasses broken script) ---
# Primary: csebuetnlp/xnli_bn (391K Bengali NLI pairs)
# Fallback: Divyanshu/indicxnli (392K Bengali NLI pairs, EMNLP 2022)
print(f"\nDownloading Bengali NLI dataset...")

try:
    print(f"Trying csebuetnlp/xnli_bn...")
    train_path = hf_hub_download(
        repo_id="csebuetnlp/xnli_bn",
        repo_type="dataset",
        filename="data/train-00000-of-00001.parquet",
    )
    test_path = hf_hub_download(
        repo_id="csebuetnlp/xnli_bn",
        repo_type="dataset",
        filename="data/test-00000-of-00001.parquet",
    )
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    # xnli_bn columns: sentence1, sentence2, label (0=contradiction, 1=entailment, 2=neutral)
    SENT1_COL, SENT2_COL = "sentence1", "sentence2"
    LABEL_MAP_FN = lambda x: x  # Already 0=contra, 1=entail, 2=neutral
    print(f"Loaded xnli_bn: Train={len(train_df):,}, Test={len(test_df):,}")

except Exception as e:
    print(f"  xnli_bn failed: {e}")
    print(f"Falling back to Divyanshu/indicxnli (Bengali subset)...")
    train_path = hf_hub_download(
        repo_id="Divyanshu/indicxnli",
        repo_type="dataset",
        filename="forward/bn_train.json",
    )
    test_path = hf_hub_download(
        repo_id="Divyanshu/indicxnli",
        repo_type="dataset",
        filename="forward/bn_test.json",
    )
    train_df = pd.read_json(train_path)
    test_df = pd.read_json(test_path)
    # indicxnli columns: premise, hypothesis, label (0=entailment, 1=neutral, 2=contradiction)
    SENT1_COL, SENT2_COL = "premise", "hypothesis"
    # Remap labels: 0=entailment→1, 1=neutral→2, 2=contradiction→0
    LABEL_REMAP = {0: 1, 1: 2, 2: 0}
    train_df["label"] = train_df["label"].map(LABEL_REMAP)
    test_df["label"] = test_df["label"].map(LABEL_REMAP)
    print(f"Loaded indicxnli Bengali: Train={len(train_df):,}, Test={len(test_df):,}")

print(f"Columns: {list(train_df.columns)}")
print(f"Label distribution (train):")
label_names = {0: "contradiction", 1: "entailment", 2: "neutral"}
for label_val, count in train_df["label"].value_counts().sort_index().items():
    print(f"  {label_names.get(label_val, label_val)}: {count:,}")

xnli = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "validation": Dataset.from_pandas(test_df),
})
print(f"Dataset: {xnli}")

print(f"\nLoading {NLI_MODEL} tokenizer...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)

# --- Tokenize ---
MAX_LEN_NLI = 128


def tokenize_nli(examples):
    return nli_tokenizer(
        examples[SENT1_COL],
        examples[SENT2_COL],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN_NLI,
    )


xnli_tokenized = xnli.map(
    tokenize_nli, batched=True, remove_columns=[SENT1_COL, SENT2_COL]
)
xnli_tokenized.set_format("torch")

if "validation" not in xnli_tokenized:
    split = xnli_tokenized["train"].train_test_split(test_size=0.1, seed=42)
    xnli_tokenized["train"] = split["train"]
    xnli_tokenized["validation"] = split["test"]
    print(f"Split into train={len(split['train'])}, val={len(split['test'])}")
else:
    print(
        f"Train: {len(xnli_tokenized['train'])}, "
        f"Val: {len(xnli_tokenized['validation'])}"
    )

# --- Fine-tune ---


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL, num_labels=3)

training_args = TrainingArguments(
    output_dir=str(FINETUNED_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=True,
    seed=42,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=xnli_tokenized["train"],
    eval_dataset=xnli_tokenized["validation"],
    compute_metrics=compute_metrics,
)

print("\nStarting fine-tuning...")
t0 = time.time()
trainer.train()
train_time = time.time() - t0
print(f"\nTraining complete in {train_time / 60:.1f} min")

eval_results = trainer.evaluate()
print(f"Evaluation: {eval_results}")

trainer.save_model(str(FINETUNED_DIR))
nli_tokenizer.save_pretrained(str(FINETUNED_DIR))
print(f"Model saved to {FINETUNED_DIR}")

metrics = {
    "base_model": NLI_MODEL,
    "dataset": NLI_DATASET,
    "training_time_minutes": train_time / 60,
    "eval_accuracy": eval_results["eval_accuracy"],
    "eval_f1_macro": eval_results["eval_f1_macro"],
}
with open(OUTPUT_DIR / "banglabert_nli_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

# =====================================================================
# STAGE 2: DUAL-MODEL ENSEMBLE SCORING
# =====================================================================
print("\n" + "=" * 60)
print("STAGE 2: Dual-Model Ensemble Scoring")
print("=" * 60)

MODELS = [
    {
        "name": "mDeBERTa",
        "model_id": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        "weight": 0.6,
    },
    {
        "name": "BanglaBERT",
        "model_id": str(FINETUNED_DIR),
        "weight": 0.4,
    },
]

MAX_LEN = 512
BATCH_SIZE = 32
CONTRADICTION_THRESHOLD = 0.4

MDEBERTA_LABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}
BBLABEL_MAP = {0: "contradiction", 1: "entailment", 2: "neutral"}

# --- Load models ---
loaded_models = []
for mcfg in MODELS:
    print(f"Loading {mcfg['name']} ({mcfg['model_id']})...")
    tok = AutoTokenizer.from_pretrained(mcfg["model_id"])
    mdl = AutoModelForSequenceClassification.from_pretrained(mcfg["model_id"])
    mdl = mdl.to(DEVICE)
    mdl.eval()
    loaded_models.append({
        "name": mcfg["name"],
        "model": mdl,
        "tokenizer": tok,
        "weight": mcfg["weight"],
        "label_map": MDEBERTA_LABEL_MAP
        if "mDeBERTa" in mcfg["name"]
        else BBLABEL_MAP,
    })
    print(f"  GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

print(
    f"\nLoaded {len(loaded_models)} models. "
    f"Total GPU mem: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
)


# --- Score batch ---


def score_batch_ensemble(pairs, models):
    """Score a batch using weighted ensemble of NLI models."""
    all_probs = []
    total_weight = sum(m["weight"] for m in models)

    for m in models:
        inputs = m["tokenizer"](
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            logits = m["model"](**inputs).logits
            probs = torch.softmax(logits, dim=-1)
        all_probs.append(
            (probs.cpu().numpy(), m["weight"] / total_weight, m["label_map"])
        )

    # Weighted average
    ensemble = None
    for probs, w, _ in all_probs:
        if ensemble is None:
            ensemble = probs * w
        else:
            ensemble += probs * w

    LABEL_NAMES = {0: "contradiction", 1: "entailment", 2: "neutral"}

    results = []
    for i in range(len(pairs)):
        pred_idx = int(ensemble[i].argmax())
        results.append({
            "label": LABEL_NAMES[pred_idx],
            "probs": {
                "contradiction": float(ensemble[i][0]),
                "entailment": float(ensemble[i][1]),
                "neutral": float(ensemble[i][2]),
            },
            "per_model": {
                m["name"]: {
                    "contradiction_prob": float(all_probs[j][0][i][0]),
                }
                for j, m in enumerate(models)
            },
        })
    return results


# --- Score all samples ---
pairs = [(row["original_text"], row["text"]) for row in perturbed]
print(f"Scoring {len(pairs)} pairs with ensemble ({BATCH_SIZE}/batch)...")

all_results = []
t0 = time.time()
for i in range(0, len(pairs), BATCH_SIZE):
    batch = pairs[i : i + BATCH_SIZE]
    results = score_batch_ensemble(batch, loaded_models)
    all_results.extend(results)
    if (i // BATCH_SIZE) % 10 == 0:
        elapsed = time.time() - t0
        pct = min(100, 100 * (i + len(batch)) / len(pairs))
        rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (len(pairs) - i - len(batch)) / rate if rate > 0 else 0
        print(
            f"  [{pct:.0f}%] {i + len(batch)}/{len(pairs)} "
            f"— {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
        )

elapsed = time.time() - t0
print(f"\nScoring complete in {elapsed:.0f}s ({elapsed / len(pairs) * 1000:.1f}ms/sample)")

# --- Analysis ---
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

label_counts = Counter(r["label"] for r in all_results)
print("Overall ensemble distribution:")
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

print("\nPer-model contradiction rates:")
for mcfg in MODELS:
    mname = mcfg["name"]
    contra_probs = [
        r["per_model"][mname]["contradiction_prob"] for r in all_results
    ]
    above_thresh = sum(1 for p in contra_probs if p >= CONTRADICTION_THRESHOLD)
    print(
        f"  {mname}: mean={np.mean(contra_probs):.3f}, "
        f"above {CONTRADICTION_THRESHOLD}="
        f"{above_thresh}/{len(contra_probs)} "
        f"({100 * above_thresh / len(contra_probs):.0f}%)"
    )

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
        row["ensemble_label"] = result["label"]
        row["mDeBERTa_contra_prob"] = (
            f"{result['per_model']['mDeBERTa']['contradiction_prob']:.4f}"
        )
        row["BanglaBERT_contra_prob"] = (
            f"{result['per_model']['BanglaBERT']['contradiction_prob']:.4f}"
        )
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
        "ensemble_label": result["label"],
        "ensemble_contra_prob": result["probs"]["contradiction"],
        "mDeBERTa_contra_prob": result["per_model"]["mDeBERTa"]["contradiction_prob"],
        "BanglaBERT_contra_prob": result["per_model"]["BanglaBERT"]["contradiction_prob"],
    })
scores_path = OUTPUT_DIR / "dual_model_scores.json"
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
    "models_used": [m["name"] for m in loaded_models],
    "ensemble_weights": {m["name"]: m["weight"] for m in loaded_models},
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
report_path = OUTPUT_DIR / "dual_model_report.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"Saved report: {report_path}")

# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("DUAL-MODEL QUALITY FILTER — COMPLETE")
print("=" * 60)
print(f"Input:        {len(perturbed)} rule-filtered perturbations")
print(f"Output:       {len(kept)} quality-filtered perturbations")
print(f"Discarded:    {discarded} ({100 * discarded / len(perturbed):.1f}%)")
print(f"Threshold:    {CONTRADICTION_THRESHOLD}")
print(f"Models:       {', '.join(m['name'] for m in loaded_models)}")
print(f"Training:     {metrics.get('eval_accuracy', 'N/A')} accuracy on xnli_bn")
print()
print("Files written to /kaggle/working/:")
print("  1. finfact_bd_perturbed_filtered.csv")
print("  2. dual_model_scores.json")
print("  3. dual_model_report.json")
print("  4. banglabert_nli_finetuned/ (model checkpoint)")
print("  5. banglabert_nli_metrics.json")
print("=" * 60)

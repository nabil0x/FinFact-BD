#!/usr/bin/env python3
"""
Fine-tune BanglaBERT on Bengali NLI (xnli_bn) dataset.
This script fine-tunes the base BanglaBERT model for 3-class NLI.
"""

import sys
import json
import time
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_output_dir

# =============================================================================
# CONFIG
# =============================================================================

config = get_config()
OUTPUT_DIR = get_output_dir()

# Model config
BASE_MODEL = "csebuetnlp/banglabert"
DATASET_NAME = "csebuetnlp/xnli_bn"
OUTPUT_MODEL_DIR = OUTPUT_DIR / "banglabert_nli_finetuned"
METRICS_PATH = OUTPUT_DIR / "banglabert_nli_metrics.json"

# Training config
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 32,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "logging_dir": str(OUTPUT_DIR / "logs"),
    "logging_steps": 100,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "fp16": torch.cuda.is_available(),
    "seed": 42,
}


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision_macro = precision_score(labels, predictions, average="macro")
    recall_macro = recall_score(labels, predictions, average="macro")
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }


def main():
    """Main function."""
    print(f"Fine-tuning {BASE_MODEL} on {DATASET_NAME}...")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(DATASET_NAME)
    print(f"Dataset loaded: {dataset}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Load model
    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=3  # contradiction, entailment, neutral
    )
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=["premise", "hypothesis"],
    )
    
    # Rename label column if needed
    if "label" not in tokenized_dataset["train"].column_names:
        # Map label names to indices
        label_map = {"contradiction": 0, "entailment": 1, "neutral": 2}
        def map_labels(example):
            example["label"] = label_map[example["label"]]
            return example
        tokenized_dataset = tokenized_dataset.map(map_labels)
    
    # Set format for PyTorch
    tokenized_dataset.set_format("torch")
    
    # Split validation set if not present
    if "validation" not in tokenized_dataset:
        split = tokenized_dataset["train"].train_test_split(test_size=0.1, seed=42)
        tokenized_dataset["train"] = split["train"]
        tokenized_dataset["validation"] = split["test"]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_MODEL_DIR),
        **TRAINING_CONFIG,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f}s")
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")
    
    # Save model
    print(f"\nSaving model to {OUTPUT_MODEL_DIR}...")
    trainer.save_model(str(OUTPUT_MODEL_DIR))
    tokenizer.save_pretrained(str(OUTPUT_MODEL_DIR))
    
    # Save metrics
    metrics = {
        "base_model": BASE_MODEL,
        "dataset": DATASET_NAME,
        "training_time_seconds": training_time,
        "eval_results": eval_results,
        "training_config": TRAINING_CONFIG,
    }
    
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved metrics to {METRICS_PATH}")
    
    print("\n✓ Fine-tuning complete!")
    print(f"Model saved to: {OUTPUT_MODEL_DIR}")
    print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Macro F1: {eval_results['eval_f1_macro']:.4f}")


if __name__ == "__main__":
    main()

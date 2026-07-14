#!/usr/bin/env python3
"""
FinFact-BD: Extract Original Articles from BENI v2
Extracts the 10K original articles that were used to generate perturbations.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Set

import zstandard as zstd

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_data_dir, get_output_dir

# =============================================================================
# CONFIG
# =============================================================================

config = get_config()
DATA_DIR = get_data_dir()
OUTPUT_DIR = get_output_dir()
BENI_V2_PATH = DATA_DIR / "beni_v2_deduped.csv.zst"

# Output paths
ORIGINALS_PATH = OUTPUT_DIR / "finfact_bd_originals.csv"
PERTURBED_PATH = OUTPUT_DIR / "finfact_bd_perturbed_10k.csv"


def load_original_ids(perturbed_path: Path) -> Set[str]:
    """Load original article IDs from perturbed CSV."""
    original_ids = set()
    with open(perturbed_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_ids.add(row["original_id"])
    print(f"Found {len(original_ids)} unique original article IDs")
    return original_ids


def extract_from_beni(beni_path: Path, original_ids: Set[str], output_path: Path):
    """Extract matching articles from BENI v2 dataset (streaming)."""
    print(f"Extracting from {beni_path}...")
    
    # Open compressed file with streaming
    dctx = zstd.ZstdDecompressor()
    extracted = []
    header = None
    
    with open(beni_path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            # Read line by line
            buffer = ""
            while True:
                chunk = reader.read(8192)  # Read 8KB chunks
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                
                # Process complete lines
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue
                    
                    # Get header
                    if header is None and line.startswith("article_id,"):
                        header = line
                        continue
                    
                    if header is None:
                        continue
                    
                    # Parse article_id (first field)
                    article_id = line.split(",")[0]
                    if article_id in original_ids:
                        extracted.append(line)
    
    print(f"Extracted {len(extracted)} articles from BENI v2")
    
    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for line in extracted:
            f.write(line + "\n")
    
    print(f"Saved to {output_path}")
    return len(extracted)


def main():
    """Main function."""
    # Check if perturbed file exists
    if not PERTURBED_PATH.exists():
        print(f"ERROR: {PERTURBED_PATH} not found.")
        print("Run: zstd -d finfact_bd_perturbed.csv.zst -o finfact_bd_perturbed_10k.csv")
        return
    
    # Load original IDs
    original_ids = load_original_ids(PERTURBED_PATH)
    
    # Extract from BENI v2
    count = extract_from_beni(BENI_V2_PATH, original_ids, ORIGINALS_PATH)
    
    # Create combined CSV
    combined_path = OUTPUT_DIR / "finfact_bd_combined.csv"
    create_combined(original_ids, count, combined_path)
    
    # Update metadata
    update_metadata(count)


def create_combined(original_ids: Set[str], count: int, output_path: Path):
    """Create combined CSV with originals and perturbed."""
    print(f"Creating combined CSV...")
    
    # Read originals
    originals = []
    with open(ORIGINALS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            originals.append(row)
    
    # Read perturbed
    perturbed = []
    with open(PERTURBED_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            perturbed.append(row)
    
    # Get all unique fieldnames
    all_fields = set()
    for row in originals:
        all_fields.update(row.keys())
    for row in perturbed:
        all_fields.update(row.keys())
    
    # Use perturbed fieldnames as base (has all columns)
    fieldnames = list(perturbed[0].keys()) if perturbed else list(originals[0].keys())
    
    # Add any missing fields from originals
    for field in all_fields:
        if field not in fieldnames:
            fieldnames.append(field)
    
    # Combine
    combined = []
    for row in originals:
        combined.append(row)
    for row in perturbed:
        combined.append(row)
    
    # Write
    if combined:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined)
    
    print(f"Combined CSV: {len(combined)} samples saved to {output_path}")


def update_metadata(original_count: int):
    """Update metadata.json with new counts."""
    metadata_path = OUTPUT_DIR / "metadata.json"
    
    # Load existing metadata or create new
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update counts
    metadata["original_count"] = original_count
    metadata["perturbed_count"] = 10000  # We have 10K perturbed
    metadata["total_samples"] = original_count + 10000
    metadata["version"] = "2.0"
    metadata["status"] = "raw_extracted"
    
    # Save
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Updated metadata.json")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    zstd = None  # type: ignore[assignment]

OUTPUT_FIELDS = ["id", "sample_id", "text", "label", "split", "original_id", "rewrite_family", "target_sentence_index", "target_span", "desired_change", "difficulty", "importance_score", "editability_score", "diversity_bonus", "generation_model", "regeneration_attempts", "verification_result", "perturbation_plan", "metadata"]


def _flatten(row: Dict[str, Any]) -> Dict[str, str]:
    flat: Dict[str, str] = {}
    for k, v in row.items():
        if isinstance(v, (dict, list)):
            flat[k] = json.dumps(v, ensure_ascii=False)
        elif v is None:
            flat[k] = ""
        else:
            flat[k] = str(v)
    return flat


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("id\n", encoding="utf-8")
        logger.warning("No rows for %s — wrote empty file", path.name)
        return
    fieldnames: List[str] = []
    seen: set = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fieldnames.append(k)
    for f in reversed(OUTPUT_FIELDS):
        if f in seen:
            fieldnames.remove(f); fieldnames.insert(0, f)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(_flatten(r) for r in rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


def _maybe_compress(path: Path, threshold_mb: int = 50, level: int = 3) -> None:
    if not HAS_ZSTD or path.stat().st_size < threshold_mb * 1024 * 1024:
        return
    compressed = path.with_suffix(path.suffix + ".zst")
    try:
        with open(path, "rb") as src, open(compressed, "wb") as dst:
            zstd.ZstdCompressor(level=level).copy_stream(src, dst)
        path.unlink()
        logger.info("Compressed %s → %s (level %d)", path.name, compressed.name, level)
    except Exception:
        logger.exception("Compression failed for %s", path)


def export_results(raw_outputs: List[Dict[str, Any]], filtered_outputs: List[Dict[str, Any]], output_dir: str, config: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None, dataset_version: str = "1.0") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ec = (config or {}).get("export", {})
    thresh = int(ec.get("compression_threshold_mb", 50))
    level = int(ec.get("compression_level", 3))

    raw_path = out / ec.get("raw_output", "finfact_bd_rewritten_raw.csv")
    _write_csv(raw_path, raw_outputs)
    _maybe_compress(raw_path, thresh, level)

    filt_path = out / ec.get("filtered_output", "finfact_bd_rewritten_filtered.csv")
    _write_csv(filt_path, filtered_outputs)
    _maybe_compress(filt_path, thresh, level)

    fam_dist = dict(Counter(r.get("rewrite_family", "unknown") for r in filtered_outputs))
    diff_dist = dict(Counter(r.get("difficulty", "unknown") for r in filtered_outputs))
    split_dist = dict(Counter(r.get("split", "unknown") for r in filtered_outputs))

    def _safe(v: Any) -> Any:
        if isinstance(v, Path): return str(v)
        if isinstance(v, (str, int, float, bool)): return v
        if isinstance(v, (list, tuple)): return list(v)
        if isinstance(v, dict): return {k: _safe(v) for k, v in v.items()}
        return str(v)

    metadata = {
        "dataset_version": dataset_version, "dataset_name": "FinFact-BD",
        "release_tag": f"FinFact-BD-v{dataset_version}",
        "generation_date": datetime.utcnow().isoformat() + "Z",
        "pipeline_type": "claim_guided_rewriting",
        "total_raw": len(raw_outputs), "total_filtered": len(filtered_outputs),
        "filter_rate": round(1.0 - len(filtered_outputs) / max(len(raw_outputs), 1), 4),
        "family_distribution": fam_dist, "difficulty_distribution": diff_dist,
        "split_distribution": split_dist,
        "stats": stats or {}, "config": _safe(config or {}),
    }
    meta_path = out / ec.get("metadata", "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Metadata written to %s", meta_path)

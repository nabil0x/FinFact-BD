# Kaggle Smoke Test and Full Run Commands

This runbook assumes the repository is available on Kaggle and commands are run
from the repository root.

The active pipeline is:

```text
Article
  -> Claim Extraction
  -> Claim Ranking
  -> Rewrite Planning
  -> Controlled LLM Rewrite
  -> Verification
  -> Regeneration
  -> Export
  -> Human Validation Workbook
```

## Preconditions

- Enable a Kaggle GPU runtime.
- Accept the Hugging Face access terms for `CohereLabs/aya-expanse-8b`.
- Add a Kaggle secret named `HF_TOKEN` with a Hugging Face token that can read
  the gated Aya model.
- Confirm the input file exists at `data/finfact_bd/finfact_bd_originals.csv`.
- Run all commands from the repository root.

## 1. Get the Repository

Use this when starting from a clean Kaggle notebook:

```bash
git clone https://github.com/nabil0x/FinFact-BD.git
cd FinFact-BD
git pull
```

If the repository is already attached to the notebook, run:

```bash
cd /kaggle/working/FinFact-BD
git pull
```

Adjust the path if the attached repository directory has a different name.

## 2. Set Kaggle Environment

In a Kaggle notebook Python cell:

```python
import os
from kaggle_secrets import UserSecretsClient

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_HOME"] = "/kaggle/temp/huggingface"
```

If running from a terminal instead of a notebook cell:

```bash
export HF_TOKEN="<your_huggingface_token>"
export HF_HOME=/kaggle/temp/huggingface
```

Do not commit tokens or paste them into repository files.

## 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the Bangla normalizer git dependency fails transiently, rerun:

```bash
pip install git+https://github.com/csebuetnlp/normalizer.git
```

## 4. Hardware Check

```bash
nvidia-smi
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

Expected result: CUDA is available on a Kaggle GPU runtime.

## 5. Static Integrity Checks

Run these before downloading large models:

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

Validate the production config shape:

```bash
python - <<'PY'
from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("configs/rewrite_pipeline.yaml").read_text(encoding="utf-8"))
required = [
    ("paths", "input_csv"),
    ("paths", "output_dir"),
    ("input", "id_column"),
    ("models", "extractor"),
    ("models", "planner"),
    ("models", "generator"),
    ("models", "embedding"),
    ("models", "nli"),
    ("models", "fluency"),
]

missing = []
for section, key in required:
    if section not in cfg or key not in cfg[section]:
        missing.append(f"{section}.{key}")

if missing:
    raise SystemExit(f"Missing config keys: {missing}")

print("config ok")
print("input:", cfg["paths"]["input_csv"])
print("default output:", cfg["paths"]["output_dir"])
PY
```

## 6. Hugging Face Access Check

This checks model metadata access without downloading full weights:

```bash
python - <<'PY'
import os
from huggingface_hub import hf_hub_download

repos = [
    "Qwen/Qwen3-8B",
    "CohereLabs/aya-expanse-8b",
    "intfloat/multilingual-e5-large",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    "csebuetnlp/banglabert",
]

token = os.environ.get("HF_TOKEN")
for repo in repos:
    path = hf_hub_download(repo_id=repo, filename="config.json", token=token)
    print("OK", repo, "->", path)
PY
```

If `CohereLabs/aya-expanse-8b` fails with `401` or `403`, the run is not ready.
Accept the model terms on Hugging Face and verify `HF_TOKEN`.

## 7. Smoke Test

Run a clean five-article smoke test:

```bash
rm -rf data/generated/rewrite_generation_smoke
mkdir -p logs

python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation_smoke \
  --num-samples 5 \
  --seed 42 \
  --log-level INFO \
  2>&1 | tee logs/rewrite_smoke.log
```

The pipeline writes the checkpoint inside the supplied output directory:

```text
data/generated/rewrite_generation_smoke/checkpoint.json
```

This keeps smoke, pilot, and full runs isolated.

## 8. Inspect Smoke Outputs

```bash
ls -lh data/generated/rewrite_generation_smoke
```

Expected files:

```text
checkpoint.json
finfact_bd_rewritten.csv
finfact_bd_rewritten.jsonl
human_validation.xlsx
metadata.json
```

Inspect run counts and first accepted samples:

```bash
python - <<'PY'
from pathlib import Path
import csv
import json

out = Path("data/generated/rewrite_generation_smoke")
meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))

print("metadata:", meta)
print("processed:", len(ckpt.get("processed_ids", [])))
print("accepted:", len(ckpt.get("samples", [])))
print("failures:", len(ckpt.get("failures", [])))

csv_path = out / "finfact_bd_rewritten.csv"
rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
print("csv rows:", len(rows))
for row in rows[:3]:
    print({
        "sample_id": row["sample_id"],
        "article_id": row["article_id"],
        "family": row["perturbation_family"],
        "attempts": row["regeneration_attempts"],
    })

if ckpt.get("failures"):
    print("first failures:")
    for failure in ckpt["failures"][:5]:
        print(failure)
PY
```

Manually inspect the validation workbook before scaling:

```bash
python - <<'PY'
from openpyxl import load_workbook

path = "data/generated/rewrite_generation_smoke/human_validation.xlsx"
wb = load_workbook(path, read_only=True)
print(wb.sheetnames)
ws = wb["Samples"]
for row in ws.iter_rows(min_row=2, max_row=min(ws.max_row, 4), values_only=True):
    print({
        "sample_id": row[0],
        "headline": row[1],
        "claim_focus": row[2],
        "context_window": row[3],
    })
PY
```

## 9. Smoke Acceptance Gate

Do not start a full run unless all checks pass:

- `compileall` passes.
- `pytest` passes.
- Hugging Face access check passes for all five model repositories.
- Smoke run finishes without Python exceptions or CUDA OOM.
- `checkpoint.json` shows at least one accepted sample.
- `metadata.json` reports the same accepted count as the CSV/JSONL files.
- The workbook is claim-first and readable.
- Manual inspection confirms the Bangla rewrite is fluent enough for the paper.
- Verification failures, if any, are explainable rather than systemic.

If the smoke run accepts zero samples, inspect `logs/rewrite_smoke.log` and
`checkpoint.json` before changing thresholds. Do not silently relax verification.

## 10. Fix and Rerun Loop

After code or config fixes:

```bash
git pull
python -m compileall -q src scripts tests
python -m pytest -q

rm -rf data/generated/rewrite_generation_smoke
python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation_smoke \
  --num-samples 5 \
  --seed 42 \
  --log-level INFO \
  2>&1 | tee logs/rewrite_smoke.log
```

Use a clean smoke output directory after implementation changes. Use resume only
for interrupted runs with the same code/config.

## 11. Pilot Run

After smoke passes, run a medium pilot:

```bash
rm -rf data/generated/rewrite_generation_pilot
mkdir -p logs

python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation_pilot \
  --num-samples 100 \
  --seed 42 \
  --log-level INFO \
  2>&1 | tee logs/rewrite_pilot.log
```

Inspect pilot stats:

```bash
python - <<'PY'
from pathlib import Path
import json

out = Path("data/generated/rewrite_generation_pilot")
meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
print("stats:", meta["stats"])
print("processed:", len(ckpt.get("processed_ids", [])))
print("accepted:", len(ckpt.get("samples", [])))
print("failed:", len(ckpt.get("failures", [])))
PY
```

## 12. Full Run

Run over the complete input file by omitting `--num-samples`:

```bash
rm -rf data/generated/rewrite_generation_full
mkdir -p logs

python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation_full \
  --seed 42 \
  --log-level INFO \
  2>&1 | tee logs/rewrite_full.log
```

## 13. Resume an Interrupted Run

Resume by rerunning the same command with the same output directory:

```bash
python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation_full \
  --seed 42 \
  --log-level INFO \
  2>&1 | tee -a logs/rewrite_full.log
```

The pipeline reads:

```text
data/generated/rewrite_generation_full/checkpoint.json
```

and skips already processed article IDs.

## 14. Final Output Check

```bash
python - <<'PY'
from pathlib import Path
import csv
import json

out = Path("data/generated/rewrite_generation_full")
required = [
    "checkpoint.json",
    "finfact_bd_rewritten.csv",
    "finfact_bd_rewritten.jsonl",
    "metadata.json",
    "human_validation.xlsx",
]

for name in required:
    path = out / name
    print(name, "OK" if path.exists() else "MISSING", path.stat().st_size if path.exists() else 0)

meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
ckpt = json.loads((out / "checkpoint.json").read_text(encoding="utf-8"))
rows = list(csv.DictReader((out / "finfact_bd_rewritten.csv").open(encoding="utf-8")))
jsonl_rows = sum(1 for _ in (out / "finfact_bd_rewritten.jsonl").open(encoding="utf-8"))

print("metadata stats:", meta["stats"])
print("checkpoint accepted:", len(ckpt.get("samples", [])))
print("csv rows:", len(rows))
print("jsonl rows:", jsonl_rows)

assert meta["total_samples"] == len(rows) == jsonl_rows == len(ckpt.get("samples", []))
print("final output counts match")
PY
```

## Common Failures

`401` or `403` for Aya:

- Accept `CohereLabs/aya-expanse-8b` terms on Hugging Face.
- Confirm `HF_TOKEN` is set in the Kaggle environment.

`KeyError: qwen3` or unsupported architecture:

- Ensure `transformers>=4.51.0` is installed from `requirements.txt`.

CUDA OOM:

- Confirm `lazy: true` and `unload_after_call: true` are present for Qwen and
  Aya in `configs/rewrite_pipeline.yaml`.
- Restart the Kaggle kernel to clear fragmented VRAM.
- Rerun the smoke test before scaling again.

Zero accepted samples:

- Inspect verifier reasons in `checkpoint.json`.
- Inspect generated text in failed attempt logs if available.
- Do not weaken verification thresholds until the failure mode is understood.

Bangla quality is poor:

- Stop after smoke or pilot.
- Treat the generator choice as a research risk and validate an alternative
  generator before full-scale generation.

# Kaggle Script Workflow

Use the repository scripts instead of copying long command blocks. All scripts
must be run from inside the repository or can be called by path after cloning.

The scripts never accept Hugging Face tokens as command-line arguments. They read
`HF_TOKEN` from the environment and, on Kaggle, try to load the `HF_TOKEN` Kaggle
Secret automatically.

## Preconditions

- Enable a Kaggle GPU runtime.
- Accept Hugging Face access for `CohereLabs/aya-expanse-8b`.
- Add a Kaggle Secret named `HF_TOKEN`.
- Confirm `data/finfact_bd/finfact_bd_originals.csv` exists.

## Script Map

| Script | Purpose |
|---|---|
| `scripts/kaggle_setup.sh` | Pull latest repo and install requirements |
| `scripts/kaggle_check.sh` | Run compile, unit tests, and config validation |
| `scripts/kaggle_gpu.sh` | Print `nvidia-smi` and Torch CUDA status |
| `scripts/kaggle_preflight_metadata.sh` | Check Hugging Face model authorization only |
| `scripts/kaggle_preflight_download.sh` | Download/cache configured model artifacts |
| `scripts/kaggle_preflight_load.sh` | Load each configured model sequentially on GPU |
| `scripts/kaggle_preflight_all.sh` | Run metadata, download, and load preflight |
| `scripts/kaggle_smoke.sh` | Run clean 5-sample smoke generation |
| `scripts/kaggle_pilot20.sh` | Run clean 20-sample pilot generation |
| `scripts/kaggle_pilot.sh` | Run clean 100-sample pilot generation |
| `scripts/kaggle_stress1k.sh` | Run clean 1k stress generation |
| `scripts/kaggle_metrics.sh` | Summarize throughput, retry, memory, and verifier timing metrics |
| `scripts/kaggle_full.sh` | Run full generation; does not clean by default |
| `scripts/kaggle_resume.sh` | Resume full generation from checkpoint |
| `scripts/kaggle_inspect.sh` | Inspect exported dataset/checkpoint/workbook |
| `scripts/kaggle_all_smoke.sh` | Setup, check, preflight, smoke, and inspect |

All wrappers call the orchestrator:

```bash
python scripts/kaggle_run.py <subcommand>
```

Inside a Kaggle notebook, prefer IPython `%run` if shell wrappers cannot see
Kaggle Secrets:

```python
%run scripts/kaggle_run.py preflight --stage metadata
%run scripts/kaggle_run.py smoke
```

## Recommended Kaggle Sequence

Clone or enter the repo:

```bash
cd /kaggle/working/FinFact-BD
```

Run setup:

```bash
scripts/kaggle_setup.sh
```

Check local integrity:

```bash
scripts/kaggle_check.sh
scripts/kaggle_gpu.sh
```

Run staged model checks:

```bash
scripts/kaggle_preflight_metadata.sh
scripts/kaggle_preflight_download.sh
scripts/kaggle_preflight_load.sh
```

Run smoke generation:

```bash
scripts/kaggle_smoke.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_smoke --fast
```

If smoke passes, run pilot:

```bash
scripts/kaggle_pilot20.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_pilot20 --fast
```

If the 20-sample pilot passes, run the larger pilot:

```bash
scripts/kaggle_pilot.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_pilot --fast
scripts/kaggle_metrics.sh --output-dir data/generated/rewrite_generation_pilot --log logs/rewrite_pilot.log --write
```

If pilot passes, run a 1k stress test:

```bash
scripts/kaggle_stress1k.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_stress1k --fast
scripts/kaggle_metrics.sh --output-dir data/generated/rewrite_generation_stress1k --log logs/rewrite_stress1k.log --write
```

If the 1k stress test passes, run full generation:

```bash
scripts/kaggle_full.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_full --fast
scripts/kaggle_metrics.sh --output-dir data/generated/rewrite_generation_full --log logs/rewrite_full.log --write
```

## Expected Model Lifecycle

The pipeline is staged by model role, not by article:

```text
all pending articles -> Qwen extraction/planning -> release Qwen
all planned articles -> Aya rewrite/regeneration
verification starts -> lazy-load e5/NLI/BanglaBERT -> batched verification
run complete -> release verifier stack -> release Aya
```

For a five-sample smoke or a 20-sample pilot, Qwen, Aya, e5, NLI, and BanglaBERT
should each load once per pipeline run. If logs show any of these models loading
once per article, stop the run and update the repository before scaling.

Verifier batch size is controlled by `verification.batch_size` in
`configs/rewrite_pipeline.yaml`. The default is `8`, which is intentionally
conservative for Kaggle T4 memory. If a verifier batch hits CUDA OOM, the
pipeline clears CUDA cache, splits the batch, and retries without regenerating
the candidates.

Resume interrupted full generation:

```bash
scripts/kaggle_resume.sh
```

One-command smoke path:

```bash
scripts/kaggle_all_smoke.sh
```

## Python Orchestrator

Equivalent direct calls:

```bash
python scripts/kaggle_run.py setup
python scripts/kaggle_run.py check
python scripts/kaggle_run.py gpu
python scripts/kaggle_run.py preflight --stage all
python scripts/kaggle_run.py smoke
python scripts/kaggle_run.py pilot --num-samples 20 --output-dir data/generated/rewrite_generation_pilot20
python scripts/kaggle_run.py pilot
python scripts/kaggle_run.py stress1k
python scripts/kaggle_run.py full
python scripts/kaggle_run.py resume
python scripts/kaggle_run.py inspect --output-dir data/generated/rewrite_generation_smoke
python scripts/kaggle_run.py metrics --output-dir data/generated/rewrite_generation_stress1k --log logs/rewrite_stress1k.log --write
python scripts/kaggle_run.py inspect --output-dir data/generated/rewrite_generation_smoke --fast
```

Useful options:

```bash
python scripts/kaggle_run.py smoke --num-samples 2
python scripts/kaggle_run.py pilot --num-samples 50
python scripts/kaggle_run.py full --clean
python scripts/kaggle_run.py inspect --output-dir data/generated/rewrite_generation_smoke --skip-workbook
python scripts/kaggle_run.py preflight --stage metadata
python scripts/kaggle_run.py preflight --stage download
python scripts/kaggle_run.py preflight --stage load
```

By default, Xet is disabled for preflight and pipeline runs because Kaggle/HF
downloads can otherwise appear stalled. Pass `--enable-xet` to use Xet.

## Logs

Scripts write logs under `logs/`:

| Log | Created by |
|---|---|
| `logs/pip_install.log` | `kaggle_setup.sh` |
| `logs/pytest.log` | `kaggle_check.sh` |
| `logs/model_metadata_preflight.log` | metadata preflight |
| `logs/model_download_preflight.log` | download preflight |
| `logs/model_load_preflight.log` | load preflight |
| `logs/rewrite_smoke.log` | smoke generation |
| `logs/rewrite_pilot.log` | pilot generation |
| `logs/rewrite_full.log` | full/resume generation |

## Outputs

Smoke:

```text
data/generated/rewrite_generation_smoke/
```

20-sample pilot:

```text
data/generated/rewrite_generation_pilot20/
```

Pilot:

```text
data/generated/rewrite_generation_pilot/
```

1k stress:

```text
data/generated/rewrite_generation_stress1k/
```

Full:

```text
data/generated/rewrite_generation_full/
```

Each output directory should contain:

```text
checkpoint.json
planned_articles.jsonl
finfact_bd_rewritten.csv
finfact_bd_rewritten.jsonl
human_validation.xlsx
metadata.json
metrics_summary.json
```

`metadata.json` contains runtime timing under `stats.runtime`, including
planning, generation, and per-verifier timing. `planned_articles.jsonl` allows
resume to skip completed Qwen extraction/planning work.
`metrics_summary.json` contains throughput, retry, memory, verification timing,
OOM recovery, and checkpoint health summaries for stress/full-run decisions.

## Acceptance Gate

Do not start the full run unless:

- `scripts/kaggle_check.sh` passes.
- `scripts/kaggle_preflight_all.sh` passes.
- `scripts/kaggle_smoke.sh` finishes without CUDA OOM or Python exceptions.
- `scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_smoke` shows at least one accepted sample.
- The human validation workbook is claim-first and readable.
- Manual inspection confirms Bangla output quality is acceptable.
- `metrics_summary.json` shows acceptable throughput, rare or zero OOM
  recoveries, and no verifier bottleneck severe enough to dominate generation.

If smoke accepts zero samples, inspect `logs/rewrite_smoke.log` and
`data/generated/rewrite_generation_smoke/checkpoint.json`. Do not weaken
verification thresholds until the failure mode is understood.

## Common Failures

Aya `401` or `403`:

- The `HF_TOKEN` secret is missing or belongs to the wrong Hugging Face account.
- The account has not accepted access to `CohereLabs/aya-expanse-8b`.

Qwen architecture error:

- Install current requirements with `scripts/kaggle_setup.sh`.
- The pipeline requires `transformers>=4.51.0`.

Download appears stuck:

- Use `scripts/kaggle_preflight_download.sh`.
- Check `logs/model_download_preflight.log`.
- Keep Xet disabled unless there is a reason to enable it.

CUDA OOM:

- Restart the Kaggle kernel.
- Run `scripts/kaggle_preflight_load.sh`.
- Confirm Qwen and Aya load sequentially rather than together.
- If only verifier OOM appears, reduce `verification.batch_size`; the pipeline
  will already retry smaller verifier sub-batches automatically.

Interrupted full run:

- Do not delete the output directory.
- Run `scripts/kaggle_resume.sh`.

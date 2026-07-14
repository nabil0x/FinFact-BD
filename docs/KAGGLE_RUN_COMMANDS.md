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
| `scripts/kaggle_pilot.sh` | Run clean 100-sample pilot generation |
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
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_smoke
```

If smoke passes, run pilot:

```bash
scripts/kaggle_pilot.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_pilot
```

If pilot passes, run full generation:

```bash
scripts/kaggle_full.sh
scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_full
```

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
python scripts/kaggle_run.py pilot
python scripts/kaggle_run.py full
python scripts/kaggle_run.py resume
python scripts/kaggle_run.py inspect --output-dir data/generated/rewrite_generation_smoke
```

Useful options:

```bash
python scripts/kaggle_run.py smoke --num-samples 2
python scripts/kaggle_run.py pilot --num-samples 50
python scripts/kaggle_run.py full --clean
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

Pilot:

```text
data/generated/rewrite_generation_pilot/
```

Full:

```text
data/generated/rewrite_generation_full/
```

Each output directory should contain:

```text
checkpoint.json
finfact_bd_rewritten.csv
finfact_bd_rewritten.jsonl
human_validation.xlsx
metadata.json
```

## Acceptance Gate

Do not start the full run unless:

- `scripts/kaggle_check.sh` passes.
- `scripts/kaggle_preflight_all.sh` passes.
- `scripts/kaggle_smoke.sh` finishes without CUDA OOM or Python exceptions.
- `scripts/kaggle_inspect.sh --output-dir data/generated/rewrite_generation_smoke` shows at least one accepted sample.
- The human validation workbook is claim-first and readable.
- Manual inspection confirms Bangla output quality is acceptable.

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

Interrupted full run:

- Do not delete the output directory.
- Run `scripts/kaggle_resume.sh`.

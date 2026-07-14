#!/usr/bin/env bash
set -euo pipefail
python scripts/kaggle_run.py pilot \
  --num-samples 20 \
  --output-dir data/generated/rewrite_generation_pilot20 \
  "$@"

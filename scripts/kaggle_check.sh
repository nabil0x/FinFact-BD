#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/kaggle_run.py check "$@"

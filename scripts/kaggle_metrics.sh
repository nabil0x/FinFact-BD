#!/usr/bin/env bash
set -euo pipefail
python scripts/kaggle_run.py metrics "$@"

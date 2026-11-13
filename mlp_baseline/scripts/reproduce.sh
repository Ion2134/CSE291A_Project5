#!/usr/bin/env bash
set -euo pipefail
BASELINE_DIR="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$BASELINE_DIR"
python -c 'import sys, site; assert sys.prefix != sys.base_prefix and not site.ENABLE_USER_SITE, "Activate a fully isolated venv first"'
for seed in 42 43 44; do
  python -u -m mlp_baseline.cli train-injection \
    --data data/injection --out "runs/seed${seed}" \
    --seed "$seed" --epochs 40 --batch-size 64 --threads 4
done

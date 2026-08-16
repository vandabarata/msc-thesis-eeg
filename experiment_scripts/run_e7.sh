#!/bin/bash
# E7: Subject-Identity Analysis (Privacy Axis)
# - E7a: Linear probe on real data (baseline detector)
# - E7b: Synthetic transfer — do generators memorize subject identity?
# - E7c: Augmented model — does augmentation change subject reliance?
# - Proximity check: cosine distance synthetic-to-real vs real-to-real
#
# Requires GPU for embedding extraction. Lightweight training (linear probe only).
# Expected runtime: ~30-60 minutes (23 folds x extraction + probe training).

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "============================================================"
echo "  E7: Subject-Identity Analysis"
echo "  $(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

python -m training.run_e7 "$@"

echo ""
echo "============================================================"
echo "  E7 complete — $(date)"
echo "  Results: results/e7/"
echo "============================================================"

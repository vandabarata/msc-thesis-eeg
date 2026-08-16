#!/bin/bash
# E6: Cross-Generator Comparison (Statistical Analysis)
# - Wilcoxon signed-rank tests between all method pairs
# - Ratio sensitivity analysis (0.50 vs 1.00 for each generator)
# - Cost-benefit analysis
#
# Runs on existing LOPO results (no training, no GPU needed).
# Expected runtime: ~2 minutes.

set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "============================================================"
echo "  E6: Cross-Generator Comparison"
echo "  $(date)"
echo "============================================================"

python -m training.run_e6 "$@"

echo ""
echo "============================================================"
echo "  E6 complete — $(date)"
echo "  Results: results/e6/"
echo "============================================================"

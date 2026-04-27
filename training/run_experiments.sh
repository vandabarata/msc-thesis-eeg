#!/usr/bin/env bash
# run_experiments.sh — Run Phase 1 experiments (E1 + E2).
#
# Usage (from project root or from training/):
#   bash training/run_experiments.sh quick
#   bash training/run_experiments.sh full
#   bash training/run_experiments.sh e1-single
#   bash training/run_experiments.sh e1-lopo
#   bash training/run_experiments.sh e2-single

set -euo pipefail

# Always run from the project root (one level above training/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

MODE="${1:-quick}"

echo "============================================================"
echo "  Phase 1 Experiments — mode: $MODE"
echo "  Project root: $PROJECT_ROOT"
echo "============================================================"

case "$MODE" in

  quick)
    echo ""
    echo "--- E1: Baseline (single-split, seed 42) ---"
    python -m training.train --experiment e1 --mode single --seeds 42

    echo ""
    echo "--- E2: SMOTE (single-split, seed 42) ---"
    python -m training.train --experiment e2 --augmentation smote --mode single --seeds 42

    echo ""
    echo "--- E2: ADASYN (single-split, seed 42) ---"
    python -m training.train --experiment e2 --augmentation adasyn --mode single --seeds 42
    ;;

  full)
    echo ""
    echo "--- E1: Baseline (LOPO, 3 seeds) ---"
    python -m training.train --experiment e1 --mode lopo --seeds 42 123 456

    echo ""
    echo "--- E2: SMOTE (LOPO, 3 seeds) ---"
    python -m training.train --experiment e2 --augmentation smote --mode lopo --seeds 42 123 456

    echo ""
    echo "--- E2: ADASYN (LOPO, 3 seeds) ---"
    python -m training.train --experiment e2 --augmentation adasyn --mode lopo --seeds 42 123 456
    ;;

  e1-single)
    echo ""
    echo "--- E1: Baseline (single-split, 3 seeds) ---"
    python -m training.train --experiment e1 --mode single --seeds 42 123 456
    ;;

  e1-lopo)
    echo ""
    echo "--- E1: Baseline (LOPO, 3 seeds) ---"
    python -m training.train --experiment e1 --mode lopo --seeds 42 123 456
    ;;

  e2-single)
    echo ""
    echo "--- E2: SMOTE (single-split, 3 seeds) ---"
    python -m training.train --experiment e2 --augmentation smote --mode single --seeds 42 123 456

    echo ""
    echo "--- E2: ADASYN (single-split, 3 seeds) ---"
    python -m training.train --experiment e2 --augmentation adasyn --mode single --seeds 42 123 456
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Usage: bash training/run_experiments.sh {quick|full|e1-single|e1-lopo|e2-single}"
    exit 1
    ;;

esac

echo ""
echo "============================================================"
echo "  Done! Results saved to: results/"
echo "============================================================"

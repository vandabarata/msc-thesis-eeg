#!/usr/bin/env bash
# retrain_e4_cvae.sh — Retrain E4 CVAE checkpoints (no synthetic data or detector).
#
# The original LOPO run deleted cvae.pt after each fold to save disk.
# E5 (LDM) needs these checkpoints. This script retrains only the CVAE
# and saves cvae.pt to each fold directory, then launches E5 LOPO.
#
# Usage:
#   nohup bash experiment_scripts/retrain_e4_cvae.sh &> retrain_cvae.log &

set -uo pipefail
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

source "$PROJECT_ROOT/.venv/bin/activate"

SEEDS=(42 123 456)

echo "============================================"
echo "  Retrain E4 CVAE checkpoints for E5"
echo "  $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================"

for seed in "${SEEDS[@]}"; do
    echo ""
    echo ">>> Seed $seed — $(date)"

    for fold in $(seq 0 22); do
        fold_dir="results/e4/seed_${seed}/fold_$(printf '%02d' $fold)"
        cvae_path="${fold_dir}/cvae.pt"

        if [ -f "$cvae_path" ]; then
            echo "  Fold $fold — cvae.pt exists, skipping"
            continue
        fi

        echo "  Fold $fold — training CVAE..."
        python -m training.generate --model cvae --mode lopo --seed "$seed" --folds "$fold" --checkpoint-only
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "  ERROR: Fold $fold seed $seed failed (exit $rc)"
            exit 1
        fi
    done

    echo ">>> Seed $seed — done ($(date))"
done

echo ""
echo "============================================"
echo "  All E4 CVAE checkpoints retrained"
echo "  Starting E5 LOPO..."
echo "============================================"

# Now run E5 LOPO
exec bash experiment_scripts/run_lopo.sh e5

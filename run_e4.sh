#!/bin/bash
set -e

cd ~/msc_thesis_code
LOG=e4_train.log
mkdir -p logs results

echo "============================================" | tee "$LOG"
echo "  E4 Training — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for SEED in 42 123 456; do
  echo "=== E4 CVAE seed $SEED: generate ===" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 .venv/bin/python -m training.generate \
      --model cvae --mode single --seed $SEED --device cuda \
      --ratio 1.0 \
      >> "$LOG" 2>&1
  echo "=== E4 CVAE seed $SEED: generate done — $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  echo "" | tee -a "$LOG"

  SYNTH=results/e4/seed_${SEED}/single_split/synthetic_ratio_1.00.npz

  echo "=== E4 CVAE seed $SEED: train ===" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 .venv/bin/python -m training.train \
      --experiment e4 --mode single --seeds $SEED --device cuda \
      --synthetic-windows "$SYNTH" \
      >> "$LOG" 2>&1
  echo "=== E4 CVAE seed $SEED: train done — $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
done

echo "============================================" | tee -a "$LOG"
echo "  ALL E4 COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

#!/bin/bash
set -e

cd ~/msc_thesis_code
LOG=e5_train.log
mkdir -p logs results

echo "============================================" | tee "$LOG"
echo "  E5 Training — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for SEED in 42 123 456; do
  CVAE_CKPT=results/e4/seed_${SEED}/single_split/cvae.pt

  echo "=== E5 LDM seed $SEED: generate (cvae from $CVAE_CKPT) ===" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 .venv/bin/python -m training.generate \
      --model ldm --mode single --seed $SEED --device cuda \
      --cvae-checkpoint "$CVAE_CKPT" \
      --ratio 1.0 \
      >> "$LOG" 2>&1
  echo "=== E5 LDM seed $SEED: generate done — $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  echo "" | tee -a "$LOG"

  SYNTH=results/e5/seed_${SEED}/single_split/synthetic_ratio_1.00.npz

  echo "=== E5 LDM seed $SEED: train ===" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 .venv/bin/python -m training.train \
      --experiment e5 --mode single --seeds $SEED --device cuda \
      --synthetic-windows "$SYNTH" \
      >> "$LOG" 2>&1
  echo "=== E5 LDM seed $SEED: train done — $(date '+%H:%M:%S') ===" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
done

echo "============================================" | tee -a "$LOG"
echo "  ALL E5 COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

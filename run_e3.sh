#!/bin/bash
set -e

cd ~/msc_thesis_code
LOG=logs/e3_training.log
mkdir -p logs

echo "============================================" | tee "$LOG"
echo "  E3 Training — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

for SEED in 42 123 456; do
  echo ">>> Seed $SEED: Training TimeGAN generator — $(date '+%H:%M:%S')" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 .venv/bin/python -m training.generate \
      --model timegan --mode single --seed $SEED --device cuda \
      --ratio 1.0 \
      >> "$LOG" 2>&1
  echo ">>> Seed $SEED: TimeGAN generator done — $(date '+%H:%M:%S')" | tee -a "$LOG"
  echo "" | tee -a "$LOG"

  SYNTH=results/e3/seed_${SEED}/single_split/synthetic_ratio_1.00.npz

  echo ">>> Seed $SEED: Training detector with synthetic data — $(date '+%H:%M:%S')" | tee -a "$LOG"
  env PYTHONUNBUFFERED=1 .venv/bin/python -m training.train \
      --experiment e3 --mode single --seeds $SEED --device cuda \
      --synthetic-windows "$SYNTH" \
      >> "$LOG" 2>&1
  echo ">>> Seed $SEED: Detector training done — $(date '+%H:%M:%S')" | tee -a "$LOG"
  echo "" | tee -a "$LOG"
done

echo "============================================" | tee -a "$LOG"
echo "  ALL E3 COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

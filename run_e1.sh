#!/bin/bash
set -e

cd ~/msc_thesis_code
LOG=logs/e1_training.log
mkdir -p logs

echo "============================================" | tee "$LOG"
echo "  E1 Training — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

echo ">>> Starting E1 (baseline, no augmentation) — $(date '+%H:%M:%S')" | tee -a "$LOG"
env PYTHONUNBUFFERED=1 .venv/bin/python -m training.train \
    --experiment e1 --mode single --seeds 42 123 456 \
    >> "$LOG" 2>&1
echo ">>> E1 done — $(date '+%H:%M:%S')" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  E1 COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

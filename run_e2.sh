#!/bin/bash
set -e

cd ~/msc_thesis_code
LOG=logs/e2_training.log
mkdir -p logs

echo "============================================" | tee "$LOG"
echo "  E2 Training — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# E2: SMOTE
echo ">>> Starting E2 SMOTE — $(date '+%H:%M:%S')" | tee -a "$LOG"
env PYTHONUNBUFFERED=1 .venv/bin/python -m training.train \
    --experiment e2 --augmentation smote --mode single --seeds 42 123 456 \
    >> "$LOG" 2>&1
echo ">>> E2 SMOTE done — $(date '+%H:%M:%S')" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# E2: ADASYN
echo ">>> Starting E2 ADASYN — $(date '+%H:%M:%S')" | tee -a "$LOG"
env PYTHONUNBUFFERED=1 .venv/bin/python -m training.train \
    --experiment e2 --augmentation adasyn --mode single --seeds 42 123 456 \
    >> "$LOG" 2>&1
echo ">>> E2 ADASYN done — $(date '+%H:%M:%S')" | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"
echo "  ALL E2 COMPLETE — $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "============================================" | tee -a "$LOG"

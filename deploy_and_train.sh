#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="data/signal_cache"
REMOTE="uni"
REMOTE_DIR="~/msc_thesis_code"
LOG="/tmp/deploy_and_train.log"

exec > >(tee -a "$LOG") 2>&1
echo "=== deploy_and_train.sh started at $(date) ==="

# --- Step 1: Wait for cache build to finish ---
echo "Waiting for build_cache.py to finish..."
while pgrep -f "build_cache.py" > /dev/null 2>&1; do
    count=$(ls "$CACHE_DIR"/*_signals.npy 2>/dev/null | wc -l)
    echo "  $(date +%H:%M:%S) — $count/24 signal files complete"
    sleep 60
done
echo "Cache build process finished at $(date)"

# --- Step 2: Verify all 24 cases ---
count=$(ls "$CACHE_DIR"/*_signals.npy 2>/dev/null | wc -l)
echo "Signal files found: $count"
if [ "$count" -ne 24 ]; then
    echo "ERROR: Expected 24 signal files, found $count. Aborting."
    exit 1
fi

# Check no zero-byte files
bad=0
for f in "$CACHE_DIR"/*_signals.npy; do
    if [ ! -s "$f" ]; then
        echo "ERROR: Zero-byte file: $f"
        bad=1
    fi
done
if [ "$bad" -eq 1 ]; then
    echo "ERROR: Found zero-byte signal files. Aborting."
    exit 1
fi

# Check all index files exist
idx_count=$(ls "$CACHE_DIR"/*_index.npz 2>/dev/null | wc -l)
if [ "$idx_count" -ne 24 ]; then
    echo "ERROR: Expected 24 index files, found $idx_count. Aborting."
    exit 1
fi

echo "All 24 cases verified (signals + indices, no zero-byte files)."
du -sh "$CACHE_DIR"

# --- Step 3: Transfer signal_cache to uni ---
echo "=== Transferring signal_cache to uni at $(date) ==="
ssh "$REMOTE" "mkdir -p $REMOTE_DIR/data/signal_cache"
scp "$CACHE_DIR"/*_signals.npy "$CACHE_DIR"/*_index.npz "$REMOTE:$REMOTE_DIR/data/signal_cache/"
echo "Transfer complete at $(date)"

# Verify remote file count
remote_count=$(ssh "$REMOTE" "ls $REMOTE_DIR/data/signal_cache/*_signals.npy 2>/dev/null | wc -l")
echo "Remote signal files: $remote_count"
if [ "$remote_count" -ne 24 ]; then
    echo "ERROR: Remote has $remote_count signal files, expected 24. Aborting."
    exit 1
fi

# --- Step 4: Launch E1 on uni ---
echo "=== Launching E1 training on uni at $(date) ==="
ssh "$REMOTE" "cd $REMOTE_DIR && mkdir -p logs && nohup .venv/bin/python -m training.train --experiment e1 --mode single --seeds 42 123 456 > logs/e1_training.log 2>&1 &"
sleep 2
running=$(ssh "$REMOTE" "pgrep -f 'training.train.*e1' | head -1")
if [ -n "$running" ]; then
    echo "E1 training launched successfully on uni (PID: $running)"
else
    echo "WARNING: Could not confirm E1 process on uni. Check manually."
fi

echo "=== deploy_and_train.sh finished at $(date) ==="
echo "Monitor with: ssh uni 'tail -f ~/msc_thesis_code/logs/e1_training.log'"

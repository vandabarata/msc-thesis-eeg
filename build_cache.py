"""Build flat-signal caches for all patients, one at a time.

Usage:
    python build_cache.py

Or equivalently:
    python -m data.loader --build-cache

This is a thin wrapper around _build_signal_cache() from data/loader.py.
"""
import gc
import json
from pathlib import Path
from data.loader import SIGNAL_CACHE_DIR, SPLIT_CONFIG_PATH, _build_signal_cache

SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

with open(SPLIT_CONFIG_PATH) as f:
    config = json.load(f)

all_cases: set = set()
for fold in config["lopo_folds"]:
    all_cases.update(fold["train_cases"])
    all_cases.update(fold["test_cases"])

for case_id in sorted(all_cases):
    sig_path = SIGNAL_CACHE_DIR / f"{case_id}_signals.npy"
    if sig_path.exists():
        print(f"  {case_id}: already cached, skipping")
        continue
    _build_signal_cache(case_id)
    gc.collect()

print("Done! All signal caches built.")

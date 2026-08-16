"""
extract_precision_recall.py — Extract precision and recall at optimal threshold
for all LOPO fold results.

Strategy: iterate the test DataLoader ONCE per fold. For each batch, run all
27 model checkpoints (9 configs x 3 seeds) and collect predictions. This
avoids re-reading the mmap 27 times.

Usage:
    python -m training.extract_precision_recall [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.loader import CHBMITDataset
from models.detector import SeizureDetector

RESULTS_DIR = _PROJECT_ROOT / "results"
SEEDS = [42, 123, 456]
N_FOLDS = 23

CONFIGS = [
    ("e1", None, None),
    ("e5", "0.50", "ratio"),
    ("e3", "0.50", "ratio"),
    ("e4", "0.50", "ratio"),
    ("e5", "1.00", "ratio"),
    ("e4", "1.00", "ratio"),
    ("e3", "1.00", "ratio"),
    ("e2", "adasyn", "aug"),
    ("e2", "smote", "aug"),
]


def get_results_path(exp, val, mode, seed, fold):
    if exp == "e1":
        return RESULTS_DIR / exp / f"seed_{seed}" / f"fold_{fold:02d}" / "results.json"
    elif mode == "aug":
        return RESULTS_DIR / exp / f"seed_{seed}" / f"fold_{fold:02d}" / f"results_{val}.json"
    else:
        return RESULTS_DIR / exp / f"seed_{seed}" / f"fold_{fold:02d}" / f"results_ratio_{val}.json"


def get_model_path(exp, val, mode, seed, fold):
    base = RESULTS_DIR / exp / f"seed_{seed}" / f"fold_{fold:02d}"
    if exp == "e1":
        return base / "best_model.pt"
    elif mode == "aug":
        return base / f"best_model_{val}.pt"
    else:
        return base / f"best_model_ratio_{val}.pt"


def label_for(exp, val):
    return f"{exp} {val or 'baseline'}"


def compute_precision_recall(y_true, y_prob):
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.argmax(f1_scores)
    return float(precision[best_idx]), float(recall[best_idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("=" * 70, flush=True)
    print("  Extracting Precision & Recall — single-pass per fold", flush=True)
    print("=" * 70, flush=True)

    all_results = {}

    for fold in range(N_FOLDS):
        # Load all models for this fold up front
        models_info = []
        for exp, val, mode in CONFIGS:
            for seed in SEEDS:
                model_path = get_model_path(exp, val, mode, seed, fold)
                if not model_path.exists():
                    continue
                model = SeizureDetector()
                ckpt = torch.load(model_path, map_location=device, weights_only=False)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                else:
                    model.load_state_dict(ckpt)
                model.to(device).eval()
                models_info.append({
                    "model": model,
                    "exp": exp, "val": val, "mode": mode, "seed": seed,
                    "probs": [],
                })

        if not models_info:
            print(f"  Fold {fold}: no models found, skipping", flush=True)
            continue

        print(f"  Fold {fold}: {len(models_info)} models loaded, running inference...",
              end="", flush=True)

        # Create test loader and iterate ONCE
        test_ds = CHBMITDataset(split="test", fold=fold, normalize=True, seed=42)
        test_loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=0)

        all_labels = []
        n_batches = 0
        with torch.no_grad():
            for x, y, _pid in test_loader:
                x = x.to(device)
                all_labels.append(y.numpy() if isinstance(y, torch.Tensor) else np.asarray(y))
                for info in models_info:
                    logits = info["model"](x)
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    info["probs"].append(probs)
                n_batches += 1

        y_true = np.concatenate(all_labels)
        print(f" {len(y_true)} samples, {n_batches} batches", flush=True)

        # Compute precision/recall and save
        for info in models_info:
            y_prob = np.concatenate(info["probs"])
            prec, rec = compute_precision_recall(y_true, y_prob)
            label = label_for(info["exp"], info["val"])

            if label not in all_results:
                all_results[label] = {"precision": [], "recall": []}
            all_results[label]["precision"].append(prec)
            all_results[label]["recall"].append(rec)

            # Update results JSON
            results_path = get_results_path(
                info["exp"], info["val"], info["mode"], info["seed"], fold)
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                tm = data.get("test_metrics", data)
                tm["precision_optimal"] = prec
                tm["recall_optimal"] = rec
                with open(results_path, "w") as f:
                    json.dump(data, f, indent=2)

        # Free memory
        del models_info, test_ds, test_loader
        print(f"    -> done", flush=True)

    # Summary
    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)

    summary = {}
    for label in [label_for(e, v) for e, v, _ in CONFIGS]:
        if label not in all_results:
            continue
        vals = all_results[label]
        prec_arr = np.array([p for p in vals["precision"] if not np.isnan(p)])
        rec_arr = np.array([r for r in vals["recall"] if not np.isnan(r)])
        summary[label] = {
            "precision_mean": float(np.mean(prec_arr)),
            "precision_fold_std": float(np.std(prec_arr)),
            "recall_mean": float(np.mean(rec_arr)),
            "recall_fold_std": float(np.std(rec_arr)),
            "n": len(prec_arr),
        }
        print(f"  {label:<18s}: prec={np.mean(prec_arr):.4f}+/-{np.std(prec_arr):.4f}  "
              f"rec={np.mean(rec_arr):.4f}+/-{np.std(rec_arr):.4f}", flush=True)

    output_path = RESULTS_DIR / "precision_recall_summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()

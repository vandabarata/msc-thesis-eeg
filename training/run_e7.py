"""
run_e7.py — E7: Subject-Identity Analysis (Privacy Axis).

Tests whether synthetic augmentation preserves, reduces, or amplifies
subject-specific patterns using a linear probe on frozen detector embeddings.

Sub-experiments:
  E7a: Real baseline — how identifiable are subjects from real EEG?
  E7b: Synthetic transfer — do synthetic windows inherit subject identity?
       (requires regenerating synthetic data; skipped if .npz files absent)
  E7c: Augmented model — does augmentation change detector's reliance on
       subject signatures? Compares E1 baseline vs E5 LDM (best augmentation).

Also computes proximity check when synthetic data is available.

Usage:
    python -m training.run_e7 [--folds 0,1,2,...] [--seeds 42,123,456]
                              [--regenerate] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.loader import CHBMITDataset, get_lopo_dataloaders
from models.detector import SeizureDetector
from training.subject_identity import (
    extract_embeddings,
    train_probe,
    compute_proximity,
    SubjectIdentityProbe,
)

RESULTS_DIR = _PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "e7"
N_FOLDS = 23
SEEDS = [42, 123, 456]


def load_detector(checkpoint_path: Path, device: str = "cpu") -> SeizureDetector:
    """Load a detector from a checkpoint."""
    model = SeizureDetector()
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def run_e7a_e7c_fold(
    fold: int,
    seed: int,
    device: str = "cpu",
) -> Dict:
    """Run E7a (baseline) and E7c (augmented) for a single fold/seed."""
    results = {"fold": fold, "seed": seed}

    # Load data (training set for embedding extraction)
    train_loader, _, _ = get_lopo_dataloaders(fold=fold, batch_size=256, seed=seed)
    n_subjects = len(set(int(pid) for _, _, pid in train_loader.dataset))

    # ── E7a: Baseline detector embeddings ─────────────────────────────
    baseline_path = RESULTS_DIR / "e1" / f"seed_{seed}" / f"fold_{fold:02d}" / "best_model.pt"
    if not baseline_path.exists():
        print(f"    [SKIP] No E1 checkpoint: {baseline_path}")
        return results

    detector_baseline = load_detector(baseline_path, device)
    emb_real, pids_real = extract_embeddings(detector_baseline, train_loader, device)
    n_subjects = len(np.unique(pids_real))

    _, e7a = train_probe(emb_real, pids_real, n_subjects, seed=seed, device=device)
    results["e7a"] = e7a

    # ── E7c: LDM-augmented detector embeddings ────────────────────────
    aug_path = RESULTS_DIR / "e5" / f"seed_{seed}" / f"fold_{fold:02d}" / "best_model_ratio_0.50.pt"
    if aug_path.exists():
        detector_aug = load_detector(aug_path, device)
        emb_aug, pids_aug = extract_embeddings(detector_aug, train_loader, device)
        _, e7c = train_probe(emb_aug, pids_aug, n_subjects, seed=seed, device=device)
        results["e7c"] = e7c
        results["e7c_vs_e7a_diff"] = e7c["val_accuracy"] - e7a["val_accuracy"]
    else:
        print(f"    [SKIP] No E5 checkpoint: {aug_path}")

    return results


def run_e7b_fold(
    fold: int,
    seed: int,
    generator: str = "ldm",
    device: str = "cpu",
) -> Optional[Dict]:
    """
    Run E7b (synthetic transfer) for a single fold/seed.
    Requires synthetic .npz files or --regenerate flag.
    """
    # Check for existing synthetic data
    exp_map = {"timegan": "e3", "cvae": "e4", "ldm": "e5"}
    exp = exp_map[generator]
    synth_path = RESULTS_DIR / exp / f"seed_{seed}" / f"fold_{fold:02d}" / "synthetic_ratio_0.50.npz"

    if not synth_path.exists():
        return None

    # Load synthetic data
    synth_data = np.load(str(synth_path))
    synthetic_windows = synth_data["windows"]
    synthetic_pids = synth_data["patient_ids"]

    # Load baseline detector for embedding extraction
    baseline_path = RESULTS_DIR / "e1" / f"seed_{seed}" / f"fold_{fold:02d}" / "best_model.pt"
    if not baseline_path.exists():
        return None

    detector = load_detector(baseline_path, device)

    # Get real embeddings for training the probe
    train_loader, _, _ = get_lopo_dataloaders(fold=fold, batch_size=256, seed=seed)
    emb_real, pids_real = extract_embeddings(detector, train_loader, device)
    n_subjects = len(np.unique(pids_real))

    # Train probe on real data
    probe, _ = train_probe(emb_real, pids_real, n_subjects, seed=seed, device=device, val_fraction=0.01)

    # Get synthetic embeddings
    synth_tensor = torch.from_numpy(synthetic_windows).float()
    from torch.utils.data import TensorDataset
    synth_ds = TensorDataset(
        synth_tensor,
        torch.ones(len(synth_tensor), dtype=torch.long),
        torch.from_numpy(synthetic_pids).long(),
    )
    synth_loader = DataLoader(synth_ds, batch_size=256, shuffle=False)
    emb_synth, pids_synth = extract_embeddings(detector, synth_loader, device)

    # Evaluate probe on synthetic embeddings
    unique_pids = np.unique(pids_real)
    pid_map = {int(pid): i for i, pid in enumerate(unique_pids)}
    known_mask = np.array([int(p) in pid_map for p in synthetic_pids])

    if known_mask.sum() == 0:
        return {"fold": fold, "seed": seed, "accuracy": float("nan"), "n_known": 0}

    probe.eval()
    with torch.no_grad():
        known_emb = torch.from_numpy(emb_synth[known_mask]).float().to(device)
        known_pids_mapped = torch.tensor(
            [pid_map[int(p)] for p in synthetic_pids[known_mask]],
            dtype=torch.long, device=device,
        )
        logits = probe(known_emb)
        preds = logits.argmax(dim=1)
        acc = (preds == known_pids_mapped).float().mean().item()

    # Proximity check
    proximity = compute_proximity(emb_real, emb_synth[known_mask])

    return {
        "fold": fold,
        "seed": seed,
        "accuracy": float(acc),
        "n_known": int(known_mask.sum()),
        "proximity": proximity,
    }


def main():
    parser = argparse.ArgumentParser(description="E7: Subject-Identity Analysis")
    parser.add_argument("--folds", type=str, default=None,
                        help="Comma-separated fold indices (default: all 23)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated seeds (default: 42,123,456)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--regenerate", action="store_true",
                        help="Regenerate synthetic data for E7b (slow)")
    args = parser.parse_args()

    folds = list(range(N_FOLDS)) if args.folds is None else [int(f) for f in args.folds.split(",")]
    seeds = SEEDS if args.seeds is None else [int(s) for s in args.seeds.split(",")]
    device = args.device

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Device: {device}")
    print(f"  Folds: {len(folds)} ({folds[0]}..{folds[-1]})")
    print(f"  Seeds: {seeds}")

    # ── E7a + E7c: Real baseline vs augmented model ───────────────────
    print("\n" + "=" * 60)
    print("  E7a: Subject-ID probe on real data (baseline detector)")
    print("  E7c: Subject-ID probe on real data (LDM-augmented detector)")
    print("=" * 60)

    all_results = []
    t0 = time.time()

    for fold in folds:
        for seed in seeds:
            print(f"\n  Fold {fold:02d}, Seed {seed}:")
            result = run_e7a_e7c_fold(fold, seed, device)
            all_results.append(result)

            if "e7a" in result:
                print(f"    E7a accuracy: {result['e7a']['val_accuracy']:.4f} "
                      f"(chance: {result['e7a']['chance_level']:.4f})")
            if "e7c" in result:
                diff = result["e7c_vs_e7a_diff"]
                direction = "LOWER (less subject-dependent)" if diff < 0 else "HIGHER (more subject-dependent)"
                print(f"    E7c accuracy: {result['e7c']['val_accuracy']:.4f} "
                      f"(diff: {diff:+.4f} = {direction})")

    elapsed = time.time() - t0
    print(f"\n  E7a+E7c completed in {elapsed / 60:.1f} min")

    # ── Aggregate results ─────────────────────────────────────────────
    e7a_accs = [r["e7a"]["val_accuracy"] for r in all_results if "e7a" in r]
    e7c_accs = [r["e7c"]["val_accuracy"] for r in all_results if "e7c" in r]
    e7a_chance = [r["e7a"]["chance_level"] for r in all_results if "e7a" in r]
    diffs = [r["e7c_vs_e7a_diff"] for r in all_results if "e7c_vs_e7a_diff" in r]

    print("\n" + "=" * 60)
    print("  E7 Summary")
    print("=" * 60)

    if e7a_accs:
        print(f"\n  E7a (baseline detector, real data):")
        print(f"    Subject-ID accuracy: {np.mean(e7a_accs):.4f} +/- {np.std(e7a_accs):.4f}")
        print(f"    Chance level:        {np.mean(e7a_chance):.4f}")
        print(f"    Above-chance ratio:  {np.mean(e7a_accs) / np.mean(e7a_chance):.1f}x")

    if e7c_accs:
        print(f"\n  E7c (LDM-augmented detector, real data):")
        print(f"    Subject-ID accuracy: {np.mean(e7c_accs):.4f} +/- {np.std(e7c_accs):.4f}")
        print(f"    Diff vs E7a:         {np.mean(diffs):+.4f} +/- {np.std(diffs):.4f}")
        if np.mean(diffs) < 0:
            print(f"    Interpretation:      Augmentation REDUCES subject reliance (better generalization)")
        else:
            print(f"    Interpretation:      Augmentation AMPLIFIES subject patterns (worse generalization)")

    # ── E7b: Synthetic transfer (if data available) ───────────────────
    e7b_results = []
    has_synthetic = any(
        (RESULTS_DIR / "e5" / f"seed_{s}" / f"fold_{f:02d}" / "synthetic_ratio_0.50.npz").exists()
        for f in folds for s in seeds
    )

    if has_synthetic:
        print("\n" + "=" * 60)
        print("  E7b: Synthetic transfer (probe trained on real, tested on synthetic)")
        print("=" * 60)

        for fold in folds:
            for seed in seeds:
                result = run_e7b_fold(fold, seed, "ldm", device)
                if result:
                    e7b_results.append(result)
                    print(f"    Fold {fold:02d} Seed {seed}: acc={result['accuracy']:.4f}, "
                          f"proximity={result['proximity']['proximity_ratio']:.3f}")

        if e7b_results:
            e7b_accs = [r["accuracy"] for r in e7b_results if not np.isnan(r["accuracy"])]
            prox_ratios = [r["proximity"]["proximity_ratio"] for r in e7b_results if "proximity" in r]
            print(f"\n  E7b summary:")
            print(f"    Synthetic subject-ID: {np.mean(e7b_accs):.4f} +/- {np.std(e7b_accs):.4f}")
            print(f"    Proximity ratio:      {np.mean(prox_ratios):.4f} +/- {np.std(prox_ratios):.4f}")
            print(f"    (<1.0 = closer than real = possible memorisation)")
    else:
        print("\n  E7b: SKIPPED (synthetic .npz files not available)")
        print("    To run E7b, regenerate with: python -m training.run_e7 --regenerate")

    # ── Save results ──────────────────────────────────────────────────
    output = {
        "e7a_e7c_results": all_results,
        "e7b_results": e7b_results,
        "summary": {
            "e7a_mean_accuracy": float(np.mean(e7a_accs)) if e7a_accs else None,
            "e7a_std_accuracy": float(np.std(e7a_accs)) if e7a_accs else None,
            "e7a_chance_level": float(np.mean(e7a_chance)) if e7a_chance else None,
            "e7c_mean_accuracy": float(np.mean(e7c_accs)) if e7c_accs else None,
            "e7c_std_accuracy": float(np.std(e7c_accs)) if e7c_accs else None,
            "e7c_vs_e7a_mean_diff": float(np.mean(diffs)) if diffs else None,
            "e7c_vs_e7a_std_diff": float(np.std(diffs)) if diffs else None,
            "interpretation_e7c": (
                "augmentation_reduces_subject_reliance" if diffs and np.mean(diffs) < 0
                else "augmentation_amplifies_subject_patterns" if diffs
                else None
            ),
            "n_folds": len(folds),
            "n_seeds": len(seeds),
            "device": device,
        },
    }

    output_path = OUTPUT_DIR / "e7_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()

"""
run_per_patient.py — Per-patient breakdown of augmentation effects.

Answers: which patients benefit from augmentation? Is there a pattern?
Correlates per-fold AUPRC improvement with patient characteristics
(seizure count, recording duration, baseline difficulty).

Usage:
    python -m training.run_per_patient
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "e6"
SPLIT_CONFIG = _PROJECT_ROOT / "data" / "split_config.json"
SEEDS = [42, 123, 456]
N_FOLDS = 23


def load_split_config():
    with open(SPLIT_CONFIG) as f:
        return json.load(f)


def load_fold_results(experiment: str, ratio: str = None, seed: int = None) -> dict:
    """Load per-fold results for a given experiment/ratio/seed."""
    results = {}
    seeds = [seed] if seed else SEEDS
    for s in seeds:
        for fold in range(N_FOLDS):
            if experiment == "e1":
                path = RESULTS_DIR / experiment / f"seed_{s}" / f"fold_{fold:02d}" / "results.json"
            elif experiment == "e2":
                path = RESULTS_DIR / experiment / f"seed_{s}" / f"fold_{fold:02d}" / f"results_{ratio}.json"
            else:
                path = RESULTS_DIR / experiment / f"seed_{s}" / f"fold_{fold:02d}" / f"results_ratio_{ratio}.json"

            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                tm = data.get("test_metrics", data)
                results.setdefault(fold, []).append(tm.get("auprc", float("nan")))

    return {fold: float(np.mean(vals)) for fold, vals in results.items()}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_split_config()

    print("\n" + "=" * 70)
    print("  Per-Patient Augmentation Analysis")
    print("=" * 70)

    # Load baseline and best augmentation (LDM 0.50) per fold
    baseline = load_fold_results("e1")
    ldm_050 = load_fold_results("e5", "0.50")
    cvae_050 = load_fold_results("e4", "0.50")
    timegan_050 = load_fold_results("e3", "0.50")

    # Get test patient for each fold
    fold_patients = {}
    for fold_cfg in config["lopo_folds"]:
        fold_idx = fold_cfg["fold"]
        fold_patients[fold_idx] = fold_cfg["test_cases"]

    # Per-fold comparison
    print("\n  Per-fold AUPRC: Baseline vs LDM(0.50) vs CVAE(0.50) vs TimeGAN(0.50)")
    print(f"  {'Fold':>4s}  {'Test patient':<14s}  {'Baseline':>8s}  {'LDM':>8s}  {'CVAE':>8s}  {'TimeGAN':>8s}  {'LDM diff':>9s}  {'Best aug':>9s}")
    print(f"  {'─' * 80}")

    improvements = []
    patient_data = []

    for fold in range(N_FOLDS):
        bl = baseline.get(fold, float("nan"))
        ldm = ldm_050.get(fold, float("nan"))
        cvae = cvae_050.get(fold, float("nan"))
        tgan = timegan_050.get(fold, float("nan"))
        patients = fold_patients.get(fold, ["?"])
        patient_str = "+".join(patients)

        ldm_diff = ldm - bl if not (np.isnan(ldm) or np.isnan(bl)) else float("nan")
        best_aug = max(ldm, cvae, tgan) if not any(np.isnan(x) for x in [ldm, cvae, tgan]) else float("nan")
        best_diff = best_aug - bl if not np.isnan(best_aug) else float("nan")

        best_name = "LDM" if best_aug == ldm else "CVAE" if best_aug == cvae else "TimeGAN"
        marker = "+" if ldm_diff > 0 else " "

        print(f"  {fold:>4d}  {patient_str:<14s}  {bl:>8.4f}  {ldm:>8.4f}  {cvae:>8.4f}  {tgan:>8.4f}  {ldm_diff:>+9.4f}{marker} {best_name:>9s}")

        improvements.append({
            "fold": fold,
            "test_patients": patients,
            "baseline_auprc": bl,
            "ldm_050_auprc": ldm,
            "cvae_050_auprc": cvae,
            "timegan_050_auprc": tgan,
            "ldm_diff": ldm_diff,
            "best_augmentation": best_name,
            "best_aug_diff": best_diff,
        })

        patient_data.append({
            "fold": fold,
            "patients": patients,
            "baseline": bl,
            "ldm_diff": ldm_diff,
        })

    # Analysis
    valid_diffs = [d["ldm_diff"] for d in improvements if not np.isnan(d["ldm_diff"])]
    positive = [d for d in improvements if not np.isnan(d["ldm_diff"]) and d["ldm_diff"] > 0]
    negative = [d for d in improvements if not np.isnan(d["ldm_diff"]) and d["ldm_diff"] < 0]

    print(f"\n  {'─' * 70}")
    print(f"\n  Summary:")
    print(f"    Folds where LDM improves over baseline: {len(positive)}/{len(valid_diffs)}")
    print(f"    Folds where LDM hurts:                  {len(negative)}/{len(valid_diffs)}")
    print(f"    Mean improvement when positive:         {np.mean([d['ldm_diff'] for d in positive]):+.4f}" if positive else "")
    print(f"    Mean degradation when negative:         {np.mean([d['ldm_diff'] for d in negative]):+.4f}" if negative else "")

    # Correlate with baseline difficulty
    print(f"\n  Correlation: baseline AUPRC vs LDM improvement")
    baselines = np.array([d["baseline_auprc"] for d in improvements if not np.isnan(d["ldm_diff"])])
    diffs = np.array([d["ldm_diff"] for d in improvements if not np.isnan(d["ldm_diff"])])

    if len(baselines) > 5:
        corr = np.corrcoef(baselines, diffs)[0, 1]
        print(f"    Pearson r = {corr:.4f}")
        if corr < -0.3:
            print(f"    Interpretation: augmentation tends to help EASY patients more (or hurt hard ones)")
        elif corr > 0.3:
            print(f"    Interpretation: augmentation tends to help HARD patients more")
        else:
            print(f"    Interpretation: no clear relationship between difficulty and augmentation benefit")

    # Quartile analysis
    print(f"\n  By baseline difficulty quartile:")
    sorted_idx = np.argsort(baselines)
    q_size = len(sorted_idx) // 4

    for q, label in enumerate(["Hardest (Q1)", "Q2", "Q3", "Easiest (Q4)"]):
        start = q * q_size
        end = start + q_size if q < 3 else len(sorted_idx)
        q_idx = sorted_idx[start:end]
        q_bl = baselines[q_idx]
        q_diff = diffs[q_idx]
        pos_count = (q_diff > 0).sum()
        print(f"    {label:<14s}: baseline={np.mean(q_bl):.3f}, mean LDM diff={np.mean(q_diff):+.4f}, "
              f"improved={pos_count}/{len(q_idx)}")

    # Best case for each generator
    print(f"\n  Folds where EACH generator is the best augmentation:")
    best_counts = {"LDM": 0, "CVAE": 0, "TimeGAN": 0}
    for d in improvements:
        if d["best_augmentation"] in best_counts:
            best_counts[d["best_augmentation"]] += 1
    for gen, count in sorted(best_counts.items(), key=lambda x: -x[1]):
        print(f"    {gen:<10s}: {count}/{N_FOLDS} folds")

    # Save
    output = {
        "per_fold": improvements,
        "summary": {
            "n_folds_ldm_improves": len(positive),
            "n_folds_ldm_hurts": len(negative),
            "n_folds_total": len(valid_diffs),
            "mean_ldm_diff": float(np.mean(valid_diffs)),
            "correlation_baseline_vs_diff": float(corr) if len(baselines) > 5 else None,
            "best_generator_counts": best_counts,
        },
    }

    output_path = OUTPUT_DIR / "per_patient_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()

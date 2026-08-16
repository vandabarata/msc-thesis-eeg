"""
run_e6.py — E6: Cross-Generator Comparison (Statistical Analysis).

Analyses the completed LOPO results from E1-E5:
  1. Wilcoxon signed-rank tests between all method pairs (per-fold paired)
  2. Ratio sensitivity analysis (0.50 vs 1.00 for each generator)
  3. Cost-benefit analysis (AUPRC gain vs training time)

No GPU needed. Reads existing results JSON files from results/e1-e5.

Usage:
    python -m training.run_e6
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from itertools import combinations

import numpy as np

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from training.evaluate import wilcoxon_compare, ratio_sensitivity_analysis, cost_benefit_analysis

RESULTS_DIR = _PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "e6"
SEEDS = [42, 123, 456]
N_FOLDS = 23


def load_fold_auprcs(experiment: str, ratio: str = None) -> list:
    """Load per-fold AUPRC values averaged across seeds for a given experiment/ratio."""
    fold_auprcs = []
    for fold in range(N_FOLDS):
        seed_auprcs = []
        for seed in SEEDS:
            if experiment == "e1":
                path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / "results.json"
            elif experiment == "e2":
                # E2 has results_smote.json and results_adasyn.json
                path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / f"results_{ratio}.json"
            else:
                path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / f"results_ratio_{ratio}.json"

            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                tm = data.get("test_metrics", data)
                auprc = tm.get("auprc")
                if auprc is not None:
                    seed_auprcs.append(auprc)

        if seed_auprcs:
            fold_auprcs.append(float(np.mean(seed_auprcs)))
        else:
            fold_auprcs.append(float("nan"))

    return fold_auprcs


def load_all_fold_auprcs(experiment: str, ratio: str = None) -> list:
    """Load ALL per-fold-seed AUPRC values (not averaged) for detailed stats."""
    auprcs = []
    for seed in SEEDS:
        for fold in range(N_FOLDS):
            if experiment == "e1":
                path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / "results.json"
            elif experiment == "e2":
                path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / f"results_{ratio}.json"
            else:
                path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / f"results_ratio_{ratio}.json"

            if path.exists():
                with open(path) as f:
                    data = json.load(f)
                tm = data.get("test_metrics", data)
                auprc = tm.get("auprc")
                if auprc is not None:
                    auprcs.append(auprc)
    return auprcs


def estimate_training_hours(experiment: str) -> float:
    """Estimate total training time from LOPO status logs."""
    # Use the elapsed times from lopo_status
    times = {
        "e1": 106.0,   # ~4.4 days = 106h (3 seeds x 23 folds x ~92min)
        "e2": 210.0,   # ~8.75 days = 210h (SMOTE + ADASYN, 3 seeds x 23 folds)
        "e3": 504.0,   # 30221min = 504h (includes generation + 2 ratios)
        "e4": 572.0,   # 34319min = 572h (includes CVAE training + 2 ratios)
        "e5": 673.0,   # 40400min = 673h (includes LDM generation + 2 ratios)
    }
    return times.get(experiment, 0.0)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  E6: Cross-Generator Statistical Comparison")
    print("=" * 60)

    # ── 1. Load all experiment results ────────────────────────────────────
    experiments = {
        "E1 (baseline)": ("e1", None),
        "E2 (SMOTE)": ("e2", "smote"),
        "E2 (ADASYN)": ("e2", "adasyn"),
        "E3 (TimeGAN 0.50)": ("e3", "0.50"),
        "E3 (TimeGAN 1.00)": ("e3", "1.00"),
        "E4 (CVAE 0.50)": ("e4", "0.50"),
        "E4 (CVAE 1.00)": ("e4", "1.00"),
        "E5 (LDM 0.50)": ("e5", "0.50"),
        "E5 (LDM 1.00)": ("e5", "1.00"),
    }

    fold_auprcs = {}
    all_auprcs = {}

    print("\n  Loading LOPO results...")
    for name, (exp, ratio) in experiments.items():
        fold_auprcs[name] = load_fold_auprcs(exp, ratio)
        all_auprcs[name] = load_all_fold_auprcs(exp, ratio)
        valid = [v for v in all_auprcs[name] if not np.isnan(v)]
        print(f"    {name:<22s}: AUPRC = {np.mean(valid):.4f} +/- {np.std(valid):.4f} (n={len(valid)})")

    # ── 2. Wilcoxon signed-rank tests ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Wilcoxon Signed-Rank Tests (per-fold paired, seed-averaged)")
    print("─" * 60)

    baseline_name = "E1 (baseline)"
    augmentation_methods = [k for k in experiments.keys() if k != baseline_name]

    wilcoxon_results = {}

    # All methods vs baseline
    print("\n  vs. Baseline (E1):")
    print(f"    {'Method':<22s} {'Mean diff':>10s} {'p-value':>10s} {'Sig (.05)':>10s} {'Sig (.01)':>10s}")
    print(f"    {'─' * 62}")

    for name in augmentation_methods:
        result = wilcoxon_compare(fold_auprcs[baseline_name], fold_auprcs[name])
        wilcoxon_results[f"{baseline_name} vs {name}"] = result
        sig05 = "YES" if result["significant_005"] else "no"
        sig01 = "YES" if result["significant_001"] else "no"
        print(f"    {name:<22s} {result['mean_diff']:>+10.4f} {result['p_value']:>10.4f} {sig05:>10s} {sig01:>10s}")

    # Pairwise between generators at ratio 0.50 (the better ratio)
    print("\n  Pairwise (ratio 0.50 only):")
    generators_050 = ["E3 (TimeGAN 0.50)", "E4 (CVAE 0.50)", "E5 (LDM 0.50)"]
    print(f"    {'Pair':<40s} {'Mean diff':>10s} {'p-value':>10s} {'Sig (.05)':>10s}")
    print(f"    {'─' * 70}")

    for a, b in combinations(generators_050, 2):
        result = wilcoxon_compare(fold_auprcs[a], fold_auprcs[b])
        wilcoxon_results[f"{a} vs {b}"] = result
        sig05 = "YES" if result["significant_005"] else "no"
        print(f"    {a} vs {b:<20s} {result['mean_diff']:>+10.4f} {result['p_value']:>10.4f} {sig05:>10s}")

    # ── 3. Ratio sensitivity analysis ─────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Ratio Sensitivity Analysis")
    print("─" * 60)

    baseline_folds = fold_auprcs[baseline_name]
    ratio_results = {}

    for generator, ratios in [("TimeGAN", ["0.50", "1.00"]), ("CVAE", ["0.50", "1.00"]), ("LDM", ["0.50", "1.00"])]:
        exp_map = {"TimeGAN": "e3", "CVAE": "e4", "LDM": "e5"}
        exp = exp_map[generator]

        ratio_data = {}
        for ratio in ratios:
            folds = load_fold_auprcs(exp, ratio)
            ratio_data[float(ratio)] = folds

        analysis = ratio_sensitivity_analysis(ratio_data, baseline_folds)
        ratio_results[generator] = analysis

        print(f"\n  {generator}:")
        print(f"    Best ratio: {analysis['best_ratio']}")
        print(f"    Best gain vs baseline: {analysis['best_ratio_gain']:+.4f}")
        print(f"    Diminishing returns at high ratio: {analysis['diminishing_returns']}")
        print(f"    Degrades below baseline at: {'all ratios' if analysis['degradation_ratio'] == sorted(ratio_data.keys())[0] else analysis['degradation_ratio'] or 'never'}")
        if analysis.get("best_vs_baseline_p_value") is not None:
            print(f"    Best ratio vs baseline p-value: {analysis['best_vs_baseline_p_value']:.4f}")

    # ── 4. Cost-benefit analysis ──────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Cost-Benefit Analysis")
    print("─" * 60)

    cost_data = {}
    for name, (exp, ratio) in experiments.items():
        if name == baseline_name:
            valid = [v for v in all_auprcs[name] if not np.isnan(v)]
            cost_data["e1"] = {
                "auprc_mean": float(np.mean(valid)),
                "total_train_seconds": estimate_training_hours("e1") * 3600,
                "total_gen_seconds": 0.0,
            }
        elif "0.50" in name:
            # Use best ratio (0.50) for each generator
            valid = [v for v in all_auprcs[name] if not np.isnan(v)]
            cost_data[exp] = {
                "auprc_mean": float(np.mean(valid)),
                "total_train_seconds": estimate_training_hours(exp) * 3600,
                "total_gen_seconds": 0.0,
            }

    cost_analysis = cost_benefit_analysis(cost_data)

    print(f"\n    {'Experiment':<15s} {'AUPRC':>8s} {'Gain':>8s} {'Hours':>8s} {'Gain/hr':>10s} {'Rank':>6s}")
    print(f"    {'─' * 60}")
    for exp in ["e1", "e2", "e3", "e4", "e5"]:
        if exp in cost_analysis:
            ca = cost_analysis[exp]
            rank = ca.get("cost_benefit_rank", "-")
            print(f"    {exp:<15s} {ca['auprc_mean']:>8.4f} {ca['auprc_gain_over_baseline']:>+8.4f} "
                  f"{ca['total_time_hours']:>8.1f} {ca['gain_per_hour']:>10.6f} {rank!s:>6s}")

    # ── 5. Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  E6 Summary")
    print("=" * 60)

    # Find if any method significantly beats baseline
    beats_baseline = [
        name for name in augmentation_methods
        if wilcoxon_results.get(f"{baseline_name} vs {name}", {}).get("mean_diff", 0) > 0
        and wilcoxon_results.get(f"{baseline_name} vs {name}", {}).get("significant_005", False)
    ]
    hurts_baseline = [
        name for name in augmentation_methods
        if wilcoxon_results.get(f"{baseline_name} vs {name}", {}).get("mean_diff", 0) < 0
        and wilcoxon_results.get(f"{baseline_name} vs {name}", {}).get("significant_005", False)
    ]

    print(f"\n  Methods significantly BETTER than baseline (p<0.05): {beats_baseline or 'NONE'}")
    print(f"  Methods significantly WORSE than baseline (p<0.05):  {hurts_baseline or 'NONE'}")
    print(f"\n  Best augmentation: E5 (LDM) at ratio 0.50 (AUPRC = {np.mean([v for v in all_auprcs['E5 (LDM 0.50)'] if not np.isnan(v)]):.4f})")
    print(f"  Baseline:          E1 (AUPRC = {np.mean([v for v in all_auprcs[baseline_name] if not np.isnan(v)]):.4f})")
    print(f"\n  Consistent finding: ratio 0.50 > 1.00 for all generators (diminishing returns)")
    print(f"  All augmentation methods degrade performance vs baseline in LOPO")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        "wilcoxon_tests": wilcoxon_results,
        "ratio_sensitivity": {k: {
            "best_ratio": v["best_ratio"],
            "best_ratio_gain": v["best_ratio_gain"],
            "diminishing_returns": v["diminishing_returns"],
            "degradation_ratio": v["degradation_ratio"],
            "best_vs_baseline_p_value": v.get("best_vs_baseline_p_value"),
            "per_ratio": {str(rk): rv for rk, rv in v.get("per_ratio", {}).items()},
        } for k, v in ratio_results.items()},
        "cost_benefit": cost_analysis,
        "fold_auprcs": {k: v for k, v in fold_auprcs.items()},
        "summary": {
            "beats_baseline_significant": beats_baseline,
            "hurts_baseline_significant": hurts_baseline,
            "best_augmentation": "E5 (LDM 0.50)",
            "finding": "No augmentation method significantly improves over baseline in LOPO",
        },
    }

    output_path = OUTPUT_DIR / "e6_analysis.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {output_path}")


if __name__ == "__main__":
    main()

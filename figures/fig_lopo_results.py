"""
Generate publication-ready LOPO results figures for the thesis.

Produces:
  1. AUPRC bar chart (all methods, grouped by ratio)
  2. Multi-metric comparison (AUPRC, Precision, Recall, F1, AUROC, Sens@95%Spec)
  3. Ratio sensitivity plot (0.50 vs 1.00 per generator)
  4. Per-patient heatmap (LDM 0.50 vs baseline per fold)

Reads from: results/e6/e6_analysis.json, results/precision_recall_summary.json

Usage:
    python -m figures.fig_lopo_results [--output-dir figures/output]
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = _PROJECT_ROOT / "results"

METHODS_ORDER = [
    "E1 (baseline)",
    "E5 (LDM 0.50)",
    "E3 (TimeGAN 0.50)",
    "E4 (CVAE 0.50)",
    "E5 (LDM 1.00)",
    "E4 (CVAE 1.00)",
    "E3 (TimeGAN 1.00)",
    "E2 (ADASYN)",
    "E2 (SMOTE)",
]

SHORT_LABELS = [
    "Baseline",
    "LDM 0.50",
    "TimeGAN 0.50",
    "CVAE 0.50",
    "LDM 1.00",
    "CVAE 1.00",
    "TimeGAN 1.00",
    "ADASYN",
    "SMOTE",
]

COLORS = {
    "E1 (baseline)": "#6B7280",
    "E5 (LDM 0.50)": "#10B981",
    "E5 (LDM 1.00)": "#059669",
    "E3 (TimeGAN 0.50)": "#F59E0B",
    "E3 (TimeGAN 1.00)": "#D97706",
    "E4 (CVAE 0.50)": "#06B6D4",
    "E4 (CVAE 1.00)": "#0891B2",
    "E2 (ADASYN)": "#EF4444",
    "E2 (SMOTE)": "#DC2626",
}

PREC_REC_KEY_MAP = {
    "E1 (baseline)": "e1 baseline",
    "E5 (LDM 0.50)": "e5 0.50",
    "E3 (TimeGAN 0.50)": "e3 0.50",
    "E4 (CVAE 0.50)": "e4 0.50",
    "E5 (LDM 1.00)": "e5 1.00",
    "E4 (CVAE 1.00)": "e4 1.00",
    "E3 (TimeGAN 1.00)": "e3 1.00",
    "E2 (ADASYN)": "e2 adasyn",
    "E2 (SMOTE)": "e2 smote",
}

LOPO_METRICS = {
    "E1 (baseline)": {"auprc": 0.3941, "auroc": 0.8438, "f1": 0.4333, "sens95": 0.6459},
    "E5 (LDM 0.50)": {"auprc": 0.3259, "auroc": 0.7851, "f1": 0.3730, "sens95": 0.5684},
    "E3 (TimeGAN 0.50)": {"auprc": 0.3064, "auroc": 0.7779, "f1": 0.3500, "sens95": 0.5671},
    "E4 (CVAE 0.50)": {"auprc": 0.2976, "auroc": 0.7769, "f1": 0.3480, "sens95": 0.5402},
    "E5 (LDM 1.00)": {"auprc": 0.2830, "auroc": 0.7715, "f1": 0.3384, "sens95": 0.5300},
    "E4 (CVAE 1.00)": {"auprc": 0.2507, "auroc": 0.7288, "f1": 0.3060, "sens95": 0.5140},
    "E3 (TimeGAN 1.00)": {"auprc": 0.2145, "auroc": 0.7287, "f1": 0.2778, "sens95": 0.4895},
    "E2 (ADASYN)": {"auprc": 0.1844, "auroc": 0.6458, "f1": 0.2514, "sens95": 0.4046},
    "E2 (SMOTE)": {"auprc": 0.1580, "auroc": 0.6283, "f1": 0.2295, "sens95": 0.3835},
}

LOPO_STDS = {
    "E1 (baseline)": {"auprc": 0.3308, "auroc": 0.1907, "f1": 0.3078, "sens95": 0.3367},
    "E5 (LDM 0.50)": {"auprc": 0.3185, "auroc": 0.2216, "f1": 0.3008, "sens95": 0.3478},
    "E3 (TimeGAN 0.50)": {"auprc": 0.2990, "auroc": 0.2235, "f1": 0.2831, "sens95": 0.3335},
    "E4 (CVAE 0.50)": {"auprc": 0.2894, "auroc": 0.2147, "f1": 0.2802, "sens95": 0.3353},
    "E5 (LDM 1.00)": {"auprc": 0.2767, "auroc": 0.2097, "f1": 0.2713, "sens95": 0.3198},
    "E4 (CVAE 1.00)": {"auprc": 0.2771, "auroc": 0.2451, "f1": 0.2671, "sens95": 0.3507},
    "E3 (TimeGAN 1.00)": {"auprc": 0.2499, "auroc": 0.2374, "f1": 0.2501, "sens95": 0.3246},
    "E2 (ADASYN)": {"auprc": 0.2270, "auroc": 0.2639, "f1": 0.2494, "sens95": 0.3364},
    "E2 (SMOTE)": {"auprc": 0.2016, "auroc": 0.2532, "f1": 0.2241, "sens95": 0.3187},
}


def load_e6_data():
    path = RESULTS_DIR / "e6" / "e6_analysis.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def load_precision_recall():
    path = RESULTS_DIR / "precision_recall_summary.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fig1_auprc_bar(output_dir, e6_data):
    """Bar chart of AUPRC for all methods."""
    fig, ax = plt.subplots(figsize=(10, 5))

    means = [LOPO_METRICS[m]["auprc"] for m in METHODS_ORDER]
    stds = [LOPO_STDS[m]["auprc"] for m in METHODS_ORDER]
    colors = [COLORS[m] for m in METHODS_ORDER]

    bars = ax.bar(range(len(means)), means, yerr=stds, color=colors,
                  edgecolor='white', linewidth=0.5, capsize=3, error_kw={'linewidth': 1})

    ax.axhline(means[0], color=COLORS["E1 (baseline)"], linestyle='--',
               linewidth=1, alpha=0.7, label=f'Baseline ({means[0]:.3f})')

    ax.set_xticks(range(len(SHORT_LABELS)))
    ax.set_xticklabels(SHORT_LABELS, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("AUPRC (mean across 69 runs)")
    ax.set_title("LOPO AUPRC - All Methods (23 folds x 3 seeds)")
    ax.set_ylim(0, 0.8)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, axis='y', alpha=0.2)

    for i, (m, s) in enumerate(zip(means, stds)):
        pct = (m - means[0]) / means[0] * 100
        label = f"{pct:+.0f}%" if i > 0 else "ref"
        ax.text(i, m + s + 0.02, label, ha='center', fontsize=8,
                color='#EF4444' if pct < 0 else '#6B7280')

    plt.tight_layout()
    fig.savefig(output_dir / "lopo_auprc_bar.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "lopo_auprc_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: lopo_auprc_bar.pdf/.png")


def fig2_multi_metric(output_dir, prec_rec_data):
    """Grouped bar chart with all 6 metrics."""
    metrics = ["AUPRC", "Precision", "Recall", "F1", "AUROC", "Sens@95%Spec"]
    methods_short = ["Baseline", "LDM 0.50", "TimeGAN 0.50", "CVAE 0.50",
                     "LDM 1.00", "CVAE 1.00", "TimeGAN 1.00", "ADASYN", "SMOTE"]

    data = np.zeros((len(METHODS_ORDER), 6))
    for i, m in enumerate(METHODS_ORDER):
        data[i, 0] = LOPO_METRICS[m]["auprc"]
        data[i, 3] = LOPO_METRICS[m]["f1"]
        data[i, 4] = LOPO_METRICS[m]["auroc"]
        data[i, 5] = LOPO_METRICS[m]["sens95"]
        if prec_rec_data:
            pk = PREC_REC_KEY_MAP[m]
            if pk in prec_rec_data:
                data[i, 1] = prec_rec_data[pk]["precision_mean"]
                data[i, 2] = prec_rec_data[pk]["recall_mean"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(metrics))
    width = 0.09
    n_methods = len(METHODS_ORDER)

    for i, m in enumerate(METHODS_ORDER):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[i], width, label=methods_short[i],
                      color=COLORS[m], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("LOPO Results - All Metrics (23 folds x 3 seeds)")
    ax.set_ylim(0, 1.0)
    ax.legend(loc='upper right', fontsize=8, ncol=3)
    ax.grid(True, axis='y', alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "lopo_multi_metric.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "lopo_multi_metric.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: lopo_multi_metric.pdf/.png")


def fig3_ratio_sensitivity(output_dir):
    """Line plot showing ratio 0.50 vs 1.00 per generator."""
    generators = ["LDM", "CVAE", "TimeGAN"]
    gen_colors = ["#10B981", "#06B6D4", "#F59E0B"]
    ratios = [0.50, 1.00]

    auprc_050 = [LOPO_METRICS[f"E5 (LDM 0.50)"]["auprc"],
                 LOPO_METRICS[f"E4 (CVAE 0.50)"]["auprc"],
                 LOPO_METRICS[f"E3 (TimeGAN 0.50)"]["auprc"]]
    auprc_100 = [LOPO_METRICS[f"E5 (LDM 1.00)"]["auprc"],
                 LOPO_METRICS[f"E4 (CVAE 1.00)"]["auprc"],
                 LOPO_METRICS[f"E3 (TimeGAN 1.00)"]["auprc"]]

    fig, ax = plt.subplots(figsize=(6, 4))

    baseline = LOPO_METRICS["E1 (baseline)"]["auprc"]
    ax.axhline(baseline, color='#6B7280', linestyle='--', linewidth=1.5,
               label=f'Baseline ({baseline:.3f})')

    for i, (gen, color) in enumerate(zip(generators, gen_colors)):
        values = [auprc_050[i], auprc_100[i]]
        ax.plot(ratios, values, 'o-', color=color, linewidth=2, markersize=8, label=gen)
        for r, v in zip(ratios, values):
            pct = (v - baseline) / baseline * 100
            ax.annotate(f"{pct:+.0f}%", (r, v), textcoords="offset points",
                        xytext=(10, 5), fontsize=8, color=color)

    ax.set_xticks(ratios)
    ax.set_xticklabels(["0.50", "1.00"])
    ax.set_xlabel("Synthetic Ratio")
    ax.set_ylabel("AUPRC")
    ax.set_title("Ratio Sensitivity - All Generators Under LOPO")
    ax.set_ylim(0.15, 0.45)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "lopo_ratio_sensitivity.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "lopo_ratio_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: lopo_ratio_sensitivity.pdf/.png")


def fig4_per_patient(output_dir, e6_data):
    """Heatmap of per-fold AUPRC difference (LDM 0.50 vs baseline)."""
    if e6_data is None or "fold_auprcs" not in e6_data:
        print("  Skipping per-patient heatmap (no fold data)")
        return

    baseline_folds = np.array(e6_data["fold_auprcs"]["E1 (baseline)"])
    ldm_folds = np.array(e6_data["fold_auprcs"]["E5 (LDM 0.50)"])
    diff = ldm_folds - baseline_folds

    sorted_idx = np.argsort(diff)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors_bar = ['#10B981' if d > 0 else '#EF4444' for d in diff[sorted_idx]]

    y_pos = np.arange(len(diff))
    ax.barh(y_pos, diff[sorted_idx], color=colors_bar, edgecolor='white', linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Fold {i:02d}" for i in sorted_idx], fontsize=8)
    ax.set_xlabel("AUPRC Difference (LDM 0.50 - Baseline)")
    ax.set_title("Per-Fold Impact of LDM Augmentation (ratio 0.50)")
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.2)

    n_pos = sum(1 for d in diff if d > 0)
    n_neg = sum(1 for d in diff if d <= 0)
    ax.text(0.98, 0.95, f"Improved: {n_pos}/23\nHurt: {n_neg}/23",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_dir / "lopo_per_patient_ldm.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "lopo_per_patient_ldm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: lopo_per_patient_ldm.pdf/.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="figures/output")
    args = parser.parse_args()

    output_dir = _PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    e6_data = load_e6_data()
    prec_rec_data = load_precision_recall()

    print("Generating figures...")
    fig1_auprc_bar(output_dir, e6_data)
    fig2_multi_metric(output_dir, prec_rec_data)
    fig3_ratio_sensitivity(output_dir)
    fig4_per_patient(output_dir, e6_data)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == '__main__':
    main()

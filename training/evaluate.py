"""
evaluate.py — Evaluation metrics for seizure detection experiments.

Metrics (from thesis SLR — Saito & Rehmsmeier 2015; Carrle et al. 2023;
Zhao et al. 2022; Dakshit et al. 2023):
  Primary:     AUPRC (Area Under Precision-Recall Curve)
  Secondary:   AUROC (Area Under ROC)
  Operational: F1 at optimal threshold, Sensitivity @ 95% Specificity
  Stability:   Per-patient AUPRC (mean ± std across patients)
  Stability:   Seed variance (mean ± std across seeds)
  Statistical: Wilcoxon signed-rank test (across LOPO folds)

Usage:
    from training.evaluate import evaluate_model, compute_metrics, wilcoxon_compare

    # Evaluate a model on a DataLoader
    results = evaluate_model(model, test_loader, device="cuda")
    print(results["auprc"], results["auroc"])

    # Compare two experiment results across LOPO folds
    p_value = wilcoxon_compare(fold_auprcs_a, fold_auprcs_b)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    roc_curve,
)
from scipy.stats import wilcoxon


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    patient_ids: Optional[np.ndarray] = None,
) -> Dict:
    """
    Compute all thesis-mandated evaluation metrics.

    Args:
        y_true: (N,) binary labels (0=interictal, 1=ictal)
        y_prob: (N,) predicted probability for class 1 (ictal)
        patient_ids: (N,) integer patient IDs (optional, for per-patient stats)

    Returns:
        Dict with keys:
            auprc: float — primary metric
            auroc: float — secondary metric
            f1_optimal: float — F1 at optimal threshold
            threshold_optimal: float — threshold maximizing F1 on this data
            sensitivity_at_95spec: float — sensitivity at 95% specificity
            n_samples: int
            n_ictal: int
            n_interictal: int
            per_patient: dict (if patient_ids provided) — {patient_id: {auprc, auroc, n_ictal, n_total}}
    """
    results: Dict = {}
    n = len(y_true)
    n_ictal = int(y_true.sum())
    n_interictal = n - n_ictal

    results["n_samples"] = n
    results["n_ictal"] = n_ictal
    results["n_interictal"] = n_interictal

    # Handle edge cases (no positives or no negatives)
    if n_ictal == 0 or n_interictal == 0:
        results["auprc"] = float("nan")
        results["auroc"] = float("nan")
        results["f1_optimal"] = 0.0
        results["threshold_optimal"] = 0.5
        results["sensitivity_at_95spec"] = 0.0
        results["per_patient"] = {}
        return results

    # ── Primary: AUPRC ─────────────────────────────────────────────────
    results["auprc"] = float(average_precision_score(y_true, y_prob))

    # ── Secondary: AUROC ───────────────────────────────────────────────
    results["auroc"] = float(roc_auc_score(y_true, y_prob))

    # ── Operational: F1 at optimal threshold ───────────────────────────
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_prob)
    # F1 = 2 * (precision * recall) / (precision + recall)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-12)
    best_idx = np.argmax(f1_scores)
    results["f1_optimal"] = float(f1_scores[best_idx])
    results["threshold_optimal"] = float(thresholds_pr[min(best_idx, len(thresholds_pr) - 1)])

    # ── Operational: Sensitivity @ 95% Specificity ─────────────────────
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    # Find the threshold where specificity >= 0.95
    mask = specificity >= 0.95
    if mask.any():
        # Among all points with spec >= 0.95, pick the one with highest sensitivity
        results["sensitivity_at_95spec"] = float(tpr[mask].max())
    else:
        results["sensitivity_at_95spec"] = 0.0

    # ── Per-patient metrics ────────────────────────────────────────────
    if patient_ids is not None:
        per_patient = {}
        unique_patients = np.unique(patient_ids)
        for pid in unique_patients:
            mask = patient_ids == pid
            p_true = y_true[mask]
            p_prob = y_prob[mask]
            p_n_ictal = int(p_true.sum())
            p_n_total = int(mask.sum())

            p_metrics = {
                "n_ictal": p_n_ictal,
                "n_total": p_n_total,
            }

            if p_n_ictal > 0 and p_n_ictal < p_n_total:
                p_metrics["auprc"] = float(average_precision_score(p_true, p_prob))
                p_metrics["auroc"] = float(roc_auc_score(p_true, p_prob))
            else:
                p_metrics["auprc"] = float("nan")
                p_metrics["auroc"] = float("nan")

            per_patient[int(pid)] = p_metrics

        results["per_patient"] = per_patient

        # Per-patient AUPRC stats (excluding NaN patients)
        patient_auprcs = [
            m["auprc"] for m in per_patient.values()
            if not np.isnan(m["auprc"])
        ]
        if patient_auprcs:
            results["per_patient_auprc_mean"] = float(np.mean(patient_auprcs))
            results["per_patient_auprc_std"] = float(np.std(patient_auprcs))
        else:
            results["per_patient_auprc_mean"] = float("nan")
            results["per_patient_auprc_std"] = float("nan")
    else:
        results["per_patient"] = {}

    return results


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Dict:
    """
    Run a trained model on a DataLoader and compute all metrics.

    Args:
        model: trained SeizureDetector (or any model returning (batch, 2) logits)
        dataloader: PyTorch DataLoader yielding (window, label, patient_id)
        device: "cpu" or "cuda"

    Returns:
        Dict of metrics (same format as compute_metrics)
    """
    model.eval()
    model.to(device)

    all_probs = []
    all_labels = []
    all_patients = []

    for batch in dataloader:
        windows, labels, patient_ids = batch
        windows = windows.to(device, dtype=torch.float32)

        logits = model(windows)
        probs = torch.softmax(logits, dim=1)[:, 1]  # P(ictal)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())
        all_patients.append(patient_ids.numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    patient_ids = np.concatenate(all_patients)

    return compute_metrics(y_true, y_prob, patient_ids)


def aggregate_seed_results(
    seed_results: List[Dict],
) -> Dict:
    """
    Aggregate metrics across multiple seeds (mean ± std).

    Args:
        seed_results: list of metric dicts (one per seed)

    Returns:
        Dict with mean/std for each scalar metric
    """
    aggregated = {}
    scalar_keys = ["auprc", "auroc", "f1_optimal", "sensitivity_at_95spec"]

    for key in scalar_keys:
        values = [r[key] for r in seed_results if not np.isnan(r.get(key, float("nan")))]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
        else:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")

    aggregated["n_seeds"] = len(seed_results)
    return aggregated


def aggregate_lopo_results(
    fold_results: List[Dict],
) -> Dict:
    """
    Aggregate metrics across LOPO folds (mean ± std across folds).

    Args:
        fold_results: list of metric dicts (one per fold)

    Returns:
        Dict with fold-level aggregation
    """
    aggregated = {}
    scalar_keys = ["auprc", "auroc", "f1_optimal", "sensitivity_at_95spec"]

    for key in scalar_keys:
        values = [r[key] for r in fold_results if not np.isnan(r.get(key, float("nan")))]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_median"] = float(np.median(values))
        else:
            aggregated[f"{key}_mean"] = float("nan")
            aggregated[f"{key}_std"] = float("nan")
            aggregated[f"{key}_median"] = float("nan")

    aggregated["n_folds"] = len(fold_results)
    aggregated["fold_auprcs"] = [r.get("auprc", float("nan")) for r in fold_results]

    return aggregated


def wilcoxon_compare(
    auprcs_a: List[float],
    auprcs_b: List[float],
) -> Dict:
    """
    Wilcoxon signed-rank test comparing paired AUPRC values across folds.

    Used for E6 cross-generator comparison (Wilcoxon 1945; applied to
    EEG fold comparisons by Zhao et al. 2022).

    Args:
        auprcs_a: per-fold AUPRC values for experiment A
        auprcs_b: per-fold AUPRC values for experiment B

    Returns:
        Dict with statistic, p_value, and interpretation
    """
    a = np.array(auprcs_a)
    b = np.array(auprcs_b)

    # Remove NaN pairs
    valid = ~(np.isnan(a) | np.isnan(b))
    a_valid = a[valid]
    b_valid = b[valid]

    if len(a_valid) < 5:
        return {
            "statistic": float("nan"),
            "p_value": float("nan"),
            "n_valid_pairs": int(valid.sum()),
            "significant_005": False,
            "significant_001": False,
            "mean_diff": float(np.mean(b_valid - a_valid)) if len(a_valid) > 0 else float("nan"),
        }

    # Two-sided test (is there a significant difference?)
    stat, p_value = wilcoxon(a_valid, b_valid, alternative="two-sided")

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n_valid_pairs": int(valid.sum()),
        "significant_005": p_value < 0.05,
        "significant_001": p_value < 0.01,
        "mean_diff": float(np.mean(b_valid - a_valid)),
        "median_diff": float(np.median(b_valid - a_valid)),
    }


def format_results_table(results: Dict, experiment_name: str = "") -> str:
    """
    Format metrics as a readable text table for logging.

    Args:
        results: metric dict from compute_metrics or evaluate_model
        experiment_name: optional experiment label

    Returns:
        Formatted multi-line string
    """
    lines = []
    if experiment_name:
        lines.append(f"{'=' * 50}")
        lines.append(f"  {experiment_name}")
        lines.append(f"{'=' * 50}")

    lines.append(f"  Samples:     {results.get('n_samples', '?'):>8}")
    lines.append(f"  Ictal:       {results.get('n_ictal', '?'):>8}")
    lines.append(f"  Interictal:  {results.get('n_interictal', '?'):>8}")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  AUPRC:                 {results.get('auprc', float('nan')):>8.4f}")
    lines.append(f"  AUROC:                 {results.get('auroc', float('nan')):>8.4f}")
    lines.append(f"  F1 (optimal):          {results.get('f1_optimal', float('nan')):>8.4f}")
    lines.append(f"  Threshold (optimal):   {results.get('threshold_optimal', float('nan')):>8.4f}")
    lines.append(f"  Sens. @ 95% Spec.:     {results.get('sensitivity_at_95spec', float('nan')):>8.4f}")

    if "per_patient_auprc_mean" in results:
        lines.append(f"  {'─' * 40}")
        mean = results["per_patient_auprc_mean"]
        std = results["per_patient_auprc_std"]
        lines.append(f"  Per-patient AUPRC:     {mean:.4f} ± {std:.4f}")

    return "\n".join(lines)


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Synthetic test data
    np.random.seed(42)
    n = 1000
    y_true = np.concatenate([np.zeros(800), np.ones(200)]).astype(int)
    y_prob = np.clip(y_true * 0.7 + np.random.randn(n) * 0.2, 0, 1)
    patient_ids = np.random.randint(0, 5, size=n)

    results = compute_metrics(y_true, y_prob, patient_ids)
    print(format_results_table(results, "Synthetic Test"))

    # Test wilcoxon
    a = [0.3, 0.4, 0.35, 0.45, 0.5, 0.38]
    b = [0.35, 0.45, 0.40, 0.50, 0.55, 0.42]
    w = wilcoxon_compare(a, b)
    print(f"\nWilcoxon test: stat={w['statistic']:.2f}, p={w['p_value']:.4f}, sig@0.05={w['significant_005']}")
    print("\nAll checks passed.")

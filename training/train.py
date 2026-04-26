"""
train.py — Training pipeline for seizure detection experiments.

Supports:
  E1: Baseline detector (real data only, class-weighted CE)
  E2: Non-synthetic controls (SMOTE, ADASYN augmentation)
  E3-E5: Generator augmentation (synthetic windows injected into training)

Training follows standard practices from the thesis SLR:
  - Optimizer: Adam, lr=1e-3
  - Batch size: 64
  - Early stopping on validation AUPRC (patience=10; Prechelt 1998)
  - Max 100 epochs
  - Loss: Class-weighted cross-entropy (Zhao et al. 2022)
  - 3 seeds per experiment for stability (seeds: 42, 123, 456)
  - Single-split for development, LOPO for final results (Carrle 2023)

Usage:
    from training.train import train_single_split, train_lopo, Trainer

    # E1: Quick single-split baseline
    results = train_single_split(experiment="e1", seed=42)

    # E1: Full LOPO evaluation
    results = train_lopo(experiment="e1", seeds=[42, 123, 456])

    # E2b: SMOTE augmentation
    results = train_single_split(experiment="e2", augmentation="smote", seed=42)
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path so imports work when running this file directly
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.detector import SeizureDetector
from data.loader import (
    CHBMITDataset,
    CaseAwareSampler,
    get_dataloaders,
    get_lopo_dataloaders,
    ALL_CASES,
    N_CHANNELS,
    WINDOW_SAMPLES,
    SPLIT_CONFIG_PATH,
)
from training.evaluate import evaluate_model, format_results_table


# ── Configuration ──────────────────────────────────────────────────────────
RESULTS_DIR = _PROJECT_ROOT / "results"


@dataclass
class TrainConfig:
    """Training hyperparameters (frozen across all experiments)."""
    lr: float = 1e-3
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 10           # Early stopping patience (on val AUPRC)
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Augmentation (E2 controls)
    augmentation: Optional[str] = None  # None, "smote", or "adasyn"
    smote_k: int = 5                    # k neighbors for SMOTE/ADASYN

    # Synthetic windows (E3-E5 generators)
    synthetic_windows: Optional[list] = field(default=None, repr=False)

    # Output
    experiment: str = "e1"
    save_checkpoints: bool = True


# ── SMOTE / ADASYN Augmentation ───────────────────────────────────────────
_INTERICTAL_RATIO = 10  # interictal subsample = 10× ictal count

def apply_oversampling(
    dataset: CHBMITDataset,
    method: str = "smote",
    k: int = 5,
    seed: int = 42,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Apply SMOTE or ADASYN oversampling to the ictal class in the dataset.

    Loads all ictal windows plus a random subsample of interictal windows
    (10x ictal count) to keep memory feasible on 32 GB machines. SMOTE
    only uses k-NN from the minority class, so the full majority set is
    not needed. Returns NEW synthetic windows only (not the originals).
    """
    try:
        if method == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(k_neighbors=min(k, dataset._n_ictal - 1), random_state=seed)
        elif method == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(n_neighbors=min(k, dataset._n_ictal - 1), random_state=seed)
        else:
            raise ValueError(f"Unknown oversampling method: {method}")
    except ImportError:
        raise ImportError(
            f"imbalanced-learn is required for {method}: pip install imbalanced-learn"
        )

    rng = np.random.RandomState(seed)
    flat_dim = N_CHANNELS * WINDOW_SAMPLES

    ictal_idx = np.where(dataset._all_labels == 1)[0]
    interictal_idx = np.where(dataset._all_labels == 0)[0]
    n_sub = min(len(interictal_idx), len(ictal_idx) * _INTERICTAL_RATIO)
    interictal_sub = rng.choice(interictal_idx, size=n_sub, replace=False)

    selected = np.concatenate([ictal_idx, interictal_sub])
    n = len(selected)
    print(f"  {method.upper()}: loading {len(ictal_idx)} ictal + {n_sub} interictal subsample ({n} total)")

    X = np.zeros((n, flat_dim), dtype=np.float32)
    y = np.zeros(n, dtype=int)
    patient_ids = np.zeros(n, dtype=int)

    for i, real_idx in enumerate(selected):
        window, label, pid = dataset._get_real_window(int(real_idx))
        X[i] = window.ravel()
        y[i] = label
        patient_ids[i] = pid

    n_before = len(X)
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    n_new = len(X_resampled) - n_before
    if n_new <= 0:
        print(f"Warning: {method} generated 0 new samples (class may already be balanced)")
        return []

    ictal_pids = patient_ids[y == 1]
    synthetic = []
    for i in range(n_before, len(X_resampled)):
        window = X_resampled[i].reshape(N_CHANNELS, WINDOW_SAMPLES)
        pid = int(rng.choice(ictal_pids))
        synthetic.append((window, 1, pid))

    print(f"  {method.upper()} generated {len(synthetic)} synthetic ictal windows")
    return synthetic


# ── Trainer ────────────────────────────────────────────────────────────────
class Trainer:
    """
    Training loop with early stopping on validation AUPRC.

    This class handles the core training cycle:
      1. Train for one epoch
      2. Evaluate on validation set
      3. Check early stopping criterion
      4. Save best checkpoint

    The same Trainer is used for all experiments (E1-E5).
    """

    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: torch.Tensor,
        checkpoint_dir: Optional[Path] = None,
    ) -> Tuple[SeizureDetector, Dict]:
        """
        Train a SeizureDetector to convergence with early stopping.

        Args:
            train_loader: training DataLoader
            val_loader: validation DataLoader
            class_weights: (2,) tensor [w_interictal, w_ictal]
            checkpoint_dir: directory to save best model checkpoint

        Returns:
            (best_model, training_history)
        """
        model = SeizureDetector().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )

        best_val_auprc = -1.0
        best_epoch = 0
        patience_counter = 0
        best_state = None

        history = {
            "train_loss": [],
            "val_auprc": [],
            "val_auroc": [],
            "epoch_times": [],
        }

        for epoch in range(self.config.max_epochs):
            t0 = time.time()
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)

            # ── Train one epoch ────────────────────────────────────────
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                windows, labels, _ = batch
                windows = windows.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                logits = model(windows)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # ── Validate ───────────────────────────────────────────────
            val_metrics = evaluate_model(model, val_loader, device=str(self.device))
            val_auprc = val_metrics["auprc"]
            val_auroc = val_metrics["auroc"]

            epoch_time = time.time() - t0
            history["train_loss"].append(avg_loss)
            history["val_auprc"].append(val_auprc)
            history["val_auroc"].append(val_auroc)
            history["epoch_times"].append(epoch_time)

            # Handle NaN AUPRC (no positives in val set)
            if np.isnan(val_auprc):
                print(f"  Epoch {epoch+1:3d}/{self.config.max_epochs}: "
                      f"loss={avg_loss:.4f}, val_auprc=NaN, val_auroc=NaN "
                      f"({epoch_time:.1f}s)")
                patience_counter += 1
            else:
                improved = val_auprc > best_val_auprc
                if improved:
                    best_val_auprc = val_auprc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1

                print(f"  Epoch {epoch+1:3d}/{self.config.max_epochs}: "
                      f"loss={avg_loss:.4f}, val_auprc={val_auprc:.4f}, "
                      f"val_auroc={val_auroc:.4f} "
                      f"{'*' if improved else ''} ({epoch_time:.1f}s)")

            # ── Early stopping ─────────────────────────────────────────
            if patience_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={self.config.patience})")
                break

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)
            print(f"  Restored best model from epoch {best_epoch} (val_auprc={best_val_auprc:.4f})")
        else:
            print("  Warning: no improvement was observed during training")

        # Save checkpoint
        if self.config.save_checkpoints and checkpoint_dir is not None and best_state is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = checkpoint_dir / "best_model.pt"
            torch.save({
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "best_val_auprc": best_val_auprc,
                "config": asdict(self.config),
            }, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")

        history["best_epoch"] = best_epoch
        history["best_val_auprc"] = best_val_auprc
        history["total_epochs"] = epoch + 1

        return model, history


# ── Experiment Runners ─────────────────────────────────────────────────────
def train_single_split(
    experiment: str = "e1",
    augmentation: Optional[str] = None,
    synthetic_windows: Optional[list] = None,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict:
    """
    Run an experiment using the single fixed split (development mode).

    Args:
        experiment: experiment label ("e1", "e2", etc.)
        augmentation: None, "smote", or "adasyn" (for E2)
        synthetic_windows: pre-generated synthetic windows (for E3-E5)
        seed: random seed
        device: "cpu" or "cuda" (auto-detected if None)

    Returns:
        Dict with train_history, val_metrics, test_metrics
    """
    config = TrainConfig(
        experiment=experiment,
        augmentation=augmentation,
        seed=seed,
    )
    if device:
        config.device = device

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment.upper()}")
    print(f"  Mode: Single-split (development)")
    print(f"  Augmentation: {augmentation or 'None'}")
    print(f"  Seed: {seed}")
    print(f"  Device: {config.device}")
    print(f"{'=' * 60}")

    # Build datasets
    print("\nLoading datasets...")

    # Apply oversampling if requested (E2)
    # IMPORTANT: SMOTE/ADASYN must operate on unnormalized (µV-scale) windows,
    # because synthetic_windows passed to CHBMITDataset will be normalized by the
    # dataset constructor. Operating on already-normalized windows would cause
    # double normalization (bug fixed 2026-04-14).
    if augmentation in ("smote", "adasyn") and synthetic_windows is not None:
        raise ValueError(
            "Cannot combine oversampling (E2) with synthetic windows (E3-E5). "
            "These are mutually exclusive experiments."
        )

    oversample_windows = None
    if augmentation in ("smote", "adasyn"):
        train_ds_unnorm = CHBMITDataset(split="train", seed=seed, normalize=False)
        oversample_windows = apply_oversampling(
            train_ds_unnorm, method=augmentation, seed=seed,
        )

    train_ds = CHBMITDataset(
        split="train", seed=seed,
        synthetic_windows=oversample_windows,
    )
    val_ds = CHBMITDataset(split="val", seed=seed)
    test_ds = CHBMITDataset(split="test", seed=seed)

    # Inject generator-produced synthetic windows (E3-E5)
    if synthetic_windows is not None:
        train_ds = CHBMITDataset(
            split="train", seed=seed,
            synthetic_windows=synthetic_windows,
        )

    n_inter, n_ictal = train_ds.get_class_counts()
    class_weights = train_ds.get_class_weights()
    print(f"  Train: {len(train_ds)} windows ({n_ictal} ictal, {n_inter} interictal)")
    print(f"  Val:   {len(val_ds)} windows")
    print(f"  Test:  {len(test_ds)} windows")
    print(f"  Class weights: {class_weights.tolist()}")

    # Build data loaders
    train_sampler = CaseAwareSampler(train_ds, seed=seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=True,
    )

    # Train
    print("\nTraining...")
    trainer = Trainer(config)
    checkpoint_dir = RESULTS_DIR / experiment / f"seed_{seed}" / "single_split"
    model, history = trainer.train(train_loader, val_loader, class_weights, checkpoint_dir)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device=config.device)
    print(format_results_table(test_metrics, f"{experiment.upper()} — Test (single-split)"))

    # Save results
    results = {
        "experiment": experiment,
        "mode": "single_split",
        "augmentation": augmentation,
        "seed": seed,
        "train_history": history,
        "test_metrics": test_metrics,
    }

    results_dir = RESULTS_DIR / experiment / f"seed_{seed}" / "single_split"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def train_lopo(
    experiment: str = "e1",
    augmentation: Optional[str] = None,
    synthetic_windows_fn=None,
    seeds: List[int] = None,
    device: Optional[str] = None,
    folds: Optional[List[int]] = None,
) -> Dict:
    """
    Run an experiment using full LOPO cross-validation (final results).

    23 folds × N seeds = total runs.

    Args:
        experiment: experiment label ("e1", "e2", etc.)
        augmentation: None, "smote", or "adasyn" (for E2)
        synthetic_windows_fn: callable(fold, seed) -> list of synthetic windows
            Used for E3-E5 where each fold may need different synthetic data.
        seeds: list of random seeds (default: [42, 123, 456])
        device: "cpu" or "cuda"
        folds: specific fold indices to run (default: all 23)

    Returns:
        Dict with per-fold, per-seed results + aggregated statistics
    """
    if seeds is None:
        seeds = [42, 123, 456]
    if folds is None:
        folds = list(range(23))

    config_base = TrainConfig(experiment=experiment, augmentation=augmentation)
    if device:
        config_base.device = device

    print(f"\n{'=' * 60}")
    print(f"  Experiment: {experiment.upper()}")
    print(f"  Mode: LOPO ({len(folds)} folds × {len(seeds)} seeds = {len(folds) * len(seeds)} runs)")
    print(f"  Augmentation: {augmentation or 'None'}")
    print(f"  Seeds: {seeds}")
    print(f"  Device: {config_base.device}")
    print(f"{'=' * 60}")

    # Load split config to get correct fold-to-subject mapping
    with open(str(SPLIT_CONFIG_PATH)) as _f:
        _split_config = json.load(_f)
    lopo_folds_config = _split_config["lopo_folds"]

    all_results = {}

    for seed in seeds:
        seed_results = {}

        for fold in folds:
            # Resume: skip completed folds
            fold_result_path = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}" / "results.json"
            if fold_result_path.exists():
                print(f"\n  Fold {fold}/22, Seed {seed} — skipping (already complete)")
                try:
                    with open(fold_result_path) as _rf:
                        seed_results[fold] = json.load(_rf)
                    continue
                except (json.JSONDecodeError, KeyError):
                    pass  # corrupted, re-run

            print(f"\n{'─' * 50}")
            print(f"  Fold {fold}/22, Seed {seed}")
            print(f"{'─' * 50}")

            config = TrainConfig(
                experiment=experiment,
                augmentation=augmentation,
                seed=seed,
                device=config_base.device,
            )

            # Build datasets for this fold
            synthetic = None
            if synthetic_windows_fn is not None:
                synthetic = synthetic_windows_fn(fold, seed)

            # Apply oversampling if requested (E2)
            # IMPORTANT: SMOTE/ADASYN must operate on unnormalized (µV-scale)
            # windows to avoid double normalization (bug fixed 2026-04-14).
            oversample_windows = None
            if augmentation in ("smote", "adasyn"):
                train_ds_unnorm = CHBMITDataset(
                    split="train", fold=fold, seed=seed, normalize=False,
                )
                if train_ds_unnorm._n_ictal >= 2:
                    oversample_windows = apply_oversampling(
                        train_ds_unnorm, method=augmentation, seed=seed,
                    )

            # Combine synthetic sources: oversampling + generator outputs
            all_synthetic = oversample_windows
            if synthetic is not None:
                if all_synthetic is not None:
                    all_synthetic = all_synthetic + synthetic
                else:
                    all_synthetic = synthetic

            train_ds = CHBMITDataset(
                split="train", fold=fold, seed=seed,
                synthetic_windows=all_synthetic,
            )

            val_ds = CHBMITDataset(split="val", fold=fold, seed=seed)
            test_ds = CHBMITDataset(split="test", fold=fold, seed=seed)

            class_weights = train_ds.get_class_weights()
            n_inter, n_ictal = train_ds.get_class_counts()
            print(f"  Train: {len(train_ds)} ({n_ictal} ictal)")
            print(f"  Val:   {len(val_ds)}")
            print(f"  Test:  {len(test_ds)}")

            # Build data loaders
            train_sampler = CaseAwareSampler(train_ds, seed=seed)
            train_loader = DataLoader(
                train_ds, batch_size=config.batch_size, sampler=train_sampler,
                num_workers=config.num_workers, pin_memory=True, drop_last=True,
            )
            val_loader = DataLoader(
                val_ds, batch_size=config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True,
            )
            test_loader = DataLoader(
                test_ds, batch_size=config.batch_size, shuffle=False,
                num_workers=config.num_workers, pin_memory=True,
            )

            # Train
            trainer = Trainer(config)
            checkpoint_dir = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}"
            model, history = trainer.train(
                train_loader, val_loader, class_weights, checkpoint_dir,
            )

            # Evaluate on test set (held-out patient)
            test_metrics = evaluate_model(model, test_loader, device=config.device)
            test_metrics["fold"] = fold
            test_metrics["test_subject"] = lopo_folds_config[fold]["test_subject"]

            print(format_results_table(
                test_metrics,
                f"Fold {fold} — {test_metrics['test_subject']}",
            ))

            seed_results[fold] = {
                "train_history": history,
                "test_metrics": test_metrics,
            }

            # Save per-fold results incrementally
            fold_dir = RESULTS_DIR / experiment / f"seed_{seed}" / f"fold_{fold:02d}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            with open(fold_dir / "results.json", "w") as f:
                json.dump(seed_results[fold], f, indent=2, default=str)

        all_results[seed] = seed_results

    # ── Aggregate results ──────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  LOPO Summary — {experiment.upper()}")
    print(f"{'=' * 60}")

    # Per-seed fold aggregation
    for seed in seeds:
        fold_auprcs = [
            all_results[seed][f]["test_metrics"]["auprc"]
            for f in folds
            if f in all_results[seed]
        ]
        valid = [a for a in fold_auprcs if not np.isnan(a)]
        if valid:
            print(f"  Seed {seed}: AUPRC = {np.mean(valid):.4f} ± {np.std(valid):.4f} "
                  f"(median={np.median(valid):.4f}, {len(valid)}/{len(folds)} folds)")

    # Cross-seed aggregation
    all_auprcs = []
    for seed in seeds:
        for f in folds:
            if f in all_results[seed]:
                a = all_results[seed][f]["test_metrics"]["auprc"]
                if not np.isnan(a):
                    all_auprcs.append(a)

    if all_auprcs:
        print(f"\n  Overall: AUPRC = {np.mean(all_auprcs):.4f} ± {np.std(all_auprcs):.4f} "
              f"(N={len(all_auprcs)} fold-seed pairs)")

    # Save aggregated results
    summary = {
        "experiment": experiment,
        "mode": "lopo",
        "augmentation": augmentation,
        "seeds": seeds,
        "folds": folds,
        "all_results": all_results,
        "summary": {
            "overall_auprc_mean": float(np.mean(all_auprcs)) if all_auprcs else float("nan"),
            "overall_auprc_std": float(np.std(all_auprcs)) if all_auprcs else float("nan"),
            "n_fold_seed_pairs": len(all_auprcs),
        },
    }

    summary_dir = RESULTS_DIR / experiment
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "lopo_summary.json"

    # Merge with existing summary (allows split-machine runs)
    if summary_path.exists():
        try:
            with open(summary_path) as _f:
                existing = json.load(_f)
            for s_key, s_val in existing.get("all_results", {}).items():
                s_key_int = int(s_key) if isinstance(s_key, str) and s_key.isdigit() else s_key
                if s_key_int not in summary["all_results"]:
                    summary["all_results"][s_key_int] = s_val
                else:
                    for f_key, f_val in s_val.items():
                        f_key_int = int(f_key) if isinstance(f_key, str) and f_key.isdigit() else f_key
                        if f_key_int not in summary["all_results"][s_key_int]:
                            summary["all_results"][s_key_int][f_key_int] = f_val
        except (json.JSONDecodeError, KeyError):
            pass

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train seizure detection experiments (E1/E2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # E1 baseline — single-split (quick development)
  python train.py --experiment e1 --mode single

  # E1 baseline — full LOPO (final results)
  python train.py --experiment e1 --mode lopo

  # E2b: SMOTE augmentation — single-split
  python train.py --experiment e2 --augmentation smote --mode single

  # E2c: ADASYN augmentation — single-split
  python train.py --experiment e2 --augmentation adasyn --mode single

  # Specific seeds and folds
  python train.py --experiment e1 --mode lopo --seeds 42 123 --folds 0 1 2
        """,
    )
    parser.add_argument("--experiment", type=str, default="e1",
                        choices=["e1", "e2", "e3", "e4", "e5"],
                        help="Experiment label (default: e1)")
    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "lopo"],
                        help="Evaluation mode (default: single)")
    parser.add_argument("--augmentation", type=str, default=None,
                        choices=["smote", "adasyn"],
                        help="Oversampling method for E2 (default: None)")
    parser.add_argument("--synthetic-windows", type=str, default=None,
                        help="Path to synthetic windows .npz file (for E3-E5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456],
                        help="Random seeds (default: 42 123 456)")
    parser.add_argument("--folds", type=int, nargs="+", default=None,
                        help="Specific LOPO folds to run (default: all 23)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")

    args = parser.parse_args()

    # Load synthetic windows if provided (E3-E5)
    synthetic_windows = None
    if args.synthetic_windows:
        from training.generate import load_synthetic_windows
        print(f"Loading synthetic windows from {args.synthetic_windows}...")
        synthetic_windows = load_synthetic_windows(args.synthetic_windows)
        print(f"  Loaded {len(synthetic_windows)} synthetic windows")

    if args.mode == "single":
        for seed in args.seeds:
            train_single_split(
                experiment=args.experiment,
                augmentation=args.augmentation,
                synthetic_windows=synthetic_windows,
                seed=seed,
                device=args.device,
            )
    else:
        # For LOPO, wrap the static synthetic windows in a callable
        # that returns the same windows for every fold.
        # NOTE: For proper per-fold generator training (where each fold needs
        # its own synthetic data), use the Python API with a custom
        # synthetic_windows_fn(fold, seed) callable instead of the CLI.
        syn_fn = None
        if synthetic_windows is not None:
            syn_fn = lambda fold, seed: synthetic_windows  # noqa: E731
            print("  Note: same synthetic windows will be used for all LOPO folds.")
            print("  For per-fold generation, use the Python API with synthetic_windows_fn.")

        train_lopo(
            experiment=args.experiment,
            augmentation=args.augmentation,
            synthetic_windows_fn=syn_fn,
            seeds=args.seeds,
            device=args.device,
            folds=args.folds,
        )

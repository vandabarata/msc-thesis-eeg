"""
generate.py — Generator training pipeline for E3, E4, E5.

Unified CLI to train any generator (TimeGAN, CVAE, LDM) on ictal windows
and produce synthetic data in the format expected by train.py.

Supports:
  - Training generators on single-split or specific LOPO folds
  - Generating synthetic windows at various ratios (25%, 50%, 100%, 200%)
  - Saving generator checkpoints for reproducibility
  - Plugging directly into train.py via saved .npy synthetic window files

Usage:
    # Train CVAE on single-split, generate at 100% ratio
    python -m training.generate --model cvae --mode single --ratio 100

    # Train TimeGAN on LOPO fold 0
    python -m training.generate --model timegan --mode lopo --folds 0

    # Train LDM (requires pretrained CVAE checkpoint)
    python -m training.generate --model ldm --cvae-checkpoint results/cvae/seed_42/single_split/cvae.pt
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from data.loader import (
    CHBMITDataset,
    ALL_CASES,
    N_CHANNELS,
    WINDOW_SAMPLES,
)


RESULTS_DIR = _PROJECT_ROOT / "results"


def extract_ictal_windows(
    dataset: CHBMITDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract all ictal windows from a dataset as numpy arrays.

    Returns:
        (windows, labels, patient_ids)
        windows: (N_ictal, 23, 1024)
        labels: (N_ictal,) — all 1s
        patient_ids: (N_ictal,)
    """
    windows = []
    labels = []
    pids = []
    for w, l, p in dataset.windows:
        if l == 1:
            windows.append(w)
            labels.append(l)
            pids.append(p)

    return (
        np.array(windows, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(pids, dtype=np.int64),
    )


def extract_all_windows(
    dataset: CHBMITDataset,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract all windows (ictal + interictal) from a dataset."""
    windows = np.array([w for w, _, _ in dataset.windows], dtype=np.float32)
    labels = np.array([l for _, l, _ in dataset.windows], dtype=np.int64)
    pids = np.array([p for _, _, p in dataset.windows], dtype=np.int64)
    return windows, labels, pids


def train_timegan(
    train_dataset: CHBMITDataset,
    seed: int = 42,
    device: str = "cpu",
    n_epochs: Tuple[int, int, int] = (600, 600, 600),
    verbose: bool = True,
) -> "TimeGAN":
    """Train TimeGAN on ictal windows from the training dataset."""
    from models.timegan import TimeGAN

    torch.manual_seed(seed)
    np.random.seed(seed)

    windows, labels, pids = extract_ictal_windows(train_dataset)
    if verbose:
        print(f"  Training TimeGAN on {len(windows)} ictal windows")

    model = TimeGAN(n_channels=N_CHANNELS, seq_len=WINDOW_SAMPLES)
    history = model.train_model(
        windows,
        n_epochs_ae=n_epochs[0],
        n_epochs_sup=n_epochs[1],
        n_epochs_joint=n_epochs[2],
        device=device,
        verbose=verbose,
    )
    return model


def train_cvae(
    train_dataset: CHBMITDataset,
    seed: int = 42,
    device: str = "cpu",
    n_epochs: int = 500,
    verbose: bool = True,
) -> "CVAE":
    """Train CVAE on all windows (ictal + interictal) from the training dataset."""
    from models.cvae import CVAE

    torch.manual_seed(seed)
    np.random.seed(seed)

    # CVAE trains on all windows (conditioned on label)
    windows, labels, pids = extract_all_windows(train_dataset)
    if verbose:
        n_ictal = (labels == 1).sum()
        print(f"  Training CVAE on {len(windows)} windows ({n_ictal} ictal)")

    model = CVAE(n_channels=N_CHANNELS, seq_len=WINDOW_SAMPLES, latent_dim=128)
    history = model.train_model(
        windows, labels,
        n_epochs=n_epochs,
        device=device,
        verbose=verbose,
    )
    return model


def train_ldm(
    train_dataset: CHBMITDataset,
    cvae_checkpoint: str,
    seed: int = 42,
    device: str = "cpu",
    n_epochs: int = 500,
    verbose: bool = True,
) -> "LatentDiffusion":
    """
    Train LDM using a pretrained CVAE checkpoint.

    Args:
        cvae_checkpoint: path to saved CVAE state dict (.pt file)
    """
    from models.cvae import CVAE
    from models.ldm import LatentDiffusion

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load pretrained CVAE
    cvae = CVAE(n_channels=N_CHANNELS, seq_len=WINDOW_SAMPLES, latent_dim=128)
    ckpt = torch.load(cvae_checkpoint, map_location="cpu", weights_only=True)
    cvae.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    if verbose:
        print(f"  Loaded pretrained CVAE from {cvae_checkpoint}")

    # Train on all windows
    windows, labels, pids = extract_all_windows(train_dataset)
    if verbose:
        n_ictal = (labels == 1).sum()
        print(f"  Training LDM on {len(windows)} windows ({n_ictal} ictal)")

    model = LatentDiffusion(cvae=cvae, latent_dim=128)
    history = model.train_model(
        windows, labels,
        n_epochs=n_epochs,
        device=device,
        verbose=verbose,
    )
    return model


def generate_synthetic(
    model,
    n_real_ictal: int,
    ratio: float = 1.0,
    device: str = "cpu",
    patient_id: int = -1,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Generate synthetic ictal windows at a given ratio.

    Args:
        model: trained generator (TimeGAN, CVAE, or LatentDiffusion)
        n_real_ictal: number of real ictal windows in training set
        ratio: synthetic:real ratio (0.25, 0.5, 1.0, 2.0)
        device: torch device
        patient_id: patient ID to assign to synthetic windows (-1 = synthetic sentinel)

    Returns:
        List of (window, label=1, patient_id) tuples
    """
    n_synthetic = max(1, int(n_real_ictal * ratio))
    print(f"  Generating {n_synthetic} synthetic windows (ratio={ratio}, real_ictal={n_real_ictal})")

    synthetic = model.generate(
        n_samples=n_synthetic,
        device=device,
        patient_id=patient_id,
    )
    return synthetic


def save_generator(model, save_dir: Path, model_name: str):
    """Save generator checkpoint."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{model_name}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
    }, path)
    print(f"  Saved {model_name} to {path}")


def save_synthetic_windows(
    synthetic: List[Tuple[np.ndarray, int, int]],
    save_dir: Path,
    ratio: float,
):
    """Save synthetic windows to disk as .npz for reuse."""
    save_dir.mkdir(parents=True, exist_ok=True)
    windows = np.array([w for w, _, _ in synthetic], dtype=np.float32)
    labels = np.array([l for _, l, _ in synthetic], dtype=np.int64)
    pids = np.array([p for _, _, p in synthetic], dtype=np.int64)

    path = save_dir / f"synthetic_ratio_{ratio:.2f}.npz"
    np.savez_compressed(path, windows=windows, labels=labels, patient_ids=pids)
    print(f"  Saved {len(synthetic)} synthetic windows to {path}")


def load_synthetic_windows(path: str) -> List[Tuple[np.ndarray, int, int]]:
    """Load synthetic windows from .npz file."""
    data = np.load(path)
    windows = data["windows"]
    labels = data["labels"]
    pids = data["patient_ids"]
    return [(windows[i], int(labels[i]), int(pids[i])) for i in range(len(windows))]


# ── CLI ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train generators and produce synthetic EEG windows.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train CVAE on single-split, generate at 100% ratio
  python -m training.generate --model cvae --ratio 1.0

  # Train TimeGAN
  python -m training.generate --model timegan --ratio 1.0

  # Train LDM (needs pretrained CVAE)
  python -m training.generate --model ldm --cvae-checkpoint results/e4/seed_42/single_split/cvae.pt

  # Generate at multiple ratios
  python -m training.generate --model cvae --ratio 0.25 0.5 1.0 2.0
        """,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["timegan", "cvae", "ldm"])
    parser.add_argument("--ratio", type=float, nargs="+", default=[1.0],
                        help="Synthetic:real ratios (default: 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cvae-checkpoint", type=str, default=None,
                        help="Path to pretrained CVAE checkpoint (required for LDM)")
    parser.add_argument("--n-epochs", type=int, default=None,
                        help="Override default epoch count")

    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Map model → experiment
    exp_map = {"timegan": "e3", "cvae": "e4", "ldm": "e5"}
    experiment = exp_map[args.model]

    print(f"\n{'=' * 60}")
    print(f"  Generator: {args.model.upper()}")
    print(f"  Experiment: {experiment.upper()}")
    print(f"  Ratios: {args.ratio}")
    print(f"  Seed: {args.seed}")
    print(f"  Device: {device}")
    print(f"{'=' * 60}")

    # Load training data
    train_ds = CHBMITDataset(split="train", seed=args.seed)
    n_ictal = train_ds._n_ictal
    print(f"\n  Training set: {len(train_ds)} windows ({n_ictal} ictal)")

    # Train generator
    save_dir = RESULTS_DIR / experiment / f"seed_{args.seed}" / "single_split"

    if args.model == "timegan":
        epochs = (args.n_epochs or 600, args.n_epochs or 600, args.n_epochs or 600)
        model = train_timegan(train_ds, args.seed, device, epochs)
    elif args.model == "cvae":
        model = train_cvae(train_ds, args.seed, device, args.n_epochs or 500)
    elif args.model == "ldm":
        if not args.cvae_checkpoint:
            parser.error("--cvae-checkpoint is required for LDM training")
        model = train_ldm(train_ds, args.cvae_checkpoint, args.seed, device, args.n_epochs or 500)

    save_generator(model, save_dir, args.model)

    # Generate at each ratio
    for ratio in args.ratio:
        synthetic = generate_synthetic(model, n_ictal, ratio, device)
        save_synthetic_windows(synthetic, save_dir, ratio)

    print(f"\nDone. Outputs in {save_dir}/")

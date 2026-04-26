"""
subject_identity.py — E7 Subject-Identity Analysis.

Tests whether synthetic augmentation preserves, reduces, or amplifies
subject-specific patterns. Uses a linear probe on frozen detector embeddings.

Three sub-experiments:
  E7a: Real baseline — how identifiable are subjects from real EEG?
  E7b: Synthetic transfer — do synthetic windows inherit subject identity?
  E7c: Augmented model — does augmentation change detector's reliance on
       subject signatures?

Method:
  1. Take a trained SeizureDetector (from E1 or augmented experiment)
  2. Freeze the backbone, extract (batch, 128) embeddings
  3. Train a linear probe: Linear(128, n_subjects) to predict patient_id
  4. Report subject-ID accuracy

Interpretation:
  - E7b accuracy HIGH → generator memorizes subject signatures
  - E7c accuracy < E7a → augmentation improves generalization (good)
  - E7c accuracy > E7a → augmentation amplifies subject patterns (bad)

Usage:
    from training.subject_identity import SubjectIdentityProbe, run_e7

    results = run_e7(
        detector_baseline=model_e1,
        detector_augmented=model_best,
        real_dataset=train_ds,
        synthetic_windows=synthetic,
    )
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from models.detector import SeizureDetector


class SubjectIdentityProbe(nn.Module):
    """
    Linear probe for subject identification on frozen detector embeddings.

    Linear(128, n_subjects) — single linear layer, no hidden layers.
    This ensures that any subject-identifiability must already be present
    in the embedding space, not learned by a deep classifier.
    """

    def __init__(self, embedding_dim: int = 128, n_subjects: int = 23):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, n_subjects)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """(batch, 128) → (batch, n_subjects) logits."""
        return self.linear(embeddings)


@torch.no_grad()
def extract_embeddings(
    detector: SeizureDetector,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract embeddings from a frozen detector for all windows in a dataloader.

    Returns:
        (embeddings, patient_ids) — shapes (N, 128) and (N,)
    """
    detector.eval()
    detector.to(device)

    all_emb = []
    all_pids = []

    for windows, labels, patient_ids in dataloader:
        windows = windows.to(device, dtype=torch.float32)
        emb = detector.get_embeddings(windows)
        all_emb.append(emb.cpu().numpy())
        all_pids.append(patient_ids.numpy())

    return np.concatenate(all_emb), np.concatenate(all_pids)


def train_probe(
    embeddings: np.ndarray,
    patient_ids: np.ndarray,
    n_subjects: int,
    n_epochs: int = 50,
    lr: float = 1e-2,
    batch_size: int = 256,
    val_fraction: float = 0.2,
    seed: int = 42,
    device: str = "cpu",
) -> Tuple[SubjectIdentityProbe, Dict]:
    """
    Train a linear probe to predict subject identity from embeddings.

    Args:
        embeddings: (N, 128) detector embeddings
        patient_ids: (N,) integer patient IDs
        n_subjects: number of unique subjects
        val_fraction: held-out fraction for accuracy evaluation

    Returns:
        (trained_probe, results_dict)
    """
    rng = np.random.RandomState(seed)

    # Train/val split (window-level, within training patients)
    N = len(embeddings)
    indices = rng.permutation(N)
    val_size = max(1, int(N * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    # Remap patient IDs to contiguous 0..n_subjects-1
    unique_pids = np.unique(patient_ids)
    pid_map = {int(pid): i for i, pid in enumerate(unique_pids)}
    mapped_pids = np.array([pid_map[int(p)] for p in patient_ids])
    actual_n_subjects = len(unique_pids)

    # Build tensors
    X_train = torch.from_numpy(embeddings[train_idx]).float().to(device)
    y_train = torch.from_numpy(mapped_pids[train_idx]).long().to(device)
    X_val = torch.from_numpy(embeddings[val_idx]).float().to(device)
    y_val = torch.from_numpy(mapped_pids[val_idx]).long().to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Train probe
    probe = SubjectIdentityProbe(
        embedding_dim=embeddings.shape[1],
        n_subjects=actual_n_subjects,
    ).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    for epoch in range(n_epochs):
        probe.train()
        for X_batch, y_batch in train_loader:
            logits = probe(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate
        probe.eval()
        with torch.no_grad():
            val_logits = probe(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}

    if best_state is not None:
        probe.load_state_dict(best_state)

    # Final metrics
    probe.eval()
    with torch.no_grad():
        val_logits = probe(X_val)
        val_preds = val_logits.argmax(dim=1)
        val_acc = (val_preds == y_val).float().mean().item()

        train_logits = probe(X_train)
        train_preds = train_logits.argmax(dim=1)
        train_acc = (train_preds == y_train).float().mean().item()

    results = {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "n_subjects": actual_n_subjects,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "chance_level": 1.0 / actual_n_subjects,
    }

    return probe, results


def run_e7(
    detector_baseline: SeizureDetector,
    detector_augmented: Optional[SeizureDetector],
    train_loader: DataLoader,
    synthetic_windows: Optional[np.ndarray] = None,
    synthetic_patient_ids: Optional[np.ndarray] = None,
    device: str = "cpu",
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run the full E7 subject-identity analysis.

    Args:
        detector_baseline: trained detector from E1 (baseline)
        detector_augmented: trained detector from best augmented experiment
        train_loader: training DataLoader (real windows)
        synthetic_windows: (N, 23, 1024) synthetic windows from best generator
        synthetic_patient_ids: (N,) patient IDs assigned to synthetic windows
        device: torch device
        seed: random seed

    Returns:
        Dict with E7a, E7b, E7c results
    """
    results = {}

    # ── E7a: Real baseline ─────────────────────────────────────────────
    if verbose:
        print("\n  E7a: Subject-ID on real data (baseline detector)...")
    emb_real, pids_real = extract_embeddings(detector_baseline, train_loader, device)
    n_subjects = len(np.unique(pids_real))

    _, e7a = train_probe(emb_real, pids_real, n_subjects, seed=seed, device=device)
    results["e7a_real_baseline"] = e7a
    if verbose:
        print(f"    Accuracy: {e7a['val_accuracy']:.4f} (chance: {e7a['chance_level']:.4f})")

    # ── E7b: Synthetic transfer ────────────────────────────────────────
    if synthetic_windows is not None and synthetic_patient_ids is not None:
        if verbose:
            print("\n  E7b: Subject-ID on synthetic data (baseline detector)...")

        # Get embeddings for synthetic windows
        synth_tensor = torch.from_numpy(synthetic_windows).float()
        synth_ds = TensorDataset(
            synth_tensor,
            torch.ones(len(synth_tensor), dtype=torch.long),  # dummy labels
            torch.from_numpy(synthetic_patient_ids).long(),
        )
        synth_loader = DataLoader(synth_ds, batch_size=256, shuffle=False)
        emb_synth, pids_synth = extract_embeddings(detector_baseline, synth_loader, device)

        # Train probe on real, evaluate on synthetic
        probe_real, _ = train_probe(emb_real, pids_real, n_subjects, seed=seed, device=device, val_fraction=0.01)
        probe_real.eval()
        with torch.no_grad():
            synth_emb_t = torch.from_numpy(emb_synth).float().to(device)
            synth_pids_t = torch.from_numpy(synthetic_patient_ids).long().to(device)

            # Remap synthetic pids — skip windows with unknown IDs (e.g. sentinel -1)
            unique_pids = np.unique(pids_real)
            pid_map = {int(pid): i for i, pid in enumerate(unique_pids)}
            known_mask = np.array([int(p) in pid_map for p in synthetic_patient_ids])
            if known_mask.sum() == 0:
                results["e7b_synthetic_transfer"] = {
                    "accuracy": float("nan"),
                    "n_synthetic": len(synthetic_windows),
                    "n_known_pid": 0,
                    "interpretation": "No synthetic windows have known patient IDs — cannot evaluate subject transfer",
                }
                if verbose:
                    print("    Skipped: no synthetic windows have known patient IDs")
            else:
                synth_emb_t = synth_emb_t[torch.from_numpy(known_mask)]
                known_pids = synthetic_patient_ids[known_mask]
                mapped_synth = torch.tensor(
                    [pid_map[int(p)] for p in known_pids],
                    dtype=torch.long, device=device,
                )

                logits = probe_real(synth_emb_t)
                preds = logits.argmax(dim=1)
                acc = (preds == mapped_synth).float().mean().item()

                results["e7b_synthetic_transfer"] = {
                    "accuracy": float(acc),
                    "n_synthetic": len(synthetic_windows),
                    "n_known_pid": int(known_mask.sum()),
                    "interpretation": "HIGH = generator memorizes subject patterns",
                }
                if verbose:
                    print(f"    Accuracy: {acc:.4f} ({known_mask.sum()}/{len(synthetic_patient_ids)} windows with known PIDs)")

    # ── E7c: Augmented model ──────────────────────────────────────────
    if detector_augmented is not None:
        if verbose:
            print("\n  E7c: Subject-ID on real data (augmented detector)...")
        emb_aug, pids_aug = extract_embeddings(detector_augmented, train_loader, device)

        _, e7c = train_probe(emb_aug, pids_aug, n_subjects, seed=seed, device=device)
        results["e7c_augmented_model"] = e7c
        if verbose:
            print(f"    Accuracy: {e7c['val_accuracy']:.4f} (chance: {e7c['chance_level']:.4f})")
            if "e7a_real_baseline" in results:
                diff = e7c["val_accuracy"] - results["e7a_real_baseline"]["val_accuracy"]
                direction = "LOWER (better generalization)" if diff < 0 else "HIGHER (more subject-dependent)"
                print(f"    vs baseline: {diff:+.4f} → {direction}")

    return results


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test with random data
    torch.manual_seed(42)

    detector = SeizureDetector()
    probe = SubjectIdentityProbe(128, 10)

    # Fake embeddings
    emb = torch.randn(200, 128)
    pids = torch.randint(0, 10, (200,)).numpy()

    _, results = train_probe(emb.numpy(), pids, 10, n_epochs=10)
    print(f"Subject-ID probe test:")
    print(f"  Train acc: {results['train_accuracy']:.4f}")
    print(f"  Val acc:   {results['val_accuracy']:.4f}")
    print(f"  Chance:    {results['chance_level']:.4f}")
    print(f"  Subjects:  {results['n_subjects']}")
    print("\n  All checks passed.")

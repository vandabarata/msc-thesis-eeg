"""
detector.py — Frozen 1D-CNN seizure detector for all experiments (E1–E5).

Architecture:
    Input: (batch, 23, 1024)
    → Conv1d(23, 32, kernel=7, stride=2) + BN + ReLU + Dropout(0.3)
    → Conv1d(32, 64, kernel=5, stride=2) + BN + ReLU + Dropout(0.3)
    → Conv1d(64, 128, kernel=3, stride=2) + BN + ReLU + Dropout(0.3)
    → AdaptiveAvgPool1d(1) → Flatten
    → Linear(128, 64) + ReLU + Dropout(0.5)
    → Linear(64, 2)

This architecture is intentionally simple:
  - Deep enough to learn temporal patterns in 4s EEG windows
  - Simple enough to train quickly across 23 LOPO folds × 3 seeds
  - No attention layers or recurrence
  - Serves as a controlled variable, not a confound

The same architecture and hyperparameters are used identically across all
experiments so that downstream performance differences can only be attributed
to the training data, not the detector architecture.

Training hyperparameters (Section 4.4):
  - Optimizer: Adam, lr=1e-3
  - Batch size: 64
  - Early stopping on validation AUPRC (patience=10; Prechelt 1998)
  - Max 100 epochs
  - Loss: Class-weighted cross-entropy (weights inversely proportional
    to class frequency in training set; Zhao et al. 2022)

Usage:
    from models.detector import SeizureDetector

    model = SeizureDetector()
    logits = model(x)  # x: (batch, 23, 1024) → logits: (batch, 2)

    # For embeddings (E7 subject-identity analysis):
    embeddings = model.get_embeddings(x)  # → (batch, 128)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeizureDetector(nn.Module):
    """
    1D-CNN seizure detector — frozen architecture for E1–E5.

    Input:  (batch, 23, 1024) — 23 EEG channels, 4s window at 256 Hz
    Output: (batch, 2) — logits for [interictal, ictal]
    """

    def __init__(
        self,
        n_channels: int = 23,
        n_classes: int = 2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # ── Convolutional backbone ─────────────────────────────────────
        # Block 1: (batch, 23, 1024) → (batch, 32, 512)
        self.conv1 = nn.Conv1d(n_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.3)

        # Block 2: (batch, 32, 512) → (batch, 64, 256)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)

        # Block 3: (batch, 64, 256) → (batch, 128, 128)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(0.3)

        # ── Pooling → Flatten ──────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool1d(1)  # (batch, 128, 1)

        # ── Classification head ────────────────────────────────────────
        self.fc1 = nn.Linear(128, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, n_classes)

        self.relu = nn.ReLU(inplace=True)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run convolutional backbone + pooling, return (batch, 128) embeddings."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.drop3(x)

        # Pool → flatten
        x = self.pool(x)       # (batch, 128, 1)
        x = x.squeeze(-1)      # (batch, 128)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: input → logits.

        Args:
            x: (batch, 23, 1024) EEG windows

        Returns:
            (batch, 2) logits for [interictal, ictal]
        """
        x = self._backbone(x)          # (batch, 128)
        x = self.fc1(x)                # (batch, 64)
        x = self.relu(x)
        x = self.drop_fc(x)
        x = self.fc2(x)                # (batch, 2)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from the backbone (before classification head).
        Used for E7 subject-identity analysis (linear probe on frozen embeddings).

        Args:
            x: (batch, 23, 1024) EEG windows

        Returns:
            (batch, 128) embeddings from AdaptiveAvgPool1d output
        """
        return self._backbone(x)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    model = SeizureDetector()
    print(f"SeizureDetector architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(4, 23, 1024)
    logits = model(x)
    print(f"\nInput shape:      {x.shape}")
    print(f"Output shape:     {logits.shape}")
    print(f"Output (logits):  {logits.detach()}")

    # Test embeddings
    emb = model.get_embeddings(x)
    print(f"Embedding shape:  {emb.shape}")

    # Verify output shapes
    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"
    assert emb.shape == (4, 128), f"Expected (4, 128), got {emb.shape}"
    print("\nAll checks passed.")

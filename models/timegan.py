"""
timegan.py — TimeGAN for synthetic EEG generation (E3).

Implements the TimeGAN framework from Yoon et al. (2019):
  "Time-series Generative Adversarial Networks" (NeurIPS 2019)

TimeGAN has 5 sub-networks, all GRU-based:
  1. Embedder:    Real space → latent space
  2. Recovery:    Latent space → real space (reconstruction)
  3. Supervisor:  Latent(t) → Latent(t+1) (temporal dynamics in latent space)
  4. Generator:   Random noise → latent space
  5. Discriminator: Latent sequence → real/fake score

Training has 3 phases:
  Phase 1: Embedding (autoencoder: embedder + recovery)
  Phase 2: Supervised (supervisor learns temporal dynamics on real latent codes)
  Phase 3: Joint (generator + discriminator + supervisor + embedder, adversarial)

For EEG:
  - Input: (batch, 23, 1024) — 23 channels, 1024 timesteps
  - We treat each window as a multivariate time series of length T
  - To keep GRU tractable, we downsample the temporal axis by reshaping
    1024 samples into T=64 steps of 16-sample segments (or use strided input)
  - Hidden dim: 128 (matches detector embedding size)
  - Latent dim: 128

Reference:
  Yoon, J., Jarrett, D., & van der Schaar, M. (2019).
  Time-series Generative Adversarial Networks. NeurIPS 2019.

Usage:
    from models.timegan import TimeGAN

    model = TimeGAN(n_channels=23, seq_len=1024)
    model.train_model(real_windows, n_epochs=2000)
    synthetic = model.generate(n_samples=500)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ── Sub-networks ───────────────────────────────────────────────────────────

class Embedder(nn.Module):
    """Maps real-space features to latent space. GRU-based."""

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            input_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, T, input_dim) → (batch, T, hidden_dim)"""
        h, _ = self.rnn(x)
        return self.activation(self.fc(h))


class Recovery(nn.Module):
    """Maps latent space back to real space. GRU-based."""

    def __init__(self, hidden_dim: int, output_dim: int, n_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, T, hidden_dim) → (batch, T, output_dim)"""
        out, _ = self.rnn(h)
        return self.fc(out)


class Supervisor(nn.Module):
    """Learns temporal dynamics: predicts next latent step from current."""

    def __init__(self, hidden_dim: int, n_layers: int = 2):
        super().__init__()
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, T, hidden_dim) → (batch, T, hidden_dim)"""
        out, _ = self.rnn(h)
        return self.activation(self.fc(out))


class Generator(nn.Module):
    """Maps random noise to latent space. GRU-based."""

    def __init__(self, noise_dim: int, hidden_dim: int, n_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            noise_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """(batch, T, noise_dim) → (batch, T, hidden_dim)"""
        h, _ = self.rnn(z)
        return self.activation(self.fc(h))


class Discriminator(nn.Module):
    """Classifies latent sequences as real or fake. GRU-based."""

    def __init__(self, hidden_dim: int, n_layers: int = 3):
        super().__init__()
        self.rnn = nn.GRU(
            hidden_dim, hidden_dim, num_layers=n_layers,
            batch_first=True, dropout=0.1 if n_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, T, hidden_dim) → (batch, T, 1)"""
        out, _ = self.rnn(h)
        return self.fc(out)


# ── TimeGAN ────────────────────────────────────────────────────────────────

class TimeGAN(nn.Module):
    """
    TimeGAN (Yoon et al. 2019) adapted for multichannel EEG.

    Input EEG windows are (batch, n_channels, seq_len) = (batch, 23, 1024).
    Internally reshaped to (batch, T, feature_dim) for GRU processing:
      - T = seq_len // segment_size = 1024 // 16 = 64 timesteps
      - feature_dim = n_channels * segment_size = 23 * 16 = 368

    Args:
        n_channels: number of EEG channels (23)
        seq_len: samples per window (1024)
        hidden_dim: latent/hidden dimension for all sub-networks (128)
        noise_dim: dimension of input noise for generator (128)
        segment_size: samples per GRU timestep (16 → T=64 steps)
        n_layers: GRU layers per sub-network
    """

    def __init__(
        self,
        n_channels: int = 23,
        seq_len: int = 1024,
        hidden_dim: int = 128,
        noise_dim: int = 128,
        segment_size: int = 16,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim
        self.segment_size = segment_size

        # Derived dimensions
        self.T = seq_len // segment_size         # 64 timesteps
        self.feature_dim = n_channels * segment_size  # 368

        # Sub-networks
        self.embedder = Embedder(self.feature_dim, hidden_dim, n_layers)
        self.recovery = Recovery(hidden_dim, self.feature_dim, n_layers)
        self.supervisor = Supervisor(hidden_dim, n_layers=max(n_layers - 1, 1))
        self.generator = Generator(noise_dim, hidden_dim, n_layers)
        self.discriminator = Discriminator(hidden_dim, n_layers)

    def _reshape_to_seq(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, n_channels, seq_len) → (batch, T, feature_dim)"""
        batch = x.shape[0]
        # (batch, 23, 1024) → (batch, 23, 64, 16) → (batch, 64, 23, 16) → (batch, 64, 368)
        x = x.reshape(batch, self.n_channels, self.T, self.segment_size)
        x = x.permute(0, 2, 1, 3)  # (batch, T, n_channels, segment_size)
        x = x.reshape(batch, self.T, self.feature_dim)
        return x

    def _reshape_to_eeg(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, T, feature_dim) → (batch, n_channels, seq_len)"""
        batch = x.shape[0]
        # (batch, 64, 368) → (batch, 64, 23, 16) → (batch, 23, 64, 16) → (batch, 23, 1024)
        x = x.reshape(batch, self.T, self.n_channels, self.segment_size)
        x = x.permute(0, 2, 1, 3)  # (batch, n_channels, T, segment_size)
        x = x.reshape(batch, self.n_channels, self.seq_len)
        return x

    def _autoencoder_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """Embed → recover (for Phase 1 training)."""
        h = self.embedder(x_seq)
        x_hat = self.recovery(h)
        return x_hat

    def _supervisor_forward(self, x_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed → supervise (for Phase 2 training). Returns (h_real, h_supervised)."""
        h = self.embedder(x_seq)
        h_sup = self.supervisor(h)
        return h, h_sup

    def _generator_forward(
        self, z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full generator path (for Phase 3 joint training).
        Returns: (x_hat, h_fake, h_fake_sup)
        """
        h_fake = self.generator(z)
        h_fake_sup = self.supervisor(h_fake)
        x_hat = self.recovery(h_fake_sup)
        return x_hat, h_fake, h_fake_sup

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        device: str = "cpu",
        patient_id: int = 0,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate synthetic EEG windows.

        Args:
            n_samples: number of windows to generate
            device: torch device
            patient_id: patient ID to assign to synthetic windows

        Returns:
            List of (window, label=1, patient_id) tuples
            window shape: (23, 1024), dtype float32
        """
        self.eval()
        synthetic = []
        batch_size = min(64, n_samples)

        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            z = torch.randn(n, self.T, self.noise_dim, device=device)

            h_fake = self.generator(z)
            h_fake_sup = self.supervisor(h_fake)
            x_fake_seq = self.recovery(h_fake_sup)
            x_fake = self._reshape_to_eeg(x_fake_seq)

            for i in range(n):
                window = x_fake[i].cpu().numpy().astype(np.float32)
                synthetic.append((window, 1, patient_id))

        return synthetic

    def train_model(
        self,
        real_windows: np.ndarray,
        n_epochs_ae: int = 600,
        n_epochs_sup: int = 600,
        n_epochs_joint: int = 600,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict:
        """
        Train all 3 phases of TimeGAN.

        Args:
            real_windows: (N, 23, 1024) array of real EEG windows
            n_epochs_ae: epochs for Phase 1 (autoencoder)
            n_epochs_sup: epochs for Phase 2 (supervisor)
            n_epochs_joint: epochs for Phase 3 (joint adversarial)
            batch_size: training batch size
            lr: learning rate for all optimizers
            device: torch device
            verbose: print progress

        Returns:
            Dict with training losses per phase
        """
        self.to(device)
        N = len(real_windows)

        # Reshape all real data to sequence format
        real_tensor = torch.from_numpy(real_windows).float().to(device)
        real_seq = self._reshape_to_seq(real_tensor)  # (N, T, feature_dim)

        history = {"phase1_loss": [], "phase2_loss": [], "phase3_g_loss": [], "phase3_d_loss": []}

        # ── Phase 1: Autoencoder (Embedder + Recovery) ─────────────────
        if verbose:
            print(f"  Phase 1: Autoencoder ({n_epochs_ae} epochs)")

        opt_ae = torch.optim.Adam(
            list(self.embedder.parameters()) + list(self.recovery.parameters()),
            lr=lr,
        )

        for epoch in range(n_epochs_ae):
            self.train()
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                x = real_seq[idx]

                x_hat = self._autoencoder_forward(x)
                loss = nn.functional.mse_loss(x_hat, x)

                opt_ae.zero_grad()
                loss.backward()
                opt_ae.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            history["phase1_loss"].append(avg)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs_ae}: recon_loss={avg:.6f}")

        # ── Phase 2: Supervisor ────────────────────────────────────────
        if verbose:
            print(f"  Phase 2: Supervisor ({n_epochs_sup} epochs)")

        opt_sup = torch.optim.Adam(self.supervisor.parameters(), lr=lr)

        for epoch in range(n_epochs_sup):
            self.train()
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                x = real_seq[idx]

                h_real, h_sup = self._supervisor_forward(x)
                # Supervisor should predict next step: h_sup[:, :-1] ≈ h_real[:, 1:]
                loss = nn.functional.mse_loss(h_sup[:, :-1, :], h_real[:, 1:, :].detach())

                opt_sup.zero_grad()
                loss.backward()
                opt_sup.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            history["phase2_loss"].append(avg)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs_sup}: supervisor_loss={avg:.6f}")

        # ── Phase 3: Joint Training ────────────────────────────────────
        if verbose:
            print(f"  Phase 3: Joint adversarial ({n_epochs_joint} epochs)")

        opt_g = torch.optim.Adam(
            list(self.generator.parameters()) +
            list(self.supervisor.parameters()) +
            list(self.embedder.parameters()) +
            list(self.recovery.parameters()),
            lr=lr,
        )
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        for epoch in range(n_epochs_joint):
            self.train()
            perm = torch.randperm(N)
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                x = real_seq[idx]
                b = x.shape[0]

                # Generate fake
                z = torch.randn(b, self.T, self.noise_dim, device=device)
                h_real = self.embedder(x)
                h_fake = self.generator(z)
                h_fake_sup = self.supervisor(h_fake)
                h_real_sup = self.supervisor(h_real)

                # ── Discriminator step ─────────────────────────────────
                d_real = self.discriminator(h_real.detach())
                d_fake = self.discriminator(h_fake_sup.detach())

                d_loss_real = nn.functional.binary_cross_entropy_with_logits(
                    d_real, torch.ones_like(d_real),
                )
                d_loss_fake = nn.functional.binary_cross_entropy_with_logits(
                    d_fake, torch.zeros_like(d_fake),
                )
                d_loss = d_loss_real + d_loss_fake

                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

                # ── Generator step (2 updates per D update) ────────────
                for _ in range(2):
                    z = torch.randn(b, self.T, self.noise_dim, device=device)
                    h_real = self.embedder(x)
                    h_fake = self.generator(z)
                    h_fake_sup = self.supervisor(h_fake)
                    h_real_sup = self.supervisor(h_real)

                    # Adversarial loss (fool discriminator)
                    d_fake = self.discriminator(h_fake_sup)
                    g_loss_adv = nn.functional.binary_cross_entropy_with_logits(
                        d_fake, torch.ones_like(d_fake),
                    )

                    # Supervised loss (temporal consistency)
                    g_loss_sup = nn.functional.mse_loss(
                        h_fake_sup[:, :-1, :], h_fake[:, 1:, :],
                    )

                    # Moment matching loss (mean + variance)
                    real_mean = h_real.mean(dim=0)
                    fake_mean = h_fake_sup.mean(dim=0)
                    real_var = h_real.var(dim=0)
                    fake_var = h_fake_sup.var(dim=0)
                    g_loss_moment = (
                        nn.functional.mse_loss(fake_mean, real_mean) +
                        nn.functional.mse_loss(fake_var, real_var)
                    )

                    # Reconstruction loss (autoencoder integrity)
                    x_hat = self.recovery(h_real)
                    g_loss_recon = nn.functional.mse_loss(x_hat, x)

                    # Total generator loss
                    g_loss = (
                        g_loss_adv +
                        10.0 * g_loss_sup +
                        10.0 * g_loss_moment +
                        10.0 * g_loss_recon
                    )

                    opt_g.zero_grad()
                    g_loss.backward()
                    opt_g.step()

                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                n_batches += 1

            avg_g = epoch_g_loss / max(n_batches, 1)
            avg_d = epoch_d_loss / max(n_batches, 1)
            history["phase3_g_loss"].append(avg_g)
            history["phase3_d_loss"].append(avg_d)
            if verbose and (epoch + 1) % 100 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs_joint}: g_loss={avg_g:.4f}, d_loss={avg_d:.4f}")

        return history


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    model = TimeGAN(n_channels=23, seq_len=1024)
    total = sum(p.numel() for p in model.parameters())
    print(f"TimeGAN: {total:,} parameters")
    print(f"  Embedder:      {sum(p.numel() for p in model.embedder.parameters()):,}")
    print(f"  Recovery:      {sum(p.numel() for p in model.recovery.parameters()):,}")
    print(f"  Supervisor:    {sum(p.numel() for p in model.supervisor.parameters()):,}")
    print(f"  Generator:     {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"  Discriminator: {sum(p.numel() for p in model.discriminator.parameters()):,}")

    # Test generation (without training)
    synthetic = model.generate(n_samples=4)
    w, l, p = synthetic[0]
    print(f"\n  Generated {len(synthetic)} windows")
    print(f"  Shape: {w.shape}, label: {l}, patient_id: {p}")
    assert w.shape == (23, 1024)
    print("\n  All checks passed.")

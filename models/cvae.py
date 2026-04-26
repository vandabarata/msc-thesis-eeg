"""
cvae.py — Conditional Variational Autoencoder for synthetic EEG generation (E4).

Architecture:
  - 1D-Conv encoder → 128-dim latent → 1D-Conv decoder
  - Conditioned on class label (broadcast as extra channel)
  - Optional patient conditioning (learned embedding, extra channel)
  - Loss: reconstruction (MSE) + KL divergence (with beta warmup)

Conditioning approach:
  The encoder receives conditioning as extra channels broadcast across the
  temporal dimension. The decoder receives conditioning concatenated to the
  latent vector. This is the standard CVAE approach (Sohn et al. 2015).

  Class label (always): 1 channel (encoder), 1 scalar (decoder)
  Patient ID (optional): 1 channel (encoder), patient_embed_dim scalars (decoder)

  Patient conditioning is off by default — the primary E4 experiments use
  class-only conditioning. Patient conditioning is enabled for the E6
  patient-specific vs pooled comparison (Carrle 2023; You 2025).

Latent dim: 128.

KL annealing: beta ramps from 0 to 1 over the first 50 epochs
  (Bowman et al. 2016; thesis preprocessing section specifies 50).

The CVAE encoder is reused by the LDM in Phase 4 — the encoder weights
are frozen and latent codes extracted for diffusion training.

Usage:
    from models.cvae import CVAE

    # Class-only conditioning (default, for E4)
    model = CVAE(n_channels=23, seq_len=1024, latent_dim=128)

    # With patient conditioning (for E6 comparison)
    model = CVAE(n_channels=23, latent_dim=128, n_patients=24)

    model.train_model(real_windows, labels, n_epochs=500)
    synthetic = model.generate(n_samples=500)

    # For LDM: extract latent codes
    mu, log_var = model.encode(windows, labels)
    z = model.reparameterize(mu, log_var)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """1D-Conv encoder: (batch, in_channels, seq_len) → (batch, latent_dim) mu/logvar."""

    def __init__(self, in_channels: int, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool1d(1),  # (batch, 512, 1)
        )

        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).squeeze(-1)  # (batch, 512)
        return self.fc_mu(h), self.fc_logvar(h)


class Decoder(nn.Module):
    """1D-Conv decoder: (batch, cond_input_dim) → (batch, n_channels, seq_len)."""

    def __init__(self, n_channels: int, cond_input_dim: int, seq_len: int = 1024):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len

        self.fc = nn.Linear(cond_input_dim, 512 * 8)

        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose1d(32, n_channels, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.reshape(-1, 512, 8)
        x = self.deconv(h)
        if x.shape[2] > self.seq_len:
            x = x[:, :, :self.seq_len]
        elif x.shape[2] < self.seq_len:
            x = F.pad(x, (0, self.seq_len - x.shape[2]))
        return x


class CVAE(nn.Module):
    """
    Conditional Variational Autoencoder for EEG generation.

    Args:
        n_channels: number of EEG channels (23)
        seq_len: samples per window (1024)
        latent_dim: latent space dimensionality (128)
        n_patients: number of patients for patient conditioning (None=disabled)
        patient_embed_dim: learned embedding size per patient (default 16)
    """

    def __init__(
        self,
        n_channels: int = 23,
        seq_len: int = 1024,
        latent_dim: int = 128,
        n_patients: Optional[int] = None,
        patient_embed_dim: int = 16,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.n_patients = n_patients
        self.patient_embed_dim = patient_embed_dim if n_patients else 0

        # Encoder input: n_channels + 1 (label) + 1 (patient channel, if enabled)
        enc_in = n_channels + 1 + (1 if n_patients else 0)
        self.encoder = Encoder(enc_in, latent_dim)

        # Decoder input: latent_dim + 1 (label) + patient_embed_dim (if enabled)
        dec_in = latent_dim + 1 + self.patient_embed_dim
        self.decoder = Decoder(n_channels, dec_in, seq_len)

        # Patient embedding (learned, if enabled)
        if n_patients:
            self.patient_embed = nn.Embedding(n_patients, patient_embed_dim)

    def _build_encoder_input(
        self, x: torch.Tensor, label: torch.Tensor,
        patient_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build conditioning channels and concatenate to EEG input."""
        batch = x.shape[0]
        # Label channel
        label_ch = label.float().reshape(batch, 1, 1).expand(batch, 1, self.seq_len)
        channels = [x, label_ch]

        # Patient channel (learned embedding broadcast across time)
        if self.n_patients and patient_id is not None:
            p_emb = self.patient_embed(patient_id)       # (batch, patient_embed_dim)
            # Use first dim as a single channel, broadcast
            p_ch = p_emb[:, 0:1].unsqueeze(-1).expand(batch, 1, self.seq_len)
            channels.append(p_ch)

        return torch.cat(channels, dim=1)

    def _build_decoder_input(
        self, z: torch.Tensor, label: torch.Tensor,
        patient_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Concatenate conditioning to latent vector."""
        parts = [z, label.float().reshape(-1, 1)]

        if self.n_patients and patient_id is not None:
            p_emb = self.patient_embed(patient_id)  # (batch, patient_embed_dim)
            parts.append(p_emb)
        elif self.patient_embed_dim > 0:
            # No patient_id given but model expects it — use zeros
            parts.append(torch.zeros(z.shape[0], self.patient_embed_dim, device=z.device))

        return torch.cat(parts, dim=1)

    def encode(
        self, x: torch.Tensor, label: torch.Tensor,
        patient_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        x_cond = self._build_encoder_input(x, label, patient_id)
        return self.encoder(x_cond)

    @staticmethod
    def reparameterize(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * eps."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(
        self, z: torch.Tensor, label: torch.Tensor,
        patient_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent code to EEG window."""
        z_cond = self._build_decoder_input(z, label, patient_id)
        return self.decoder(z_cond)

    def forward(
        self, x: torch.Tensor, label: torch.Tensor,
        patient_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode → reparameterize → decode."""
        mu, log_var = self.encode(x, label, patient_id)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, label, patient_id)
        return x_recon, mu, log_var

    @staticmethod
    def loss_function(
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
        beta: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ELBO loss = reconstruction + beta * KL divergence.

        Beta warmup prevents posterior collapse: start beta=0, linearly
        increase to 1 over the first 50 epochs (Bowman et al. 2016).
        """
        recon = F.mse_loss(x_recon, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        total = recon + beta * kl
        return total, recon, kl

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        device: str = "cpu",
        patient_id: int = 0,
        label: int = 1,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """Generate synthetic EEG windows by sampling from the prior."""
        self.eval()
        synthetic = []
        batch_size = min(64, n_samples)

        for start in range(0, n_samples, batch_size):
            n = min(batch_size, n_samples - start)
            z = torch.randn(n, self.latent_dim, device=device)
            labels = torch.full((n,), label, device=device, dtype=torch.long)

            pid = None
            if self.n_patients:
                pid = torch.full((n,), patient_id, device=device, dtype=torch.long)

            x_gen = self.decode(z, labels, pid)

            for i in range(n):
                window = x_gen[i].cpu().numpy().astype(np.float32)
                synthetic.append((window, label, patient_id))

        return synthetic

    def train_model(
        self,
        real_windows: np.ndarray,
        labels: np.ndarray,
        patient_ids: Optional[np.ndarray] = None,
        n_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-3,
        beta_warmup: int = 50,
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict:
        """
        Train the CVAE.

        Args:
            real_windows: (N, 23, 1024) array
            labels: (N,) array of 0/1 labels
            patient_ids: (N,) array of patient IDs (optional, for patient conditioning)
            n_epochs: training epochs
            batch_size: batch size
            lr: learning rate
            beta_warmup: epochs over which beta ramps 0→1 (default 50, Bowman 2016)
            device: torch device
            verbose: print progress
        """
        self.to(device)
        N = len(real_windows)

        # Keep data on CPU, move per-batch to GPU to avoid VRAM OOM
        x_cpu = torch.from_numpy(real_windows).float()
        y_cpu = torch.from_numpy(labels).long()
        p_cpu = None
        if patient_ids is not None and self.n_patients:
            p_cpu = torch.from_numpy(patient_ids).long()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history = {"total_loss": [], "recon_loss": [], "kl_loss": [], "beta": []}

        for epoch in range(n_epochs):
            self.train()
            perm = torch.randperm(N)
            epoch_total = epoch_recon = epoch_kl = 0.0
            n_batches = 0

            beta = min(1.0, epoch / max(beta_warmup, 1))

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                x = x_cpu[idx].to(device)
                y = y_cpu[idx].to(device)
                p = p_cpu[idx].to(device) if p_cpu is not None else None

                x_recon, mu, log_var = self.forward(x, y, p)
                total, recon, kl = self.loss_function(x, x_recon, mu, log_var, beta)

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                epoch_total += total.item()
                epoch_recon += recon.item()
                epoch_kl += kl.item()
                n_batches += 1

            avg_t = epoch_total / max(n_batches, 1)
            avg_r = epoch_recon / max(n_batches, 1)
            avg_k = epoch_kl / max(n_batches, 1)
            history["total_loss"].append(avg_t)
            history["recon_loss"].append(avg_r)
            history["kl_loss"].append(avg_k)
            history["beta"].append(beta)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_t:.4f} "
                      f"(recon={avg_r:.4f}, kl={avg_k:.4f}, beta={beta:.2f})")

        return history


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test class-only conditioning (default)
    model = CVAE(n_channels=23, seq_len=1024, latent_dim=128)
    total = sum(p.numel() for p in model.parameters())
    print(f"CVAE (class-only): {total:,} parameters")

    x = torch.randn(4, 23, 1024)
    y = torch.tensor([1, 0, 1, 0])
    x_recon, mu, log_var = model(x, y)
    assert x_recon.shape == (4, 23, 1024)
    assert mu.shape == (4, 128)
    print(f"  Forward OK: {x_recon.shape}, mu={mu.shape}")

    synthetic = model.generate(n_samples=4)
    assert synthetic[0][0].shape == (23, 1024)
    print(f"  Generate OK: {len(synthetic)} windows")

    # Test with patient conditioning
    model_p = CVAE(n_channels=23, latent_dim=128, n_patients=24)
    total_p = sum(p.numel() for p in model_p.parameters())
    print(f"\nCVAE (patient-conditioned): {total_p:,} parameters")

    pids = torch.tensor([0, 1, 2, 3])
    x_recon_p, mu_p, _ = model_p(x, y, pids)
    assert x_recon_p.shape == (4, 23, 1024)
    print(f"  Forward OK: {x_recon_p.shape}")

    synthetic_p = model_p.generate(n_samples=4, patient_id=5)
    assert synthetic_p[0][0].shape == (23, 1024)
    print(f"  Generate OK: {len(synthetic_p)} windows")

    print("\n  All checks passed.")

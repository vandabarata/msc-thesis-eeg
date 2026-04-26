"""
ldm.py — Latent Diffusion Model for synthetic EEG generation (E5).

Architecture (Rombach et al. 2022, adapted for 1D EEG):
  1. Freeze the CVAE encoder from E4
  2. Extract latent codes z for all ictal training windows
  3. Train a 1D UNet denoiser in latent space (DDPM)
  4. Sample: UNet denoises random noise → latent z → CVAE decoder → EEG

The denoiser operates on the 128-dim latent vector from the CVAE. Since a
128-dim vector has no spatial structure, we reshape it to (channels, length)
= (16, 8) to give the UNet a 1D "spatial" axis for convolutions.

UNet-1D Denoiser:
  - Input: (batch, 16, 8) noisy latent + sinusoidal timestep embedding
  - 3 down blocks: 16→32→64→128 channels
  - Bottleneck: 128 channels
  - 3 up blocks: 128→64→32→16 channels (with skip connections)
  - Output: (batch, 16, 8) predicted noise

Diffusion:
  - T=1000 timesteps, cosine noise schedule
  - Loss: MSE on predicted noise (simplified DDPM loss)
  - Sampling: DDIM (50 steps) for fast inference, DDPM (1000) for quality

Class conditioning: timestep embedding + label embedding added together.

Usage:
    from models.ldm import LatentDiffusion
    from models.cvae import CVAE

    # Load pretrained CVAE
    cvae = CVAE()
    cvae.load_state_dict(...)

    # Build LDM with frozen CVAE
    ldm = LatentDiffusion(cvae=cvae)
    ldm.train_model(real_windows, labels, n_epochs=500)
    synthetic = ldm.generate(n_samples=500)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Noise Schedule ─────────────────────────────────────────────────────────

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine noise schedule (Nichol & Dhariwal 2021).
    More gradual than linear — better for small latent spaces.
    """
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999).float()


class DiffusionSchedule:
    """Precomputed diffusion schedule quantities."""

    def __init__(self, T: int = 1000):
        self.T = T
        betas = cosine_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Register as buffers (not parameters)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

    def to(self, device: torch.device):
        """Move all tensors to device."""
        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas", "posterior_variance",
        ]:
            setattr(self, attr, getattr(self, attr).to(device))
        return self


# ── Sinusoidal Timestep Embedding ──────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (batch,) integer timesteps → (batch, dim) embedding."""
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# ── 1D UNet Denoiser ──────────────────────────────────────────────────────

class ResBlock1d(nn.Module):
    """Residual block with time/class conditioning."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.cond_proj = nn.Linear(cond_dim, out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, in_ch, L)
        cond: (batch, cond_dim) — combined time+class embedding
        """
        h = self.act(self.bn1(self.conv1(x)))
        # Add conditioning (broadcast over spatial dim)
        h = h + self.cond_proj(cond).unsqueeze(-1)
        h = self.act(self.bn2(self.conv2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    """
    1D UNet for denoising in latent space.

    Input: (batch, latent_channels, latent_length) + timestep + label
    Output: (batch, latent_channels, latent_length) predicted noise

    The latent vector (128-dim) is reshaped to (16, 8) for UNet processing.
    """

    def __init__(
        self,
        in_channels: int = 16,
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4),
        time_dim: int = 128,
        n_classes: int = 2,
        n_patients: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        cond_dim = time_dim

        # Timestep and class embeddings — additive conditioning
        # (standard for small latent spaces; Ho et al. 2020 DDPM paper).
        # The thesis mentions cross-attention for patient conditioning, but
        # cross-attention requires a sequence of conditioning tokens and is
        # designed for complex conditioning signals like text (Rombach et al.
        # 2022 Stable Diffusion). For scalar labels and patient IDs, additive
        # embedding is the standard approach and computationally cheaper.
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.class_embed = nn.Embedding(n_classes, time_dim)
        self.patient_embed = nn.Embedding(n_patients, time_dim) if n_patients else None

        # Down path
        self.down_blocks = nn.ModuleList()
        self.down_pools = nn.ModuleList()
        ch = in_channels
        channels = []
        for mult in channel_mults:
            out_ch = base_channels * mult
            self.down_blocks.append(ResBlock1d(ch, out_ch, cond_dim))
            self.down_pools.append(nn.Conv1d(out_ch, out_ch, kernel_size=2, stride=2))
            channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        self.bottleneck = ResBlock1d(ch, ch, cond_dim)

        # Up path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            self.up_samples.append(nn.ConvTranspose1d(ch, out_ch, kernel_size=2, stride=2))
            # Skip connection doubles channels
            self.up_blocks.append(ResBlock1d(out_ch * 2, out_ch, cond_dim))
            ch = out_ch

        # Output projection
        self.out_conv = nn.Conv1d(ch, in_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, label: torch.Tensor,
        patient_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch, in_channels, L) noisy latent
        t: (batch,) integer timesteps
        label: (batch,) class labels
        patient_id: (batch,) patient IDs (optional)

        Returns: (batch, in_channels, L) predicted noise
        """
        # Conditioning: additive embeddings
        cond = self.time_embed(t) + self.class_embed(label)
        if self.patient_embed is not None and patient_id is not None:
            cond = cond + self.patient_embed(patient_id)

        # Down path (collect skip connections)
        skips = []
        h = x
        for block, pool in zip(self.down_blocks, self.down_pools):
            h = block(h, cond)
            skips.append(h)
            h = pool(h)

        # Bottleneck
        h = self.bottleneck(h, cond)

        # Up path (use skip connections)
        for upsample, block, skip in zip(self.up_samples, self.up_blocks, reversed(skips)):
            h = upsample(h)
            # Handle size mismatch from rounding
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        return self.out_conv(h)


# ── Latent Diffusion Model ────────────────────────────────────────────────

class LatentDiffusion(nn.Module):
    """
    Latent Diffusion Model for EEG generation.

    Uses a pretrained (frozen) CVAE encoder/decoder and trains a UNet
    denoiser in the CVAE's latent space.

    Args:
        cvae: pretrained CVAE instance (encoder+decoder will be frozen)
        latent_dim: CVAE latent dimensionality (128)
        latent_channels: channels after reshape (16)
        latent_length: length after reshape (8)
        T: number of diffusion timesteps (1000)
    """

    def __init__(
        self,
        cvae,
        latent_dim: int = 128,
        latent_channels: int = 16,
        latent_length: int = 8,
        T: int = 1000,
        n_patients: Optional[int] = None,
    ):
        super().__init__()
        assert latent_channels * latent_length == latent_dim, \
            f"latent_channels * latent_length must equal latent_dim: {latent_channels}*{latent_length} != {latent_dim}"

        self.cvae = cvae
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.latent_length = latent_length
        self.T = T

        # Freeze CVAE — only UNet is trained
        for param in self.cvae.parameters():
            param.requires_grad = False

        # Denoiser
        self.unet = UNet1D(
            in_channels=latent_channels,
            base_channels=64,
            channel_mults=(1, 2, 4),
            time_dim=128,
            n_classes=2,
            n_patients=n_patients,
        )

        # Diffusion schedule
        self.schedule = DiffusionSchedule(T)

    def to(self, *args, **kwargs):
        """Override to() so that DiffusionSchedule tensors also move to device."""
        result = super().to(*args, **kwargs)
        # Infer device from the first argument (handles both .to("cuda") and .to(device=...))
        device = None
        if args:
            device = args[0]
        elif "device" in kwargs:
            device = kwargs["device"]
        if device is not None:
            self.schedule = self.schedule.to(device)
        return result

    def _z_to_spatial(self, z: torch.Tensor) -> torch.Tensor:
        """(batch, latent_dim) → (batch, latent_channels, latent_length)"""
        return z.reshape(-1, self.latent_channels, self.latent_length)

    def _spatial_to_z(self, h: torch.Tensor) -> torch.Tensor:
        """(batch, latent_channels, latent_length) → (batch, latent_dim)"""
        return h.reshape(-1, self.latent_dim)

    @torch.no_grad()
    def encode_dataset(
        self,
        windows: np.ndarray,
        labels: np.ndarray,
        device: str = "cpu",
        batch_size: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode all windows to latent codes using the frozen CVAE encoder.

        Returns:
            (latent_codes, labels) — latent_codes shape (N, latent_dim)
        """
        self.cvae.eval()
        all_z = []
        N = len(windows)

        for i in range(0, N, batch_size):
            x = torch.from_numpy(windows[i:i+batch_size]).float().to(device)
            y = torch.from_numpy(labels[i:i+batch_size]).long().to(device)

            mu, log_var = self.cvae.encode(x, y)
            # Use mu (not sampled z) for stable latent codes
            all_z.append(mu.cpu().numpy())

        return np.concatenate(all_z, axis=0), labels

    def q_sample(
        self, z_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to clean latent at timestep t."""
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].reshape(-1, 1, 1)
        sqrt_one_minus = self.schedule.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1)

        return sqrt_alpha * z_0 + sqrt_one_minus * noise

    def p_losses(
        self,
        z_0: torch.Tensor,
        label: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss: MSE between predicted and actual noise.

        Args:
            z_0: (batch, latent_channels, latent_length) clean latent
            label: (batch,) class labels
            t: (batch,) timesteps (random if None)
        """
        batch = z_0.shape[0]
        device = z_0.device

        if t is None:
            t = torch.randint(0, self.T, (batch,), device=device)

        noise = torch.randn_like(z_0)
        z_noisy = self.q_sample(z_0, t, noise)
        noise_pred = self.unet(z_noisy, t, label)

        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def sample_ddpm(
        self,
        n_samples: int,
        label: int = 1,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        DDPM sampling (T steps — slow but high quality).

        Returns: (n_samples, latent_dim) sampled latent codes
        """
        self.eval()
        labels = torch.full((n_samples,), label, device=device, dtype=torch.long)

        z = torch.randn(n_samples, self.latent_channels, self.latent_length, device=device)

        for t_val in reversed(range(self.T)):
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            noise_pred = self.unet(z, t, labels)

            alpha = self.schedule.alphas[t_val]
            alpha_cum = self.schedule.alphas_cumprod[t_val]
            beta = self.schedule.betas[t_val]

            # Predict z_{t-1}
            z = (1 / alpha.sqrt()) * (
                z - (beta / (1 - alpha_cum).sqrt()) * noise_pred
            )

            if t_val > 0:
                noise = torch.randn_like(z)
                z = z + self.schedule.posterior_variance[t_val].sqrt() * noise

        return self._spatial_to_z(z)

    @torch.no_grad()
    def sample_ddim(
        self,
        n_samples: int,
        label: int = 1,
        device: str = "cpu",
        n_steps: int = 50,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """
        DDIM sampling (fewer steps — faster inference).

        Args:
            n_steps: number of denoising steps (50-100 recommended)
            eta: 0.0 = deterministic DDIM, 1.0 = DDPM-equivalent

        Returns: (n_samples, latent_dim) sampled latent codes
        """
        self.eval()
        labels = torch.full((n_samples,), label, device=device, dtype=torch.long)

        # Subsequence of timesteps
        step_size = self.T // n_steps
        timesteps = list(range(0, self.T, step_size))[::-1]

        z = torch.randn(n_samples, self.latent_channels, self.latent_length, device=device)

        for i, t_val in enumerate(timesteps):
            t = torch.full((n_samples,), t_val, device=device, dtype=torch.long)
            noise_pred = self.unet(z, t, labels)

            alpha_cum_t = self.schedule.alphas_cumprod[t_val]

            if i < len(timesteps) - 1:
                t_prev_val = timesteps[i + 1]
                alpha_cum_prev = self.schedule.alphas_cumprod[t_prev_val]
            else:
                alpha_cum_prev = torch.tensor(1.0, device=device)

            # Predict x_0
            pred_z0 = (z - (1 - alpha_cum_t).sqrt() * noise_pred) / alpha_cum_t.sqrt()

            # Direction pointing to z_t
            sigma = eta * ((1 - alpha_cum_prev) / (1 - alpha_cum_t) * (1 - alpha_cum_t / alpha_cum_prev)).sqrt()
            dir_zt = (1 - alpha_cum_prev - sigma ** 2).sqrt() * noise_pred

            z = alpha_cum_prev.sqrt() * pred_z0 + dir_zt
            if sigma > 0 and i < len(timesteps) - 1:
                z = z + sigma * torch.randn_like(z)

        return self._spatial_to_z(z)

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        device: str = "cpu",
        patient_id: int = 0,
        label: int = 1,
        use_ddim: bool = True,
        ddim_steps: int = 50,
    ) -> List[Tuple[np.ndarray, int, int]]:
        """
        Generate synthetic EEG windows via diffusion → CVAE decode.

        Args:
            n_samples: number of windows to generate
            device: torch device
            patient_id: patient ID to assign
            label: class label to condition on
            use_ddim: use DDIM (fast) or DDPM (slow)
            ddim_steps: number of DDIM steps

        Returns:
            List of (window, label, patient_id) tuples
        """
        self.eval()
        self.cvae.eval()

        if use_ddim:
            z = self.sample_ddim(n_samples, label, device, ddim_steps)
        else:
            z = self.sample_ddpm(n_samples, label, device)

        # Decode latent codes with frozen CVAE decoder
        labels_tensor = torch.full((n_samples,), label, device=device, dtype=torch.long)
        x_gen = self.cvae.decode(z, labels_tensor)

        synthetic = []
        for i in range(n_samples):
            window = x_gen[i].cpu().numpy().astype(np.float32)
            synthetic.append((window, label, patient_id))

        return synthetic

    def train_model(
        self,
        real_windows: np.ndarray,
        labels: np.ndarray,
        n_epochs: int = 500,
        batch_size: int = 64,
        lr: float = 1e-4,
        device: str = "cpu",
        verbose: bool = True,
    ) -> dict:
        """
        Train the UNet denoiser in latent space.

        Steps:
        1. Encode all real windows to latent codes using frozen CVAE encoder
        2. Train UNet to predict noise on latent codes

        Args:
            real_windows: (N, 23, 1024) array
            labels: (N,) array of 0/1 labels
            n_epochs: training epochs
            batch_size: batch size
            lr: learning rate (lower than CVAE — denoiser is sensitive)
            device: torch device
            verbose: print progress

        Returns:
            Dict with loss history
        """
        self.to(device)

        # Step 1: Encode all windows to latent codes
        if verbose:
            print("  Encoding dataset to latent space...")
        z_all, labels_all = self.encode_dataset(real_windows, labels, device, batch_size)
        # Keep latent codes on CPU, move per-batch to GPU
        z_cpu = torch.from_numpy(z_all).float()
        z_cpu = self._z_to_spatial(z_cpu)  # (N, latent_channels, latent_length)
        y_cpu = torch.from_numpy(labels_all).long()
        N = len(z_cpu)

        if verbose:
            print(f"  Encoded {N} windows to latent space ({z_cpu.shape})")
            print(f"  Training UNet denoiser ({n_epochs} epochs)")

        # Only optimize UNet parameters
        optimizer = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        history = {"loss": []}

        for epoch in range(n_epochs):
            self.unet.train()
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, N, batch_size):
                idx = perm[i:i + batch_size]
                z_0 = z_cpu[idx].to(device)
                y = y_cpu[idx].to(device)

                loss = self.p_losses(z_0, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg = epoch_loss / max(n_batches, 1)
            history["loss"].append(avg)

            if verbose and (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{n_epochs}: loss={avg:.6f}")

        return history


# ── Quick sanity check ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from models.cvae import CVAE

    # Create a dummy CVAE (untrained — just for shape testing)
    cvae = CVAE(n_channels=23, seq_len=1024, latent_dim=128)
    ldm = LatentDiffusion(cvae=cvae, T=100)  # T=100 for faster test

    unet_params = sum(p.numel() for p in ldm.unet.parameters())
    total_params = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    print(f"LDM UNet denoiser: {unet_params:,} parameters")
    print(f"LDM trainable (UNet only): {total_params:,} parameters")
    print(f"CVAE frozen: {sum(p.numel() for p in ldm.cvae.parameters()):,} parameters (not trained)")

    # Test forward (noise prediction)
    z = torch.randn(4, 16, 8)
    t = torch.randint(0, 100, (4,))
    labels = torch.tensor([1, 1, 0, 1])
    noise_pred = ldm.unet(z, t, labels)
    print(f"\n  Noise pred shape: {noise_pred.shape}")
    assert noise_pred.shape == (4, 16, 8)

    # Test generation (DDIM, few steps — untrained model, just shapes)
    ldm.schedule.to("cpu")
    synthetic = ldm.generate(n_samples=2, use_ddim=True, ddim_steps=10)
    w, l, p = synthetic[0]
    print(f"  Generated {len(synthetic)} windows via DDIM (10 steps)")
    print(f"  Shape: {w.shape}, label: {l}, patient_id: {p}")
    assert w.shape == (23, 1024)

    print("\n  All checks passed.")

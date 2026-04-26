"""
visualize.py — Visualization tools for generator evaluation (E3–E5).

Mandatory fidelity plots for each generator (Carrle et al. 2023;
You et al. 2025; Zhao et al. 2022):
  1. PSD comparison: real vs synthetic (per frequency band)
  2. t-SNE: real interictal, real ictal, synthetic ictal
  3. Amplitude distribution: real vs synthetic

Usage:
    from training.visualize import plot_psd_comparison, plot_tsne, plot_amplitude_dist

    plot_psd_comparison(real_windows, synthetic_windows, save_path="results/e3/psd.png")
    plot_tsne(real_windows, real_labels, synthetic_windows, save_path="results/e3/tsne.png")
    plot_amplitude_dist(real_windows, synthetic_windows, save_path="results/e3/amplitude.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from scipy.signal import welch


# EEG frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}
FS = 256  # Sampling frequency


def compute_psd(windows: np.ndarray, fs: int = FS) -> tuple:
    """
    Compute average PSD across all windows and channels.

    Args:
        windows: (N, n_channels, seq_len)
        fs: sampling frequency

    Returns:
        (freqs, psd_mean, psd_std)
        freqs: (n_freqs,)
        psd_mean: (n_freqs,) — mean PSD across windows/channels
        psd_std: (n_freqs,) — std
    """
    all_psd = []
    for i in range(len(windows)):
        for ch in range(windows.shape[1]):
            freqs, psd = welch(windows[i, ch], fs=fs, nperseg=min(256, windows.shape[2]))
            all_psd.append(psd)

    all_psd = np.array(all_psd)
    return freqs, all_psd.mean(axis=0), all_psd.std(axis=0)


def compute_band_powers(windows: np.ndarray, fs: int = FS) -> dict:
    """
    Compute mean power per frequency band.

    Returns:
        Dict mapping band_name → (mean_power, std_power)
    """
    freqs, psd_mean, psd_std = compute_psd(windows, fs)
    freq_res = freqs[1] - freqs[0]

    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_powers[name] = (
            float(np.sum(psd_mean[mask]) * freq_res),
            float(np.sum(psd_std[mask]) * freq_res),
        )
    return band_powers


def compute_psd_kl_divergence(real: np.ndarray, synthetic: np.ndarray, fs: int = FS) -> dict:
    """
    KL divergence between real and synthetic PSD per frequency band.
    Lower = more similar spectral profile.

    Returns:
        Dict mapping band_name → KL divergence
    """
    freqs_r, psd_r, _ = compute_psd(real, fs)
    freqs_s, psd_s, _ = compute_psd(synthetic, fs)
    freq_res = freqs_r[1] - freqs_r[0]

    kl_per_band = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs_r >= lo) & (freqs_r <= hi)
        p = psd_r[mask] + 1e-12  # real
        q = psd_s[mask] + 1e-12  # synthetic
        # Normalize to distributions
        p = p / p.sum()
        q = q / q.sum()
        kl = float(np.sum(p * np.log(p / q)))
        kl_per_band[name] = kl

    return kl_per_band


def _band_powers_from_psd(freqs: np.ndarray, psd_mean: np.ndarray, psd_std: np.ndarray) -> dict:
    """Compute per-band power from pre-computed PSD (avoids redundant recomputation)."""
    freq_res = freqs[1] - freqs[0]
    band_powers = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        band_powers[name] = (
            float(np.sum(psd_mean[mask]) * freq_res),
            float(np.sum(psd_std[mask]) * freq_res),
        )
    return band_powers


def plot_psd_comparison(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "PSD Comparison: Real vs Synthetic",
):
    """
    Plot PSD comparison between real and synthetic EEG windows.

    Creates two subplots:
      - Left: full PSD curves with confidence bands
      - Right: per-band power bar chart
    """
    freqs_r, psd_r, std_r = compute_psd(real_windows)
    freqs_s, psd_s, std_s = compute_psd(synthetic_windows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: PSD curves
    ax1.semilogy(freqs_r, psd_r, label="Real", color="#2196F3", linewidth=1.5)
    ax1.fill_between(freqs_r, np.maximum(psd_r - std_r, 1e-12), psd_r + std_r, alpha=0.2, color="#2196F3")
    ax1.semilogy(freqs_s, psd_s, label="Synthetic", color="#F44336", linewidth=1.5)
    ax1.fill_between(freqs_s, np.maximum(psd_s - std_s, 1e-12), psd_s + std_s, alpha=0.2, color="#F44336")

    # Shade frequency bands
    colors = {"delta": "#E3F2FD", "theta": "#FFF3E0", "alpha": "#E8F5E9", "beta": "#F3E5F5", "gamma": "#FCE4EC"}
    for name, (lo, hi) in BANDS.items():
        ax1.axvspan(lo, hi, alpha=0.15, color=colors.get(name, "#eee"), label=name)

    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power Spectral Density (µV²/Hz)")
    ax1.set_title("PSD")
    ax1.set_xlim(0, 45)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right: band power comparison (reuse already-computed PSDs)
    bands_r = _band_powers_from_psd(freqs_r, psd_r, std_r)
    bands_s = _band_powers_from_psd(freqs_s, psd_s, std_s)

    names = list(BANDS.keys())
    x = np.arange(len(names))
    width = 0.35

    r_vals = [bands_r[n][0] for n in names]
    s_vals = [bands_s[n][0] for n in names]
    r_err = [bands_r[n][1] for n in names]
    s_err = [bands_s[n][1] for n in names]

    ax2.bar(x - width/2, r_vals, width, yerr=r_err, label="Real", color="#2196F3", alpha=0.8, capsize=3)
    ax2.bar(x + width/2, s_vals, width, yerr=s_err, label="Synthetic", color="#F44336", alpha=0.8, capsize=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Band Power (µV²)")
    ax2.set_title("Per-Band Power")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved PSD plot to {save_path}")
    plt.close(fig)

    return fig


def plot_tsne(
    real_windows: np.ndarray,
    real_labels: np.ndarray,
    synthetic_windows: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "t-SNE: Real vs Synthetic",
    n_samples: int = 2000,
    perplexity: int = 30,
    seed: int = 42,
):
    """
    t-SNE visualization with 3 groups:
      - Real interictal (blue)
      - Real ictal (orange)
      - Synthetic ictal (red)

    Subsamples to n_samples for tractability.
    """
    from sklearn.manifold import TSNE

    # Separate real into ictal/interictal
    real_ictal_mask = real_labels == 1
    real_interictal = real_windows[~real_ictal_mask]
    real_ictal = real_windows[real_ictal_mask]

    # Subsample each group
    rng = np.random.RandomState(seed)
    n_per_group = n_samples // 3

    def subsample(arr, n):
        if len(arr) <= n:
            return arr
        idx = rng.choice(len(arr), n, replace=False)
        return arr[idx]

    ri = subsample(real_interictal, n_per_group)
    rc = subsample(real_ictal, n_per_group)
    sc = subsample(synthetic_windows, n_per_group)

    # Flatten windows to feature vectors
    all_windows = np.concatenate([ri, rc, sc], axis=0)
    all_flat = all_windows.reshape(len(all_windows), -1)

    # Group labels for coloring
    group_labels = (
        ["Real interictal"] * len(ri) +
        ["Real ictal"] * len(rc) +
        ["Synthetic ictal"] * len(sc)
    )

    # Run t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, max_iter=1000)
    embedding = tsne.fit_transform(all_flat)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    colors = {"Real interictal": "#2196F3", "Real ictal": "#FF9800", "Synthetic ictal": "#F44336"}

    for label in ["Real interictal", "Real ictal", "Synthetic ictal"]:
        mask = np.array([g == label for g in group_labels])
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=colors[label], label=label, alpha=0.5, s=10, edgecolors="none",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=10)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved t-SNE plot to {save_path}")
    plt.close(fig)

    return fig


def plot_amplitude_dist(
    real_windows: np.ndarray,
    synthetic_windows: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Amplitude Distribution: Real vs Synthetic",
    n_samples: int = 5000,
    seed: int = 42,
):
    """
    Compare amplitude distributions of real vs synthetic EEG.

    Plots histogram of sample amplitudes across all channels.
    """
    rng = np.random.RandomState(seed)

    # Subsample windows first, then flatten (avoids full ravel memory spike)
    def flatten_subsample(arr, n_vals):
        n_windows = min(len(arr), max(1, n_vals // (arr.shape[1] * arr.shape[2])))
        idx = rng.choice(len(arr), n_windows, replace=False)
        return arr[idx].ravel()

    real_flat = flatten_subsample(real_windows, n_samples * 1000)
    synth_flat = flatten_subsample(synthetic_windows, n_samples * 1000)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    bins = np.linspace(
        min(real_flat.min(), synth_flat.min()),
        max(real_flat.max(), synth_flat.max()),
        100,
    )
    ax1.hist(real_flat, bins=bins, alpha=0.6, label="Real", color="#2196F3", density=True)
    ax1.hist(synth_flat, bins=bins, alpha=0.6, label="Synthetic", color="#F44336", density=True)
    ax1.set_xlabel("Amplitude (normalized)")
    ax1.set_ylabel("Density")
    ax1.set_title("Amplitude Histogram")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # QQ-style: sorted quantiles
    n_q = min(1000, len(real_flat), len(synth_flat))
    q_real = np.quantile(real_flat, np.linspace(0, 1, n_q))
    q_synth = np.quantile(synth_flat, np.linspace(0, 1, n_q))
    ax2.scatter(q_real, q_synth, s=3, alpha=0.5, color="#9C27B0")
    lims = [min(q_real.min(), q_synth.min()), max(q_real.max(), q_synth.max())]
    ax2.plot(lims, lims, "k--", linewidth=0.8, label="y=x (perfect match)")
    ax2.set_xlabel("Real Quantiles")
    ax2.set_ylabel("Synthetic Quantiles")
    ax2.set_title("Q-Q Plot")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved amplitude plot to {save_path}")
    plt.close(fig)

    return fig


def generate_all_plots(
    real_windows: np.ndarray,
    real_labels: np.ndarray,
    synthetic_windows: np.ndarray,
    output_dir: str,
    generator_name: str = "generator",
):
    """
    Generate all mandatory evaluation plots for a generator.

    Saves: psd.png, tsne.png, amplitude.png + kl_divergence.json
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Real ictal only for fair PSD comparison
    real_ictal = real_windows[real_labels == 1]

    print(f"\n  Generating evaluation plots for {generator_name}...")

    plot_psd_comparison(
        real_ictal, synthetic_windows,
        save_path=str(out / "psd.png"),
        title=f"PSD: Real vs {generator_name}",
    )

    plot_tsne(
        real_windows, real_labels, synthetic_windows,
        save_path=str(out / "tsne.png"),
        title=f"t-SNE: Real vs {generator_name}",
    )

    plot_amplitude_dist(
        real_ictal, synthetic_windows,
        save_path=str(out / "amplitude.png"),
        title=f"Amplitude: Real vs {generator_name}",
    )

    # PSD KL divergence
    kl = compute_psd_kl_divergence(real_ictal, synthetic_windows)
    import json
    with open(out / "kl_divergence.json", "w") as f:
        json.dump(kl, f, indent=2)
    print(f"  KL divergence per band: {kl}")


# ── Quick test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    real = np.random.randn(100, 23, 1024).astype(np.float32)
    labels = np.concatenate([np.zeros(80), np.ones(20)]).astype(np.int64)
    synthetic = np.random.randn(20, 23, 1024).astype(np.float32)

    generate_all_plots(real, labels, synthetic, "/tmp/viz_test", "TestGen")
    print("\n  All visualization checks passed.")

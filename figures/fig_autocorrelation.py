"""
Generate publication-ready autocorrelation figure.
Adapted from the EPIA 2026 paper script.

Compares real vs synthetic ictal EEG autocorrelation for all 3 generators.
Requires: synthetic .npz files from single-split (on the uni machine).

Usage:
    python -m figures.fig_autocorrelation [--output figures/output/autocorrelation.pdf]
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

RESULTS_DIR = _PROJECT_ROOT / "results"
FS = 256
MAX_LAG = 128
N_SAMPLES = 500
SEEDS = [42, 123, 456]


def avg_autocorr(windows, n_samples, rng):
    idx = rng.choice(len(windows), min(n_samples, len(windows)), replace=False)
    subset = windows[idx]
    acfs = []
    for w in subset:
        for ch in range(w.shape[0]):
            sig = w[ch] - w[ch].mean()
            full_acf = np.correlate(sig, sig, mode="full")
            full_acf = full_acf[len(sig) - 1:]
            if full_acf[0] != 0:
                full_acf = full_acf / full_acf[0]
            acfs.append(full_acf[:MAX_LAG + 1])
    acfs = np.array(acfs)
    return acfs.mean(axis=0), acfs.std(axis=0)


def load_real_ictal():
    from data.loader import CHBMITDataset
    from torch.utils.data import DataLoader
    ds = CHBMITDataset(split='train', normalize=True, ictal_only=True)
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    all_windows = []
    for batch in dl:
        all_windows.append(batch[0].numpy())
    return np.concatenate(all_windows)


def load_synthetic(experiment, seeds=SEEDS):
    all_windows = []
    for seed in seeds:
        path = RESULTS_DIR / experiment / f'seed_{seed}' / 'single_split' / 'synthetic_ratio_1.00.npz'
        if not path.exists():
            print(f"  Warning: {path} not found, skipping")
            continue
        data = np.load(path)
        all_windows.append(data['windows'])
    if not all_windows:
        return None
    return np.concatenate(all_windows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="figures/output/autocorrelation.pdf")
    args = parser.parse_args()

    output_path = _PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading real ictal windows...")
    real_ictal = load_real_ictal()
    print(f"  Got {len(real_ictal)} real ictal windows")

    generators = [
        ('e3', 'TimeGAN'),
        ('e4', 'CVAE'),
        ('e5', 'LDM'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    lags = np.arange(MAX_LAG + 1) / FS * 1000  # ms

    for i, (exp, name) in enumerate(generators):
        ax = axes[i]
        rng = np.random.RandomState(42)

        print(f"  Computing autocorrelation for {name}...")
        synthetic = load_synthetic(exp)
        if synthetic is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha='center')
            ax.set_title(f"({chr(97+i)}) {name}", fontsize=11)
            continue

        acf_r_mean, acf_r_std = avg_autocorr(real_ictal, N_SAMPLES, rng)
        rng2 = np.random.RandomState(42)
        acf_s_mean, acf_s_std = avg_autocorr(synthetic, N_SAMPLES, rng2)

        ax.plot(lags, acf_r_mean, label="Real", color='black', linewidth=1.5, linestyle='-')
        ax.fill_between(lags, acf_r_mean - acf_r_std, acf_r_mean + acf_r_std,
                        alpha=0.15, color='black')

        ax.plot(lags, acf_s_mean, label="Synthetic", color='#D32F2F', linewidth=1.5, linestyle='--')
        ax.fill_between(lags, acf_s_mean - acf_s_std, acf_s_mean + acf_s_std,
                        alpha=0.15, color='#D32F2F')

        ax.set_xlabel("Lag (ms)")
        if i == 0:
            ax.set_ylabel("Autocorrelation")
        ax.set_title(f"({chr(97+i)}) {name}", fontsize=11)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(True, alpha=0.2)
        ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    print(f"\nSaved to {output_path}")

    png_path = output_path.with_suffix('.png')
    fig.savefig(png_path, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved to {png_path}")
    plt.close(fig)


if __name__ == '__main__':
    main()

# Synthetic Data in Healthcare: A Focus on EEG Signals

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="website/images/ista_horizontal_negative.png">
  <source media="(prefers-color-scheme: light)" srcset="website/images/ista_horizontal_main.png">
  <img src="website/images/ista_horizontal_main.png" align="right" width="180" alt="ISCTE-IUL ISTA">
</picture>

_MSc thesis by Vanda Barata at ISCTE-IUL._\
_Supervised by [Ana de Almeida](https://ciencia.iscte-iul.pt/authors/ana-de-almeida) and [Luis Nunes](https://ciencia.iscte-iul.pt/authors/luis-miguel-martins-nunes)._

---

![](https://img.shields.io/badge/language-python-3776AB?style=for-the-badge&logo=python&logoColor=ffffff)
![](https://img.shields.io/badge/framework-pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=ffffff)
![](https://img.shields.io/badge/dataset-CHB--MIT-0d28c2?style=for-the-badge)

![](https://img.shields.io/github/last-commit/vandabarata/msc-thesis-eeg/main?logo=github)

[![](https://img.shields.io/badge/project%20website-visit-00a9e0?style=for-the-badge&logo=github-pages&logoColor=ffffff)](https://vandabarata.github.io/msc-thesis-eeg/)

> Three generative models (**TimeGAN**, **Conditional VAE**, and **Latent Diffusion**) are trained to synthesize ictal (seizure) EEG windows. A frozen 49K-param 1D-CNN detector is then trained on real + synthetic data and evaluated with **leave-one-patient-out (LOPO)** cross-validation on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/). The detector is the same across all experiments, so any performance change comes from the data alone.
>
> Methodology, preprocessing, dataset exploration, architecture choices, and literature references are on the **[project website](https://vandabarata.github.io/msc-thesis-eeg/)**.

---

## Project Structure

```
msc_thesis_code/
├── data/                        Data pipeline
│   ├── loader.py                  Dataset, preprocessing, windowing, splits
│   ├── split_config.json          Patient-level splits (single + 23 LOPO folds)
│   ├── homogenize.py              EDF cleaning pipeline (already run)
│   └── build_cache.py             Builds the flat-signal mmap cache from clean_edfs/
│
├── models/                      Neural network architectures
│   ├── detector.py                1D-CNN seizure detector (frozen, 49K params)
│   ├── timegan.py                 TimeGAN (5 GRU sub-networks, 1.6M params)
│   ├── cvae.py                    Conditional VAE (1D-Conv, 1.9M params)
│   └── ldm.py                     Latent Diffusion (UNet-1D + DDPM/DDIM, 2.3M params)
│
├── training/                    Training, evaluation, and analysis
│   ├── train.py                   Detector training (E1-E5), TSTR, early stopping
│   ├── generate.py                Generator training + synthetic window production (E3-E5)
│   ├── evaluate.py                Metrics (AUPRC, AUROC, F1, sens@95%spec), Wilcoxon, E6 analysis
│   ├── visualize.py               Fidelity evaluation (PSD, KL, C2ST, autocorrelation, t-SNE)
│   └── subject_identity.py        E7 linear probe + proximity check
│
├── experiment_scripts/          Shell scripts for running on the remote GPU machine
│   ├── run_e1.sh ... run_e5.sh    Per-experiment single-split launchers
│   ├── run_lopo.sh                Full LOPO evaluation (E1-E5, 23 folds x 3 seeds)
│   └── deploy_and_train.sh        Deploy code to remote and start training
│
├── notebooks/                   Exploration
│   └── chb-mit-analysis.ipynb     Full EDA notebook
│
├── results/                     Experiment outputs (metrics + model checkpoints)
├── index.html                   Project website (GitHub Pages)
├── website/                     Website assets
│   └── images/                    ISCTE/ISTA logos
├── requirements.txt             Python dependencies (CPU)
└── requirements-gpu.txt         Python dependencies (CUDA)
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

<details>
<summary>GPU support</summary>

The default installation uses PyTorch CPU. For CUDA:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
</details>

<details>
<summary>Dataset</summary>

Download the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) and place it under `chb-mit-scalp-eeg-database-1.0.0/`. The homogenization script (`data/homogenize.py`) produces the `clean_edfs/` directory with 683 standardized EDF+ files. Both directories are large and excluded from git.
</details>

---

## Experiments

| Experiment | Description | Status |
|:----------:|-------------|:------:|
| **E1** | Baseline 1D-CNN detector (real data, class-weighted cross-entropy) | LOPO complete |
| **E2** | Non-synthetic controls (SMOTE, ADASYN) | LOPO running |
| **E3** | TimeGAN augmentation (4 ratios: 25/ 50/ 100/ 200%) | LOPO running |
| **E4** | CVAE augmentation (4 ratios: 25/ 50/ 100/ 200%) | LOPO running |
| **E5** | Latent Diffusion augmentation (4 ratios, reuses CVAE encoder) | LOPO running |
| **E6** | Cross-generator comparison (Wilcoxon, ratio sensitivity, cost-benefit) | After LOPO |
| **E7** | Subject-identity analysis (linear probe + proximity check) | After E6 |

> **Note (5 May 2026):** First LOPO run failed due to OOM (E2) and disk-full (E3). Fixed: reduced SMOTE interictal subsample from 10x to 5x, rewrote pipeline to generate+train per fold instead of all-at-once. Relaunched.

### Results (single-split, 3 seeds)

| Experiment | Generator | AUPRC | Per-Patient AUPRC | Gen. train (avg/ seed) | Det. train (avg/ seed) |
|:----------:|-----------|:-----:|:-----------------:|:----------------------:|:----------------------:|
| **E5** | LDM | **0.2272 +/- 0.0193** | **0.3759 +/- 0.0419** | ~58 min | ~72 min |
| **E1** | None (baseline) | 0.1766 +/- 0.0542 | 0.2264 +/- 0.0646 | - | ~205 min |
| **E4** | CVAE | 0.1750 +/- 0.0732 | 0.2877 +/- 0.1239 | ~42 min | ~99 min |
| **E3** | TimeGAN | 0.1742 +/- 0.0842 | 0.1943 +/- 0.1063 | ~17 min | ~84 min |
| **E2** | ADASYN | 0.1078 +/- 0.0732 | 0.1302 +/- 0.0713 | - | ~79 min |

LDM augmentation improves AUPRC by +29% over baseline with the lowest cross-seed variance (single-split).

### Results (LOPO, 23 folds x 3 seeds)

E1 baseline LOPO complete (4 May 2026). E2-E5 LOPO relaunched 5 May 2026 (after OOM/disk-full fixes).

| Experiment | Generator | AUPRC (cross-seed) | AUROC | F1 | Sens. @ 95% Spec. | Det. train (avg/ seed/ fold) |
|:----------:|-----------|:-------------------:|:-----:|:--:|:-----------------:|:---------------------------:|
| **E1** | None (baseline) | 0.3941 +/- 0.0228 | 0.8438 +/- 0.0147 | 0.4333 +/- 0.0194 | 0.6459 +/- 0.0123 | ~92 min |

Remaining experiments will be added as they complete. Detailed results, per-fold breakdowns, and fidelity analysis on the [project website](https://vandabarata.github.io/msc-thesis-eeg/).

<details>
<summary>Protocol rules enforced in code</summary>

- Patient-level splits only (no leakage)
- Normalization from training data only
- Synthetic data in training only (val/test raise `ValueError`)
- Same frozen detector across E1-E5
- TSTR: synthetic ictal + real interictal only (no real ictal in training)
- 4 synthetic ratios (25%, 50%, 100%, 200%) per generator in LOPO
- 3 seeds (42, 123, 456), mean +/- std
- AUPRC as primary metric
</details>

### Running Experiments

```bash
source .venv/bin/activate

# E1: Baseline
python -m training.train --experiment e1 --mode single --seeds 42 123 456

# E2: SMOTE/ ADASYN controls
python -m training.train --experiment e2 --augmentation smote --mode single --seeds 42 123 456
python -m training.train --experiment e2 --augmentation adasyn --mode single --seeds 42 123 456

# E3: TimeGAN (single-split, ratio 1.0)
python -m training.generate --model timegan --mode single --seed 42
python -m training.train --experiment e3 --mode single --seeds 42 123 456

# E4: CVAE
python -m training.generate --model cvae --mode single --seed 42
python -m training.train --experiment e4 --mode single --seeds 42 123 456

# E5: LDM (needs pretrained CVAE)
python -m training.generate --model ldm --mode single --seed 42 \
    --cvae-checkpoint results/e4/seed_42/single_split/cvae.pt
python -m training.train --experiment e5 --mode single --seeds 42 123 456

# TSTR evaluation (E3-E5, after generator LOPO completes)
python -m training.train --experiment e3 --mode tstr
python -m training.train --experiment e4 --mode tstr
python -m training.train --experiment e5 --mode tstr

# Full LOPO evaluation (E1-E5, 23 folds x 3 seeds x 4 ratios, includes TSTR)
bash experiment_scripts/run_lopo.sh

# Manual multi-ratio LOPO (generate at 4 ratios, train at 4 ratios)
python -m training.generate --model cvae --mode lopo --seed 42 --ratio 0.25 0.5 1.0 2.0
python -m training.train --experiment e4 --mode lopo --seeds 42 --ratio 0.25 0.5 1.0 2.0
```

### Results Structure

```
results/<experiment>/
├── seed_42/
│   ├── single_split/
│   │   ├── best_model.pt           Model checkpoint (or best_model_<aug>.pt for E2)
│   │   ├── results.json            Metrics + history (or results_<aug>.json for E2)
│   │   └── plots/                  Fidelity plots (E3-E5 only)
│   ├── fold_00/ ... fold_22/       LOPO folds (same structure per fold)
│   │   ├── results.json              AUGM results (ratio 1.0, or single-ratio default)
│   │   ├── results_ratio_0.25.json   AUGM at 25% ratio (multi-ratio runs)
│   │   ├── results_ratio_0.50.json   AUGM at 50% ratio
│   │   ├── results_ratio_1.00.json   AUGM at 100% ratio
│   │   ├── results_ratio_2.00.json   AUGM at 200% ratio
│   │   ├── tstr_results.json         TSTR results (E3-E5 only)
│   │   ├── tstr_model.pt             TSTR model checkpoint (E3-E5 only)
│   │   └── plots/                    Fidelity plots (E3-E5 only)
│   └── ...
├── lopo_summary.json               Aggregated LOPO results
├── tstr_summary.json               Aggregated TSTR results (E3-E5 only)
└── lopo_status/                    Checkpoint files from run_lopo.sh
```

---

## Hardware

All experiments run on a single university workstation via SSH:

| | Spec |
|---|---|
| **GPU** | NVIDIA RTX 3080 Ti (12 GB VRAM) |
| **CUDA** | 13.2 (Driver 595.58) |
| **RAM** | 32 GB |
| **Disk** | ~98 GB (shared with OS) |
| **Python** | 3.12, PyTorch + CUDA |

The 12 GB VRAM and 32 GB RAM constraints shaped the data pipeline design (memory-mapped flat-signal cache, on-the-fly windowing, per-batch GPU transfer for generators).

---

## Dataset

**[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)**: 24 cases, 23 unique patients, 23-channel bipolar montage at 256 Hz. 683 EDF files after homogenization (3 dropped from original 686), <0.4% seizure windows.

Dataset exploration, patient demographics, and the preprocessing pipeline are documented on the [project website](https://vandabarata.github.io/msc-thesis-eeg/).

---

## License

This repository contains code for an MSc thesis. The CHB-MIT dataset is available from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) under the PhysioNet Credentialed Health Data License.

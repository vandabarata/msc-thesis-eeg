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
│   ├── run_e6.sh                  E6 statistical analysis (no GPU needed)
│   ├── run_e7.sh                  E7 subject-identity analysis (GPU)
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
| **E1** | Baseline 1D-CNN detector (real data, class-weighted cross-entropy) | LOPO complete (4 May) |
| **E2** | Non-synthetic controls (SMOTE, ADASYN) | LOPO complete (14 May) |
| **E3** | TimeGAN augmentation (2 ratios: 50/ 100%) | LOPO complete (7 Jun) |
| **E4** | CVAE augmentation (2 ratios: 50/ 100%) | LOPO complete (1 Jul) |
| **E5** | Latent Diffusion augmentation (2 ratios, reuses CVAE encoder) | LOPO complete (31 Jul) |
| **E6** | Cross-generator comparison (Wilcoxon, ratio sensitivity, cost-benefit) | Complete (16 Aug) |
| **E7** | Subject-identity analysis (linear probe + proximity check) | Complete (16 Aug) |

> **All experiments complete (16 Aug 2026).** Total LOPO runtime: 673 hours. Key finding: **no augmentation method significantly improves over baseline under LOPO** (all p<0.01 vs baseline, Wilcoxon signed-rank). LDM at ratio 0.50 is the least harmful (-17%), but the single-split +29% improvement (reported in the EPIA paper) does not generalize across patients. This reversal validates the paper's own warning that single-split results require broader confirmation.

### Results (single-split, 3 seeds)

Single patient-level split (18 train/ 2 val/ 4 test patients), ratio 1.0 for generators. These are the results reported in the EPIA 2026 paper.

| Experiment | Generator | AUPRC | Precision | Recall | F1 | Sens.@95%Spec. | Per-Patient AUPRC |
|:----------:|-----------|:-----:|:---------:|:------:|:--:|:--------------:|:-----------------:|
| **E5** | LDM | **0.227 +/- 0.019** | **0.407 +/- 0.045** | **0.336 +/- 0.005** | **0.367 +/- 0.020** | **0.717 +/- 0.077** | **0.376 +/- 0.042** |
| **E1** | None (baseline) | 0.177 +/- 0.054 | 0.324 +/- 0.071 | 0.259 +/- 0.078 | 0.287 +/- 0.077 | 0.618 +/- 0.097 | 0.226 +/- 0.065 |
| **E4** | CVAE | 0.175 +/- 0.073 | 0.264 +/- 0.075 | 0.273 +/- 0.132 | 0.263 +/- 0.103 | 0.602 +/- 0.207 | 0.288 +/- 0.124 |
| **E3** | TimeGAN | 0.120 +/- 0.028 | 0.204 +/- 0.044 | 0.217 +/- 0.048 | 0.214 +/- 0.020 | 0.435 +/- 0.184 | 0.218 +/- 0.097 |
| **E2** | ADASYN | 0.108 +/- 0.073 | - | - | - | - | 0.130 +/- 0.071 |

LDM augmentation improves AUPRC by +29% over baseline with the lowest cross-seed variance (single-split). This result does NOT generalize under LOPO (see below).

### Results (LOPO, 23 folds x 3 seeds)

| Experiment | Generator | Ratio | AUPRC | Precision | Recall | F1 | AUROC | Sens. @ 95% Spec. |
|:----------:|-----------|:-----:|:-----:|:---------:|:------:|:--:|:-----:|:-----------------:|
| **E1** | None (baseline) | - | **0.3941 +/- 0.3308** | 0.4848 +/- 0.3438 | 0.4849 +/- 0.2706 | 0.4333 +/- 0.3078 | 0.8438 +/- 0.1907 | 0.6459 +/- 0.3367 |
| **E5** | LDM | 0.50 | 0.3259 +/- 0.3185 | 0.4402 +/- 0.3415 | 0.4351 +/- 0.2971 | 0.3730 +/- 0.3008 | 0.7851 +/- 0.2216 | 0.5684 +/- 0.3478 |
| **E3** | TimeGAN | 0.50 | 0.3064 +/- 0.2990 | 0.4104 +/- 0.3429 | 0.4506 +/- 0.2817 | 0.3500 +/- 0.2831 | 0.7779 +/- 0.2235 | 0.5671 +/- 0.3335 |
| **E4** | CVAE | 0.50 | 0.2976 +/- 0.2894 | 0.4120 +/- 0.3389 | 0.3990 +/- 0.2731 | 0.3480 +/- 0.2802 | 0.7769 +/- 0.2147 | 0.5402 +/- 0.3353 |
| **E5** | LDM | 1.00 | 0.2830 +/- 0.2767 | 0.4119 +/- 0.3323 | 0.4072 +/- 0.2750 | 0.3384 +/- 0.2713 | 0.7715 +/- 0.2097 | 0.5300 +/- 0.3198 |
| **E4** | CVAE | 1.00 | 0.2507 +/- 0.2771 | 0.3620 +/- 0.3119 | 0.4049 +/- 0.3042 | 0.3060 +/- 0.2671 | 0.7288 +/- 0.2451 | 0.5140 +/- 0.3507 |
| **E3** | TimeGAN | 1.00 | 0.2145 +/- 0.2499 | 0.3321 +/- 0.3123 | 0.3625 +/- 0.2712 | 0.2778 +/- 0.2501 | 0.7287 +/- 0.2374 | 0.4895 +/- 0.3246 |
| **E2** | ADASYN | - | 0.1844 +/- 0.2270 | 0.2817 +/- 0.2849 | 0.4283 +/- 0.3299 | 0.2514 +/- 0.2494 | 0.6458 +/- 0.2639 | 0.4046 +/- 0.3364 |
| **E2** | SMOTE | - | 0.1580 +/- 0.2016 | 0.2862 +/- 0.2857 | 0.3700 +/- 0.3058 | 0.2295 +/- 0.2241 | 0.6283 +/- 0.2532 | 0.3835 +/- 0.3187 |

**No augmentation method beats the E1 baseline.** All differences are statistically significant (Wilcoxon signed-rank, p<0.01). LDM at ratio 0.50 is the best augmentation but still -17% below baseline. Ratio 0.50 consistently outperforms 1.00 for all generators (diminishing returns). Detailed results, per-fold breakdowns, and fidelity analysis on the [project website](https://vandabarata.github.io/msc-thesis-eeg/).

### E6: Statistical Analysis

- **Wilcoxon tests:** All methods significantly worse than baseline (p<0.01). Pairwise differences between generators at ratio 0.50 are NOT significant (p>0.3) - they are statistically indistinguishable.
- **Ratio sensitivity:** 0.50 > 1.00 for all generators. All ratios degrade below baseline.
- **Per-patient:** LDM improves only 7/23 folds (mostly hard patients with baseline AUPRC < 0.05). Correlation between baseline difficulty and improvement: r = -0.33.
- **Cost-benefit:** LDM has the smallest loss per compute hour (-0.0001 AUPRC/h), but all augmentation methods have negative ROI.

<details>
<summary>Protocol rules enforced in code</summary>

- Patient-level splits only (no leakage)
- Normalization from training data only
- Synthetic data in training only (val/test raise `ValueError`)
- Same frozen detector across E1-E5
- TSTR: synthetic ictal + real interictal only (no real ictal in training)
- 2 synthetic ratios (50%, 100%) per generator in LOPO
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

# Full LOPO evaluation (E1-E5, 23 folds x 3 seeds x 2 ratios, includes TSTR)
bash experiment_scripts/run_lopo.sh

# E6: Cross-generator statistical comparison (no GPU needed)
python -m training.run_e6

# E7: Subject-identity analysis (needs GPU for embedding extraction)
python -m training.run_e7 --device cuda

# Per-patient augmentation analysis
python -m training.run_per_patient
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
├── lopo_status/                    Checkpoint files from run_lopo.sh
├── e6/                             E6 statistical analysis outputs
│   ├── e6_analysis.json              Wilcoxon, ratio sensitivity, cost-benefit
│   └── per_patient_analysis.json     Per-fold augmentation breakdown
└── e7/                             E7 subject-identity analysis
    └── e7_analysis.json              Probe accuracy, proximity ratios
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

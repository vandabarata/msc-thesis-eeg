# Synthetic Data in Healthcare: A Focus on EEG Signals

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="website/images/ista_horizontal_negative.png">
  <source media="(prefers-color-scheme: light)" srcset="website/images/ista_horizontal_main.png">
  <img src="website/images/ista_horizontal_main.png" align="right" width="180" alt="ISCTE-IUL ISTA">
</picture>

_MSc thesis by Vanda Barata at ISCTE-IUL._\
_Supervised by [Ana de Almeida](https://ciencia.iscte-iul.pt/authors/ana-de-almeida) and [Luís Nunes](https://ciencia.iscte-iul.pt/authors/luis-miguel-martins-nunes)._

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
│   └── homogenize.py              EDF cleaning pipeline (already run)
│
├── models/                      Neural network architectures
│   ├── detector.py                1D-CNN seizure detector (frozen, 49K params)
│   ├── timegan.py                 TimeGAN (5 GRU sub-networks, 1.6M params)
│   ├── cvae.py                    Conditional VAE (1D-Conv, 1.9M params)
│   └── ldm.py                     Latent Diffusion (UNet-1D + DDPM/DDIM, 2.3M params)
│
├── training/                    Training, evaluation, and analysis
│   ├── train.py                   Detector training loop (E1/E2), early stopping
│   ├── generate.py                Generator training + synthetic window production (E3-E5)
│   ├── evaluate.py                AUPRC, AUROC, F1, per-patient, Wilcoxon test
│   ├── visualize.py               PSD comparison, t-SNE, amplitude distributions
│   ├── subject_identity.py        E7 linear probe for subject-ID analysis
│   └── run_experiments.sh         Runs all Phase 1 experiments
│
├── notebooks/                   Exploration
│   └── chb-mit-analysis.ipynb     Full EDA notebook
│
├── clean_edfs/                  686 homogenized EDF+ files (not in git)
├── results/                     Experiment outputs (not in git)
├── index.html                   Project website (GitHub Pages)
├── website/                     Website assets
│   └── images/                    ISCTE/ISTA logos
└── requirements.txt             Python dependencies
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

Download the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) and place it under `chb-mit-scalp-eeg-database-1.0.0/`. The homogenization script (`data/homogenize.py`) produces the `clean_edfs/` directory with 686 standardized EDF+ files. Both directories are large and excluded from git.
</details>

---

## Experiments

| Experiment | Description | Status |
|:----------:|-------------|:------:|
| **E1** | Baseline detector (real data, class-weighted CE) | Ready |
| **E2** | Non-synthetic controls (SMOTE, ADASYN) | Ready |
| **E3** | TimeGAN augmentation (25%, 50%, 100%, 200%) | Ready |
| **E4** | CVAE augmentation (same ratios) | Ready |
| **E5** | Latent Diffusion augmentation (same ratios, needs trained CVAE) | After E4 |
| **E6** | Cross-generator comparison (Wilcoxon signed-rank test) | After E1-E5 |
| **E7** | Subject-identity analysis (linear probe on embeddings) | After E6 |

<details>
<summary>Protocol rules enforced in code</summary>

- Patient-level splits only (no leakage)
- Normalization from training data only
- Synthetic data in training only (val/test raise `ValueError`)
- Same frozen detector across E1-E5
- 3 seeds (42, 123, 456), mean +/- std
- AUPRC as primary metric
</details>

### Running Experiments

```bash
source .venv/bin/activate

# E1: Baseline
python -m training.train --experiment e1 --mode single --seeds 42

# E2: SMOTE / ADASYN controls
python -m training.train --experiment e2 --augmentation smote --mode single --seeds 42
python -m training.train --experiment e2 --augmentation adasyn --mode single --seeds 42

# E3: TimeGAN
python -m training.generate --model timegan --ratio 0.25 0.5 1.0 2.0

# E4: CVAE
python -m training.generate --model cvae --ratio 0.25 0.5 1.0 2.0

# E5: LDM (needs pretrained CVAE)
python -m training.generate --model ldm \
    --cvae-checkpoint results/e4/seed_42/single_split/cvae.pt \
    --ratio 0.25 0.5 1.0 2.0

# Full LOPO evaluation (23 folds x 3 seeds)
bash training/run_experiments.sh full
```

### Results Structure

```
results/<experiment>/
├── seed_42/
│   ├── single_split/
│   │   ├── best_model.pt        Model checkpoint
│   │   └── results.json         Metrics + history
│   ├── fold_00/ ... fold_22/    LOPO folds
└── lopo_summary.json            Aggregated results
```

---

## Dataset

**[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)**: 24 cases, 23 unique patients, 23-channel bipolar montage at 256 Hz. 686 EDF files after homogenization, ~5% seizure windows.

Dataset exploration, patient demographics, and the preprocessing pipeline are documented on the [project website](https://vandabarata.github.io/msc-thesis-eeg/).

---

## License

This repository contains code for an MSc thesis. The CHB-MIT dataset is available from [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) under the PhysioNet Credentialed Health Data License.

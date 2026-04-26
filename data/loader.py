"""
data_loader.py - PyTorch Dataset/DataLoader for CHB-MIT EEG windows.

Reads homogenized clean EDF+ files from clean_edfs/, segments them into
sliding windows, labels each window as ictal (1) or interictal (0) based
on seizure timestamps from the original summary files, and provides
PyTorch-compatible Datasets for training and evaluation.

Signal preprocessing (applied before windowing):
  1. Notch filter at 60 Hz + 120 Hz harmonic (US powerline)
  2. Bandpass filter 0.5–40 Hz (4th-order Butterworth, zero-phase)
  3. Amplitude clipping at ±800 µV

Window quality control (applied after windowing):
  - Reject windows where any channel is flat (std < 0.01 µV)
  - Reject windows where >25% of samples in any channel are clipped

Per-case flat-signal caching:
  On first load, preprocessed signals (in µV) are cached as flat
  (23, N) float16 arrays to data/signal_cache/<case_id>_signals.npy,
  alongside a small index file with precomputed valid window positions.
  Windows are sliced on the fly from memory-mapped signals — no overlap
  duplication. Delete data/signal_cache/ to rebuild.

Literature references (thesis SLR numbering):
  [6]  You et al. (2025) — filtering is standard across EEG generation pipelines
  [7]  Sobhani et al. (2025) — FIR bandpass + ICA; FIR for phase preservation
  [23] Dakshit et al. (2023) — exclusion bias from aggressive artefact removal
  [25] Zhao et al. (2022) — artefact-free segment selection for seizure work
  [26] Carrle et al. (2023) — 1–40 Hz bandpass + outlier window rejection + ICLabel
  [29] Chaibi et al. (2024) — band-specific filtering for epileptic patterns

Usage:
    from data.loader import CHBMITDataset, get_dataloaders

    # Quick single-split loaders
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

    # Or build dataset directly
    ds = CHBMITDataset(split="train")
    window, label, patient_id = ds[0]
    # window: Tensor(23, 1024), label: int (0 or 1), patient_id: int (0..23)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, iirnotch

try:
    import pyedflib
except ImportError:
    raise ImportError("pyedflib is required: pip install pyedflib")


# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent            # data/
PROJECT_DIR = BASE_DIR.parent               # project root
CLEAN_EDFS_DIR = PROJECT_DIR / "clean_edfs"
RAW_DB_DIR = PROJECT_DIR / "chb-mit-scalp-eeg-database-1.0.0"
SPLIT_CONFIG_PATH = BASE_DIR / "split_config.json"
NORM_PARAMS_PATH = BASE_DIR / "norm_params.npz"
WINDOW_CACHE_DIR = BASE_DIR / "window_cache"  # Per-case cached windows (µV)

FS = 256                    # Sampling frequency (Hz)
N_CHANNELS = 23             # Number of EEG channels after homogenization
WINDOW_SEC = 4              # Window duration in seconds
WINDOW_SAMPLES = FS * WINDOW_SEC  # 1024 samples per window
OVERLAP = 0.5               # 50% overlap between consecutive windows
STEP_SAMPLES = int(WINDOW_SAMPLES * (1 - OVERLAP))  # 512 samples
ICTAL_THRESHOLD = 0.5       # Fraction of window that must be ictal to label as seizure

# Preprocessing parameters (literature-grounded, see module docstring)
NOTCH_FREQS = [60.0, 120.0] # US powerline 60 Hz + 2nd harmonic
NOTCH_Q = 30.0              # Quality factor for notch filter
BANDPASS_LOW = 0.5           # High-pass cutoff (Hz) — removes DC drift
BANDPASS_HIGH = 40.0         # Low-pass cutoff (Hz) — removes EMG/high-freq noise
BANDPASS_ORDER = 4           # Butterworth filter order
CLIP_UV = 800.0              # Amplitude clipping threshold (µV)

# Window quality control thresholds
QC_FLAT_THRESHOLD = 0.01     # Std below this (µV) = flat channel
QC_CLIP_FRACTION = 0.25      # Reject if >25% of samples in a channel are clipped

# All 24 case IDs in order (used for patient_id integer encoding)
ALL_CASES = [
    "chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08",
    "chb09", "chb10", "chb11", "chb12", "chb13", "chb14", "chb15", "chb16",
    "chb17", "chb18", "chb19", "chb20", "chb21", "chb22", "chb23", "chb24",
]
CASE_TO_ID = {c: i for i, c in enumerate(ALL_CASES)}


# ── Summary Parsing ────────────────────────────────────────────────────────
def parse_seizure_times(summary_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """
    Parse a chbXX-summary.txt file and return seizure times (in seconds)
    for each EDF filename.

    Returns:
        dict mapping edf_filename -> list of (start_sec, end_sec) tuples
    """
    seizures: Dict[str, List[Tuple[int, int]]] = {}

    with open(summary_path, "r") as f:
        lines = f.readlines()

    current_file: Optional[str] = None
    n_seizures = 0
    starts: List[int] = []
    ends: List[int] = []

    for line in lines:
        line = line.strip()

        # New file block
        if line.startswith("File Name:"):
            # Save previous file's data
            if current_file is not None:
                if len(starts) != len(ends):
                    print(f"Warning: {current_file} has {len(starts)} start(s) but {len(ends)} end(s). Truncating to pairs.")
                pairs = list(zip(starts, ends))
                seizures[current_file] = pairs

            current_file = line.split(":")[-1].strip()
            n_seizures = 0
            starts = []
            ends = []
            continue

        if line.startswith("Number of Seizures in File:"):
            try:
                n_seizures = int(line.split(":")[-1].strip())
            except ValueError:
                n_seizures = 0
            continue

        # Handle both formats:
        #   "Seizure Start Time: 2996 seconds"
        #   "Seizure 1 Start Time: 2996 seconds"
        if "Start Time:" in line and line.startswith("Seizure"):
            match = re.search(r"(\d+)\s+seconds", line)
            if match:
                starts.append(int(match.group(1)))
            continue

        if "End Time:" in line and line.startswith("Seizure"):
            match = re.search(r"(\d+)\s+seconds", line)
            if match:
                ends.append(int(match.group(1)))
            continue

    # Don't forget the last file
    if current_file is not None:
        if len(starts) != len(ends):
            print(f"Warning: {current_file} has {len(starts)} start(s) but {len(ends)} end(s). Truncating to pairs.")
        pairs = list(zip(starts, ends))
        seizures[current_file] = pairs

    return seizures


def get_all_seizure_times() -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
    """
    Parse all summary files and return nested dict:
        case_id -> edf_filename -> [(start_sec, end_sec), ...]
    """
    all_times: Dict[str, Dict[str, List[Tuple[int, int]]]] = {}

    for case_id in ALL_CASES:
        summary_path = RAW_DB_DIR / case_id / f"{case_id}-summary.txt"
        if summary_path.exists():
            all_times[case_id] = parse_seizure_times(str(summary_path))
        else:
            all_times[case_id] = {}

    return all_times


# ── EDF Reading ────────────────────────────────────────────────────────────
def read_edf_signals(edf_path: str) -> np.ndarray:
    """
    Read all channels from a clean EDF file.

    Returns:
        np.ndarray of shape (N_CHANNELS, n_samples), dtype float32
    """
    reader = pyedflib.EdfReader(edf_path)
    try:
        n_signals = reader.signals_in_file
        n_samples = reader.getNSamples()[0]
        signals = np.zeros((n_signals, n_samples), dtype=np.float32)
        for i in range(n_signals):
            signals[i] = reader.readSignal(i).astype(np.float32)
    finally:
        reader.close()

    return signals


# ── Signal Preprocessing ───────────────────────────────────────────────────
def preprocess_signals(signals: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Apply literature-grounded preprocessing to raw EEG signals.

    Pipeline (in order):
      1. Notch filter at 60 Hz + 120 Hz (US powerline and 2nd harmonic)
      2. Bandpass filter 0.5–40 Hz (4th-order Butterworth, zero-phase)
      3. Amplitude clipping at ±800 µV

    Args:
        signals: (n_channels, n_samples) array, float32, in µV
        fs: sampling frequency in Hz

    Returns:
        Preprocessed signals, same shape and dtype.

    Literature backing:
      - Notch: You 2025 [6]; standard across EEG generation pipelines
      - Bandpass 0.5–40 Hz: Carrle 2023 [26] (1–40 Hz); Sobhani 2025 [7] (0.3–70 Hz)
      - Zero-phase (filtfilt): preserves seizure onset timing
      - Clipping: Carrle 2023 [26] rejects outlier windows; Dakshit 2023 [23]
        warns against exclusion bias — clipping is a softer alternative
    """
    out = signals.copy()

    # 1. Notch filter (60 Hz + 120 Hz harmonic)
    for freq in NOTCH_FREQS:
        if freq < fs / 2:  # Only apply if below Nyquist
            b, a = iirnotch(freq, NOTCH_Q, fs)
            out = filtfilt(b, a, out, axis=1).astype(np.float32)

    # 2. Bandpass filter 0.5–40 Hz (4th-order Butterworth, zero-phase)
    nyq = fs / 2.0
    low = BANDPASS_LOW / nyq
    high = BANDPASS_HIGH / nyq
    b, a = butter(BANDPASS_ORDER, [low, high], btype="band")
    out = filtfilt(b, a, out, axis=1).astype(np.float32)

    # 3. Amplitude clipping at ±800 µV
    np.clip(out, -CLIP_UV, CLIP_UV, out=out)

    return out


def window_passes_qc(window: np.ndarray) -> bool:
    """
    Check whether a window passes quality control.

    Rejection criteria:
      - Flat channel: if any channel has std < 0.01 µV across the full window
        (catches zero-padded channels from homogenization)
      - Excessive clipping: if >25% of samples in any channel sit at
        the ±800 µV clip boundary

    Args:
        window: (n_channels, window_samples) array

    Returns:
        True if the window passes QC, False if it should be rejected.

    Literature backing:
      - Zhao 2022 [25]: clinical experts selected artefact-free segments
      - Carrle 2023 [26]: rejects windows with extreme amplitude stats
    """
    n_channels, n_samples = window.shape

    for ch in range(n_channels):
        channel = window[ch]

        # Check for flat channel (zero-padded or dead electrode)
        if np.std(channel) < QC_FLAT_THRESHOLD:
            return False

        # Check for excessive clipping
        n_clipped = np.sum(np.abs(channel) >= CLIP_UV - 0.1)
        if n_clipped / n_samples > QC_CLIP_FRACTION:
            return False

    return True


# ── Windowing ──────────────────────────────────────────────────────────────
def create_windows_from_file(
    signals: np.ndarray,
    seizure_times: List[Tuple[int, int]],
    case_id: str,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Segment signals into sliding windows and label each.

    Args:
        signals: (N_CHANNELS, n_samples) array
        seizure_times: list of (start_sec, end_sec) seizure intervals
        case_id: patient case ID string

    Returns:
        List of (window, label, patient_id) tuples where:
            window: np.ndarray(N_CHANNELS, WINDOW_SAMPLES) float32
            label: 0 (interictal) or 1 (ictal)
            patient_id: integer index into ALL_CASES
    """
    n_samples = signals.shape[1]
    patient_id = CASE_TO_ID[case_id]

    # Build sample-level seizure mask
    seizure_mask = np.zeros(n_samples, dtype=bool)
    for start_sec, end_sec in seizure_times:
        start_sample = start_sec * FS
        end_sample = end_sec * FS
        start_sample = max(0, min(start_sample, n_samples))
        end_sample = max(0, min(end_sample, n_samples))
        seizure_mask[start_sample:end_sample] = True

    windows = []
    pos = 0
    while pos + WINDOW_SAMPLES <= n_samples:
        window = signals[:, pos : pos + WINDOW_SAMPLES]
        # Label: ictal if >= ICTAL_THRESHOLD of window samples are in seizure
        ictal_frac = seizure_mask[pos : pos + WINDOW_SAMPLES].mean()
        label = 1 if ictal_frac >= ICTAL_THRESHOLD else 0
        windows.append((window, label, patient_id))
        pos += STEP_SAMPLES

    return windows


# ── Normalization ──────────────────────────────────────────────────────────
def compute_normalization_params(
    cases: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and std from all files in the given cases.
    Uses Welford's online algorithm (vectorized per-file batch update)
    to avoid loading everything into memory.

    Prefers signal caches (preprocessed float16 mmap) when available, falling
    back to reading and preprocessing clean EDFs.  This allows norm params to
    be computed on machines that only have the caches (no clean_edfs/).

    Returns:
        (means, stds) each of shape (N_CHANNELS,)
    """
    count = np.zeros(N_CHANNELS, dtype=np.float64)
    mean = np.zeros(N_CHANNELS, dtype=np.float64)
    m2 = np.zeros(N_CHANNELS, dtype=np.float64)

    def _welford_update(signals_f64: np.ndarray) -> None:
        n_samples = signals_f64.shape[1]
        for ch in range(N_CHANNELS):
            ch_data = signals_f64[ch]
            new_count = count[ch] + n_samples
            ch_mean = ch_data.mean()
            ch_m2 = ch_data.var() * n_samples
            delta = ch_mean - mean[ch]
            mean[ch] = (count[ch] * mean[ch] + n_samples * ch_mean) / new_count
            m2[ch] += ch_m2 + delta ** 2 * count[ch] * n_samples / new_count
            count[ch] = new_count

    CHUNK = 256 * 3600  # ~1 hour of signal at 256 Hz

    for case_id in cases:
        sig_path = SIGNAL_CACHE_DIR / f"{case_id}_signals.npy"
        if sig_path.exists():
            sig = np.load(str(sig_path), mmap_mode='r')
            if sig.shape[0] == N_CHANNELS and sig.shape[1] > 0:
                for start in range(0, sig.shape[1], CHUNK):
                    chunk = sig[:, start:start + CHUNK].astype(np.float64)
                    _welford_update(chunk)
            continue

        edf_dirs = _get_edf_dirs_for_case(case_id)
        for edf_dir in edf_dirs:
            for edf_file in sorted(edf_dir.glob("*.edf")):
                signals = read_edf_signals(str(edf_file))
                if signals.shape[0] != N_CHANNELS:
                    continue
                signals = preprocess_signals(signals)
                _welford_update(signals.astype(np.float64))

    stds = np.sqrt(m2 / np.maximum(count - 1, 1))
    stds = np.maximum(stds, 1e-8)

    return mean.astype(np.float32), stds.astype(np.float32)


def _get_edf_dirs_for_case(case_id: str) -> List[Path]:
    """
    Get all clean_edfs subdirectories for a given case.
    Handles both simple (chb01/) and multi-set (chb04_set0/, chb04_set1/).
    """
    dirs = []
    simple = CLEAN_EDFS_DIR / case_id
    if simple.is_dir():
        dirs.append(simple)

    # Multi-set directories: chbXX_set0, chbXX_set1, ...
    for d in sorted(CLEAN_EDFS_DIR.iterdir()):
        if d.is_dir() and d.name.startswith(case_id + "_set"):
            dirs.append(d)

    return dirs


# ── Flat-Signal Cache ─────────────────────────────────────────────────────
# Instead of caching pre-windowed arrays (which double storage due to 50%
# overlap), we cache flat preprocessed signals per case and compute windows
# on the fly via memory-mapped slicing.
#
# Per case the cache contains:
#   <case_id>_signals.npy  — (23, total_samples) float16, uncompressed (mmap)
#   <case_id>_index.npz    — compressed; keys: starts, labels, pid, qc_rejected
#     starts:  (N,) int64  — valid window start positions (QC-passed)
#     labels:  (N,) int8   — 0=interictal, 1=ictal
#     pid:     scalar int8 — patient id for this case
#     qc_rejected: scalar  — count of QC-rejected windows

SIGNAL_CACHE_DIR = BASE_DIR / "signal_cache"


def _build_signal_cache(case_id: str) -> None:
    """Build flat-signal cache for one case from its clean EDF files."""
    import tempfile

    SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    all_seizure_times = get_all_seizure_times()
    seizure_times_by_file = all_seizure_times.get(case_id, {})
    edf_dirs = _get_edf_dirs_for_case(case_id)
    patient_id = CASE_TO_ID[case_id]

    file_signals = []
    file_boundaries = []
    file_seizure_times = []
    cumulative_samples = 0

    for edf_dir in edf_dirs:
        for edf_file in sorted(edf_dir.glob("*.edf")):
            seizure_times = seizure_times_by_file.get(edf_file.name, [])
            try:
                signals = read_edf_signals(str(edf_file))
            except Exception as e:
                print(f"    Warning: failed to read {edf_file}: {e}")
                continue
            if signals.shape[0] != N_CHANNELS:
                print(f"    Warning: {edf_file} has {signals.shape[0]} channels, skipping")
                continue

            signals = preprocess_signals(signals)
            n_samples = signals.shape[1]

            file_signals.append(signals.astype(np.float16))
            file_boundaries.append((cumulative_samples, cumulative_samples + n_samples))
            file_seizure_times.append(seizure_times)
            cumulative_samples += n_samples
            del signals

    if cumulative_samples == 0:
        flat = np.empty((N_CHANNELS, 0), dtype=np.float16)
    else:
        flat = np.concatenate(file_signals, axis=1)
    del file_signals

    # Build seizure mask over full concatenated signal
    seizure_mask = np.zeros(cumulative_samples, dtype=bool)
    for (offset, _), times in zip(file_boundaries, file_seizure_times):
        for start_sec, end_sec in times:
            s = offset + max(0, start_sec * FS)
            e = offset + min(cumulative_samples - offset, end_sec * FS)
            seizure_mask[s:e] = True

    # Enumerate windows, run QC, record valid positions
    starts = []
    labels = []
    qc_rejected = 0
    pos = 0

    # Only create windows within file boundaries (no cross-file windows)
    for (f_start, f_end) in file_boundaries:
        pos = f_start
        while pos + WINDOW_SAMPLES <= f_end:
            window = flat[:, pos:pos + WINDOW_SAMPLES]
            if window_passes_qc(window.astype(np.float32)):
                ictal_frac = seizure_mask[pos:pos + WINDOW_SAMPLES].mean()
                label = 1 if ictal_frac >= ICTAL_THRESHOLD else 0
                starts.append(pos)
                labels.append(label)
            else:
                qc_rejected += 1
            pos += STEP_SAMPLES

    # Write signal .npy (atomic via temp file)
    # suffix must end with .npy so np.save doesn't append another .npy
    sig_path = SIGNAL_CACHE_DIR / f"{case_id}_signals.npy"
    fd, tmp = tempfile.mkstemp(dir=str(SIGNAL_CACHE_DIR), suffix=".tmp.npy")
    os.close(fd)
    try:
        np.save(tmp, flat)
        os.rename(tmp, str(sig_path))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    del flat

    # Write index .npz (small, compressed)
    idx_path = SIGNAL_CACHE_DIR / f"{case_id}_index.npz"
    np.savez_compressed(
        str(idx_path),
        starts=np.array(starts, dtype=np.int64),
        labels=np.array(labels, dtype=np.int8),
        pid=np.array(patient_id, dtype=np.int8),
        qc_rejected=np.array(qc_rejected),
    )
    print(f"  {case_id}: cached {len(starts)} windows, {qc_rejected} QC-rejected, "
          f"signal shape {(N_CHANNELS, cumulative_samples)}")


class _WindowsProxy:
    """Sequence proxy so dataset.windows[i] and iteration still work.

    Each access slices one window from the memory-mapped flat signal.
    No window data is held in memory between accesses.
    """

    def __init__(self, dataset: "CHBMITDataset"):
        self._ds = dataset

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        if idx < 0:
            idx += len(self)
        if self._ds._active_indices is not None:
            idx = int(self._ds._active_indices[idx])
        if idx < self._ds._n_real:
            return self._ds._get_real_window(idx)
        else:
            syn_idx = idx - self._ds._n_real
            return self._ds._synthetic_windows[syn_idx]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# ── Case-Aware Batch Sampler ──────────────────────────────────────────────
class CaseAwareSampler(torch.utils.data.Sampler):
    """Yields indices grouped by case to keep mmap page cache warm.

    Within each case, window indices are shuffled.  Case order is also
    shuffled each epoch.  Every index is yielded exactly once per epoch,
    so training sees the same data as a fully-shuffled sampler.
    """

    def __init__(self, dataset: "CHBMITDataset", seed: int = 42):
        self._cumulative = dataset._cumulative
        self._n_real = dataset._n_real
        self._n_synthetic = dataset._n_synthetic
        self._active_indices = dataset._active_indices
        self._seed = seed
        self._epoch = 0

    def __len__(self) -> int:
        if self._active_indices is not None:
            return len(self._active_indices)
        return self._n_real + self._n_synthetic

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.RandomState(self._seed + self._epoch)

        if self._active_indices is not None:
            perm = rng.permutation(len(self._active_indices))
            yield from perm.tolist()
            return

        n_cases = len(self._cumulative) - 1
        case_order = rng.permutation(n_cases)

        for case_idx in case_order:
            start = int(self._cumulative[case_idx])
            end = int(self._cumulative[case_idx + 1])
            case_indices = rng.permutation(np.arange(start, end))
            yield from case_indices.tolist()

        if self._n_synthetic > 0:
            syn_indices = rng.permutation(np.arange(self._n_real, self._n_real + self._n_synthetic))
            yield from syn_indices.tolist()


# ── Dataset ────────────────────────────────────────────────────────────────
class CHBMITDataset(Dataset):
    """
    PyTorch Dataset backed by memory-mapped flat signals.

    Each case's preprocessed EEG is stored as a flat (23, N) float16 array.
    Windows are sliced on the fly — no duplication from overlap. Total cache
    is ~35-40 GB (vs ~64-74 GB for pre-windowed), and RAM usage is ~10 MB
    regardless of dataset size.

    Args:
        split: "train", "val", or "test"
        split_config_path: path to split_config.json
        normalize: whether to apply z-score normalization
        ictal_only: if True, only return ictal (seizure) windows
        fold: if set, use LOPO fold N instead of single_split
        synthetic_windows: optional list of (window, label, patient_id) tuples.
            ONLY allowed for split="train".
        seed: random seed for reproducible val split

    Leakage prevention (enforced at init):
        - No patient overlap between train/val/test splits
        - Normalization params keyed to training cases (hash-based caching)
        - QC runs before normalization (thresholds in µV)
        - Synthetic data blocked from val/test splits
    """

    def __init__(
        self,
        split: str = "train",
        split_config_path: Optional[str] = None,
        normalize: bool = True,
        ictal_only: bool = False,
        fold: Optional[int] = None,
        synthetic_windows: Optional[List[Tuple[np.ndarray, int, int]]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.split = split
        self.normalize = normalize
        self.ictal_only = ictal_only
        self.seed = seed

        config_path = split_config_path or str(SPLIT_CONFIG_PATH)
        with open(config_path, "r") as f:
            config = json.load(f)

        # ── Determine cases for this split ─────────────────────────────
        if fold is not None:
            lopo = config["lopo_folds"][fold]
            all_train = lopo["train_cases"]
            test_cases = lopo["test_cases"]
            self.train_cases = all_train

            if split == "test":
                self.cases = test_cases
            else:
                rng = np.random.RandomState(seed)
                indices = rng.permutation(len(all_train))
                val_size = max(1, len(all_train) // 5)
                val_indices = set(indices[:val_size])

                if split == "val":
                    self.cases = [all_train[i] for i in sorted(val_indices)]
                else:
                    self.cases = [all_train[i] for i in range(len(all_train)) if i not in val_indices]

            assert not (set(self.cases) & set(test_cases)) or split == "test", \
                f"DATA LEAKAGE in LOPO fold {fold}: {set(self.cases) & set(test_cases)}"
        else:
            self.cases = config["single_split"][split]
            self.train_cases = config["single_split"]["train"]

            all_splits = config["single_split"]
            for a_name, b_name in [("train", "val"), ("train", "test"), ("val", "test")]:
                a_set = set(all_splits.get(a_name, []))
                b_set = set(all_splits.get(b_name, []))
                overlap = a_set & b_set
                assert not overlap, (
                    f"DATA LEAKAGE: patients {overlap} appear in both "
                    f"'{a_name}' and '{b_name}' splits!"
                )

        # ── Synthetic data safeguard ───────────────────────────────────
        if synthetic_windows is not None and split != "train":
            raise ValueError(
                f"Synthetic data can ONLY be added to the 'train' split, not '{split}'. "
                f"Evaluating on synthetic data violates the experimental protocol. "
                f"See thesis Section 1.4 and Carrle et al. (2023)."
            )

        # Load normalization params (compute if missing)
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        if normalize:
            self._load_or_compute_norm_params()

        # Build flat-signal index (lazy mmap — files opened on demand)
        self._signal_paths: List[Path] = []          # per-case signal file paths
        self._signal_cache: Dict[int, np.ndarray] = {}  # LRU cache of open mmaps
        self._signal_cache_order: List[int] = []     # access order for LRU eviction
        self._signal_cache_max: int = 4              # max open mmaps at once
        self._window_starts: List[np.ndarray] = []   # per-case arrays of start positions
        self._cumulative: np.ndarray = np.array([0], dtype=np.int64)
        self._all_labels: np.ndarray = np.empty(0, dtype=np.int8)
        self._all_pids: np.ndarray = np.empty(0, dtype=np.int8)
        self._qc_rejected = 0
        self._build_index()
        self._n_real = int(self._cumulative[-1])

        # Add synthetic windows (training only — enforced above)
        self._synthetic_windows: List[Tuple[np.ndarray, int, int]] = []
        if synthetic_windows is not None:
            for window, label, patient_id in synthetic_windows:
                if self.normalize and self.mean is not None and self.std is not None:
                    window = (window - self.mean[:, None]) / self.std[:, None]
                self._synthetic_windows.append((window, label, patient_id))
        self._n_synthetic = len(self._synthetic_windows)

        # Cache class counts
        real_ictal = int(np.sum(self._all_labels == 1)) if len(self._all_labels) > 0 else 0
        syn_ictal = sum(1 for _, l, _ in self._synthetic_windows if l == 1)
        self._n_ictal = real_ictal + syn_ictal
        self._n_interictal = (self._n_real + self._n_synthetic) - self._n_ictal

        # Filter to ictal only if requested
        self._active_indices: Optional[np.ndarray] = None
        if ictal_only:
            real_ictal_idx = np.where(self._all_labels == 1)[0]
            syn_ictal_idx = np.array([
                self._n_real + i for i, (_, l, _) in enumerate(self._synthetic_windows) if l == 1
            ], dtype=np.int64)
            self._active_indices = np.concatenate([real_ictal_idx, syn_ictal_idx]) if len(syn_ictal_idx) > 0 else real_ictal_idx
            self._n_ictal = len(self._active_indices)
            self._n_interictal = 0

        self.windows = _WindowsProxy(self)

    def _load_or_compute_norm_params(self):
        """Load or compute per-channel normalization params from training data."""
        import hashlib
        cases_key = hashlib.md5(",".join(sorted(self.train_cases)).encode()).hexdigest()[:8]
        norm_path = NORM_PARAMS_PATH.parent / f"norm_params_{cases_key}.npz"

        if norm_path.exists():
            params = np.load(str(norm_path))
            self.mean = params["mean"]
            self.std = params["std"]
        else:
            print(f"Computing normalization params from {len(self.train_cases)} training cases (key={cases_key})...")
            self.mean, self.std = compute_normalization_params(self.train_cases)
            np.savez(str(norm_path), mean=self.mean, std=self.std)
            print(f"Saved normalization params to {norm_path}")

    def _build_index(self):
        """Build index over flat-signal caches (lazy mmap).

        For each case: ensure signal cache exists, store the path (NOT
        opened yet), and load the small index file.  Mmaps are opened on
        demand in _get_mmap() with LRU eviction, so only a few cases are
        resident at any time.  Total RAM: ~10 MB for metadata arrays.
        """
        SIGNAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        all_labels = []
        all_pids = []
        cumulative = [0]

        for case_id in self.cases:
            sig_path = SIGNAL_CACHE_DIR / f"{case_id}_signals.npy"
            idx_path = SIGNAL_CACHE_DIR / f"{case_id}_index.npz"

            if not sig_path.exists() or not idx_path.exists():
                print(f"  Building signal cache for {case_id}...")
                _build_signal_cache(case_id)

            index_data = np.load(str(idx_path))

            starts = index_data["starts"]
            labels = index_data["labels"]
            pid = int(index_data["pid"])
            self._qc_rejected += int(index_data["qc_rejected"])

            n_windows = len(starts)
            self._signal_paths.append(sig_path)
            self._window_starts.append(starts)
            all_labels.append(labels)
            all_pids.append(np.full(n_windows, pid, dtype=np.int8))
            cumulative.append(cumulative[-1] + n_windows)

        self._cumulative = np.array(cumulative, dtype=np.int64)
        self._all_labels = np.concatenate(all_labels) if all_labels else np.empty(0, dtype=np.int8)
        self._all_pids = np.concatenate(all_pids) if all_pids else np.empty(0, dtype=np.int8)

    def _get_mmap(self, case_idx: int) -> np.ndarray:
        """Return mmap for a case, opening lazily with LRU eviction."""
        if case_idx in self._signal_cache:
            order = self._signal_cache_order
            order.remove(case_idx)
            order.append(case_idx)
            return self._signal_cache[case_idx]

        if len(self._signal_cache) >= self._signal_cache_max:
            evict = self._signal_cache_order.pop(0)
            del self._signal_cache[evict]

        mmap = np.load(str(self._signal_paths[case_idx]), mmap_mode='r')
        self._signal_cache[case_idx] = mmap
        self._signal_cache_order.append(case_idx)
        return mmap

    def _resolve_index(self, idx: int) -> Tuple[int, int]:
        """Map global real-window index to (case_index, local_window_index)."""
        case_idx = int(np.searchsorted(self._cumulative[1:], idx, side='right'))
        local_idx = idx - int(self._cumulative[case_idx])
        return case_idx, local_idx

    def _get_real_window(self, idx: int) -> Tuple[np.ndarray, int, int]:
        """Slice one window from the memory-mapped flat signal."""
        case_idx, local_idx = self._resolve_index(idx)
        start = int(self._window_starts[case_idx][local_idx])
        signal_mmap = self._get_mmap(case_idx)
        window = signal_mmap[:, start:start + WINDOW_SAMPLES]
        window = np.ascontiguousarray(window, dtype=np.float32)
        if self.normalize and self.mean is not None and self.std is not None:
            window = (window - self.mean[:, None]) / self.std[:, None]
        label = int(self._all_labels[idx])
        patient_id = int(self._all_pids[idx])
        return window, label, patient_id

    def __len__(self) -> int:
        if self._active_indices is not None:
            return len(self._active_indices)
        return self._n_real + self._n_synthetic

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        if self._active_indices is not None:
            idx = int(self._active_indices[idx])

        if idx < self._n_real:
            window, label, patient_id = self._get_real_window(idx)
        else:
            syn_idx = idx - self._n_real
            window, label, patient_id = self._synthetic_windows[syn_idx]

        return torch.from_numpy(np.ascontiguousarray(window)), label, patient_id

    def get_class_counts(self) -> Tuple[int, int]:
        """Return (n_interictal, n_ictal) counts."""
        return self._n_interictal, self._n_ictal

    def get_class_weights(self) -> torch.Tensor:
        """Return class weights inversely proportional to frequency."""
        n_interictal, n_ictal = self.get_class_counts()
        total = n_interictal + n_ictal
        w_interictal = total / (2 * max(n_interictal, 1))
        w_ictal = total / (2 * max(n_ictal, 1))
        return torch.tensor([w_interictal, w_ictal], dtype=torch.float32)


# ── DataLoader Factory ─────────────────────────────────────────────────────
def get_dataloaders(
    batch_size: int = 64,
    num_workers: int = 0,
    normalize: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders (single-split mode)."""
    train_ds = CHBMITDataset(split="train", normalize=normalize, seed=seed)
    val_ds = CHBMITDataset(split="val", normalize=normalize, seed=seed)
    test_ds = CHBMITDataset(split="test", normalize=normalize, seed=seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def get_lopo_dataloaders(
    fold: int,
    batch_size: int = 64,
    num_workers: int = 0,
    normalize: bool = True,
    seed: int = 42,
    synthetic_windows: Optional[List[Tuple[np.ndarray, int, int]]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, and test DataLoaders for a specific LOPO fold."""
    train_ds = CHBMITDataset(
        split="train", fold=fold, normalize=normalize,
        seed=seed, synthetic_windows=synthetic_windows,
    )
    val_ds = CHBMITDataset(split="val", fold=fold, normalize=normalize, seed=seed)
    test_ds = CHBMITDataset(split="test", fold=fold, normalize=normalize, seed=seed)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ── CLI Verification ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("CHB-MIT EEG Dataset Loader - Verification")
    print("=" * 60)

    print(f"\nConfig:")
    print(f"  Clean EDFs dir:    {CLEAN_EDFS_DIR}")
    print(f"  Signal cache dir:  {SIGNAL_CACHE_DIR}")
    print(f"  Split config:      {SPLIT_CONFIG_PATH}")
    print(f"  Window:            {WINDOW_SEC}s = {WINDOW_SAMPLES} samples")
    print(f"  Overlap:           {OVERLAP * 100:.0f}%")
    print(f"  Step:              {STEP_SAMPLES} samples")

    # Allow building caches without loading full dataset
    if "--build-cache" in sys.argv:
        print(f"\nBuilding signal caches for all cases...")
        config_path = str(SPLIT_CONFIG_PATH)
        with open(config_path) as f:
            cfg = json.load(f)
        all_cases_set: set = set()
        for fold_cfg in cfg["lopo_folds"]:
            all_cases_set.update(fold_cfg["train_cases"])
            all_cases_set.update(fold_cfg["test_cases"])
        for case_id in sorted(all_cases_set):
            sig_path = SIGNAL_CACHE_DIR / f"{case_id}_signals.npy"
            if sig_path.exists():
                print(f"  {case_id}: already cached, skipping")
                continue
            _build_signal_cache(case_id)
        print("\nDone! All signal caches built.")
        sys.exit(0)

    print(f"\nLoading training set...")
    train_ds = CHBMITDataset(split="train")
    n_inter, n_ictal = train_ds.get_class_counts()
    print(f"  Total windows:  {len(train_ds)}")
    print(f"  Interictal:     {n_inter} ({n_inter / len(train_ds) * 100:.2f}%)")
    print(f"  Ictal:          {n_ictal} ({n_ictal / len(train_ds) * 100:.2f}%)")
    print(f"  QC rejected:    {train_ds._qc_rejected}")
    print(f"  Class weights:  {train_ds.get_class_weights().tolist()}")

    w, l, p = train_ds[0]
    print(f"\n  Sample shape:   {w.shape}")
    print(f"  Sample dtype:   {w.dtype}")
    print(f"  Sample label:   {l}")
    print(f"  Sample patient: {ALL_CASES[p]} (id={p})")

    # Also test _WindowsProxy
    w2, l2, p2 = train_ds.windows[0]
    assert w2.shape == (N_CHANNELS, WINDOW_SAMPLES), f"Proxy shape mismatch: {w2.shape}"
    print(f"  Proxy test:     OK (shape {w2.shape})")

    print(f"\nLoading validation set...")
    val_ds = CHBMITDataset(split="val")
    n_inter_v, n_ictal_v = val_ds.get_class_counts()
    print(f"  Total windows:  {len(val_ds)}")
    print(f"  Interictal:     {n_inter_v}")
    print(f"  Ictal:          {n_ictal_v}")

    print(f"\nLoading test set...")
    test_ds = CHBMITDataset(split="test")
    n_inter_t, n_ictal_t = test_ds.get_class_counts()
    print(f"  Total windows:  {len(test_ds)}")
    print(f"  Interictal:     {n_inter_t}")
    print(f"  Ictal:          {n_ictal_t}")

    total = len(train_ds) + len(val_ds) + len(test_ds)
    total_ictal = n_ictal + n_ictal_v + n_ictal_t
    print(f"\n{'=' * 60}")
    print(f"TOTAL windows across all splits: {total}")
    if total > 0:
        print(f"TOTAL ictal windows:             {total_ictal} ({total_ictal / total * 100:.2f}%)")
    else:
        print(f"TOTAL ictal windows:             {total_ictal} (no windows loaded)")
    print(f"{'=' * 60}")

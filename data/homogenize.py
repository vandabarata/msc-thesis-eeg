"""
Homogenize CHB-MIT Scalp EEG EDF files into bz2-compressed pickles.

Adapted from https://github.com/bernia/chb-mit-scalp/blob/master/homogenize_signals.py

- Scans local CHB-MIT folder (e.g., chb-mit-scalp-eeg-database-1.0.0/)
- For each patient folder chbXX/, reads all *.edf files
- Uses chbXX-summary.txt to:
  - infer channel layouts ("Channels in EDF Files" + any "Channels changed" blocks)
  - map each EDF file to the active channel layout at that point
  - extract seizure counts + seizure start/end times
- For each EDF:
  - selects only the channels belonging to its layout
  - pads missing channels with zeros so every file in a layout has the same set
  - saves as: clean_signals/chbXX[/_setK]/<edf_name>.pkl.pbz2

Patients with montage changes (e.g., chb17) are written into multiple folders:
  clean_signals/chb17_set0, clean_signals/chb17_set1, ...
"""

from __future__ import annotations

import os
import glob
import bz2
import pickle
import multiprocessing as mp
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pyedflib
import pyedflib.highlevel as hl


DEFAULT_FS = 256

# The fixed 23-channel bipolar montage that every clean EDF must have.
# Derived from the majority of CHB-MIT recordings.  Sets with fewer
# channels get zero-padded (QC will reject windows on flat channels).
TARGET_MONTAGE: List[str] = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
    "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-2",
]

NDArray = np.ndarray
Metadata = Dict[str, Any]
Record = Dict[str, Union[NDArray, Metadata]]
SignalDict = Dict[str, NDArray]


# -----------------------------
# Utilities
# -----------------------------
def _unique_label(label: str, seen: Dict[str, int]) -> str:
    """Make channel labels unique: X, X-2, X-3, ..."""
    if label not in seen:
        seen[label] = 1
        return label
    seen[label] += 1
    return f"{label}-{seen[label]}"  # first repeat becomes '-2'


def _base_label(label: str) -> str:
    """Convert 'FP1-F7-2' -> 'FP1-F7' to match EDF header label."""
    parts = label.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label


def save_pbz2(path_no_ext: str, obj: Any) -> str:
    """Save obj as bz2-compressed pickle: <path_no_ext>.pbz2"""
    out_path = path_no_ext + ".pbz2"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with bz2.BZ2File(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def edf_labels_and_fs(edf_path: str) -> Tuple[List[str], int]:
    """Read labels + sampling rate without loading all signals."""
    r = pyedflib.EdfReader(edf_path)
    try:
        # pyedflib stubs are weak → normalize to list[str]
        labels = [str(x) for x in r.getSignalLabels()]
        fs = int(r.getSampleFrequency(0)) if r.signals_in_file > 0 else DEFAULT_FS
    finally:
        r.close()
    return labels, fs


# -----------------------------
# Summary parsing
# -----------------------------
def parse_summary_channel_sets_and_filemap(
    summary_path: str,
) -> Tuple[Dict[int, List[str]], Dict[str, int]]:
    """
    Parse chbXX-summary.txt to:
      - channel_sets[set_id] = ordered channel labels (unique with -2/-3)
      - file_to_set[edf_filename] = which set_id applies to that file
    """
    channel_sets: Dict[int, List[str]] = {}
    file_to_set: Dict[str, int] = {}

    current_set = 0
    in_channel_block = False
    seen_per_set: Dict[int, Dict[str, int]] = {0: {}}
    channel_sets[0] = []

    with open(summary_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Channels in EDF Files"):
                current_set = 0
                in_channel_block = True
                channel_sets.setdefault(0, [])
                seen_per_set.setdefault(0, {})
                continue

            if line.startswith("Channels changed"):
                current_set += 1
                in_channel_block = True
                channel_sets[current_set] = []
                seen_per_set[current_set] = {}
                continue

            parts = line.split()

            # Channel lines like: "Channel 1: FP1-F7"
            if in_channel_block and parts[0] == "Channel" and len(parts) >= 3:
                label = parts[2]
                if label not in {"-", "."}:
                    label_u = _unique_label(label, seen_per_set[current_set])
                    channel_sets[current_set].append(label_u)
                continue

            # File lines like: "File Name: chb17a_03.edf"
            if parts[0] == "File" and len(parts) >= 3 and parts[1] == "Name:":
                in_channel_block = False
                edf_name = parts[2]
                file_to_set[edf_name] = current_set
                continue

    return channel_sets, file_to_set


def process_metadata(summary_path: str, edf_filename: str, fs: int) -> Metadata:
    """
    Extract seizure metadata for a given EDF filename from summary text.
    """
    meta: Metadata = {"seizures": 0, "times": []}

    with open(summary_path, "r") as f:
        lines = f.readlines()

    start_i: Optional[int] = None
    for i, raw in enumerate(lines):
        if raw.startswith("File Name:"):
            parts = raw.split()
            if len(parts) >= 3 and parts[2].strip() == edf_filename:
                start_i = i
                break

    if start_i is None:
        return meta

    end_i = len(lines)
    for j in range(start_i + 1, len(lines)):
        if lines[j].startswith("File Name:"):
            end_i = j
            break

    block = lines[start_i:end_i]

    seizures = 0
    for raw in block:
        if raw.startswith("Number of Seizures in File:"):
            try:
                seizures = int(raw.split()[-1])
            except Exception:
                seizures = 0
            break
    meta["seizures"] = seizures

    times: List[Tuple[int, int]] = []
    if seizures > 0:
        start_secs: Dict[int, int] = {}
        end_secs: Dict[int, int] = {}

        for raw in block:
            if raw.startswith("Seizure") and "Start Time:" in raw:
                p = raw.split()
                try:
                    idx = int(p[1])
                    sec = int(p[-2])
                    start_secs[idx] = sec
                except Exception:
                    pass
            elif raw.startswith("Seizure") and "End Time:" in raw:
                p = raw.split()
                try:
                    idx = int(p[1])
                    sec = int(p[-2])
                    end_secs[idx] = sec
                except Exception:
                    pass

        for k in range(1, seizures + 1):
            if k in start_secs and k in end_secs:
                s = start_secs[k] * fs - 1
                e = end_secs[k] * fs - 1
                times.append((s, e))

    meta["times"] = times
    return meta


# -----------------------------
# EDF selection / normalization
# -----------------------------
def choose_indices_for_file(edf_path: str, valid_channels: List[str]) -> Optional[Tuple[List[int], int]]:
    """
    Select EDF channel indices to match valid_channels (including duplicates).
    Returns (indices, fs) or None if no matching channels were found.
    """
    try:
        labels, fs = edf_labels_and_fs(edf_path)
    except Exception:
        return None

    required: Dict[str, int] = {}
    for ch in valid_channels:
        base = _base_label(ch)
        required[base] = required.get(base, 0) + 1

    selected_counts: Dict[str, int] = {k: 0 for k in required}
    indices: List[int] = []

    for idx, label in enumerate(labels):
        if label in required and selected_counts[label] < required[label]:
            indices.append(idx)
            selected_counts[label] += 1

    if not indices:
        return None
    return indices, fs


def read_edf_by_indices(edf_path: str, indices: List[int], fs_fallback: int) -> Tuple[SignalDict, int]:
    """
    Read EDF selecting only the given channel indices, return (channel_dict, fs).
    Keys are unique labels (X, X-2, ...).
    """
    signals, signal_headers, _ = hl.read_edf(edf_path, ch_nrs=indices, digital=False)

    seen: Dict[str, int] = {}
    out: SignalDict = {}
    fs = int(fs_fallback) if fs_fallback else DEFAULT_FS

    for i, (sig, hdr) in enumerate(zip(signals, signal_headers)):
        # hdr.get('label') is typed as Unknown|None in stubs → force str
        raw_label = hdr.get("label")
        label = str(raw_label) if raw_label else f"CH{i+1}"

        label_u = _unique_label(label, seen)
        out[label_u] = np.asarray(sig)

        # Some headers may have sample_rate
        sr = hdr.get("sample_rate")
        if isinstance(sr, (int, float)) and sr:
            fs = int(sr)

    return out, fs


def homogenize_to_valid_channels(clean_dict: Record, valid_channels: List[str]) -> Record:
    """
    Ensure exactly valid_channels exist (in that order); add missing channels as zeros.
    Preserves metadata if present.
    """
    sample_len: Optional[int] = None
    for k, v in clean_dict.items():
        if k == "metadata":
            continue
        if isinstance(v, np.ndarray):
            sample_len = int(len(v))
            break
    if sample_len is None:
        raise ValueError("No signal channels found to homogenize.")

    out: Record = {}

    for ch in valid_channels:
        v = clean_dict.get(ch)
        if isinstance(v, np.ndarray):
            out[ch] = v
        else:
            out[ch] = np.zeros(sample_len, dtype=float)

    meta = clean_dict.get("metadata")
    if isinstance(meta, dict):
        out["metadata"] = cast(Metadata, meta)

    return out


# -----------------------------
# Patient processing
# -----------------------------


# Polygraphy bipolar pairs: ocular (LOC/ROC) and auricular (LUE/RAE) channels
# that appear in some CHB-MIT recordings alongside the EEG channels.
_POLYGRAPHY_CHANNELS: frozenset = frozenset({
    "LOC-ROC", "ROC-LOC", "LUE-RAE", "RAE-LUE",
})


def _is_unwanted_channel(label: str) -> bool:
    """Return True for labels that should be dropped from clean signals.

    Dropped categories (beyond the original ECG/EKG/DUMMY/"-"/"."):
      - VNS: vagal nerve stimulator pulse channel (chb09 set 1).
      - Polygraphy pairs (LOC-ROC, LUE-RAE): eye-movement / auricular
        channels, not EEG (chb12 set 3, chb13 set 1).
      - Common-reference suffixes (*-CS2, *-Ref): channels recorded in a
        referential montage rather than the standard bipolar montage
        (chb12 sets 0/1, chb15 sets 0/1).
      - Unipolar / absolute labels (no hyphen): single-electrode names
        such as "C2", "CZ", "T7" that appear in chb12 set 0 alongside
        the bipolar channels.
    """
    stripped = label.strip()
    if stripped in {"-", "."}:
        return True
    u = stripped.upper()
    if "ECG" in u or "EKG" in u:
        return True
    if "DUMMY" in u:
        return True
    # vagal nerve stimulator
    if u == "VNS":
        return True
    # polygraphy (ocular / auricular)
    if u in _POLYGRAPHY_CHANNELS:
        return True
    # common-reference montage suffixes — not bipolar EEG
    if u.endswith("-CS2") or u.endswith("-REF"):
        return True
    # unipolar / absolute channel: no hyphen means it is not a bipolar pair
    if "-" not in stripped:
        return True
    return False


def process_patient_auto(pacient: str, signals_path: str, clean_path: str,
                         force: bool = False) -> None:
    """Process one patient's raw EDFs into bz2-compressed pickles.

    Set *force=True* to overwrite existing output files (e.g. after updating
    the channel-filtering logic).  The default (False) skips files that have
    already been written, which is useful for resuming an interrupted run.
    """
    patient_in_dir = os.path.join(signals_path, f"chb{pacient}")
    summary_path = os.path.join(patient_in_dir, f"chb{pacient}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"[chb{pacient}] Missing summary: {summary_path}")
        return

    channel_sets, file_to_set = parse_summary_channel_sets_and_filemap(summary_path)

    # filter out any ECG/dummy channels from the layouts
    for set_id, chans in list(channel_sets.items()):
        filtered = [ch for ch in chans if not _is_unwanted_channel(ch)]
        if len(filtered) != len(chans):
            print(f"[chb{pacient}] removing {len(chans)-len(filtered)} unwanted channel(s) "
                  f"from set {set_id}")
        channel_sets[set_id] = filtered

    edf_paths = sorted(glob.glob(os.path.join(patient_in_dir, "*.edf")))
    if not edf_paths:
        print(f"[chb{pacient}] No EDF files found.")
        return

    groups: Dict[int, List[str]] = {}
    for p in edf_paths:
        base = os.path.basename(p)
        set_id = file_to_set.get(base, 0)
        groups.setdefault(set_id, []).append(p)

    multi_set = len(groups) > 1
    processed = 0
    skipped = 0

    for set_id, paths in groups.items():
        valid_channels = channel_sets.get(set_id)
        if not valid_channels:
            print(f"[chb{pacient}] No channels parsed for set {set_id}; skipping this set.")
            continue

        out_dir = os.path.join(clean_path, f"chb{pacient}_set{set_id}" if multi_set else f"chb{pacient}")
        os.makedirs(out_dir, exist_ok=True)

        for edf_path in paths:
            base = os.path.basename(edf_path)

            sel = choose_indices_for_file(edf_path, valid_channels)
            if sel is None:
                skipped += 1
                continue
            indices, fs_guess = sel

            try:
                sig_dict, fs = read_edf_by_indices(edf_path, indices, fs_fallback=fs_guess)
            except Exception as e:
                print(f"[chb{pacient}] Failed reading {base}: {e}")
                skipped += 1
                continue

            # just in case the EDF header had an ECG/dummy channel we didn't
            # anticipate, drop it here so the downstream homogenization never
            # sees it.  (valid_channels was already filtered above.)
            for ch in list(sig_dict.keys()):
                if _is_unwanted_channel(ch):
                    sig_dict.pop(ch, None)

            meta = process_metadata(summary_path, base, fs=fs)
            meta["channels"] = valid_channels
            meta["fs"] = fs
            meta["montage_set"] = set_id

            record: Record = dict(sig_dict)
            record["metadata"] = meta

            try:
                record = homogenize_to_valid_channels(record, valid_channels)
            except Exception as e:
                print(f"[chb{pacient}] Failed homogenizing {base}: {e}")
                skipped += 1
                continue

            out_base = os.path.join(out_dir, f"{base}.pkl")
            if not force and os.path.exists(out_base + ".pbz2"):
                skipped += 1
                continue
            save_pbz2(out_base, record)
            processed += 1

    print(f"[chb{pacient}] done. processed={processed}, skipped={skipped}, sets={sorted(groups.keys())}")


# -----------------------------
# Main
# -----------------------------
def main(force: bool = False) -> None:
    signals_path = "chb-mit-scalp-eeg-database-1.0.0"
    clean_path = "clean_signals"

    if not os.path.isdir(signals_path):
        raise FileNotFoundError(f"signals_path not found: {signals_path}")

    os.makedirs(clean_path, exist_ok=True)

    patient_dirs = sorted(
        d for d in os.listdir(signals_path)
        if d.startswith("chb") and len(d) == 5 and d[3:].isdigit()
        and os.path.isdir(os.path.join(signals_path, d))
    )
    patients = [d[3:] for d in patient_dirs]

    workers = max(1, mp.cpu_count() - 1)
    print(f"Found {len(patients)} patients. Using {workers} workers.")
    with mp.Pool(processes=workers) as pool:
        pool.starmap(process_patient_auto, [(p, signals_path, clean_path, force) for p in patients])


def clean_pickles(clean_path: str) -> None:
    """Walk *clean_path* and strip ECG/dummy channels from existing pickles.

    This helper lets you fix a previously‑generated ``clean_signals`` tree
    without rerunning the whole EDF conversion. It edits files in‑place and
    prints a message for each record that was altered.
    """
    for root, _, files in os.walk(clean_path):
        for f in files:
            if not f.endswith(".pkl.pbz2"):
                continue
            path = os.path.join(root, f)
            try:
                with bz2.BZ2File(path, "rb") as fin:
                    rec = pickle.load(fin)
            except Exception:
                print(f"failed to load {path}")
                continue

            meta = rec.get("metadata", {})
            chans = list(meta.get("channels", []))
            filtered = [ch for ch in chans if not _is_unwanted_channel(ch)]
            if len(filtered) == len(chans):
                continue

            # remove the stray signals too
            for k in list(rec.keys()):
                if k != "metadata" and _is_unwanted_channel(k):
                    rec.pop(k, None)
            meta["channels"] = filtered
            rec["metadata"] = meta

            with bz2.BZ2File(path, "wb") as fout:
                pickle.dump(rec, fout, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"cleaned {path}: dropped {len(chans)-len(filtered)} channel(s)")


# -----------------------------
# EDF output path
# -----------------------------

def _signal_header(label: str, sig: NDArray, fs: int) -> dict:
    pmin = float(sig.min())
    pmax = float(sig.max())
    if pmin == pmax:        # flat signal — give it a nonzero range
        pmin -= 1.0
        pmax += 1.0
    
    # Round physical min/max to avoid EDF+ truncation warnings
    # EDF+ allows only 8 characters for these values
    pmin = round(pmin)
    pmax = round(pmax)
    
    return {
        "label":            label,
        "dimension":        "uV",
        "sample_frequency": fs,
        "physical_min":     pmin,
        "physical_max":     pmax,
        "digital_min":      -32768,
        "digital_max":       32767,
    }


def write_clean_edf(out_path: str, valid_channels: List[str],
                    sig_dict: SignalDict, meta: Metadata) -> None:
    """Write the homogenized signals as an EDF+ file.

    Missing channels (present in valid_channels but not in sig_dict) are
    written as zero-padded signals, exactly like the pickle path.
    Seizure start/end sample pairs in meta['times'] become EDF+ annotations.
    """
    fs: int = int(meta["fs"])

    # Determine length from the first real signal
    sample_len: int = 0
    for ch in valid_channels:
        arr = sig_dict.get(ch)
        if isinstance(arr, np.ndarray):
            sample_len = len(arr)
            break

    signals: List[NDArray] = []
    headers: List[dict] = []
    for ch in valid_channels:
        arr = sig_dict.get(ch)
        sig = np.asarray(arr, dtype=np.float64) if isinstance(arr, np.ndarray) \
              else np.zeros(sample_len, dtype=np.float64)
        signals.append(sig)
        headers.append(_signal_header(ch, sig, fs))

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with pyedflib.EdfWriter(out_path, len(valid_channels),
                            file_type=pyedflib.FILETYPE_EDFPLUS) as writer:
        writer.setSignalHeaders(headers)
        montage = meta.get("montage_set", 0)
        writer.setRecordingAdditional(f"montage_set={montage}")
        writer.writeSamples(signals)
        for start_s, end_s in meta.get("times", []):
            writer.writeAnnotation(start_s / fs, (end_s - start_s) / fs, "seizure")


def process_patient_to_edf(pacient: str, signals_path: str, clean_edf_path: str,
                            force: bool = False) -> None:
    """Like process_patient_auto but writes clean EDF+ files instead of pickles.

    Every output EDF has exactly TARGET_MONTAGE (23 bipolar channels) in a
    fixed order.  Channels present in the raw EDF are selected; missing
    channels are zero-padded (downstream QC rejects windows with flat
    channels).  Files where *none* of the target channels can be found are
    skipped entirely.

    Set *force=True* to overwrite existing output files (e.g. after updating
    the channel-filtering logic).  The default (False) skips files that have
    already been written, which is useful for resuming an interrupted run.
    """
    patient_in_dir = os.path.join(signals_path, f"chb{pacient}")
    summary_path = os.path.join(patient_in_dir, f"chb{pacient}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"[chb{pacient}] Missing summary: {summary_path}")
        return

    channel_sets, file_to_set = parse_summary_channel_sets_and_filemap(summary_path)

    for set_id, chans in list(channel_sets.items()):
        filtered = [ch for ch in chans if not _is_unwanted_channel(ch)]
        if len(filtered) != len(chans):
            print(f"[chb{pacient}] removing {len(chans)-len(filtered)} unwanted channel(s) "
                  f"from set {set_id}")
        channel_sets[set_id] = filtered

    edf_paths = sorted(glob.glob(os.path.join(patient_in_dir, "*.edf")))
    if not edf_paths:
        print(f"[chb{pacient}] No EDF files found.")
        return

    groups: Dict[int, List[str]] = {}
    for p in edf_paths:
        base = os.path.basename(p)
        set_id = file_to_set.get(base, 0)
        groups.setdefault(set_id, []).append(p)

    multi_set = len(groups) > 1
    processed = 0
    skipped = 0

    for set_id, paths in groups.items():
        valid_channels = channel_sets.get(set_id)
        if not valid_channels:
            print(f"[chb{pacient}] No channels parsed for set {set_id}; skipping this set.")
            continue

        out_dir = os.path.join(
            clean_edf_path,
            f"chb{pacient}_set{set_id}" if multi_set else f"chb{pacient}",
        )

        for edf_path in paths:
            base = os.path.basename(edf_path)
            out_path = os.path.join(out_dir, base)

            if not force and os.path.exists(out_path):
                skipped += 1
                continue

            sel = choose_indices_for_file(edf_path, valid_channels)
            if sel is None:
                skipped += 1
                continue
            indices, fs_guess = sel

            try:
                sig_dict, fs = read_edf_by_indices(edf_path, indices, fs_fallback=fs_guess)
            except Exception as e:
                print(f"[chb{pacient}] Failed reading {base}: {e}")
                skipped += 1
                continue

            for ch in list(sig_dict.keys()):
                if _is_unwanted_channel(ch):
                    sig_dict.pop(ch, None)

            # Skip files where none of the target channels are present
            target_set = set(TARGET_MONTAGE)
            if not (set(sig_dict.keys()) & target_set):
                print(f"[chb{pacient}] {base}: no target channels found, skipping")
                skipped += 1
                continue

            meta = process_metadata(summary_path, base, fs=fs)
            meta["channels"] = TARGET_MONTAGE
            meta["fs"] = fs
            meta["montage_set"] = set_id

            try:
                write_clean_edf(out_path, TARGET_MONTAGE, sig_dict, meta)
            except Exception as e:
                print(f"[chb{pacient}] Failed writing {base}: {e}")
                skipped += 1
                continue

            processed += 1

    print(f"[chb{pacient}] done. processed={processed}, skipped={skipped}, sets={sorted(groups.keys())}")


def main_edf(force: bool = False) -> None:
    signals_path = "chb-mit-scalp-eeg-database-1.0.0"
    clean_edf_path = "clean_edfs"

    if not os.path.isdir(signals_path):
        raise FileNotFoundError(f"signals_path not found: {signals_path}")

    os.makedirs(clean_edf_path, exist_ok=True)

    patient_dirs = sorted(
        d for d in os.listdir(signals_path)
        if d.startswith("chb") and len(d) == 5 and d[3:].isdigit()
        and os.path.isdir(os.path.join(signals_path, d))
    )
    patients = [d[3:] for d in patient_dirs]

    workers = max(1, mp.cpu_count() - 1)
    print(f"Found {len(patients)} patients. Using {workers} workers.")
    with mp.Pool(processes=workers) as pool:
        pool.starmap(process_patient_to_edf, [(p, signals_path, clean_edf_path, force) for p in patients])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Homogenize CHB-MIT EDF files.")
    parser.add_argument(
        "--edf", action="store_true",
        help="Write clean EDF+ files (clean_edfs/) instead of pickles (clean_signals/).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output files. Without this flag, already-written files are skipped.",
    )
    args = parser.parse_args()

    if args.edf:
        main_edf(force=args.force)
    else:
        main(force=args.force)

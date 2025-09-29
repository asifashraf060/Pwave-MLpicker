"""
ETL: per-geometry manifest (CSV) -> WebDataset shards (.tar of .npz),
with robust SEG-Y<->JSON shot alignment.

Inputs (per geometry directory):
  geometries/<GEOM_ID>/picks_manifest.csv    # from build_manifest_from_segy.py
  geometries/<GEOM_ID>/geometry.json         # defines the JSON shot range
  geometries/<GEOM_ID>/*.segy                # station SEG-Y files

The manifest must have columns:
  geometry, station, segy_path, trace_index, shot_id, pick_time_s

We DO NOT rely on manifest trace_index mapping. Instead, we:
  1) Open each station's SEG-Y once, extract per-trace shot numbers from headers.
  2) Read the JSON shot list (geometry.json).
  3) Tolerantly align SEG-Y shot series to JSON shots (±tol; monotone).
  4) For each trace, get shot_id from alignment and lookup pick_time_s from manifest.
     (If not found -> 0.0 = no pick.)

Each sample in shards contains:
  x       : float16 (C,T)  # feature tensor (raw + noise channels)
  y_mask  : int8   (T,)    # 1 in ±POS_WIN_S around pick; all zero if no pick
  has_pick: int8   ()      # 1 if pick_time_s>0 else 0
  y_idx   : int32  ()      # pick index in samples; -1 if no pick
  fs      : float32()      # sampling rate

Usage:
  python etl_to_wds_from_manifest.py /ABS/ML-feed --out /ABS/wds
  python etl_to_wds_from_manifest.py /ABS/ML-feed --out /ABS/wds --geom SO-SG1 SO-SG2
"""

from __future__ import annotations
import argparse, io, json, sys, tarfile
from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.random import uniform as rn
import pandas as pd
from obspy.io.segy.core import _read_segy

from scipy import signal as scisig
from scipy.signal import welch
from obspy.signal.trigger import classic_sta_lta
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt

# -------------------- knobs (override by CLI) --------------------
TARGET_LEN   = 30000            # default window length (samples)
DTYPE_X      = "float16"        # on-disk dtype for x
SHARD_SIZE   = 10_000           # samples per .tar (aim ~1–2 GB)
NOISE_PRE_S  = 2.0              # seconds used for noise features
POS_WIN_S    = 0.04             # ± window (sec) to mark positives in y_mask
ALIGN_TOL    = 2                # max |SEG-Y-shot - JSON-shot| for a match

# Windowing / cropping
WIN_MODE     = "pick"            # "pick" | "center" | "none" | "fixed"
PRE_S  = 2.0                     # seconds before pick
POST_S = 3.0                     # seconds after pick
TOTAL_S = 5.0                    # keep for negatives; positives use PRE_S+POST_S

# fixed-window defaults
FIXED_START_S = 0.0   # seconds
FIXED_LEN_S   = 10.0  # seconds

# ── Noise window gating (applies when WIN_MODE="pick" and no pick is present)
NEG_STALTA_THRESH     = 2.0   # accept window only if max(STA/LTA) ≤ this
NEG_STALTA_STA_S      = 0.10  # STA window (sec)
NEG_STALTA_LTA_S      = 2.00  # LTA window (sec)
NEG_STALTA_MAX_TRIES  = 3     # how many random tries before giving up

# Negatives (no-pick traces)
NEG_MODE     = "random"   # "random" | "leading" | "trailing"
NEG_K        = 1          # how many negative windows to write per no-pick trace

# --- Negatives vs Positives control (per SEG-Y / station) ---
NEG_TO_POS_MAX_RATIO = 0.5   # keep at most 0.5× negatives vs positives within one station

PLOTS_POS_DEFAULT = 40
PLOTS_NEG_DEFAULT = 10

# ----------------------------------------------------------------
GEOM_DIRNAME  = "geometries"
MANIFEST_NAME = "picks_manifest.csv"
GEOMETRY_JSON = "geometry.json"

# Candidate SEG-Y trace header fields that may encode "shot number"
SHOT_HEADER_CANDIDATES = [
    "energy_source_point_number",         # Rev1: bytes 17–20
    "source_point",
    "ensemble_number",                    # sometimes shotpoint/CDP
    "field_record_number",                # FFID
    "original_field_record_number",
    "trace_sequence_number_within_line",  # fallback (1..N)
]

# ---------- Feature configuration ----------
STA_LTA_CONFIGS = [
    {"sta": 0.5, "lta": 10.0},   # fast
    {"sta": 1.0, "lta": 20.0},   # medium
]

FREQ_BANDS = [                     # Hz
    {"name": "mid",  "fmin": 5.0,  "fmax": 15.0}
]

MAXAMP_WINDOWS = np.array([100], dtype=int)  # samples, as in your pipeline
BANDPASS_FOR_MAXAMP = (3.0, 12.0)                      # Hz, from your pipeline notes

# ----------------------------- utils -----------------------------

def robust_std(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return float(1.4826 * mad)

def _robust_std(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return float(1.4826 * mad)

def _safe_bandpass(x: np.ndarray, fmin: float, fmax: float, fs: float) -> np.ndarray:
    nyq = fs * 0.5
    fmin = max(0.1, min(fmin, nyq * 0.95))
    fmax = max(fmin + 0.1, min(fmax, nyq * 0.98))
    try:
        return bandpass(x, fmin, fmax, fs, corners=2, zerophase=True).astype(np.float32, copy=False)
    except Exception:
        # very short signals can fail; just return zeros
        return np.zeros_like(x, dtype=np.float32)

def _pad_or_center_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    T = x.size
    if T == target_len:
        return x
    if T > target_len:
        s = (T - target_len) // 2
        return x[s:s+target_len]
    pad = target_len - T
    l = pad // 2; r = pad - l
    return np.pad(x, (l, r), mode="constant")

def _maxamp_feature(trace: np.ndarray, fs: float, target_len: int) -> np.ndarray:
    """
    Replicates your MaxAmplitudeFeature:
      - bandpass 3–12 Hz
      - |x|, normalize by max, invert (1-norm), zero out <0.01
      - sliding window std for each window in MAXAMP_WINDOWS
      - resize to target_len
    Returns shape (len(MAXAMP_WINDOWS), T)
    """
    low, high = BANDPASS_FOR_MAXAMP
    x = _safe_bandpass(trace, low, high, fs)
    x = np.abs(x)
    mx = float(np.max(x)) + 1e-9
    nrml = x / mx
    zero_mask = nrml < 0.01
    nrml_inv = 1.0 - nrml
    nrml_inv[zero_mask] = 0.0

    out = []
    for wlen in MAXAMP_WINDOWS:
        wlen = int(wlen)
        if wlen <= 1 or wlen > nrml_inv.size:
            out.append(np.zeros_like(trace))
            continue
        # sliding std via cumulative sums (fast) on CPU
        # compute moving mean
        csum = np.cumsum(nrml_inv, dtype=np.float64)
        csum = np.concatenate([[0.0], csum])
        mean = (csum[wlen:] - csum[:-wlen]) / wlen  # length N-w+1

        # compute moving mean of squares
        sq = nrml_inv.astype(np.float64) ** 2
        csum2 = np.cumsum(sq)
        csum2 = np.concatenate([[0.0], csum2])
        mean2 = (csum2[wlen:] - csum2[:-wlen]) / wlen

        std = np.sqrt(np.clip(mean2 - mean**2, 0.0, None)).astype(np.float32)
        std = _pad_or_center_trim(std, target_len)
        out.append(std)
    return np.stack(out, axis=0)  # (n_wins, T)

def _spectral_flatness_welch(x: np.ndarray, fs: float, nperseg=256, fband=None) -> float:
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, x.size))
    if fband is not None:
        m = (f >= fband[0]) & (f <= fband[1])
        if not np.any(m): return 1.0
        Pxx = Pxx[m]
    Pxx = np.clip(Pxx, 1e-16, None)
    gmean = np.exp(np.mean(np.log(Pxx)))
    amean = np.mean(Pxx)
    return float(gmean / (amean + 1e-16))

# ---- helpers used by feature builder ----
def bandpass_3_12(x: np.ndarray, fs: float) -> np.ndarray:
    # 3–12 Hz, stable even for short traces
    return _safe_bandpass(x, 3.0, 12.0, fs)

def hilbert_safe(x: np.ndarray) -> np.ndarray:
    # returns analytic signal (complex)
    try:
        return scisig.hilbert(x.astype(np.float32, copy=False))
    except Exception:
        # fallback: zero-phase bandpass + i*HT(x) approximation if ever needed
        return scisig.hilbert(np.ascontiguousarray(x, dtype=np.float32))

def stalta_ratio(x: np.ndarray, fs: float, sta_s: float, lta_s: float) -> np.ndarray:
    """Per-sample STA/LTA ratio (vector), rectangular windows."""
    x = np.abs(x.astype(np.float32, copy=False))
    n_sta = max(1, int(round(sta_s * fs)))
    n_lta = max(n_sta + 1, int(round(lta_s * fs)))
    sta = _movmean(x, n_sta)
    lta = _movmean(x, n_lta)
    eps = 1e-9
    r = sta / (lta + eps)
    r[lta <= eps] = 0.0
    return r.astype(np.float32, copy=False)

def sliding_std(x: np.ndarray, w: int) -> np.ndarray:
    """Std over a centered sliding window of length w; returns len(x)."""
    w = int(w)
    if w <= 1 or w > x.size:
        return np.zeros_like(x, dtype=np.float32)
    m1 = _movmean(x.astype(np.float32, copy=False), w)
    m2 = _movmean((x.astype(np.float32))**2, w)
    out = np.sqrt(np.clip(m2 - m1**2, 0.0, None))
    return out.astype(np.float32, copy=False)

def smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Simple moving average smoothing with window w."""
    w = max(1, int(w))
    return _movmean(x, w)

def build_features(seg: np.ndarray, fs: float, pick_time_in_seg: float = None,
                  seg_start_time: float = 0.0) -> np.ndarray:
    """
    Returns (C,T) features derived entirely from the 3–12 Hz band-passed signal.
    Channels (7):
      0: bp 3–12 Hz
      1: STA/LTA (fast)
      2: STA/LTA (med)
      3: envelope
      4: band env mid (smoothed envelope)
      5: maxamp sliding std
      6: SNR (pick-centric robust amplitude / quiet baseline, broadcast across T)

    Parameters:
    -----------
    seg : np.ndarray
        Waveform segment
    fs : float
        Sampling rate
    pick_time_in_seg : float, optional
        Pick time relative to start of segment (seconds). If None, uses 2/3 of segment length.
    seg_start_time : float
        Start time of segment relative to full trace (for debugging)
    """
    x = np.asarray(seg, dtype=np.float32)
    T = x.size
    if T < 4:
        return np.zeros((7, T), dtype=np.float32)

    # --- 3–12 Hz bandpass (basis for everything) ---
    bp = bandpass_3_12(x, fs).astype(np.float32, copy=False)

    # --- Envelope ---
    env = np.abs(hilbert_safe(bp)).astype(np.float32)

    # --- STA/LTA on bp ---
    sta_fast = stalta_ratio(bp, fs, sta_s=0.05, lta_s=0.5)
    sta_med  = stalta_ratio(bp, fs, sta_s=0.10, lta_s=1.0)

    # --- "mid-band env": smoothed envelope ---
    band_env_mid = smooth(env, int(max(1, 0.05 * fs))).astype(np.float32)  # ~50 ms

    # --- Max-amp sliding std on |bp| ---
    win_std = max(3, int(0.10 * fs))  # ~100 ms
    maxamp_std = sliding_std(np.abs(bp), win_std).astype(np.float32)

    # --- New Pick-Centric SNR Calculation ---
    if pick_time_in_seg is None:
        # Fallback: use 2/3 of segment length as time mark
        pick_time_in_seg = (2.0/3.0) * T / fs

    pick_idx = int(round(pick_time_in_seg * fs))

    # Check bounds
    if pick_idx < 0 or pick_idx >= T:
        raise ValueError(f"Pick time {pick_time_in_seg}s (idx {pick_idx}) exceeds segment bounds [0, {T/fs:.2f}s]")

    # Signal window: [pick-0.3s, pick+0.5s]
    sig_start_idx = max(0, pick_idx - int(0.3 * fs))
    sig_end_idx = min(T, pick_idx + int(0.5 * fs))

    if sig_end_idx <= sig_start_idx:
        raise ValueError(f"Invalid signal window around pick at {pick_time_in_seg}s")

    # Numerator: 92nd percentile of |bp| in signal window
    signal_window = np.abs(bp[sig_start_idx:sig_end_idx])
    signal_amplitude = float(np.percentile(signal_window, 92))

    # Denominator: Find quiet 1-second baseline using STA/LTA gating
    baseline_len_samples = int(1.0 * fs)  # 1 second baseline
    sta_lta_thresh = 2.0  # Start with threshold 2.0
    max_thresh = 10.0     # Don't go above this
    thresh_step = 0.5

    baseline_std = None
    current_thresh = sta_lta_thresh

    # Search for quiet baseline window in pre-pick region
    search_end_idx = max(0, pick_idx - int(0.1 * fs))  # End search 0.1s before pick

    while current_thresh <= max_thresh and baseline_std is None:
        # Try to find a quiet window
        for start_idx in range(0, max(1, search_end_idx - baseline_len_samples + 1)):
            end_idx = start_idx + baseline_len_samples
            if end_idx > search_end_idx:
                break

            baseline_segment = bp[start_idx:end_idx]
            if baseline_segment.size == 0:
                continue

            # Check if this window is quiet enough
            max_stalta = stalta_max(baseline_segment, fs,
                                  sta_s=NEG_STALTA_STA_S, lta_s=NEG_STALTA_LTA_S)

            if max_stalta <= current_thresh:
                # Found a quiet window
                baseline_std = robust_std(baseline_segment) + 1e-6
                break

        if baseline_std is None:
            current_thresh += thresh_step

    # If no quiet window found, use first 1 second as fallback
    if baseline_std is None:
        fallback_end = min(baseline_len_samples, search_end_idx)
        if fallback_end > 0:
            baseline_std = robust_std(bp[:fallback_end]) + 1e-6
        else:
            baseline_std = 1e-6  # Last resort

    # Calculate SNR
    snr_scalar = float(signal_amplitude / baseline_std)
    snr_arr = np.full((T,), snr_scalar, dtype=np.float32)  # broadcast across time

    feats = [
        bp,            # 0
        sta_fast,      # 1
        sta_med,       # 2
        env,           # 3
        band_env_mid,  # 4
        maxamp_std,    # 5
        snr_arr,       # 6
    ]
    X = np.vstack([f if f.shape == (T,) else np.resize(f, (T,)) for f in feats]).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X


def pad_or_center_trim(x: np.ndarray, target_len: int) -> np.ndarray:
    T = x.size
    if T == target_len: return x
    if T > target_len:
        s = (T - target_len) // 2
        return x[s:s+target_len]
    pad = target_len - T
    l = pad // 2; r = pad - l
    return np.pad(x, (l, r), mode="constant")

def load_geometry_json(geom_dir: Path) -> dict:
    path = geom_dir / GEOMETRY_JSON
    try:
        with path.open("r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path} (line {e.lineno}, col {e.colno}).") from e

def expand_json_shots(assets: dict) -> List[int]:
    sh = assets.get("shots", {})
    start = int(sh.get("start"))
    stop  = int(sh.get("stop"))
    step  = int(sh.get("step", 1))
    end_exclusive = bool(sh.get("end_exclusive", True))
    rng = range(start, stop, step) if end_exclusive else range(start, stop + step, step)
    return list(rng)

def extract_shots_from_segy(stream) -> Optional[List[int]]:
    """
    Extract per-trace 'shot numbers' from SEG-Y headers using common keys.
    Accepts the first key that yields a monotone non-decreasing sequence with sufficient diversity.
    """
    def get_series(key: str):
        vals = []
        for tr in stream:
            th = tr.stats.segy.trace_header
            if not hasattr(th, key):
                return None
            try:
                vals.append(int(getattr(th, key)))
            except Exception:
                return None
        return vals

    for key in SHOT_HEADER_CANDIDATES:
        series = get_series(key)
        if series is None:
            continue
        nondec = all(series[i] <= series[i+1] for i in range(len(series)-1))
        diverse = (len(set(series)) > max(10, len(series)//50))
        if nondec and diverse:
            return series
    return None

def align_shot_series(segy_shots: List[int], json_shots: List[int], tol: int = ALIGN_TOL):
    """
    Monotone two-pointer alignment allowing small mismatches/gaps.
    Returns (mapped_json_for_trace, stats_dict).
    mapped_json_for_trace[i] is the JSON shot id matched to trace i (or None).
    """
    i, j = 0, 0
    N, M = len(segy_shots), len(json_shots)
    mapped = [None] * N
    dropped_segy, dropped_json = [], []

    while i < N and j < M:
        a, b = segy_shots[i], json_shots[j]
        if abs(a - b) <= tol:
            mapped[i] = b
            i += 1; j += 1
        elif a < b - tol:
            dropped_segy.append((i, a)); i += 1
        else:
            dropped_json.append((j, b)); j += 1

    while i < N:
        dropped_segy.append((i, segy_shots[i])); i += 1
    while j < M:
        dropped_json.append((j, json_shots[j])); j += 1

    stats = {
        "matched": sum(1 for x in mapped if x is not None),
        "unmatched_traces": len(dropped_segy),
        "unmatched_json": len(dropped_json),
        "dropped_traces_preview": dropped_segy[:5],
        "dropped_json_preview": dropped_json[:5],
    }
    return mapped, stats

# --------------------------- ETL core ----------------------------

@dataclass
class ETLArgs:
    root: Path
    out: Path
    geoms: Optional[List[str]]
    target_len: int
    shard_size: int
    pos_win_s: float
    align_tol: int
    win_mode: str
    pre_s: float
    post_s: float
    neg_mode: str
    neg_k: int
    neg_stalta_thresh: float
    neg_stalta_sta_s: float
    neg_stalta_lta_s: float
    neg_stalta_max_tries: int
    total_win_s: float
    neg_to_pos_max: float
    plots_pos : int
    plots_neg : int
    fixed_start_s: float
    fixed_len_s: float

def _find_geom_picks_manifest(geom_dir: Path) -> Path | None:
    p = geom_dir / "picks_manifest.csv"
    return p if p.exists() else None

def _load_picks_manifest(csv_path: Path, geom_id: str):
    """
    Loads the geometry-local picks_manifest.csv with columns:
      geometry, station, segy_path, trace_index, shot_id, pick_time_s

    Returns:
      has_pick_map: {(station:str, shot:int) -> (has_pick:int, pick_time:float)}
      pos_count_by_station: {station:str -> int}
    """
    df = pd.read_csv(csv_path)

    # filter to this geometry (file already local, but do it for safety)
    if "geometry" in df.columns:
        df = df[df["geometry"].astype(str) == str(geom_id)]

    # normalize types
    df["station"] = df["station"].astype(str).str.strip()
    df["shot_id"] = df["shot_id"].astype(float).astype(int)
    df["pick_time_s"] = pd.to_numeric(df["pick_time_s"], errors="coerce").fillna(0.0)

    # has_pick: 1 if pick_time_s > 0, else 0
    df["has_pick"] = (df["pick_time_s"] > 0.0).astype(int)

    # Build maps
    has_pick_map = { (row.station, int(row.shot_id)) : (int(row.has_pick), float(row.pick_time_s))
                     for row in df.itertuples(index=False) }

    pos_count_by_station = (
        df.groupby("station")["has_pick"].sum().astype(int).to_dict()
    )

    total_pos = int(df["has_pick"].sum())
    uniq_stations = df["station"].nunique()
    print(f"[manifest] {csv_path.name}: {total_pos} positives across {uniq_stations} station(s)")
    return has_pick_map, pos_count_by_station


def _movmean(a: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return a.astype(np.float32, copy=False)
    # same-length moving average via convolution
    kernel = np.ones(n, dtype=np.float32) / float(n)
    return np.convolve(a.astype(np.float32, copy=False), kernel, mode="same")

def stalta_max(x: np.ndarray, fs: float, sta_s: float, lta_s: float) -> float:
    """
    Return max(STA/LTA) over x (1D), using rect. windows of sta_s and lta_s seconds.
    """
    x = np.abs(x.astype(np.float32, copy=False))
    n_sta = max(1, int(round(sta_s * fs)))
    n_lta = max(n_sta + 1, int(round(lta_s * fs)))

    # Handle case where window is larger than signal
    if n_lta >= len(x) or n_sta >= len(x):
        # For very short signals, just return a simple ratio
        return float(np.mean(x) / (np.mean(x) + 1e-9))

    sta = _movmean(x, n_sta)
    lta = _movmean(x, n_lta)

    # Ensure both arrays have the same length (should be the case with mode="same")
    min_len = min(len(sta), len(lta))
    sta = sta[:min_len]
    lta = lta[:min_len]

    eps = 1e-9
    ratio = sta / (lta + eps)
    # ignore edges where LTA is tiny
    return float(np.nanmax(np.where(lta > eps, ratio, 0.0)))

def choose_windows(T, fs, pick_time, args, full_trace=None, tm_ax=None, duration=None):
    """
    Returns list of (start, end, has_pick, y_idx_in_win, noise_end_idx) in samples.
    When WIN_MODE='pick' and no pick exists:
      - draws NEG_K windows according to NEG_MODE
      - if NEG_MODE='random', tries up to neg_stalta_max_tries to satisfy STA/LTA gate.
    """
    out = []
    if args.win_mode == "none":
        L = T
        out.append((0, L, int(pick_time > 0), int(round(pick_time * fs)) if pick_time > 0 else -1,
                    min(int(NOISE_PRE_S * fs), L)))
        return out

    if args.win_mode == "center":
        L = min(args.target_len, T)
        s0 = max(0, (T - L) // 2)
        e0 = s0 + L
        yidx = int(round(pick_time * fs)) - s0 if pick_time > 0 else -1
        has = int(pick_time > 0 and 0 <= yidx < L)
        out.append((s0, e0, has, yidx if has else -1, min(int(NOISE_PRE_S * fs), L)))
        return out

    # args.win_mode == "fixed": use user-specified [start, start+len)
    if args.win_mode == "fixed":
        # Compute sample counts
        W   = int(round(max(0.01, float(getattr(args, "fixed_len_s", 10.0))) * fs))
        T   = int(T)  # native samples in the full trace
        s0  = int(round(max(0.0, float(getattr(args, "fixed_start_s", 0.0))) * fs))

        # Slide inside the trace without padding:
        # - keep the requested start if it fits
        # - otherwise clamp so that [s0, s0+W) is contained in [0, T)
        s0  = max(0, min(s0, max(0, T - W)))
        e0  = min(T, s0 + W)

        # time axis is only used to find pick index; but we don't need it if we have s0/e0
        has = 0; yidx = -1
        if pick_time > 0.0:
            pidx = int(round(pick_time * fs))
            if s0 <= pidx < e0:
                has = 1
                yidx = pidx - s0  # pick index relative to the crop

        # noise_end_idx for SNR and features
        noise_end_idx = max(1, int(min(NOISE_PRE_S, W / fs) * fs))
        out.append((s0, e0, has, yidx, noise_end_idx))
        return out


    if pick_time > 0:
        # --- Time-based crop: 2 s before and 3 s after pick ---
        # We prefer using the provided time axis (from header); if missing, synthesize it.
        if tm_ax is None or duration is None:
            # Fallback: synthesize from fs if not supplied
            duration = (T / max(fs, 1e-9))
            tm_ax = np.linspace(0.0, duration, T, endpoint=False)

        # Desired time window
        t_pre  = float(getattr(args, "pre_s", 2.0))   # 2.0 by default
        t_post = float(getattr(args, "post_s", 3.0))  # 3.0 by default
        t0     = max(0.0, pick_time - t_pre)
        t1     = min(float(duration), pick_time + t_post)

        # Convert times -> sample indices using the time axis
        # searchsorted gives the first index where tm_ax[idx] >= t
        s0 = int(np.searchsorted(tm_ax, t0, side="left"))
        e0 = int(np.searchsorted(tm_ax, t1, side="left"))

        # Guardrails: keep indices in-bounds, ensure at least 2 samples
        s0 = max(0, min(s0, T - 2))
        e0 = max(s0 + 2, min(e0, T))

        # Pick index within the crop (nearest sample at/after pick_time)
        yidx = int(np.searchsorted(tm_ax[s0:e0], pick_time, side="left"))

        # Noise-only features should only “see” pre-pick part
        # If you keep NOISE_PRE_S, cap it; else use yidx
        if "NOISE_PRE_S" in globals():
            noise_end_idx = min(yidx, int(NOISE_PRE_S * fs))
        else:
            noise_end_idx = max(1, yidx)

        out.append((s0, e0, 1, yidx, noise_end_idx))
        return out



    # --- No pick: choose NEG_K 'noise' windows of the same fixed length L
    # Fixed length for negative windows - use same duration as positives (pre_s + post_s)
    negative_duration = args.pre_s + args.post_s
    L = min(int(round(negative_duration * fs)), T)
    rng = np.random.default_rng()

    # Use band-passed signal for all STA/LTA checks on negatives
    bp_full = None
    if full_trace is not None:
        bp_full = bandpass_3_12(full_trace, fs).astype(np.float32, copy=False)

    def _accept(seg: np.ndarray) -> bool:
        if seg.size == 0:
            return False
        m = stalta_max(seg, fs, args.neg_stalta_sta_s, args.neg_stalta_lta_s)
        return m <= args.neg_stalta_thresh


    for _ in range(max(1, args.neg_k)):
        if args.neg_mode in ("leading", "trailing"):
            s0 = 0 if args.neg_mode == "leading" else max(0, T - L)
            e0 = s0 + L
            # gate using band-passed segment
            if bp_full is None:
                seg_bp = bandpass_3_12(full_trace[s0:e0], fs).astype(np.float32, copy=False)
            else:
                seg_bp = bp_full[s0:e0]
            if _accept(seg_bp):
                out.append((s0, e0, 0, -1, min(int(NOISE_PRE_S * fs), L)))
            else:
                # If the edge window is too "loud", fall back to random search (below)
                pass

        else:
            # random search for a quiet window (STA/LTA on bandpassed signal)
            best = None
            best_ratio = np.inf
            max_tries = max(1, args.neg_stalta_max_tries)
            for _try in range(max_tries):
                s0 = int(rng.integers(0, max(1, T - L + 1)))
                e0 = s0 + L

                if full_trace is None:
                    out.append((s0, e0, 0, -1, min(int(NOISE_PRE_S * fs), L)))
                    break

                # use band-passed segment for STA/LTA
                if bp_full is not None:
                    seg_bp = bp_full[s0:e0]
                else:
                    seg_bp = bandpass_3_12(full_trace[s0:e0], fs).astype(np.float32, copy=False)

                m = stalta_max(seg_bp, fs, args.neg_stalta_sta_s, args.neg_stalta_lta_s)

                if m < best_ratio:
                    best_ratio, best = m, (s0, e0)

                if m <= args.neg_stalta_thresh:
                    out.append((s0, e0, 0, -1, min(int(NOISE_PRE_S * fs), L)))
                    break
            else:
                # max tries exhausted: use best candidate found
                s0, e0 = best if best is not None else (0, L)
                out.append((s0, e0, 0, -1, min(int(NOISE_PRE_S * fs), L)))

    return out

def write_shards_for_geometry(args: ETLArgs, geom_id: str):
    geom_dir = args.root / GEOM_DIRNAME / geom_id
    # ── [ADD] geometry-local picks manifest ──────────────────────────────────
    pm_path = _find_geom_picks_manifest(geom_dir)
    if pm_path is not None:
        has_pick_map, pos_count_by_station = _load_picks_manifest(pm_path, geom_id)
        print(f"[{geom_id}] Using picks manifest: {pm_path.name}")
    else:
        has_pick_map, pos_count_by_station = {}, {}
        print(f"[{geom_id}] [warn] No picks_manifest.csv found; all stations assumed 0 positives")

    # Prefer the discovered manifest path (but keep the original default name as a fallback)
    manifest_csv = pm_path if pm_path is not None else (geom_dir / MANIFEST_NAME)
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_csv}")

    gjson = load_geometry_json(geom_dir)
    json_shots = expand_json_shots(gjson.get("assets", {}))
    if not json_shots:
        raise ValueError(f"No JSON shots defined for geometry {geom_id}")

    df = pd.read_csv(manifest_csv)
    if df.empty:
        print(f"[WARN] Empty manifest: {manifest_csv}")
        return

    # Build (station, shot_id) -> pick_time_s lookup from manifest
    # If duplicates exist, prefer non-zero pick_time
    pick_lookup: Dict[Tuple[str, int], float] = {}
    for row in df.itertuples(index=False):
        station = str(row.station)
        sid = int(row.shot_id)
        t = float(row.pick_time_s)
        if sid < 0:
            continue
        if (station, sid) not in pick_lookup or (t > 0 and pick_lookup[(station, sid)] == 0.0):
            pick_lookup[(station, sid)] = t

    out_dir = args.out / geom_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Process each station SEG-Y once
    for station, gdf in df.groupby("station", sort=False):
        segy_path = Path(str(gdf["segy_path"].iloc[0]))
        if not segy_path.exists():
            raise FileNotFoundError(f"SEG-Y not found: {segy_path}")

        print(f"[{geom_id}] Station {station}: opening {segy_path.name}")
        st = _read_segy(str(segy_path))
        n_tr = len(st)

        # cap ≤ 0.5 × positives for this station
        pos_total = pos_count_by_station.get(str(station), 0)
        print(f"[{geom_id}/{station}] manifest positives = {pos_total}")
        neg_cap   = int(np.floor(args.neg_to_pos_max * pos_total))   # args.neg_to_pos_max default 0.5
        pos_written = 0
        neg_written = 0

        # Track previous pick time for traces without picks
        last_pick_time = None

        # collect up to 10 examples for sanity plots
        pos_samples_for_plots = []      # (X, y_idx, fs, meta)
        neg_samples_for_plots = []      # (X, -1,    fs, meta)
        rng = np.random.default_rng()   # reservoir sampling

        # Build per-trace mapping from THIS station's manifest rows
        df_st = gdf.sort_values("trace_index")
        n_manifest = len(df_st)

        # Tolerance check on counts: len(traces in segy) vs rows for this station in manifest
        if abs(n_tr - n_manifest) > 2:
            raise AssertionError(
                f"[{geom_id}] Station {station}: len(traces in SEG-Y)={n_tr} "
                f"!= rows in picks_manifest for this station={n_manifest} (tolerance ±2)"
            )
        else:
            if n_tr != n_manifest:
                print(f"[{geom_id}/{station}] count mismatch tolerated (SEG-Y {n_tr} vs manifest {n_manifest})")

        # Create a dict: trace_index -> (shot_id, pick_time_s)
        trace_to_shot_pick = {
            int(r.trace_index): (int(r.shot_id), float(r.pick_time_s))
            for r in df_st.itertuples(index=False)
        }

        # shard writer (per station)
        shard_id, in_shard = 0, 0
        tar = tarfile.open(out_dir / f"picks-{station}-{shard_id:05d}.tar", "w")

        def rotate_tar():
            nonlocal shard_id, in_shard, tar
            tar.close()
            shard_id += 1
            in_shard = 0
            tar = tarfile.open(out_dir / f"picks-{station}-{shard_id:05d}.tar", "w")

        for trace_idx in range(n_tr):
            tr = st[trace_idx]
            fs = float(tr.stats.sampling_rate)
            full = tr.data.astype(np.float32, copy=False)
            T_full = full.size
            # Build time axis from header duration (preferred over 1/fs)
            duration = float(tr.stats.endtime - tr.stats.starttime) 
            tm_ax    = np.linspace(0.0, duration, tr.stats.npts, endpoint=False)

            # Compute duration from ObsPy header (sanity only)
            duration = float(tr.stats.endtime - tr.stats.starttime)

            # Manifest mapping by trace_index
            if trace_idx not in trace_to_shot_pick:
                # tolerate 1–2 off; skip or continue
                # choose to skip this trace cleanly
                continue

            shot_id, pick_time = trace_to_shot_pick[trace_idx]
            has_local = 1 if pick_time > 0.0 else 0

            # Update last_pick_time if this trace has a valid pick
            if has_local:
                last_pick_time = pick_time

            # choose one or more windows for this trace
            wins = choose_windows(T_full, fs, pick_time, args, full_trace=full, tm_ax=tm_ax, duration=duration)

            for (s0, e0, has, yidx_in_win, noise_end_idx) in wins:
                seg = full[s0:e0]
                # If WIN_MODE != "pick" but target_len is set, pad/trim to target_len
                # Do NOT pad fixed windows; keep their true length
                if args.win_mode not in ("pick", "fixed") and seg.size != args.target_len:
                    seg = pad_or_center_trim(seg, args.target_len)
                T = seg.size

                # Determine pick time for SNR calculation
                if has:
                    # Trace has pick: use pick time relative to segment start
                    # pick_time is in seconds from start of full trace
                    # s0 is sample offset of segment start
                    seg_pick_time = pick_time - (s0 / fs)
                else:
                    # Trace has no pick: use last valid pick time if available
                    if last_pick_time is not None:
                        # Use the same relative position from the previous pick
                        # Assume traces are similar in structure, so use 2/3 as a proxy
                        # This avoids time alignment issues between different traces
                        seg_pick_time = (2.0/3.0) * T / fs
                    else:
                        # No previous pick available: use 2/3 of segment length
                        seg_pick_time = None

                # features with new pick-centric SNR
                X = build_features(seg, fs, pick_time_in_seg=seg_pick_time, seg_start_time=s0/fs)

                if has:  # positive
                    idx = int(np.clip(yidx_in_win, 0, T-1))
                    has_pick = np.int8(1); y_idx = np.int32(idx)
                    half = max(1, int(args.pos_win_s * fs))
                    y_mask = np.zeros((T,), np.int8)
                    y_mask[max(0, idx - half):min(T, idx + half + 1)] = 1
                    pos_written += 1
                    cap = max(0, int(args.plots_pos))
                    item = (X.copy(), int(y_idx), fs, {"shot": int(shot_id) if shot_id is not None else -1})
                    if cap > 0:
                        if len(pos_samples_for_plots) < cap:
                            pos_samples_for_plots.append(item)
                        else:
                            j = int(rng.integers(0, pos_written))
                            if j < cap:
                                pos_samples_for_plots[j] = item
                else:    # negative
                    if neg_written >= neg_cap:
                        continue  # enforce ≤ 0.5× rule
                    has_pick = np.int8(0); y_idx = np.int32(-1)
                    y_mask = np.zeros((T,), np.int8)
                    neg_written += 1
                    cap = max(0, int(args.plots_neg))
                    item = (X.copy(), -1, fs, {"shot": int(shot_id) if shot_id is not None else -1})
                    if cap > 0:
                        if len(neg_samples_for_plots) < cap:
                            neg_samples_for_plots.append(item)
                        else:
                            j = int(rng.integers(0, neg_written))
                            if j < cap:
                                neg_samples_for_plots[j] = item

                if in_shard >= args.shard_size:
                    rotate_tar()

                # key encodes crop start/end to help debugging
                key = f"{geom_id}_{station}_{int(shot_id) if shot_id is not None else -1:06d}_{trace_idx:05d}_{s0:06d}-{e0:06d}"
                buf = io.BytesIO()
                np.savez_compressed(
                    buf,
                    x=X.astype(DTYPE_X),
                    y_mask=y_mask,
                    has_pick=np.array(has_pick, dtype=np.int8),
                    y_idx=np.array(y_idx, dtype=np.int32),
                    fs=np.array(fs, dtype=np.float32),
                    # Optional minimal meta for plotting:
                    meta=np.array({"geometry": geom_id, "station": str(station), "shot_id": int(shot_id) if shot_id is not None else -1,
                                "crop_start": s0, "crop_end": e0}, dtype=object),
                )
                ti = tarfile.TarInfo(name=f"{key}.npz")
                ti.size = buf.getbuffer().nbytes
                buf.seek(0)
                tar.addfile(ti, fileobj=buf)
                in_shard += 1

        # ── sanity plots for this station ─────────────────────────────
        plots_dir = args.out / "validation_plots" / geom_id / str(station)
        for i, (Xpos, yidx, fs_val, meta) in enumerate(pos_samples_for_plots):
            title = f"{geom_id} station {station} shot {meta.get('shot','?')} (pick)"
            _plot_feature_stack(Xpos, fs=float(fs_val), pick_idx=int(yidx),
                                title=title, out_path=plots_dir / f"pos_{i:02d}.png")
        for i, (Xneg, _y, fs_val, meta) in enumerate(neg_samples_for_plots):
            title = f"{geom_id} station {station} shot {meta.get('shot','?')} (no-pick)"
            _plot_feature_stack(Xneg, fs=float(fs_val), pick_idx=-1,
                                title=title, out_path=plots_dir / f"neg_{i:02d}.png")
        print(f"[plots][{geom_id}/{station}] wrote {len(pos_samples_for_plots)} pos and "
            f"{len(neg_samples_for_plots)} neg to {plots_dir}")

        # ──────────────────────────────────────────────────────────────

        tar.close()
        print(f"    -> wrote shards in {out_dir} for station {station}")

FEATURE_NAMES = [
    "bp 3–12 Hz",        # 0
    "STA/LTA fast",      # 1
    "STA/LTA med",       # 2
    "envelope",          # 3
    "band env mid",      # 4
    "maxamp std",        # 5
    "SNR (bp, scalar)",  # 6
]

def _plot_feature_stack(xC_T: np.ndarray, fs: float, pick_idx: int = -1,
                        title: str = "", out_path: Path = None):
    """
    xC_T: (C,T) float array with all channels you save to 'x' in .npz
    """
    C, T = xC_T.shape
    t = np.arange(T, dtype=np.float32) / float(fs)
    ncols = 1
    nrows = C
    h = max(2, int(C * 0.9))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, h), sharex=True)
    if C == 1:
        axes = [axes]
    for c in range(C):
        ax = axes[c]
        ax.plot(t, xC_T[c], lw=0.8)
        if 0 <= pick_idx < T:
            ax.axvline(pick_idx / float(fs), color="r", lw=1.2, alpha=0.8)
        name = FEATURE_NAMES[c] if c < len(FEATURE_NAMES) else f"feat{c}"
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")

    # Extract SNR from channel 6 and add to title
    if title:
        if C > 6:  # SNR is channel 6
            snr_value = float(xC_T[6, 0])  # SNR is constant across time, so take first sample
            title_with_snr = f"{title} | SNR: {snr_value:.1f}"
        else:
            title_with_snr = title
        fig.suptitle(title_with_snr, fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    else:
        plt.show()

# ----------------------------- CLI ------------------------------

def main():
    ap = argparse.ArgumentParser(description="ETL: manifest -> WebDataset shards with SEG-Y/JSON shot alignment")
    ap.add_argument("root", type=Path, help="Global ML-feed root (contains 'geometries/')")
    ap.add_argument("--out", type=Path, required=True, help="Output root for shards")
    ap.add_argument("--geom", nargs="*", help="Optional subset of geometry IDs")
    ap.add_argument("--target-len", type=int, default=TARGET_LEN, help="Samples per window (default: 30000)")
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="Samples per .tar shard (default: 10000)")
    ap.add_argument("--pos-win-s", type=float, default=POS_WIN_S, help="±seconds around pick for y_mask (default: 0.04)")
    ap.add_argument("--align-tol", type=int, default=ALIGN_TOL, help="Shot alignment tolerance (default: 2)")
    ap.add_argument("--win-mode", choices=["pick", "center", "none", "fixed"], default=WIN_MODE)
    ap.add_argument("--pre-s", type=float, default=PRE_S, help="sec before pick when --win-mode=pick")
    ap.add_argument("--post-s", type=float, default=POST_S, help="sec after pick when --win-mode=pick")
    ap.add_argument("--neg-mode", choices=["random", "leading", "trailing"], default=NEG_MODE)
    ap.add_argument("--neg-k", type=int, default=NEG_K)
    ap.add_argument("--neg-stalta-thresh", type=float, default=NEG_STALTA_THRESH)
    ap.add_argument("--neg-sta-s", type=float, default=NEG_STALTA_STA_S)
    ap.add_argument("--neg-lta-s", type=float, default=NEG_STALTA_LTA_S)
    ap.add_argument("--neg-stalta-max-tries", type=int, default=NEG_STALTA_MAX_TRIES)
    ap.add_argument("--total-win-s", type=float, default=TOTAL_S, help="Total seconds per crop window (default: 5.0)")
    ap.add_argument("--neg-to-pos-max", type=float, default=NEG_TO_POS_MAX_RATIO, help="Max #no-pick / #with-pick within a single SEG-Y (station)")
    ap.add_argument("--plots-pos", type=int, default=PLOTS_POS_DEFAULT, help="Max positive examples to plot per station (default: 40)")
    ap.add_argument("--plots-neg", type=int, default=PLOTS_NEG_DEFAULT, help="Max negative examples to plot per station (default: 10)")
    ap.add_argument("--win-start-s", type=float, default=FIXED_START_S, help="Start time (s) for the fixed window, per trace.")
    ap.add_argument("--win-len-s", type=float, default=FIXED_LEN_S, help="Length (s) of the fixed window, per trace.")

    args_ns = ap.parse_args()

    args = ETLArgs(
        root=args_ns.root.resolve(),
        out=args_ns.out.resolve(),
        geoms=args_ns.geom,
        target_len=args_ns.target_len,
        shard_size=args_ns.shard_size,
        pos_win_s=args_ns.pos_win_s,
        align_tol=args_ns.align_tol,
        win_mode=args_ns.win_mode,
        # make sure pre_s/post_s are plain floats even if defaults were numpy scalars
        pre_s=float(np.asarray(args_ns.pre_s).item()) if hasattr(args_ns, "pre_s") else 2.0,
        post_s=float(np.asarray(args_ns.post_s).item()) if hasattr(args_ns, "post_s") else 3.0,
        neg_mode=args_ns.neg_mode,
        neg_k=args_ns.neg_k,
        neg_stalta_thresh=args_ns.neg_stalta_thresh,
        neg_stalta_sta_s=args_ns.neg_sta_s,
        neg_stalta_lta_s=args_ns.neg_lta_s,
        neg_stalta_max_tries=args_ns.neg_stalta_max_tries,
        total_win_s=float(args_ns.total_win_s),
        neg_to_pos_max=float(args_ns.neg_to_pos_max),
        plots_pos=int(args_ns.plots_pos),
        plots_neg=int(args_ns.plots_neg),
        fixed_start_s=float(args_ns.win_start_s),
        fixed_len_s=float(args_ns.win_len_s),
    )

    geom_root = args.root / GEOM_DIRNAME
    if not geom_root.exists():
        raise FileNotFoundError(f"{geom_root} not found")

    geoms = args.geoms or [p.name for p in geom_root.iterdir() if p.is_dir()]
    print(f"Geometries: {', '.join(geoms)}")
    args.out.mkdir(parents=True, exist_ok=True)

    for gid in geoms:
        write_shards_for_geometry(args, gid)

    print("Done.")

if __name__ == "__main__":
    main()

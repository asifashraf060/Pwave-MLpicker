"""
Build a per-geometry manifest by pairing SEG-Y traces with P-picks.

Folder layout (assumed):
  /ABS/PATH/ML-feed/
    dataset.json
    arrivals.mat      # MATLAB struct with fields: eventid, station, time
    geometries/
      SO-SG1/
        geometry.json
        CS01.segy
        CS02.segy
        ...
      SO-SG2/
        geometry.json
        ...

Output (per geometry):
  geometries/<GEOM_ID>/picks_manifest.csv
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any

import numpy as np
import pandas as pd
from scipy import io as sio
from obspy.io.segy.core import _read_segy  # fast, low-level reader
# from obspy import read  # alternative if you prefer format="SEGY"

# --------------------------
# Config knobs (edit here)
# --------------------------
ARRIVALS_MAT_NAME = "tlArrival.mat"  # change if your file is named differently
GEOMETRIES_SUBDIR = "geometries"    # relative to the global root
MANIFEST_NAME     = "picks_manifest.csv"

# -------- SEG-Y shot extraction & alignment --------

SHOT_HEADER_CANDIDATES = [
    # most to least preferred; we use the first that yields a sane sequence
    "energy_source_point_number",         # SEG-Y Rev1: byte 17–20 (shotpoint)
    "source_point",                       # alt naming
    "ensemble_number",                    # CDP/FFID depending on layout
    "field_record_number",                # FFID
    "original_field_record_number",       # some writers use this
    "trace_sequence_number_within_line",  # fallback (often just 1..N)
]

def extract_shots_from_segy(stream) -> list[int] | None:
    """
    Try to extract a 'shot number' per trace from the SEG-Y trace headers.
    Returns a list of ints (len == n_traces) or None if nothing usable found.
    """
    def get_series(key: str):
        vals = []
        for tr in stream:
            th = tr.stats.segy.trace_header
            if not hasattr(th, key):
                return None
            v = getattr(th, key)
            # sometimes numpy scalar; force int
            try:
                vals.append(int(v))
            except Exception:
                return None
        return vals

    # try each candidate; pick the first that looks monotone non-decreasing
    for key in SHOT_HEADER_CANDIDATES:
        series = get_series(key)
        if series is None:
            continue
        # accept sequences that are non-decreasing and have > ~10 distinct values
        if all(series[i] <= series[i+1] for i in range(len(series)-1)) and len(set(series)) > max(10, len(series)//50):
            return series
    return None

def align_shot_series(segy_shots: list[int], json_shots: list[int], tol: int = 2):
    """
    Monotone two-pointer alignment allowing small mismatches/gaps.
    Returns:
      mapped_json_for_trace : list[Optional[int]]  length == len(segy_shots)
      stats : dict with counts for logging
    """
    i, j = 0, 0
    N, M = len(segy_shots), len(json_shots)
    mapped = [None] * N
    dropped_segy = []   # traces we couldn't match
    dropped_json = []   # json shots we skipped

    while i < N and j < M:
        a, b = segy_shots[i], json_shots[j]
        if abs(a - b) <= tol:
            mapped[i] = b
            i += 1; j += 1
        elif a < b - tol:
            # SEG-Y has a shot number smaller than JSON → probably an extra trace
            dropped_segy.append((i, a))
            i += 1
        else:  # a > b + tol
            # JSON has a shot we didn't see in SEG-Y → skip it
            dropped_json.append((j, b))
            j += 1

    # trailing tails
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


# ---------- helpers: MATLAB struct parsing ----------

def _to_numpy(obj: Any) -> np.ndarray:
    """Best-effort convert matlab cell/char/ndarray to a 1-D numpy array."""
    arr = np.asarray(obj)
    # MATLAB structs often come as shape (1,1) object arrays
    while arr.dtype == object and arr.size == 1:
        arr = np.asarray(arr.item())
    return arr

def _decode_station_array(sta_obj: Any) -> List[str]:
    """
    Convert MATLAB 'station' field into list of strings.
    Handles cell arrays of char arrays or plain char arrays.
    """
    a = _to_numpy(sta_obj)
    out: List[str] = []
    if a.dtype == object:
        # cell array of strings
        for cell in a.ravel():
            s = "".join(np.asarray(cell).astype(str)).strip()
            out.append(s)
    elif a.dtype.kind in ("U", "S"):
        # possibly 2-D char array where each row is a fixed-width string
        if a.ndim == 2:
            for i in range(a.shape[0]):
                out.append("".join(a[i]).strip())
        else:
            out.append("".join(a).strip())
    else:
        # last resort
        out = [str(x) for x in a.ravel()]
    return out

def load_arrivals_mat(mat_path: Path) -> dict:
    """
    Load MATLAB arrivals struct and return {(station:str, eventid:int): time:float}.
    Works with your tlArrival.mat where the top-level var is 'tlArrival' and is a (1,1)
    mat_struct object containing Nx1 arrays: eventid, station (cell/char), time.
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    # Keep struct as mat_struct; don't squeeze fields too early
    data = sio.loadmat(mat_path, squeeze_me=False, struct_as_record=False)

    # Try common keys, then fall back to the first non-internal key
    candidates = ["tlArrival", "tArrival", "arrivals", "Arrival", "picks", "Picks"]
    key = next((k for k in candidates if k in data), None)
    if key is None:
        # pick the first user key if present
        user_keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(
            f"No known arrivals variable found in {mat_path}. "
            f"Keys present: {user_keys}"
        )

    tA = data[key]
    # Typical shape is (1,1) object array -> take the single element
    if isinstance(tA, np.ndarray) and tA.size == 1 and tA.dtype == object:
        tA = tA.flat[0]

    # Pull fields from mat_struct
    try:
        events  = np.asarray(tA.eventid).astype(np.int64).ravel()
        times   = np.asarray(tA.time, dtype=float).ravel()
        stations_arr = tA.station  # (N,1) object/char arrays
    except AttributeError as e:
        raise ValueError(
            f"Expected fields 'eventid', 'station', 'time' in {key} inside {mat_path}"
        ) from e

    # Decode station cell/char array to a list of Python strings
    stations: list[str] = []
    if isinstance(stations_arr, np.ndarray):
        # For shape (N,1) where each element is a 1×M char array or a scalar numpy string
        for i in range(stations_arr.shape[0]):
            cell = stations_arr[i, 0]
            if isinstance(cell, np.ndarray):
                stations.append(str(cell.item() if cell.size == 1 else "".join(cell.tolist())).strip())
            else:
                stations.append(str(cell).strip())
    else:
        # Fallback
        stations = [str(s).strip() for s in np.asarray(stations_arr).ravel().tolist()]

    if not (len(stations) == events.size == times.size):
        raise ValueError(
            f"Lengths mismatch in {mat_path}: "
            f"len(station)={len(stations)}, len(events)={events.size}, len(times)={times.size}"
        )

    # Build the lookup map
    picks: dict[tuple[str, int], float] = {}
    for s, e, t in zip(stations, events, times):
        picks[(s, int(e))] = float(t)

    return picks

# ---------- helpers: geometry JSON ----------

def load_dataset_json(path: Path) -> List[str]:
    """
    dataset.json fields:
      { "geometries": ["SO-SG1", "SO-SG2", ...], ... }
    """
    with path.open("r") as f:
        js = json.load(f)
    geoms = js.get("geometries", [])
    if not isinstance(geoms, list) or not geoms:
        raise ValueError(f"No 'geometries' found in {path}")
    return [str(g) for g in geoms]

def load_geometry_json(path: Path) -> dict:
    """
    geometry.json example:
    {
      "id": "SO-SG1",
      "assets": {
        "station_desc": "cs21_obs",
        "shotline": "MGL2104PS01B_SUB2_ext",
        "shots": {"start":40481, "stop":42755, "step":1, "end_exclusive": true},
        "pick_phase": "P"
      },
      "segy_layout": "shotGather"
    }
    """
    with path.open("r") as f:
        g = json.load(f)
    return g

def expand_shots(assets: dict) -> List[int]:
    sh = assets.get("shots", {})
    start = int(sh.get("start"))
    stop  = int(sh.get("stop"))
    step  = int(sh.get("step", 1))
    end_exclusive = bool(sh.get("end_exclusive", True))
    rng = range(start, stop, step) if end_exclusive else range(start, stop + step, step)
    return list(rng)


# ---------- main ETL per geometry ----------

def traces_in_segy(segy_path: Path) -> int:
    st = _read_segy(str(segy_path))  # returns Stream
    return len(st)

def list_station_segy_files(geom_dir: Path) -> List[Path]:
    # station files named "<station>.segy" directly inside the geometry folder
    return sorted(geom_dir.glob("*.segy"))

def build_manifest_for_geometry(root: Path, geom_id: str) -> Path:
    geom_dir = root / GEOMETRIES_SUBDIR / geom_id
    geom_json = geom_dir / "geometry.json"
    if not geom_json.exists():
        raise FileNotFoundError(f"Missing geometry.json in {geom_dir}")

    g = load_geometry_json(geom_json)
    assets = g.get("assets", {})
    shots = expand_shots(assets)
    if not shots:
        raise ValueError(f"No shots defined for geometry {geom_id}")

    # Load arrivals (picks)
    mat_path = root / ARRIVALS_MAT_NAME
    picks_map = load_arrivals_mat(mat_path)

    # Iterate station SEG-Ys
    rows = []
    segy_files = list_station_segy_files(geom_dir)
    if not segy_files:
        print(f"[WARN] No .segy files in {geom_dir}", file=sys.stderr)

    for segy_path in segy_files:
        station = segy_path.stem  # "<station>.segy"
        st = _read_segy(str(segy_path))
        n_traces = len(st)   
        # Try to read shot numbers from SEG-Y headers
        segy_shots = extract_shots_from_segy(st)     
        if segy_shots is None:
            # fallback: naive index alignment, but warn loudly if lengths differ
            if n_traces != len(shots):
                print(
                    f"[WARN] [{geom_id}] {station}: no shot header found; "
                    f"using index mapping with len(shots)={len(shots)} vs n_traces={n_traces}. "
                    f"Some traces will be unmatched.",
                    file=sys.stderr
                )
            # build mapped list by trunc/pad with None
            L = max(n_traces, len(shots))
            json_ext = shots + [None] * (L - len(shots))
            mapped_json_for_trace = json_ext[:n_traces]
            align_stats = {"matched": sum(x is not None for x in mapped_json_for_trace),
                           "unmatched_traces": mapped_json_for_trace.count(None),
                           "unmatched_json": max(0, len(shots) - n_traces)}
        else:
            # Align SEG-Y shot sequence to JSON shot list tolerantly
            mapped_json_for_trace, align_stats = align_shot_series(segy_shots, shots, tol=2)

            if align_stats["unmatched_traces"] or align_stats["unmatched_json"]:
                print(
                    f"[INFO] [{geom_id}] {station}: matched {align_stats['matched']}/{n_traces} traces "
                    f"(unmatched_traces={align_stats['unmatched_traces']}, "
                    f"unmatched_json={align_stats['unmatched_json']}).",
                    file=sys.stderr
                )
                if align_stats["dropped_traces_preview"]:
                    print(f"       dropped trace preview: {align_stats['dropped_traces_preview']}", file=sys.stderr)
                if align_stats["dropped_json_preview"]:
                    print(f"       skipped json shots preview: {align_stats['dropped_json_preview']}", file=sys.stderr)

        # ----- build rows using the mapped shot id (None means: no matching shot) -----
        for trace_idx in range(n_traces):
            shot_id = mapped_json_for_trace[trace_idx]
            # If we couldn't map this trace to a JSON shot, we can't look up a MATLAB pick
            if shot_id is None:
                t_pick = 0.0
            else:
                t_pick = picks_map.get((station, int(shot_id)), 0.0)

            rows.append({
                "geometry": geom_id,
                "station": station,
                "segy_path": str(segy_path),
                "trace_index": trace_idx,
                "shot_id": int(shot_id) if shot_id is not None else -1,
                "pick_time_s": float(t_pick)
            })

    # Write CSV
    out_csv = geom_dir / MANIFEST_NAME
    df = pd.DataFrame(rows, columns=[
        "geometry", "station", "segy_path", "trace_index", "shot_id", "pick_time_s"
    ])
    df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote manifest with {len(df):,} rows → {out_csv}")
    return out_csv


# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="Pair SEG-Y traces with P-picks into a manifest.")
    p.add_argument("root", type=Path,
                   help="Global root folder (e.g., /abs/path/ML-feed)")
    p.add_argument("--geoms", nargs="*", default=None,
                   help="Optional subset of geometry IDs to process")
    args = p.parse_args()

    root: Path = args.root.resolve()
    ds_json = root / "dataset.json"
    if not ds_json.exists():
        raise FileNotFoundError(f"dataset.json not found at {ds_json}")

    all_geoms = load_dataset_json(ds_json)
    if args.geoms:
        target = [g for g in all_geoms if g in set(args.geoms)]
        unknown = set(args.geoms) - set(target)
        if unknown:
            print(f"[WARN] Unknown geometries in --geoms: {sorted(unknown)}", file=sys.stderr)
    else:
        target = all_geoms

    print(f"Root: {root}")
    print(f"Geometries to process: {', '.join(target)}")

    out_paths = []
    for gid in target:
        out_paths.append(build_manifest_for_geometry(root, gid))

    print("\nDone.")
    for pth in out_paths:
        print(f"  - {pth}")

if __name__ == "__main__":
    main()
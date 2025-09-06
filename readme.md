# **Seismic Picker w/ Uncertainty — End-to-End**

A compact, production-oriented pipeline to:

1. build a picks manifest from SEG-Y + geometry metadata,
2. convert data to WebDataset (WDS) shards with curated features and smart negative sampling, and
3. train a noise-aware picker that outputs a pick and an uncertainty, then abstains on noisy traces.

##### **Scripts:**

-** `build_manifest_from_segy.py`** — walks each geometry, opens station SEG-Y files, aligns shots to the geometry’s JSON shot list, and writes a `picks_manifest.csv` with `geometry`, `station`, `segy_path`, `trace_index`, `shot_id`, `pick_time_s`. When a SEG-Y lacks usable shot headers it falls back to a tolerant index alignment and logs mismatches.

- **`etl_to_wds_from_manifest.py`** — one-time ETL that reads the per-geometry manifest and emits **WDS** shards (`.tar` of `.npz`). Each sample holds the feature tensor `x (C,T)`, supervision (`y_mask`, `has_pick`, `y_idx`) and `fs`. Includes options for cropping around picks, STA/LTA-gated negatives, fixed windows, and pos:neg capping. Key knobs: target length, window mode, ±pick window, and neg:pos ratio cap.

- **`ML-pipe-trns-uncrtnNoise.py`** — WDS training loop with rich tqdm bars. The model predicts (mean pick, log-σ); training uses Gaussian NLL (noise-aware) and we learn to abstain on high uncertainty. After training, a conformal pass calibrates a 90% z-quantile `q` for interval width.

##### **Data layout expected by the ETL**
```
ML-feed/
  dataset.json
  geometries/
    SO-SG1/
      geometry.json
      10001.segy
      ...
    SO-SG2/
      ...
  tlArrival.mat               # (used upstream by the manifest builder)
```

##### **Quick start**

**1) Build the per-geometry manifest (CSV)**
```
python build_manifest_from_segy.py /ABS/ML-feed
```
This scans each station SEG-Y in the geometry directory, aligns to the JSON shot list, and writes `picks_manifest.csv`. 

**2) Convert to WebDataset shards**
```
# All geometries → shards under /ABS/wds/<GEOM_ID>/
python etl_to_wds_from_manifest.py /ABS/ML-feed --out /ABS/wds

# Just some geometries + extra plots for sanity:
python etl_to_wds_from_manifest.py /ABS/ML-feed --out /ABS/wds \
  --geom SO-SG1 SO-SG2 --plots-pos 40 --plots-neg 10
```
Each shard packs .npz with: `x`, `y_mask`, `has_pick`, `y_idx`, `fs` (feature channels are band-passed + STA/LTA stacks; negatives are STA/LTA-gated and capped per station). 

**3) Train the picker with uncertainty**

```
python ML-pipe-trns-uncrtnNoise.py /ABS/ML-feed --wds /ABS/wds \
  --batch-size 16 --num-workers 2 --epochs 30 \
  --target-fs 100 --window-size 256 --num-heads 2
```
The script discovers shards, makes train/val splits, shows total batches, then trains the (mean, log-σ) head. After training it runs conformal calibration to compute `q` and saves the model. 

##### **How uncertainty & abstention work (high level)**
- The network outputs μ (pick sample) and log-σ per window.
- Training minimizes Gaussian NLL (mean & variance) so noisy windows learn larger σ.
- After training we calibrate a conformal quantile q on the validation set; the predictive band half-width is q·σ.
- Abstain if the time width (converted to seconds) is above --abstain-width-s; an extra SNR rule can also veto picks when SNR < threshold. (Implemented in training & eval scripts.)

**Requirements**

Python 3.9+, PyTorch, NumPy, SciPy, pandas, ObsPy, webdataset, matplotlib, tqdm.


**Typical folder outputs**
```
wds/
  SO-SG1/
    picks-10001-00000.tar
    picks-10002-00000.tar
  SO-SG2/
    ...
evaluate_picks/
  *.png         # qualitative pick/abstain visualizations

```

**Tips**
- Big data? Create shards once, then train many times—no need to reopen SEG-Y during training.
- On a GPU cluster, start with --batch-size 64, --num-workers 8 (scale by VRAM & I/O) and adjust.
- If you see “non-finite loss”, check for NaNs/Inf in inputs and clamp logσ during training (already handled in the pipeline).

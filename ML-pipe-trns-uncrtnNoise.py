import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm as TQDM
import os
import math
import glob
import sys

import io, tarfile, json, random
from pathlib import Path
try:
    import webdataset as wds
    _HAS_WDS = True
except Exception:
    _HAS_WDS = False

import platform
is_macos = (platform.system() == "Darwin")
is_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

pin = (torch.cuda.is_available() and not is_mps)

# Fixed crop length in seconds (matches ETL)
TOTAL_WIN_S = 5.0

# Target sample rate for training/eval
TARGET_FS = 200.0   # <- common choice; set to match your preferred fs

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ TRANSFORMER SELF-ATTENTION COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer self-attention"""
    
    def __init__(self, d_model, max_len=15000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Handle odd d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # For odd dimensions, handle the last dimension separately
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Store as (1, d_model, max_len) for broadcasting
        self.register_buffer('pe', pe.transpose(0, 1).unsqueeze(0))
    
    def forward(self, x):
        # x shape: (batch, channels, length)
        # Add positional encoding directly (broadcasting will handle batch dimension)
        # Only use the required length from positional encoding
        return x + self.pe[:, :x.size(1), :x.size(2)]


class TransformerSelfAttention(nn.Module):
    """
    Memory-efficient Transformer self-attention with sliding window
    Uses local attention to reduce memory consumption
    """
    
    def __init__(self, channels, num_heads=2, dropout=0.1, window_size=256):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size  # Local attention window
        
        # Ensure channels is divisible by num_heads
        assert channels % num_heads == 0, f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        
        self.d_k = channels // num_heads
        
        # Linear transformations for Q, K, V
        self.linear_q = nn.Linear(channels, channels)
        self.linear_k = nn.Linear(channels, channels)
        self.linear_v = nn.Linear(channels, channels)
        
        # Output projection
        self.linear_out = nn.Linear(channels, channels)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(channels)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        
        # Feed-forward network (reduced size for memory)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),  # Reduced from 4x to 2x
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def sliding_window_attention(self, Q, K, V):
        """Apply attention with sliding window to reduce memory usage"""
        batch_size, num_heads, seq_len, d_k = Q.shape
        
        # If sequence is short enough, use full attention
        if seq_len <= self.window_size * 2:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            return torch.matmul(attn_weights, V)
        
        # Otherwise, use sliding window attention
        output = torch.zeros_like(V)
        
        # Process in overlapping windows
        stride = self.window_size // 2
        for i in range(0, seq_len, stride):
            end_i = min(i + self.window_size, seq_len)
            
            # Get window queries
            Q_window = Q[:, :, i:end_i]
            
            # Determine key/value window (slightly larger for context)
            start_kv = max(0, i - stride)
            end_kv = min(seq_len, end_i + stride)
            K_window = K[:, :, start_kv:end_kv]
            V_window = V[:, :, start_kv:end_kv]
            
            # Compute attention for this window
            scores = torch.matmul(Q_window, K_window.transpose(-2, -1)) / math.sqrt(d_k)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention and accumulate
            window_output = torch.matmul(attn_weights, V_window)
            
            # Blend overlapping regions
            if i > 0 and i < seq_len - self.window_size:
                # Apply triangular blending for smooth transitions
                blend_size = stride
                blend_weights = torch.linspace(0, 1, blend_size, device=Q.device)
                blend_weights = blend_weights.view(1, 1, -1, 1)
                
                # Blend the overlapping part
                output[:, :, i:i+blend_size] = (
                    output[:, :, i:i+blend_size] * (1 - blend_weights) +
                    window_output[:, :, :blend_size] * blend_weights
                )
                output[:, :, i+blend_size:end_i] = window_output[:, :, blend_size:]
            else:
                output[:, :, i:end_i] = window_output
        
        return output
        
    def forward(self, x):
        # x shape: (batch, channels, length)
        batch_size, channels, seq_len = x.shape
        
        # Downsample if sequence is too long (>2000 samples) - more aggressive
        downsample_factor = 1
        if seq_len > 2000:
            downsample_factor = 4 if seq_len > 4000 else 2
            # Store original for skip connection
            x_original = x
            # Downsample by factor
            x = F.avg_pool1d(x, kernel_size=downsample_factor, stride=downsample_factor)
            seq_len = x.shape[2]
        
        # Add positional encoding
        x_pos = self.pos_encoding(x)
        
        # Transpose for attention: (batch, length, channels)
        x_transposed = x_pos.transpose(1, 2)
        
        # Store residual
        residual = x_transposed
        
        # Compute Q, K, V
        Q = self.linear_q(x_transposed).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.linear_k(x_transposed).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.linear_v(x_transposed).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose for attention computation: (batch, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Apply sliding window attention
        attn_output = self.sliding_window_attention(Q, K, V)
        
        # Concatenate heads: (batch, seq_len, channels)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, channels
        )
        
        # Output projection
        attn_output = self.linear_out(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Add & Norm (residual connection)
        x_attn = self.norm1(residual + attn_output)
        
        # Feed-forward network with residual
        ffn_output = self.ffn(x_attn)
        x_final = self.norm2(x_attn + ffn_output)
        
        # Transpose back: (batch, channels, length)
        x_final = x_final.transpose(1, 2)
        
        # Upsample back to original size if we downsampled
        if downsample_factor > 1:
            x_final = F.interpolate(x_final, size=x_original.shape[2], mode='linear', align_corners=False)
            # Add skip connection with original resolution
            x_final = x_final + x_original
        
        return x_final


# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ ENHANCED U-NET WITH TRANSFORMER ATTENTION
# ═══════════════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Enhanced convolution block with batch normalization and dropout"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.1):
        super().__init__()
        
        self.doubleConv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout),
            
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout1d(dropout)
        )
        
    def forward(self, x):
        return self.doubleConv(x)


class AdaptiveUNet1D(nn.Module):
    """
    ┌──────────────────────────────────────────────────────────────────────────────────────┐
    │ Memory-Efficient Adaptive 1-D U-Net with Transformer Self-Attention                  │
    │ ──────────────────────────────────────────────────────────────────────────────────── │
    │ • Uses sliding window transformer self-attention to reduce memory usage              │
    │ • Automatic downsampling for very long sequences                                     │
    │ • Multi-head self-attention for capturing long-range dependencies                    │
    │ • Positional encoding for sequence awareness                                         │
    │ • Feed-forward networks for enhanced feature transformation                          │
    │ • Can work with or without physics-informed features                                 │
    │ • Batch normalization and dropout for better generalization                          │
    └──────────────────────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, in_channels=1, out_channels=2, features=[16, 32, 64, 128], 
                 dropout=0.1, use_physics_features=True, num_heads=2, window_size=256):
        super().__init__()
        
        self.in_channels = in_channels
        self.use_physics_features = use_physics_features
        
        # ==============================
        # 1️⃣ Downsampling Path (ENCODER)
        # ==============================
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.attentions_down = nn.ModuleList()
        
        current_channels = in_channels
        for feat in features:
            self.downs.append(ConvBlock(current_channels, feat, dropout=dropout))
            self.pools.append(nn.MaxPool1d(2))
            # Use transformer self-attention with appropriate number of heads
            heads = min(num_heads, feat // 16)  # More conservative: ensure at least 16 dims per head
            heads = max(1, heads)  # At least 1 head
            self.attentions_down.append(TransformerSelfAttention(feat, num_heads=heads, 
                                                                 dropout=dropout, window_size=window_size))
            current_channels = feat
        
        # ============================================
        # 2️⃣ Bottleneck (connects ENCODER & DECODER)
        # ============================================
        self.bottleneck = ConvBlock(features[-1], features[-1]*2, dropout=dropout)
        # Transformer attention for bottleneck
        bottleneck_heads = min(num_heads, (features[-1]*2) // 16)
        bottleneck_heads = max(1, bottleneck_heads)
        self.bottleneck_attention = TransformerSelfAttention(features[-1]*2, num_heads=bottleneck_heads, 
                                                            dropout=dropout, window_size=window_size)
        
        # ==============================
        # 3️⃣ Upsampling path (DECODER)
        # ==============================
        self.ups = nn.ModuleList()
        self.attentions_up = nn.ModuleList()
        
        for feat in reversed(features):
            # Transposed convolution for upsampling
            self.ups.append(nn.ConvTranspose1d(feat*2, feat, kernel_size=2, stride=2))
            # Convolution block for feature fusion
            self.ups.append(ConvBlock(feat*2, feat, dropout=dropout))
            # Transformer attention for refined features
            heads = min(num_heads, feat // 16)
            heads = max(1, heads)
            self.attentions_up.append(TransformerSelfAttention(feat, num_heads=heads, 
                                                              dropout=dropout, window_size=window_size))
        
        # ===========================
        # 4️⃣ Final Classification Layer
        # ===========================
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
        # Physics-informed feature weighting (only if using physics features)
        self.feature_weights = None
        if use_physics_features:
            self.feature_weights = nn.Parameter(torch.ones(in_channels))
        
    def initialize_feature_weights(self, actual_channels):
        """Initialize feature weights based on actual number of input channels"""
        if self.use_physics_features:
            if self.feature_weights is None or self.feature_weights.size(0) != actual_channels:
                self.feature_weights = nn.Parameter(torch.ones(actual_channels))
                self.in_channels = actual_channels
                print(f"Initialized feature weights for {actual_channels} channels")
        
    def forward(self, x, return_feats: bool = False):
        # Apply learnable weights to input features (only if using physics features)
        if self.use_physics_features:
            # Initialize feature weights if needed
            if self.feature_weights is None:
                self.initialize_feature_weights(x.size(1))
            
            if x.size(1) == self.feature_weights.size(0):
                weighted_x = x * self.feature_weights.view(1, -1, 1)
            else:
                print(f"Warning: Input channels ({x.size(1)}) != feature weights ({self.feature_weights.size(0)})")
                # Adjust feature weights if mismatch
                self.initialize_feature_weights(x.size(1))
                weighted_x = x * self.feature_weights.view(1, -1, 1)
        else:
            weighted_x = x
        
        skip_connections = []
        enc_features_to_head = None
        # ---------------- Encoder ----------------
        for i, (down, pool, attention) in enumerate(zip(self.downs, self.pools, self.attentions_down)):
            weighted_x = down(weighted_x)
            if enc_features_to_head is None:
                enc_features_to_head = weighted_x  # (B, feat, T/2) – good, already enriched
            weighted_x = attention(weighted_x)  # Apply transformer self-attention
            skip_connections.append(weighted_x)
            weighted_x = pool(weighted_x)
        
        # --------------- Bottleneck ---------------
        weighted_x = self.bottleneck(weighted_x)
        weighted_x = self.bottleneck_attention(weighted_x)
        
        # Reverse skip connections for decoder
        skip_connections = skip_connections[::-1]
        
        # ---------------- Decoder ----------------
        for idx in range(0, len(self.ups), 2):
            # Upsampling
            weighted_x = self.ups[idx](weighted_x)
            
            # Get corresponding skip connection
            skip_conn = skip_connections[idx//2]
            
            # Handle size mismatches
            if weighted_x.shape[-1] != skip_conn.shape[-1]:
                weighted_x = F.pad(weighted_x, (0, skip_conn.shape[-1] - weighted_x.shape[-1]))
            
            # Concatenate skip connection
            weighted_x = torch.cat((skip_conn, weighted_x), dim=1)
            
            # Refine features
            weighted_x = self.ups[idx+1](weighted_x)
            
            # Apply transformer self-attention
            attention_idx = idx // 2
            if attention_idx < len(self.attentions_up):
                weighted_x = self.attentions_up[attention_idx](weighted_x)
        
        # Final classification
        out = F.softmax(self.final_conv(weighted_x), dim=1)
        return (out, enc_features_to_head) if return_feats else out         

class PickHead(nn.Module):
    """
    Take mid-level features and regress (pick_time, log_std).
    """
    def __init__(self, in_channels, hidden=128):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(256)    # down to manageable tokens
        self.conv = nn.Conv1d(in_channels, hidden, kernel_size=3, padding=1)
        self.bn   = nn.BatchNorm1d(hidden)
        self.act  = nn.GELU()
        self.proj_mean = nn.Linear(hidden, 1)
        self.proj_std  = nn.Linear(hidden, 1)

    def forward(self, feats):  # feats: (B,C,T) from encoder’s first down block output
        z = self.pool(feats)           # (B,C,256)
        z = self.act(self.bn(self.conv(z)))  # (B,H,256)
        z = z.mean(dim=-1)             # (B,H)
        pick_frac = torch.sigmoid(self.proj_mean(z)).squeeze(-1)  # (B,)
        min_s, max_s = 1.0, 200.0
        center = math.log(math.sqrt(min_s * max_s))
        halfspan = math.log(max_s / min_s) / 2.0
        log_std = (torch.tanh(self.proj_std(z)) * halfspan + center).squeeze(-1)
        return pick_frac, log_std

## NOISE

def robust_std(x):
    # σ̂ from MAD
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return 1.4826 * mad

def simple_snr(signal, noise):
    # peak-to-rms as a simple SNR proxy
    rms = np.sqrt(np.mean(noise**2)) + 1e-9
    return (np.max(np.abs(signal)) + 1e-9) / rms

def spectral_flatness_welch(x, fs, nperseg=256, band=None):
    """
    Spectral flatness (geometric mean / arithmetic mean of PSD).
    If band=(fmin,fmax) set, measure flatness in that band.
    """
    try:
        from scipy.signal import welch
    except Exception:
        # if scipy isn't available, fall back to 1.0 (maximally flat)
        return 1.0
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    if band is not None:
        fmin, fmax = band
        m = (f >= fmin) & (f <= fmax)
        if not np.any(m):
            return 1.0
        Pxx = Pxx[m]
    Pxx = np.clip(Pxx, 1e-16, None)
    gmean = np.exp(np.mean(np.log(Pxx)))
    amean = np.mean(Pxx)
    return float(gmean / (amean + 1e-16))

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🗂️  DATASET LOADING FROM WDS TARS
# ═══════════════════════════════════════════════════════════════════════════════════════

def _npz_to_tensor_dict(sample):
    """
    Accepts either raw .npz bytes (fallback tar path) or an already-decoded dict/NpzFile
    (common with WebDataset + .decode()). Returns a plain Python dict with numpy arrays/scalars.
    """
    import numpy as _np
    from io import BytesIO as _BytesIO

    def _extract(z):
        # z can be a dict-like or NpzFile
        get = z.get if hasattr(z, "get") else z.__getitem__
        x        = _np.asarray(get("x"), dtype=_np.float32)         # (C,T)
        y_mask   = _np.asarray(get("y_mask"), dtype=_np.int64)      # (T,)
        has_pick = int(_np.asarray(get("has_pick")))
        y_idx    = int(_np.asarray(get("y_idx")))
        fs       = float(_np.asarray(get("fs")))
        meta = {}
        if ("meta" in z) if hasattr(z, "__contains__") else ("meta" in getattr(z, "files", [])):
            meta_raw = get("meta")
            # meta may be stored as a 0-d object array
            if isinstance(meta_raw, _np.ndarray) and meta_raw.dtype == object:
                try:
                    meta = meta_raw.item()
                except Exception:
                    meta = {}
            elif isinstance(meta_raw, dict):
                meta = meta_raw
        return {"x": x, "y_mask": y_mask, "has_pick": has_pick, "y_idx": y_idx, "fs": fs, "meta": meta}

    # Case 1: bytes-like (.npz payload from pure tar path)
    if isinstance(sample, (bytes, bytearray, memoryview)):
        with _np.load(_BytesIO(sample), allow_pickle=True) as z:
            return _extract(z)

    # Case 2: an already-open NpzFile
    if isinstance(sample, _np.lib.npyio.NpzFile):
        return _extract(sample)

    # Case 3: dict-like (decoded by WebDataset)
    if isinstance(sample, dict) or hasattr(sample, "keys"):
        return _extract(sample)

    # Case 4: file-like object
    if hasattr(sample, "read"):
        with _np.load(sample, allow_pickle=True) as z:
            return _extract(z)

    raise TypeError(f"Unsupported npz sample type: {type(sample)}")

class WDSSimple(torch.utils.data.IterableDataset):
    """
    Lightweight WebDataset reader over *.tar shards. Falls back to a pure-tar iterator if
    webdataset isn't installed. Yields (x, y_mask, has_pick, y_idx, fs, meta).
    """
    def __init__(self, shard_glob, shuffle=1000, seed=1337):
        self.shard_glob = shard_glob if isinstance(shard_glob, (list, tuple)) else [shard_glob]
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        rng = random.Random(self.seed + wid)

        # Build the shard list from literals, directories, or glob patterns
        shards = []
        for pat in self.shard_glob:
            p = Path(pat)
            if p.is_file() and str(p).endswith(".tar"):
                shards.append(str(p.resolve()))
            elif p.is_dir():
                shards += [str(t.resolve()) for t in p.glob("picks-*.tar")]
            else:
                # glob pattern (supports absolute and relative)
                shards += [str(Path(s).resolve()) for s in glob.glob(pat)]
        shards = sorted(set(shards))
        rng.shuffle(shards)

        if not shards:
            raise RuntimeError("No shard files matched the provided patterns/paths.")

        if _HAS_WDS:
            dataset = (
                wds.WebDataset(shards, resampled=False, shardshuffle=True, empty_check=False,
                            nodesplitter=wds.split_by_worker)
                .shuffle(self.shuffle, initial=self.shuffle)
                .to_tuple("__key__", "npz")
            )
            for key, blob in dataset:
                d = _npz_to_tensor_dict(blob)               # works on raw bytes
                yield d["x"], d["y_mask"], d["has_pick"], d["y_idx"], d["fs"], d["meta"]

        else:
            for tarpath in shards:
                with tarfile.open(tarpath, "r") as tf:
                    members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith(".npz")]
                    rng.shuffle(members)
                    for m in members:
                        blob = tf.extractfile(m).read()
                        d = _npz_to_tensor_dict(blob)
                        yield d["x"], d["y_mask"], d["has_pick"], d["y_idx"], d["fs"], d["meta"]

def _rebuild_mask_from_idx(T, y_idx, has_pick, fs_hz, pos_win_s=0.04):
    m = torch.zeros(T, dtype=torch.int64)
    if has_pick and y_idx >= 0:
        half = int(round(pos_win_s * fs_hz))
        lo = max(0, int(y_idx) - half)
        hi = min(T, int(y_idx) + half + 1)
        if hi > lo:
            m[lo:hi] = 1
    return m

def _to_torch_scalar(x, dtype=None):
    # handles np scalars, Python scalars, 0-d torch tensors
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if dtype is not None else x
    try:
        v = x.item()
    except Exception:
        v = float(x)
    return torch.tensor(v, dtype=dtype if dtype is not None else torch.float32)

def make_collate_wds(target_fs=TARGET_FS, total_win_s=TOTAL_WIN_S, pos_win_s=0.04):
    """
    Collate fn that:
      - converts NumPy arrays from WDS to torch tensors
      - resamples each (C, Ti) to T_target = round(total_win_s * target_fs)
      - scales y_idx accordingly
      - rebuilds y_mask at the new length
      - replaces fs with the new target_fs
    """
    T_target = int(round(total_win_s * target_fs))

    def _collate(batch):
        xs, yms, hps, yidxs, fss, metas = [], [], [], [], [], []
        for x, y_mask, has_pick, y_idx, fs, meta in batch:
            # ---- convert to torch ----
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x)
            if isinstance(y_mask, np.ndarray):
                y_mask = torch.from_numpy(y_mask)
            has_pick = int(np.asarray(has_pick).item())  # python int
            y_idx    = int(np.asarray(y_idx).item())
            fs_val   = float(np.asarray(fs).item())

            # x: (C, Ti)  -> (1, C, Ti) for interpolate
            x = x.to(torch.float32).contiguous()
            C, Ti = x.shape[-2], x.shape[-1]
            xi = x.unsqueeze(0)  # (1, C, Ti)
            if Ti != T_target:
                xi = F.interpolate(xi, size=T_target, mode="linear", align_corners=False)
            xi = xi.squeeze(0).contiguous()  # (C, T_target)

            # scale y_idx to new grid
            y_idx_new = int(round(y_idx * T_target / Ti)) if y_idx >= 0 else -1

            # rebuild mask at new length
            ym_new = _rebuild_mask_from_idx(T_target, y_idx_new, has_pick, target_fs, pos_win_s=pos_win_s)

            xs.append(xi)
            yms.append(ym_new)
            hps.append(torch.tensor(has_pick, dtype=torch.int64))
            yidxs.append(torch.tensor(y_idx_new, dtype=torch.int64))
            fss.append(torch.tensor(float(target_fs), dtype=torch.float32))
            metas.append(meta)

        x = torch.stack(xs, dim=0)          # (B, C, T_target)
        y_mask = torch.stack(yms, dim=0)    # (B, T_target)
        has_pick = torch.stack(hps, dim=0)  # (B,)
        y_idx = torch.stack(yidxs, dim=0)   # (B,)
        fs = torch.stack(fss, dim=0)        # (B,)
        return x, y_mask, has_pick, y_idx, fs, metas

    return _collate

def snr_from_x(x, snr_channel_idx=None):
    """
    x: (B,C,T). If snr_channel_idx given, take its mean. Else compute a cheap proxy.
    """
    if snr_channel_idx is not None:
        idx = snr_channel_idx % x.size(1) # ensure valid index
        return x[:, idx, :].mean(dim=-1).abs() + 1e-6
    # proxy: peak-to-rms in the first 2s segment vs whole – rough but stable
    B, C, T = x.shape
    pre = x[:, 0, : max(400, min(2000, T//5))]  # assuming 200 Hz in ETL; adjust if needed
    rms = pre.pow(2).mean(dim=-1).sqrt() + 1e-6
    peak = x[:, 0, :].abs().amax(dim=-1) + 1e-6
    return (peak / rms).clamp(min=1e-3, max=1e3)

def peek_in_channels(shards):
    """Open one shard independently to read x.shape[1] without consuming loaders."""
    # Use a tiny one-off loader so the main loaders aren't touched
    tmp_ds = WDSSimple(shards, shuffle=0, seed=12345)
    tmp_loader = DataLoader(
        tmp_ds, batch_size=1, num_workers=0,
        collate_fn=make_collate_wds(target_fs=TARGET_FS, total_win_s=TOTAL_WIN_S)
    )
    xb, *_ = next(iter(tmp_loader))
    return int(xb.shape[1])

def count_npz_in_tars(shards):
    """Return total number of .npz entries across one or more .tar shards (reads headers only)."""
    total = 0
    for p in (shards if isinstance(shards, (list, tuple)) else [shards]):
        try:
            with tarfile.open(p, "r") as tf:
                total += sum(1 for m in tf.getmembers() if m.isfile() and m.name.endswith(".npz"))
        except tarfile.ReadError:
            # if p is a glob/dir, WDSSimple will handle expansion; skip here
            pass
    return total

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🔶 Loss: Gaussian NLL + noise-aware KL (used by train_picker)
# ═══════════════════════════════════════════════════════════════════════════════════════
def pick_loss_with_uncertainty(
    pick_frac, log_std, y_idx, has_pick, T,
    snr_scalar=None, kld_weight=0.1, min_sigma=1.0, max_sigma=200.0
    ):
    """
    pick_frac: (B,) ∈ [0,1]  -> mean_idx = pick_frac * T
    log_std:   (B,)          -> std = exp(log_std)
    y_idx:     (B,)          integer pick index, -1 if no-pick
    has_pick:  (B,)          0/1 indicator
    T:         int           number of samples (time length)
    snr_scalar:(B,)          optional SNR proxy for prior width (higher SNR => narrower prior)
    """
    device = pick_frac.device
    mean_idx = pick_frac * T
    log_std_clamped = log_std.clamp(min=float(torch.log(torch.tensor(min_sigma)).item()),
                                    max=float(torch.log(torch.tensor(max_sigma)).item()))
    std = torch.exp(log_std_clamped)

    with_pick = has_pick.bool()
    nll = torch.tensor(0.0, device=device)
    if with_pick.any():
        y = y_idx[with_pick].float()
        mu = mean_idx[with_pick]
        s  = std[with_pick]
        # Gaussian NLL (up to constant)
        nll = 0.5 * (((y - mu) ** 2) / (s ** 2) + 2.0 * log_std_clamped[with_pick])

    # Noise-aware prior on σ: σ_prior ∝ 1 / SNR  (clip to [min_sigma, max_sigma])
    if snr_scalar is None:
        snr_scalar = torch.ones_like(pick_frac)
    a = 50.0  # tune: ~samples scale; larger -> wider prior
    prior_sigma = (a / (snr_scalar + 1e-6)).clamp(min=min_sigma, max=max_sigma)
    prior_log_sigma = torch.log(prior_sigma)

    # KL( N(mu,σ) || N(mu,σ_prior) ) with matched means to avoid biasing time
    kld = prior_log_sigma - log_std_clamped + 0.5 * ((std ** 2) / (prior_sigma ** 2) - 1.0)

    zero = torch.tensor(0.0, device=device)
    # NLL for with-pick, plus KL for everyone (same weight); no extra KL on no-picks
    loss = (nll.mean() if with_pick.any() else zero) + kld_weight * kld.mean()

    return loss, mean_idx, std

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_picker(
    encoder, head, loader_train, loader_val, epochs=20, lr=1e-3, kld_weight=0.1,
    snr_ch=None, device=None, total_train_batches=None, total_val_batches=None
    ):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    encoder.to(device); head.to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    best = float("inf"); best_state = None

    for ep in range(epochs):
        encoder.train(); head.train()
        run_loss = 0.0; n = 0

        pbar = TQDM(total=total_train_batches, desc=f"Train {ep+1}/{epochs}", unit="batch", mininterval=0.3, leave=True)
        print(f"Starting training epoch {ep+1} with {total_train_batches} batches")
        for x, y_mask, has_pick, y_idx, fs, metas in loader_train:
            x = x.to(device, non_blocking=True)
            y_idx = y_idx.to(device); has_pick = has_pick.to(device)
            T = x.size(-1)

            _, feat = encoder(x, return_feats=True)
            pick_frac, log_std = head(feat)
            snr = snr_from_x(x, snr_ch).to(device)

            loss, mu, sigma = pick_loss_with_uncertainty(
                pick_frac, log_std, y_idx, has_pick, T, snr, kld_weight=kld_weight
            )

            if not torch.isfinite(loss):
                # rare but prevents a single bad batch from nuking the epoch
                print("[warn] non-finite loss; skipping batch")
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), 1.0)
            opt.step()

            loss_item = loss.detach().item()
            run_loss += loss_item; n += 1
            pbar.update(1)
            sys.stdout.flush()
            pbar.set_postfix_str(f"loss={loss_item:.4f}")
        pbar.close()

        tr = run_loss / max(1, n)

        # ----- validation -----
        encoder.eval(); head.eval()
        with torch.no_grad():
            run_loss = 0.0; n = 0
            pbarv = TQDM(total=total_val_batches, desc="Val", unit="batch", mininterval=0.3, leave=True)
            for x, y_mask, has_pick, y_idx, fs, metas in loader_val:
                x = x.to(device); y_idx = y_idx.to(device); has_pick = has_pick.to(device)
                T = x.size(-1)
                _, feat = encoder(x, return_feats=True)
                pick_frac, log_std = head(feat)
                snr = snr_from_x(x, snr_ch).to(device)
                loss, _, _ = pick_loss_with_uncertainty(
                    pick_frac, log_std, y_idx, has_pick, T, snr, kld_weight=kld_weight
                )

                if not torch.isfinite(loss):
                    # rare but prevents a single bad batch from nuking the epoch
                    print("[warn] non-finite loss; skipping batch")
                    continue

                run_loss += loss.detach().item(); n += 1
                pbarv.update(1)
            pbarv.close()

        vl = run_loss / max(1, n)
        print(f"[{ep+1:03d}/{epochs}] train={tr:.4f}  val={vl:.4f}")
        if vl < best:
            best, best_state = vl, {"encoder": encoder.state_dict(), "head": head.state_dict()}
    if best_state:
        torch.save(best_state, "best_picker_wds.pt")
    return best

def calibrate_conformal(loader_cal, encoder, head, total_cal_batches=None, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else "cpu")
    encoder.eval(); head.eval()
    errs = []
    with torch.no_grad():
        pbar = TQDM(total=total_cal_batches, desc="Calibrate", unit="batch", mininterval=0.3, leave=True)
        for x, y_mask, has_pick, y_idx, fs, metas in loader_cal:
            x = x.to(device); y_idx = y_idx.to(device); has_pick = has_pick.to(device)
            T = x.size(-1)
            _, feat = encoder(x, return_feats=True)
            pick_frac, log_std = head(feat)
            mu = pick_frac * T
            sigma = torch.exp(log_std)
            mask = has_pick.bool()
            if mask.any():
                z = (y_idx[mask].float() - mu[mask]).abs() / (sigma[mask] + 1e-6)
                errs.append(z.cpu())
            pbar.update(1)
            sys.stdout.flush()
        pbar.close()
    if not errs:
        return 1.96
    z_all = torch.cat(errs)
    return float(torch.quantile(z_all, 0.90))

def infer_on_batch(x, encoder, head, q, max_window_ms=100.0, fs=200.0, device=None):
    """
    Return predicted pick index or None (abstain) for each item.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    encoder.eval(); head.eval()
    with torch.no_grad():
        x = x.to(device)
        T = x.size(-1)
        _, feat = encoder(x, return_feats=True)
        pick_frac, log_std = head(feat)
        mu = (pick_frac * T).cpu().numpy()
        sigma = torch.exp(log_std).cpu().numpy()
        halfwin = (q * sigma) / fs * 1000.0   # ms
        max_halfwin = max_window_ms / 2.0
        picks = []
        for m, hw in zip(mu, halfwin):
            picks.append(int(m) if hw <= max_halfwin else None)
        return picks, sigma

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🧪 VALIDATION: sample N picked traces and save figures under evaluate_picks/
# ═══════════════════════════════════════════════════════════════════════════════════════
def evaluate_and_plot(
    shards, encoder, head, q, n_samples=20, out_dir="evaluate_picks", snr_channel_idx=None, seed=2025
    ):
    """
    Grabs n_samples examples *with picks* from the provided shards, runs inference,
    and saves PNGs with raw, true pick, predicted pick, and an uncertainty band.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.eval().to(device)
    head.eval().to(device)

    # one-pass iterator over shards (no heavy shuffle so we can find picked items quickly)
    ds = WDSSimple(shards, shuffle=0, seed=seed)
    loader = DataLoader(
        ds, batch_size=1, num_workers=0,
        collate_fn=make_collate_wds(target_fs=TARGET_FS, total_win_s=TOTAL_WIN_S)
    )
    saved = 0
    with torch.no_grad():
        for x, y_mask, has_pick, y_idx, fs, metas in loader:
            # require a pick
            if has_pick.item() == 0:
                continue

            x = x.to(device)                    # (1,C,T)
            T = x.size(-1)
            _, feat = encoder(x, return_feats=True)
            pick_frac, log_std = head(feat)
            mu = float((pick_frac * T).item())
            sigma = float(torch.exp(log_std).item())
            fs_val = float(fs.item())
            # conformal half-width in samples
            halfw = q * sigma
            # abstain flag (100 ms default window in samples)
            abstain = (2 * halfw / fs_val * 1000.0) > 100.0

            # plotting
            bp = x[0, 0].detach().cpu().numpy()        # ch0 = bp 3–12 Hz
            t  = np.arange(T, dtype=np.float32) / fs_val
            true_idx = int(y_idx.item()) if (has_pick.item() == 1 and y_idx.item() >= 0) else None

            plt.figure(figsize=(12, 4))
            plt.plot(t, bp, lw=0.9, label="bp 3–12 Hz")
            if true_idx is not None:
                plt.axvline(true_idx / fs_val, color="tab:green", ls="--", lw=2, label="true")
            # predicted mean
            plt.axvline(mu / fs_val, color="tab:blue", ls="-.", lw=2, label="pred")
            # uncertainty band
            lo = max(0, int(mu - halfw))
            hi = min(T - 1, int(mu + halfw))
            if hi > lo:
                plt.axvspan(lo / fs_val, hi / fs_val, color="tab:blue", alpha=0.2,
                            label=f"± q·σ  (q={q:.2f})")
            if abstain:
                plt.title("ABSTAIN (too uncertain)")
            plt.xlabel("time (s)"); plt.ylabel("norm. amp")
            plt.grid(alpha=0.3); plt.legend(loc="upper right")

            # name with station/shot if present
            meta = metas[0] if (metas and isinstance(metas[0], dict)) else {}
            tag = f"{meta.get('geometry','')}_{meta.get('station','')}_{meta.get('shot_id','')}".strip("_")
            fname = f"{saved:03d}_{tag or 'sample'}.png"
            plt.tight_layout()
            plt.savefig(Path(out_dir) / fname, dpi=150)
            plt.close()

            saved += 1
            if saved >= n_samples:
                break

    print(f"[eval] saved {saved} figure(s) to {out_dir}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# 📂 ROOT → DATASET DISCOVERY (dataset.json → geometries → wds/<geom>/picks-*.tar)
# ═══════════════════════════════════════════════════════════════════════════════════════
def discover_shards_from_root(root: str):
    """
    Given ML-feed root, read dataset.json, list geometries, and collect shard paths.
    Returns: dict {geom_id: [shard_paths...]} and a flat list of all shards.
    """
    rootp = Path(root)
    ds_path = rootp / "dataset.json"
    if not ds_path.exists():
        raise FileNotFoundError(f"dataset.json not found at {ds_path}")
    with ds_path.open("r") as f:
        ds = json.load(f)
    geoms = ds.get("geometries", [])
    if not geoms:
        raise ValueError(f"No 'geometries' in {ds_path}")
    shard_map = {}
    for gid in geoms:
        gdir = rootp / "wds" / gid
        if not gdir.exists():
            print(f"[WARN] wds directory missing for geometry {gid}: {gdir}")
            shard_map[gid] = []
            continue
        shards = sorted(str(p) for p in gdir.glob("picks-*.tar"))
        if not shards:
            print(f"[WARN] no shards found in {gdir}")
        shard_map[gid] = shards
    all_shards = [s for lst in shard_map.values() for s in lst]
    return shard_map, all_shards

def train_val_split(shards: list[str], val_ratio: float = 0.1, seed: int = 42):
    rng = random.Random(seed)
    shards = shards[:]  # copy
    rng.shuffle(shards)
    n_val = max(1, int(len(shards) * val_ratio)) if shards else 0
    val = shards[:n_val]
    train = shards[n_val:]
    return train, val

# ═══════════════════════════════════════════════════════════════════════════════════════
# 🏗️ MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════════
def main_wds(
    shards_train, shards_val, batch_size=16, num_workers=2, epochs=20, kld_weight=0.1,
    snr_channel_idx=None, num_heads=2, window_size=256, n_eval=20
    ):
    print("="*80); print("WDS PICKER w/ UNCERTAINTY (mean + logσ, selective abstention)"); print("="*80)

    # 👇 get channel count without consuming the real loaders
    in_channels = peek_in_channels(shards_train)
    print(f"Input channels (from shard): {in_channels}")

    collate_train = make_collate_wds(target_fs=TARGET_FS, total_win_s=TOTAL_WIN_S, pos_win_s=0.04)
    collate_val   = make_collate_wds(target_fs=TARGET_FS, total_win_s=TOTAL_WIN_S, pos_win_s=0.04)

    # real loaders
    loader_train = DataLoader(
        WDSSimple(shards_train, shuffle=2000, seed=42),
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_train, pin_memory=pin,
        persistent_workers=(num_workers > 0)
    )
    loader_val = DataLoader(
        WDSSimple(shards_val, shuffle=200, seed=43),
        batch_size=batch_size, num_workers=num_workers,
        collate_fn=collate_val, pin_memory=pin,
        persistent_workers=(num_workers > 0)
    )

    # --- totals for rich tqdm bars ---
    train_samples = count_npz_in_tars(shards_train)
    val_samples   = count_npz_in_tars(shards_val)
    total_train_batches = math.ceil(train_samples / batch_size) if train_samples else None
    total_val_batches   = math.ceil(val_samples   / batch_size) if val_samples   else None
    print(f"Total train batches: {total_train_batches}, Total val batches: {total_val_batches}")

    encoder = AdaptiveUNet1D(in_channels=in_channels, out_channels=2,
                             use_physics_features=False, num_heads=num_heads, window_size=window_size)
    head = PickHead(in_channels=encoder.downs[0].doubleConv[1].num_features)

    print(f"Max Epochs: {epochs}")
    
    best = train_picker(
        encoder, head, loader_train, loader_val, epochs=epochs, lr=1e-3,
        kld_weight=kld_weight, snr_ch=snr_channel_idx,
        total_train_batches=total_train_batches, total_val_batches=total_val_batches
    )

    q = calibrate_conformal(loader_val, encoder, head, total_cal_batches=total_val_batches)
    print(f"Conformal 90% z-quantile: q={q:.3f}")

    torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(), "q": q},
               "picker_wds_conformal.pt")
    print("Saved model to picker_wds_conformal.pt")

    # 🔎 run visual validation
    evaluate_and_plot(
        shards_val, encoder, head, q,
        n_samples=n_eval, out_dir="evaluate_picks",
        snr_channel_idx=snr_channel_idx
    )

if __name__ == "__main__":
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SINGLE-ENTRY RUNNER (no CLI needed): set ROOT once, script finds everything else
    # ═══════════════════════════════════════════════════════════════════════════════════
    ROOT = "/Volumes/Asif_disk1/Data-feed/ML-feed"   # ← set this to your ML-feed root (absolute or relative)

    # Model/training knobs
    NUM_HEADS     = 2
    WINDOW_SIZE   = 256
    EPOCHS        = 100
    BATCH_SIZE    = 16
    NUM_WORKERS   = 0
    KLD_WEIGHT    = 0.1
    SNR_CH_IDX    = -1   # if your ETL wrote SNR as a specific channel, set its index here (e.g., 2 or 3)

    # 1) Discover shards from dataset.json
    shard_map, all_shards = discover_shards_from_root(ROOT)
    if not all_shards:
        raise RuntimeError(f"No shards found under {ROOT}/wds/*")

    # 2) Make a simple train/val split across all shards
    train_shards, val_shards = train_val_split(all_shards, val_ratio=0.1, seed=42)
    print("Geometries:", ", ".join(k for k,v in shard_map.items() if v))
    print(f"Shards: train={len(train_shards)}  val={len(val_shards)}")

    N_EVAL_PLOTS = 20   # default; change here if you like

    # 3) Run training on WDS
    main_wds(
        shards_train=train_shards,
        shards_val=val_shards,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        epochs=EPOCHS,
        kld_weight=KLD_WEIGHT,
        snr_channel_idx=SNR_CH_IDX,
        num_heads=NUM_HEADS,
        window_size=WINDOW_SIZE,
        n_eval=N_EVAL_PLOTS,
    )

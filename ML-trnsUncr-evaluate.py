#!/usr/bin/env python3
import argparse, os, math, tarfile, json, warnings
from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --- torch / model ---
import torch
import torch.nn as nn

# --- signal ---
import scipy.signal as scisig
from obspy.io.segy.core import _read_segy

from typing import Optional, List, Tuple

# -----------------------------
# Feature helpers (mirror ETL)
# -----------------------------
def _movmean(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w == 1:
        return x.astype(np.float32, copy=False)
    k = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32, copy=False), k, mode="same")

def _safe_bandpass(x: np.ndarray, fmin: float, fmax: float, fs: float, order: int = 4) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    nyq = max(1e-6, 0.5 * float(fs))
    lo = max(1e-6, float(fmin) / nyq)
    hi = min(0.9999, float(fmax) / nyq)
    if hi <= lo or hi >= 1.0:
        return x.copy()
    b, a = scisig.butter(order, [lo, hi], btype="band")
    return scisig.filtfilt(b, a, x)

def bandpass_3_12(x: np.ndarray, fs: float) -> np.ndarray:
    return _safe_bandpass(x, 3.0, 12.0, fs)

def hilbert_safe(x: np.ndarray) -> np.ndarray:
    try:
        return scisig.hilbert(np.asarray(x, dtype=np.float32))
    except Exception:
        return scisig.hilbert(np.ascontiguousarray(x, dtype=np.float32))

def stalta_ratio(x: np.ndarray, fs: float, sta_s: float, lta_s: float) -> np.ndarray:
    x = np.abs(np.asarray(x, dtype=np.float32))
    n_sta = max(1, int(round(sta_s * fs)))
    n_lta = max(n_sta + 1, int(round(lta_s * fs)))
    sta = _movmean(x, n_sta)
    lta = _movmean(x, n_lta)
    r = sta / (lta + 1e-9)
    r[lta <= 1e-9] = 0.0
    return r.astype(np.float32, copy=False)

def sliding_std(x: np.ndarray, w: int) -> np.ndarray:
    w = int(w)
    if w <= 1 or w > x.size:
        return np.zeros_like(x, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    m1 = _movmean(x, w)
    m2 = _movmean(x**2, w)
    out = np.sqrt(np.clip(m2 - m1**2, 0.0, None))
    return out.astype(np.float32, copy=False)

def robust_std(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    mad = np.median(np.abs(x - np.median(x))) + 1e-9
    return 1.4826 * float(mad)

def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return _movmean(np.asarray(x, dtype=np.float32), max(1,int(w)))

# --- 7-channel feature stack (matches latest ETL)
#   0: bp 3–12
#   1: STA/LTA fast (0.05/0.5 s)
#   2: STA/LTA med  (0.10/1.0 s)
#   3: envelope
#   4: band env mid (smoothed env ~50 ms)
#   5: max-amp sliding std (~100 ms)
#   6: SNR scalar (rms(bp)/baseline noise std), broadcast across T
def build_features(seg: np.ndarray, fs: float, noise_end_idx: int = None) -> np.ndarray:
    x = np.asarray(seg, dtype=np.float32)
    T = x.size
    if T < 4:
        return np.zeros((7, T), dtype=np.float32)

    bp = bandpass_3_12(x, fs).astype(np.float32, copy=False)
    env = np.abs(hilbert_safe(bp)).astype(np.float32)
    sta_fast = stalta_ratio(bp, fs, 0.05, 0.5)
    sta_med  = stalta_ratio(bp, fs, 0.10, 1.0)
    band_env_mid = smooth(env, int(max(1, 0.05 * fs))).astype(np.float32)
    maxamp_std   = sliding_std(np.abs(bp), max(3, int(0.10 * fs))).astype(np.float32)

    if noise_end_idx is None or noise_end_idx <= 1:
        noise_end_idx = max(1, int(0.5 * fs))
    baseline_std = robust_std(bp[:int(noise_end_idx)]) + 1e-6
    rms_bp = float(np.sqrt(np.mean(bp**2) + 1e-12))
    snr_scalar = float(rms_bp / baseline_std)
    snr_arr = np.full((T,), snr_scalar, dtype=np.float32)

    feats = [bp, sta_fast, sta_med, env, band_env_mid, maxamp_std, snr_arr]
    X = np.vstack(feats).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X


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

# -----------------------------
# Inference utilities
# -----------------------------
@dataclass
class InferenceArgs:
    segy: Path
    model_path: Path
    out_dir: Path
    trace_step: int
    max_traces: int
    win_s: float
    hop_s: float
    target_fs: float
    abstain_width_s: float
    snr_min: float
    device: str
    mode: str          # "entire" or "window"
    win_start_s: float # only used when mode="window"
    win_len_s: float   # only used when mode="window"

def load_model(model_path: Path, device: torch.device):
    chk = torch.load(model_path, map_location=device)
    q = float(chk.get("q", 1.96))

    # Build the SAME encoder you trained, with physics features ON
    enc = AdaptiveUNet1D(
        in_channels=7, out_channels=2,
        features=[16, 32, 64, 128],
        dropout=0.1,
        use_physics_features=True,     # <-- must stay True
        num_heads=2, window_size=256,
    )

    # PickHead channel count must match your encoder’s first down block
    c_head = enc.downs[0].doubleConv[1].num_features
    head = PickHead(in_channels=c_head)

    # ---- PATCH CHECKPOINT IF 'feature_weights' IS MISSING ----
    enc_sd = chk.get("encoder", {})
    if "feature_weights" not in enc_sd:
        # get the expected tensor shape & dtype from the model
        model_fw = enc.state_dict()["feature_weights"]
        # create a default all-ones tensor on CPU, correct dtype/shape
        enc_sd["feature_weights"] = torch.ones_like(model_fw, dtype=model_fw.dtype, device="cpu")
        chk["encoder"] = enc_sd
        print("[info] checkpoint patched: added default 'feature_weights' (ones)")

    # strict load will now succeed
    enc.load_state_dict(chk["encoder"], strict=True)
    head.load_state_dict(chk["head"],  strict=True)

    enc.to(device).eval()
    head.to(device).eval()
    return enc, head, q

def resample_to_target(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32, copy=False)
    # Use polyphase for better quality
    g = math.gcd(int(round(fs_in)), int(round(fs_out)))
    up = int(round(fs_out)) // g
    dn = int(round(fs_in)) // g
    return scisig.resample_poly(x.astype(np.float32), up, dn).astype(np.float32)

def window_iter(duration: float, tm: np.ndarray, win_s: float, hop_s: float):
    """Yield (s0, e0) sample indices for windows [t, t+win_s)."""
    T = tm.size
    hop_n = max(1, int(round(hop_s * (T / duration))))
    win_n = max(2, int(round(win_s * (T / duration))))
    for s0 in range(0, max(1, T - win_n + 1), hop_n):
        e0 = s0 + win_n
        if e0 <= T:
            yield s0, e0

def predict_on_window(encoder, head, x_win: np.ndarray, device: torch.device):
    """x_win: (C,T) float32 np, at target fs. Return (mu_samples, sigma_samples)."""
    with torch.no_grad():
        xt = torch.from_numpy(x_win[None, ...]).to(device)  # (1,C,T)
        # encoder returns (softmax logits, enc_feats) when return_feats=True
        _, enc_feats = encoder(xt, return_feats=True)       # enc_feats: (B, Cenc, Tenc)
        pick_frac, log_std = head(enc_feats)                # pick_frac in [0,1], log_std log-space
        Tt = xt.shape[-1]
        mu = pick_frac.clamp(0, 1) * (Tt - 1)               # fractional to samples
        sigma = torch.exp(log_std).clamp(min=1e-6)          # in samples (already on target fs)
        return float(mu.item()), float(sigma.item())

# -----------------------------
# Plotting
# -----------------------------
def plot_pick(bp: np.ndarray, fs: float, t0_s: float, mu_samp: float, halfw_samp: float,
              abstain: bool, out_png: Path, title: str):
    T = bp.size
    t = np.arange(T, dtype=np.float32) / fs + float(t0_s)
    plt.figure(figsize=(12, 4))
    plt.plot(t, bp, lw=0.9, label="bp 3–12 Hz")
    # predicted and band
    mu_time = t0_s + (mu_samp / fs)
    plt.axvline(mu_time, color="tab:blue", ls="-.", lw=2, label="pred")
    if halfw_samp > 1:
        plt.axvspan(mu_time - halfw_samp/fs, mu_time + halfw_samp/fs, color="tab:blue", alpha=0.2, label="±q·σ")
    if abstain:
        plt.title(title + "  [ABSTAIN]")
    else:
        plt.title(title)
    plt.xlabel("time (s)"); plt.ylabel("norm. amp"); plt.grid(alpha=0.3); plt.legend(loc="upper right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close()

def plot_fulltrace(bp_full: np.ndarray, fs: float,
                   pred_time_s: Optional[float],
                   band_half_s: Optional[float],
                   abstain: bool, out_png: Path, title: str):
    """Plot the entire bp trace; overlay predicted pick + conformal band when not abstaining."""
    import matplotlib.pyplot as plt
    T = bp_full.size
    t = np.arange(T, dtype=np.float32) / fs
    plt.figure(figsize=(12, 4))
    plt.plot(t, bp_full, lw=0.9, label="bp 3–12 Hz")

    if (pred_time_s is not None) and (not abstain):
        plt.axvline(pred_time_s, color="tab:blue", ls="-.", lw=2, label="pred")
        if band_half_s is not None and band_half_s > 0:
            plt.axvspan(pred_time_s - band_half_s, pred_time_s + band_half_s,
                        color="tab:blue", alpha=0.2, label="±q·σ")

    if abstain:
        plt.title(title + "  [ABSTAIN]")
    else:
        plt.title(title)
    plt.xlabel("time (s)"); plt.ylabel("norm. amp"); plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=140, bbox_inches="tight"); plt.close()

# -----------------------------
# Main inference
# -----------------------------
def run(args: InferenceArgs):
    device = torch.device(args.device)
    enc, head, q = load_model(args.model_path, device)

    st = _read_segy(str(args.segy))
    N = len(st)
    step = max(1, args.trace_step)
    limit = args.max_traces if args.max_traces > 0 else N

    processed = 0
    for i in range(0, N, step):
        if processed >= limit:
            break
        tr = st[i]
        fs0 = float(tr.stats.sampling_rate)
        duration = float(tr.stats.endtime - tr.stats.starttime)
        data = np.asarray(tr.data, dtype=np.float32)
        tm = np.linspace(0.0, duration, tr.stats.npts, endpoint=False)

        # === MODE: "entire" or "window" ===
        npts = tr.stats.npts

        if args.mode == "entire":
            s0 = 0
            e0 = npts
        else:
            # fixed window: clamp to the trace bounds
            s0 = int(round(max(0.0, args.win_start_s) * fs0))
            e0 = s0 + int(round(max(0.01, args.win_len_s) * fs0))
            s0 = max(0, min(s0, npts - 1))
            e0 = max(s0 + 1, min(e0, npts))

        # segment to process (either whole trace or the requested window)
        seg_raw = data[s0:e0]
        bp_seg  = bandpass_3_12(seg_raw, fs0).astype(np.float32)

        # features at native fs, resample to target fs
        X = build_features(seg_raw, fs0, noise_end_idx=int(0.5 * fs0))
        Xr = np.stack([resample_to_target(X[ch], fs0, args.target_fs)
                    for ch in range(X.shape[0])], axis=0).astype(np.float32)

        # SNR gate (scalar SNR is last channel; constant across time)
        snr_scalar = float(Xr[-1, 0])
        abstain_snr = (snr_scalar < args.snr_min)

        # predict μ, σ in samples @ target_fs
        mu_t, sigma_t = predict_on_window(enc, head, Xr, device)
        mu_t = float(mu_t)
        sigma_t = float(max(1e-6, sigma_t))

        # conformal half-widths
        band_half_model = float(q * sigma_t)            # samples @ target_fs
        band_half_s     = band_half_model / args.target_fs

        # abstain decision
        abstain_model = (band_half_s >= args.abstain_width_s)
        abstain       = bool(abstain_snr or abstain_model)

        # convert μ / half-width to seconds relative to the plotted segment
        mu_seg_s        = mu_t / args.target_fs          # seconds within the segment
        band_half_seg_s = band_half_model / args.target_fs

        # Title includes global pick time (relative to full trace) for convenience
        pred_time_global_s = float((s0 / fs0) + mu_seg_s)
        title   = (f"{args.segy.name} | trace {i} | "
                f"{'entire' if args.mode=='entire' else f'win=[{s0/fs0:.2f},{e0/fs0:.2f}]s'} | "
                f"SNR={snr_scalar:.2f} | band={band_half_s:.3f}s")

        out_png = args.out_dir / f"{args.segy.stem}_trace{i:05d}.png"
        # Reuse plot_fulltrace: it plots the array you pass. For window mode we pass the window; for entire mode it's the whole trace.
        plot_fulltrace(
            bp_full=bp_seg,
            fs=fs0,
            pred_time_s=(None if abstain else mu_seg_s),
            band_half_s=(None if abstain else band_half_seg_s),
            abstain=abstain,
            out_png=out_png,
            title=(title if abstain else f"{title} | pred @ {pred_time_global_s:.2f}s"),
        )

        processed += 1


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Infer P-picks on a SEG-Y file with abstention")
    ap.add_argument("segy", type=Path, help="Path to SEG-Y file")
    ap.add_argument("--model", type=Path, default=Path("picker_wds_conformal.pt"), help="Path to trained model checkpoint")
    ap.add_argument("--out", type=Path, default=Path("inference_plots"), help="Output directory for PNGs")
    ap.add_argument("--trace-step", type=int, default=1, help="Read every k-th trace (default: 1 = every trace)")
    ap.add_argument("--max-traces", type=int, default=0, help="Limit number of traces processed (0 = all)")
    ap.add_argument("--win-s", type=float, default=5.0, help="Sliding window length in seconds")
    ap.add_argument("--hop-s", type=float, default=0.5, help="Hop size in seconds between windows")
    ap.add_argument("--target-fs", type=float, default=200.0, help="Target fs used during training (Hz)")
    ap.add_argument("--abstain-width-s", type=float, default=0.20, help="Abstain if q·σ wider than this (seconds)")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--snr-min", type=float, default=1.5, help="Abstain outright if SNR (bp) is below this threshold (default: 1.5)")
    ap.add_argument("--mode", type=str, default="entire", choices=["entire", "window"], help="Run on the entire trace, or a fixed time window.")
    ap.add_argument("--win-start-s", type=float, default=0.0, help="Window start time in seconds (used when --mode window).")
    ap.add_argument("--win-len-s", type=float, default=10.0, help="Window length in seconds (used when --mode window).")

    args_ns = ap.parse_args()

    args = InferenceArgs(
        segy=args_ns.segy,
        model_path=args_ns.model,
        out_dir=args_ns.out,
        trace_step=int(args_ns.trace_step),
        max_traces=int(args_ns.max_traces),
        win_s=float(args_ns.win_s),
        hop_s=float(args_ns.hop_s),
        target_fs=float(args_ns.target_fs),
        abstain_width_s=float(args_ns.abstain_width_s),
        snr_min=float(args_ns.snr_min),
        device=args_ns.device,
        mode=str(args_ns.mode),
        win_start_s=float(args_ns.win_start_s),
        win_len_s=float(args_ns.win_len_s),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run(args)

if __name__ == "__main__":
    main()

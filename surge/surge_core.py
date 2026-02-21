"""
SURGE: Surprise-Guided Token Reduction for efficient video understanding.

Core algorithm that is fully backbone-agnostic. Computes temporal surprise
scores for video tokens and prunes redundant (predictable) tokens.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import numpy as np
from scipy.signal import find_peaks


@dataclass
class SurgeConfig:
    """Configuration for the SURGE token pruning algorithm."""
    rho: float = 0.25
    ema_gamma: float = 0.9
    ema_var_decay: float = 0.9
    min_peak_separation: int = 8
    epsilon: float = 1e-8
    prominence: float = 0.05
    context_floor_k: int = 0
    enable_drift_correction: bool = True
    enable_variance_norm: bool = True


@dataclass
class SurgeOutput:
    """Output of the SURGE algorithm for a single video."""
    keep_mask: torch.Tensor
    surprise_scores: torch.Tensor
    surprise_curve: torch.Tensor
    smoothed_curve: torch.Tensor
    peak_indices: np.ndarray
    event_intervals: List[Tuple[int, int]]


def _build_spatial_coords(H_merged: int, W_merged: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build spatial coordinate matrix X = [1, x, y] of shape [m, 3]."""
    m = H_merged * W_merged
    ys = torch.arange(H_merged, device=device, dtype=dtype)
    xs = torch.arange(W_merged, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones(m, device=device, dtype=dtype)
    X = torch.stack([ones, grid_x.flatten(), grid_y.flatten()], dim=1)
    return X


def compute_surge(
    video_features: torch.Tensor,
    T: int,
    H_merged: int,
    W_merged: int,
    config: SurgeConfig,
) -> SurgeOutput:
    """
    Compute SURGE surprise masks for a single video's post-merger features.

    Args:
        video_features: [N, D] tensor of post-merger video embeddings for one video,
                        where N = T * H_merged * W_merged
        T: number of temporal steps
        H_merged: spatial height after merger (H / spatial_merge_size)
        W_merged: spatial width after merger (W / spatial_merge_size)
        config: SurgeConfig parameters

    Returns:
        SurgeOutput with keep_mask and diagnostics
    """
    device = video_features.device
    dtype = video_features.dtype
    m = H_merged * W_merged
    N = T * m
    D = video_features.shape[1]

    assert video_features.shape[0] == N, (
        f"Expected {N} tokens (T={T}, H'={H_merged}, W'={W_merged}), got {video_features.shape[0]}"
    )

    tokens = video_features.reshape(T, m, D)

    compute_dtype = torch.float32
    if config.enable_drift_correction:
        X = _build_spatial_coords(H_merged, W_merged, device, compute_dtype)
        XtX_inv_Xt = torch.linalg.solve(X.T @ X, X.T)

    scores = torch.zeros(T, m, device=device, dtype=compute_dtype)
    running_var = None

    for t in range(T):
        if t == 0:
            scores[t] = float("inf")
            continue

        tokens_t = tokens[t].to(compute_dtype)
        tokens_tm1 = tokens[t - 1].to(compute_dtype)

        if t == 1:
            error = tokens_t - tokens_tm1
            score = (error * error).sum(dim=1)

            running_var = score.clone()
            if config.enable_variance_norm:
                scores[t] = score / (running_var + config.epsilon)
            else:
                scores[t] = score
            continue

        tokens_tm2 = tokens[t - 2].to(compute_dtype)
        raw_delta = tokens_tm1 - tokens_tm2

        if config.enable_drift_correction:
            C = XtX_inv_Xt @ raw_delta
            drift = X @ C
            detrended_delta = raw_delta - drift
        else:
            detrended_delta = raw_delta

        pred = tokens_tm1 + detrended_delta

        error = tokens_t - pred
        score = (error * error).sum(dim=1)

        if config.enable_variance_norm:
            running_var = config.ema_var_decay * running_var + (1 - config.ema_var_decay) * score
            scores[t] = score / (running_var + config.epsilon)
        else:
            scores[t] = score

    finite_scores = scores[scores.isfinite()]
    if finite_scores.numel() > 0 and config.rho < 1.0:
        threshold = torch.quantile(finite_scores, 1.0 - config.rho)
        keep_mask = scores >= threshold
    else:
        keep_mask = torch.ones(T, m, device=device, dtype=torch.bool)

    keep_mask_flat = keep_mask.reshape(N)

    kept_per_frame = keep_mask.sum(dim=1).float()
    surprise_curve = kept_per_frame

    smoothed = torch.zeros_like(surprise_curve)
    smoothed[0] = surprise_curve[0]
    for t in range(1, T):
        smoothed[t] = config.ema_gamma * smoothed[t - 1] + (1 - config.ema_gamma) * surprise_curve[t]

    smoothed_np = smoothed.cpu().numpy()
    if T > 1:
        peaks, properties = find_peaks(
            smoothed_np,
            distance=config.min_peak_separation,
            prominence=config.prominence,
        )
    else:
        peaks = np.array([], dtype=np.int64)

    event_intervals = []
    if len(peaks) > 0:
        boundaries = [0]
        for i in range(len(peaks) - 1):
            mid = (peaks[i] + peaks[i + 1]) // 2
            boundaries.append(mid)
        boundaries.append(T)

        for i in range(len(peaks)):
            event_intervals.append((int(boundaries[i]), int(boundaries[i + 1])))

    return SurgeOutput(
        keep_mask=keep_mask_flat,
        surprise_scores=scores,
        surprise_curve=surprise_curve,
        smoothed_curve=smoothed,
        peak_indices=peaks,
        event_intervals=event_intervals,
    )


def apply_surge_to_sequence(
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    video_token_mask: torch.Tensor,
    surge_keep: torch.Tensor,
    cache_position: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Remove pruned video tokens from the assembled multimodal sequence.

    Args:
        inputs_embeds: [B, seq_len, D] — full multimodal embeddings
        position_ids: [3, B, seq_len] — 3D RoPE position IDs
        attention_mask: [B, seq_len] — 1D attention mask
        video_token_mask: [B, seq_len] — boolean, True where video tokens are
        surge_keep: [num_video_tokens] — boolean, which video tokens to keep
        cache_position: [seq_len] — optional cache position indices

    Returns:
        Tuple of pruned (inputs_embeds, position_ids, attention_mask, cache_position)
    """
    B = inputs_embeds.shape[0]

    if B == 1:
        seq_len = inputs_embeds.shape[1]
        video_positions = video_token_mask[0].nonzero(as_tuple=True)[0]

        seq_keep = torch.ones(seq_len, device=inputs_embeds.device, dtype=torch.bool)
        seq_keep[video_positions] = surge_keep.to(inputs_embeds.device)

        inputs_embeds = inputs_embeds[:, seq_keep]
        position_ids = position_ids[:, :, seq_keep]
        attention_mask = attention_mask[:, seq_keep]

        if cache_position is not None:
            new_len = inputs_embeds.shape[1]
            cache_position = torch.arange(new_len, device=inputs_embeds.device, dtype=cache_position.dtype)

        return inputs_embeds, position_ids, attention_mask, cache_position

    device = inputs_embeds.device
    new_embeds_list = []
    new_pos_list = []
    new_mask_list = []

    for b in range(B):
        seq_len = inputs_embeds.shape[1]
        video_positions = video_token_mask[b].nonzero(as_tuple=True)[0]

        seq_keep = torch.ones(seq_len, device=device, dtype=torch.bool)
        seq_keep[video_positions] = surge_keep.to(device)

        new_embeds_list.append(inputs_embeds[b, seq_keep])
        new_pos_list.append(position_ids[:, b, seq_keep])
        new_mask_list.append(attention_mask[b, seq_keep])

    max_len = max(e.shape[0] for e in new_embeds_list)
    D = inputs_embeds.shape[2]

    padded_embeds = torch.zeros(B, max_len, D, device=device, dtype=inputs_embeds.dtype)
    padded_pos = torch.zeros(3, B, max_len, device=device, dtype=position_ids.dtype)
    padded_mask = torch.zeros(B, max_len, device=device, dtype=attention_mask.dtype)

    for b in range(B):
        L = new_embeds_list[b].shape[0]
        padded_embeds[b, :L] = new_embeds_list[b]
        padded_pos[:, b, :L] = new_pos_list[b]
        padded_mask[b, :L] = new_mask_list[b]

    if cache_position is not None:
        cache_position = torch.arange(max_len, device=device, dtype=cache_position.dtype)

    return padded_embeds, padded_pos, padded_mask, cache_position

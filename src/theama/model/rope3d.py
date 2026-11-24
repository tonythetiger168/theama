# SPDX-License-Identifier: Apache-2.0
"""3D multimodal RoPE over (t, h, w) — with a time-axis scaling map so video
latent frames and audio latent frames share one timeline.

Idea reference (clean-room): MiniMax H3's 3D multimodal RoPE as described in
its public model card. Implementation here is original.
"""

from __future__ import annotations

import torch


def rope_freqs(dim: int, theta: float = 10000.0, device=None) -> torch.Tensor:
    """Standard rotary frequency vector of size dim//2."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))


def build_3d_rope(t: int, h: int, w: int, dim: int, device=None) -> torch.Tensor:
    """Position tensor for a (t, h, w) token grid, split dims as t:dim/2, h:dim/4, w:dim/4."""
    dt, dh = dim // 2, dim // 4
    ft = torch.arange(t, device=device)[:, None] * rope_freqs(dt, device=device)[None]
    fh = torch.arange(h, device=device)[:, None] * rope_freqs(dh, device=device)[None]
    fw = torch.arange(w, device=device)[:, None] * rope_freqs(dh, device=device)[None]
    ft = ft[:, None, None, :].expand(t, h, w, -1)
    fh = fh[None, :, None, :].expand(t, h, w, -1)
    fw = fw[None, None, :, :].expand(t, h, w, -1)
    return torch.cat([ft, fh, fw], dim=-1).reshape(t * h * w, -1)  # (T*H*W, dim//2)


def audio_time_map(video_t: int, audio_t: int) -> torch.Tensor:
    """Linear map from audio latent index to the video (t) axis, for shared-time RoPE."""
    return torch.linspace(0, max(video_t - 1, 1), audio_t)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding. x: (B, heads, L, dim); freqs: (L, dim//2)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)

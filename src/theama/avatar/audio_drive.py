# SPDX-License-Identifier: Apache-2.0
"""AudioDrive: audio stream -> facial/body motion conditioning with lip-sync.

M6a fallback path: LatentSync (Apache-2.0) lip-sync pipeline on top of
Wan-Animate output; native integration replaces it once the joint A/V
training from the Film line (M2) is reusable here.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AudioDrive(nn.Module):
    """Audio latent -> per-frame motion conditioning tokens. Skeleton — M6a."""

    def __init__(self, audio_dim: int = 1024, motion_dim: int = 512, window: int = 16):
        super().__init__()
        self.window = window  # audio frames of context per video frame
        self.attn_pool = nn.MultiheadAttention(motion_dim, 8, kdim=audio_dim,
                                               vdim=audio_dim, batch_first=True)
        self.frame_query = nn.Parameter(torch.randn(1, 1, motion_dim))

    def forward(self, audio_tokens: torch.Tensor, n_video_frames: int) -> torch.Tensor:
        """audio_tokens: (B, Ta, Da) -> motion cond (B, Tv, Dm), one token per video frame."""
        q = self.frame_queries(audio_tokens.shape[0], n_video_frames,
                               audio_tokens.device, audio_tokens.dtype)
        out, _ = self.attn_pool(q, audio_tokens, audio_tokens)
        return out

    def frame_queries(self, batch: int, n: int, device, dtype) -> torch.Tensor:
        # TODO(M6a): learned per-frame queries with relative-time RoPE offsets
        # aligned to the video frame grid via model.rope3d.audio_time_map.
        return self.frame_query.expand(batch, n, -1).to(device=device, dtype=dtype)

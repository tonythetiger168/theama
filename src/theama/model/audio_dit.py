# SPDX-License-Identifier: Apache-2.0
"""Parallel audio DiT with bidirectional A/V cross-attention bridges.

Architecture pattern (clean-room, ideas only): asymmetric dual-stream DiT as
in LTX-2.5 — each modality keeps its own self-attention stream, with a
bidirectional cross-attention bridge every N layers and modality-aware CFG.
Permissive reference implementations to consult while coding: Kandinsky 6.0
(MIT, 44.1kHz + lip-sync) and MAGI-2 (Apache-2.0). No H3/LTX weights, code,
or outputs are used anywhere in Theama.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalBridge(nn.Module):
    """Bidirectional cross-attention between video and audio token streams."""

    def __init__(self, video_dim: int, audio_dim: int, heads: int = 16):
        super().__init__()
        self.v2a = nn.MultiheadAttention(audio_dim, heads, kdim=video_dim, vdim=video_dim,
                                         batch_first=True)
        self.a2v = nn.MultiheadAttention(video_dim, heads, kdim=audio_dim, vdim=audio_dim,
                                         batch_first=True)
        self.v_norm = nn.LayerNorm(video_dim)
        self.a_norm = nn.LayerNorm(audio_dim)

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """v: (B, Tv, Dv), a: (B, Ta, Da). Returns updated (v, a) with residual adds."""
        dv, _ = self.a2v(self.v_norm(v), a, a)
        da, _ = self.v2a(self.a_norm(a), v, v)
        return v + dv, a + da


class AudioDiTBlock(nn.Module):
    """Single audio-stream DiT block (self-attention + MLP, adaLN modulation)."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
        self.adaln = nn.Linear(dim, dim * 6)  # scale/shift/gate x2 — conditioned on timestep+text

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        s1, b1, g1, s2, b2, g2 = self.adaln(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1[:, None]) + b1[:, None]
        x = x + g1[:, None] * self.attn(h, h, h)[0]
        h = self.norm2(x) * (1 + s2[:, None]) + b2[:, None]
        return x + g2[:, None] * self.mlp(h)


class AudioDiT(nn.Module):
    """Parallel audio stream DiT operating on DAC 44.1kHz stereo latents.

    Bridges into the video stream every `bridge_every` layers. The video
    stream itself lives in the Wan 2.2 backbone; this module owns only the
    audio tokens and the bridges (trained first via LoRA on bridges, M2).
    """

    def __init__(self, dim: int = 1024, layers: int = 12, heads: int = 16,
                 video_dim: int = 5120, bridge_every: int = 3, latent_dim: int = 128):
        super().__init__()
        self.bridge_every = bridge_every
        self.in_proj = nn.Linear(latent_dim, dim)
        self.blocks = nn.ModuleList(AudioDiTBlock(dim, heads) for _ in range(layers))
        self.bridges = nn.ModuleList(
            CrossModalBridge(video_dim, dim, heads)
            for _ in range(layers // bridge_every)
        )
        self.out = nn.Linear(dim, latent_dim)

    def forward(self, a_latent: torch.Tensor, v_tokens: torch.Tensor,
                cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One denoising step on the audio stream; returns (audio_eps, updated video tokens)."""
        a = self.in_proj(a_latent)
        for i, blk in enumerate(self.blocks):
            a = blk(a, cond)
            if (i + 1) % self.bridge_every == 0:
                v_tokens, a = self.bridges[(i + 1) // self.bridge_every - 1](v_tokens, a)
        return self.out(a), v_tokens

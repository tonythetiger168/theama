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
import torch.nn.functional as F


class CrossModalBridge(nn.Module):
    """Bidirectional cross-attention between video and audio token streams.

    Set ``norm_video_kv`` when the video tokens are un-normalised backbone
    hidden states (``wan_blocks`` internal bridging). Wan block activations
    arrive ~40x larger than VAE latents; feeding them raw as v2a
    cross-attention keys/values blows up the audio stream (issue #4 RunPod
    smoke: audio loss 79.0 / grad norm 94.0, ~40x the latent-grid path). The
    flag applies a parameter-free per-token LayerNorm to the video tokens
    used as v2a keys/values only — the a2v direction and the video residual
    keep the raw hidden states, so the frozen backbone stream is untouched.
    Leave it False for the latent-grid path: VAE latents are already ~unit
    scale and their per-token channel statistics carry content.
    """

    def __init__(self, video_dim: int, audio_dim: int, heads: int = 16,
                 norm_video_kv: bool = False):
        super().__init__()
        self.v2a = nn.MultiheadAttention(audio_dim, heads, kdim=video_dim, vdim=video_dim,
                                         batch_first=True)
        self.a2v = nn.MultiheadAttention(video_dim, heads, kdim=audio_dim, vdim=audio_dim,
                                         batch_first=True)
        self.v_norm = nn.LayerNorm(video_dim)
        self.a_norm = nn.LayerNorm(audio_dim)
        self.norm_video_kv = bool(norm_video_kv)

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """v: (B, Tv, Dv), a: (B, Ta, Da). Returns updated (v, a) with residual adds."""
        dv, _ = self.a2v(self.v_norm(v), a, a)
        v_kv = F.layer_norm(v, self.v_norm.normalized_shape) if self.norm_video_kv else v
        da, _ = self.v2a(self.a_norm(a), v_kv, v_kv)
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

    ``norm_video_kv`` must be True for ``wan_blocks`` internal bridging
    (Wan hidden states are un-normalised; see CrossModalBridge) and False
    for the default latent-grid path.
    """

    def __init__(self, dim: int = 1024, layers: int = 12, heads: int = 16,
                 video_dim: int = 5120, bridge_every: int = 3, latent_dim: int = 128,
                 norm_video_kv: bool = False):
        super().__init__()
        self.bridge_every = bridge_every
        self.in_proj = nn.Linear(latent_dim, dim)
        self.blocks = nn.ModuleList(AudioDiTBlock(dim, heads) for _ in range(layers))
        self.bridges = nn.ModuleList(
            CrossModalBridge(video_dim, dim, heads, norm_video_kv=norm_video_kv)
            for _ in range(layers // bridge_every)
        )
        self.out = nn.Linear(dim, latent_dim)

    def start_stream(self, a_latent: torch.Tensor) -> torch.Tensor:
        """Project DAC tokens into the audio DiT stream."""
        return self.in_proj(a_latent)

    def advance_to_bridge(
        self, a: torch.Tensor, cond: torch.Tensor, bridge_index: int
    ) -> torch.Tensor:
        """Run the audio-block segment immediately preceding one bridge.

        Wan's internal-bridge adapter calls this from successive backbone-block
        hooks. Advancing in segments (rather than precomputing the whole audio
        stream) keeps later audio blocks downstream of every earlier ``v2a``
        update, so both bridge directions receive useful gradients.
        """
        if not 0 <= bridge_index < len(self.bridges):
            raise IndexError(f"bridge index {bridge_index} out of range")
        start = bridge_index * self.bridge_every
        for block in self.blocks[start : start + self.bridge_every]:
            a = block(a, cond)
        return a

    def finish_stream(self, a: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Run any trailing audio blocks and project back to DAC latent size."""
        start = len(self.bridges) * self.bridge_every
        for block in self.blocks[start:]:
            a = block(a, cond)
        return self.out(a)

    def forward(self, a_latent: torch.Tensor, v_tokens: torch.Tensor,
                cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One denoising step on the audio stream; returns (audio_eps, updated video tokens)."""
        a = self.start_stream(a_latent)
        for bridge_index, bridge in enumerate(self.bridges):
            a = self.advance_to_bridge(a, cond, bridge_index)
            v_tokens, a = bridge(v_tokens, a)
        return self.finish_stream(a, cond), v_tokens

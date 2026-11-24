# SPDX-License-Identifier: Apache-2.0
"""Wan 2.2 video DiT backbone wrapper (Apache-2.0 weights, user-downloaded).

Wraps the diffusers Wan pipelines so the rest of Theama never touches
backbone internals directly. Dev/CI path uses TI2V-5B (24GB-friendly).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

A14B_REPO = "Wan-AI/Wan2.2-T2V-A14B"
TI2V5B_REPO = "Wan-AI/Wan2.2-TI2V-5B"


@dataclass
class BackboneConfig:
    repo_id: str = A14B_REPO
    dev_repo: str = TI2V5B_REPO
    dtype: torch.dtype = torch.bfloat16
    # Wan 2.2 A14B is a two-expert MoE: a high-noise and a low-noise expert,
    # switched by diffusion timestep. Boundary from the official model card.
    expert_boundary: float = 0.875


class VideoBackbone:
    """Lazy-loading wrapper around Wan 2.2 pipelines.

    Usage:
        bb = VideoBackbone(BackboneConfig())
        pipe = bb.load_t2v(device="cuda")
        frames = bb.generate(pipe, prompt="...", seconds=5, size=(1280, 720))
    """

    def __init__(self, config: BackboneConfig | None = None, dev: bool = False):
        self.config = config or BackboneConfig()
        self.dev = dev
        self._pipe = None

    def load_t2v(self):
        """Load the T2V pipeline (A14B full or TI2V-5B dev)."""
        from diffusers import AutoencoderKLWan, WanPipeline  # noqa: PLC0415

        repo = self.config.dev_repo if self.dev else self.config.repo_id
        vae = AutoencoderKLWan.from_pretrained(repo, subfolder="vae", torch_dtype=torch.float32)
        self._pipe = WanPipeline.from_pretrained(repo, vae=vae, torch_dtype=self.config.dtype)
        return self._pipe

    @torch.no_grad()
    def generate(self, pipe, prompt: str, seconds: int = 5, size=(1280, 720), fps: int = 24,
                 seed: int | None = None, negative_prompt: str = ""):
        """Run one base-generation pass. Returns decoded frames (uint8 tensor T,H,W,C)."""
        generator = torch.Generator().manual_seed(seed) if seed is not None else None
        out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=size[1],
            width=size[0],
            num_frames=seconds * fps + 1,  # Wan convention: 8n+1 frames
            num_inference_steps=40,
            generator=generator,
        )
        return out.frames[0]

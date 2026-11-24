# SPDX-License-Identifier: Apache-2.0
"""T2I backbone wrapper: Qwen-Image-2512 (quality) + Z-Image Turbo (fast).

Both Apache-2.0. Qwen-Image 2.1 (top open-weight quality) is NOT used — its
Qwen Research License is non-commercial only (see docs/license-risks.md).
Differentiator: CJK / Traditional-Chinese text rendering accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

QUALITY_REPO = "Qwen/Qwen-Image-2512"          # 20B, Apache-2.0
FAST_REPO = "Tongyi-MAI/Z-Image-Turbo"         # 6B, 8 NFE, Apache-2.0
EDIT_REPO = "Qwen/Qwen-Image-Edit-2511"        # multi-reference edit, Apache-2.0


@dataclass
class ImageConfig:
    quality_repo: str = QUALITY_REPO
    fast_repo: str = FAST_REPO
    edit_repo: str = EDIT_REPO
    dtype: torch.dtype = torch.bfloat16
    fast_steps: int = 8                        # Z-Image Turbo NFE
    quality_steps: int = 50


class ImageBackbone:
    """Lazy-loading wrapper. mode='quality' (default) or 'fast' for drafts/dev."""

    def __init__(self, config: ImageConfig | None = None, mode: str = "quality"):
        if mode not in ("quality", "fast"):
            raise ValueError(f"unknown mode: {mode}")
        self.config = config or ImageConfig()
        self.mode = mode
        self._pipe = None

    def load(self):
        """Load the diffusers pipeline for the selected mode. M7a."""
        repo = self.config.quality_repo if self.mode == "quality" else self.config.fast_repo
        # TODO(M7a): diffusers QwenImagePipeline / ZImagePipeline from_pretrained(repo)
        raise NotImplementedError("M7a: pipeline load")

    @torch.no_grad()
    def generate(self, prompt: str, size=(1664, 928), seed: int | None = None):
        """One T2I pass. steps = fast_steps in fast mode else quality_steps."""
        raise NotImplementedError("M7a: T2I generation")

    @torch.no_grad()
    def edit(self, images: list, instruction: str):
        """Multi-reference instruction editing via Qwen-Image-Edit-2511. M7b."""
        raise NotImplementedError("M7b: instruction editing")

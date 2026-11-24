# SPDX-License-Identifier: Apache-2.0
"""Joint audio-video training loop skeleton (M2+).

Objective: flow matching with logit-normal timestep sampling, applied to both
streams. Phase 1 trains only CrossModalBridge + AudioDiT via LoRA; phase 2
unfreezes for full fine-tune. See docs/training-plan.md for acceptance gates.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def logit_normal_sample(batch: int, mu: float = 0.0, sigma: float = 1.0,
                        device=None) -> torch.Tensor:
    """Logit-normal timestep sampling (flow matching)."""
    u = torch.randn(batch, device=device) * sigma + mu
    return torch.sigmoid(u)


def flow_matching_loss(model_eps: torch.Tensor, x1: torch.Tensor,
                       x0_noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """MSE between predicted velocity and target velocity (x1 - x0)."""
    target = x1 - x0_noise
    return F.mse_loss(model_eps, target)


class JointAVTrainer:
    """LoRA-first joint trainer. Skeleton — wires up in M2."""

    def __init__(self, video_backbone, audio_dit, config: dict):
        self.video = video_backbone
        self.audio = audio_dit
        self.config = config
        # TODO(M2): freeze Wan backbone; attach LoRA to audio bridges only.

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        t = logit_normal_sample(batch["video_latent"].shape[0], device=... )
        # TODO(M2): add noise at t to both streams, forward joint model,
        # sum flow_matching_loss over video + audio streams, backward.
        _ = t
        return {"loss": math.nan}  # placeholder until M2 wiring

# SPDX-License-Identifier: Apache-2.0
"""LoRA injection for M2 bridge warmup.

Bridge warmup trains only the ``CrossModalBridge`` layers while Wan and the
audio stream stay frozen. LoRA makes that warmup cheap and checkpoint-light:
the bridge projections stay frozen, a low-rank residual ``B @ A`` is learned
instead, and the adapter can later be merged back into a plain ``nn.Linear``
so the ``full`` phase fine-tunes merged weights directly (no inference-time
overhead, no dependency on PEFT for this path).

Design notes:
- Targets are the ``out_proj`` projections of both cross-attentions
  (``v2a`` / ``a2v``) in every bridge — the fused ``in_proj_weight`` of
  ``nn.MultiheadAttention`` is left dense and frozen.
- ``lora_b`` is zero-initialized, so injection is an exact identity at step 0
  (verified by tests): warmup starts from the un-bridged model behaviour.
- Scaling follows the LoRA convention ``alpha / rank``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
    "inject_lora_into_audio_dit",
    "iter_lora_modules",
    "lora_parameters",
    "lora_state_dict",
    "merge_lora",
]


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be > 0, got {alpha}")
        self.base = base
        for param in self.base.parameters():
            param.requires_grad_(False)
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / rank
        in_features, out_features = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight (base + scaled low-rank residual).

        ``nn.MultiheadAttention`` reads ``out_proj.weight``/``bias`` directly
        instead of calling the module, so fused callers see the adapted weight
        and gradients still flow to ``lora_a``/``lora_b``.
        """
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scale

    @property
    def bias(self) -> torch.nn.Parameter | None:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = F.linear(F.linear(x, self.lora_a), self.lora_b) * self.scale
        return base_out + delta.to(base_out.dtype)

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Return a plain ``nn.Linear`` with the low-rank residual folded in."""
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(self.base.weight + (self.lora_b @ self.lora_a) * self.scale)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def iter_lora_modules(module: nn.Module) -> list[tuple[str, LoRALinear]]:
    """All ``LoRALinear`` submodules of ``module`` as (qualified name, module)."""
    return [
        (name, sub)
        for name, sub in module.named_modules()
        if isinstance(sub, LoRALinear)
    ]


def lora_parameters(module: nn.Module) -> list[torch.nn.Parameter]:
    """Trainable low-rank parameters (A and B) under ``module``."""
    params: list[torch.nn.Parameter] = []
    for _, lora in iter_lora_modules(module):
        params.extend((lora.lora_a, lora.lora_b))
    return params


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    """Checkpoint-friendly state dict containing only LoRA tensors."""
    state: dict[str, torch.Tensor] = {}
    for name, lora in iter_lora_modules(module):
        state[f"{name}.lora_a"] = lora.lora_a.detach().clone()
        state[f"{name}.lora_b"] = lora.lora_b.detach().clone()
    return state


def inject_lora_into_audio_dit(
    audio_dit: nn.Module,
    rank: int = 8,
    alpha: float | None = None,
) -> int:
    """Wrap bridge ``v2a``/``a2v`` out-projections with ``LoRALinear``.

    Returns the number of injected projections (2 per bridge). Idempotent:
    projections already wrapped are left untouched.
    """
    alpha = float(alpha if alpha is not None else 2 * rank)
    injected = 0
    for bridge in audio_dit.bridges:
        for attn_name in ("v2a", "a2v"):
            attn = getattr(bridge, attn_name)
            if isinstance(attn.out_proj, LoRALinear):
                continue
            attn.out_proj = LoRALinear(attn.out_proj, rank=rank, alpha=alpha)
            injected += 1
    return injected


@torch.no_grad()
def merge_lora(audio_dit: nn.Module) -> int:
    """Fold every bridge LoRA back into a plain ``nn.Linear``. Returns merges."""
    merged_count = 0
    for bridge in audio_dit.bridges:
        for attn_name in ("v2a", "a2v"):
            attn = getattr(bridge, attn_name)
            if isinstance(attn.out_proj, LoRALinear):
                attn.out_proj = attn.out_proj.merged_linear()
                merged_count += 1
    return merged_count

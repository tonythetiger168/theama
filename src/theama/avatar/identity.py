# SPDX-License-Identifier: Apache-2.0
"""Identity encoder: reference image(s) -> identity embedding, hot-swappable.

Backbone: Wan-Animate (Apache-2.0, same Wan family as the Film line).
M6c adds VLM-agent-driven mid-stream reference swaps (Vidu S2 class feature,
clean-room implementation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class IdentityBank:
    """Named identity embeddings; supports mid-stream hot swap (M6c)."""

    entries: dict[str, torch.Tensor] = None

    def __post_init__(self):
        self.entries = self.entries or {}

    def register(self, name: str, emb: torch.Tensor) -> None:
        self.entries[name] = emb

    def swap(self, name: str) -> torch.Tensor:
        """Hot-swap target identity mid-stream. Raises on unknown name."""
        if name not in self.entries:
            raise KeyError(f"unknown identity: {name}")
        return self.entries[name]


class IdentityEncoder(nn.Module):
    """Reference image -> identity embedding. Skeleton — lands in M6a."""

    def __init__(self, dim: int = 1024):
        super().__init__()
        self.proj = nn.Linear(512, dim)  # face/appearance feature dim -> model dim

    def forward(self, ref_features: torch.Tensor) -> torch.Tensor:
        # TODO(M6a): plug Wan-Animate's reference/face encoder features here.
        return self.proj(ref_features)

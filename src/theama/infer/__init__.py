# SPDX-License-Identifier: Apache-2.0
"""Inference entry points (GPU-only at runtime; imports stay CPU-safe)."""

from theama.infer.config import (
    DEFAULT_NEGATIVE_PROMPT,
    MODELS,
    ModelSpec,
    frames_for_seconds,
)

__all__ = [
    "DEFAULT_NEGATIVE_PROMPT",
    "MODELS",
    "ModelSpec",
    "frames_for_seconds",
]

# SPDX-License-Identifier: Apache-2.0
"""Inference package — backbone wrappers (M1) + pipeline orchestration (M2+).

- config / mux / t2v / i2v / pipeline CLI: M1 working code (GPU at runtime,
  CPU-safe imports).
- extend_30s / upscale_4k: M3/M4 stage skeletons (placeholder).
"""

from theama.inference.config import (
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

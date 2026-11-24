# SPDX-License-Identifier: Apache-2.0
"""Model registry + generation config for M1 backbone reproduction.

Hard numbers from official sources (2026-11 survey, docs/landscape-2026-11.md):
- Wan 2.2 T2V/I2V-A14B: MoE 27B total / 14B active, native 81f @ 16fps,
  guidance 3.0 (high-noise expert) / 4.0 (low-noise expert),
  boundary_ratio 0.875, flow_shift 5.0 @720p (12.0 @480p).
- Wan 2.2 TI2V-5B: dense, native 1280x704, 121f @ 24fps, steps 50,
  guidance 5.0, flow_shift 5.0. 24GB-friendly dev/CI path.
"""

from __future__ import annotations

from dataclasses import dataclass

# Diffusers-layout repos (the original Wan-AI repos keep the research layout).
T2V_A14B = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
I2V_A14B = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
TI2V_5B = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Wan 2.2 3D causal VAE compresses time 4x: num_frames must be 4k + 1.
WAN_TEMPORAL_STRIDE = 4
# Spatial dims must be multiples of the VAE 16x spatial stride patch grid.
WAN_SPATIAL_MULT = 16

# Wan 2.2 official default negative prompt (model card, zh).
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
    "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    task: str                       # "t2v" | "i2v" | "ti2v"
    width: int
    height: int
    fps: int
    native_seconds: int
    num_frames: int
    steps: int
    guidance_scale: float
    guidance_scale_2: float | None  # MoE low-noise expert CFG (A14B only)
    flow_shift: float
    boundary_ratio: float | None    # MoE expert boundary (A14B only)
    min_vram_gb: int                # with model CPU offload


MODELS: dict[str, ModelSpec] = {
    # A14B native cadence is 81f @ 16fps; 24fps output comes from M3/fps-interp.
    "t2v-a14b": ModelSpec(
        "t2v-a14b", T2V_A14B, "t2v", 1280, 720, 16, 5, 81,
        steps=40, guidance_scale=3.0, guidance_scale_2=4.0,
        flow_shift=5.0, boundary_ratio=0.875, min_vram_gb=40,
    ),
    "i2v-a14b": ModelSpec(
        "i2v-a14b", I2V_A14B, "i2v", 1280, 720, 16, 5, 81,
        steps=40, guidance_scale=3.0, guidance_scale_2=4.0,
        flow_shift=5.0, boundary_ratio=0.875, min_vram_gb=40,
    ),
    # TI2V-5B native cadence is 121f @ 24fps — the dev/CI path.
    "ti2v-5b": ModelSpec(
        "ti2v-5b", TI2V_5B, "ti2v", 1280, 704, 24, 5, 121,
        steps=50, guidance_scale=5.0, guidance_scale_2=None,
        flow_shift=5.0, boundary_ratio=None, min_vram_gb=16,
    ),
}


def frames_for_seconds(spec: ModelSpec, seconds: float) -> int:
    """Frame count for `seconds` at the model's native fps, snapped to 4k+1."""
    n = int(round(seconds * spec.fps)) + 1
    if (n - 1) % WAN_TEMPORAL_STRIDE != 0:
        n += (WAN_TEMPORAL_STRIDE - (n - 1) % WAN_TEMPORAL_STRIDE) % WAN_TEMPORAL_STRIDE
    return n

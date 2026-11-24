# SPDX-License-Identifier: Apache-2.0
"""T2V/TI2V inference — lazy diffusers load (GPU-only at runtime).

Nothing in this module imports diffusers/torch at module import time, so the
CPU-only CI can still test configs, the eval set, and the mux pipeline.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from theama.inference.config import (
    DEFAULT_NEGATIVE_PROMPT,
    MODELS,
    ModelSpec,
)


def load_pipeline(spec_name: str = "ti2v-5b", offload: bool = True):
    """Load a WanPipeline. offload=True keeps 16-24GB cards viable."""
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    spec = MODELS[spec_name]
    vae = AutoencoderKLWan.from_pretrained(
        spec.repo_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanPipeline.from_pretrained(
        spec.repo_id, vae=vae, torch_dtype=torch.bfloat16
    )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe, spec


def generate_frames(
    pipe,
    spec: ModelSpec,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 42,
    num_frames: int | None = None,
) -> list[Image.Image]:
    """Run one T2V generation, return PIL frames at the model's native size."""
    import torch

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=spec.height,
        width=spec.width,
        num_frames=num_frames or spec.num_frames,
        num_inference_steps=spec.steps,
        guidance_scale=spec.guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if spec.guidance_scale_2 is not None:  # A14B MoE low-noise expert
        kwargs["guidance_scale_2"] = spec.guidance_scale_2
    return pipe(**kwargs).frames[0]


def save_frames(frames: list[Image.Image], frames_dir: str | Path) -> Path:
    """Write frames as 0001.png... — the naming mux_mp4() expects."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames, start=1):
        frame.save(frames_dir / f"{i:04d}.png")
    return frames_dir

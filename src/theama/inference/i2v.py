# SPDX-License-Identifier: Apache-2.0
"""I2V inference — lazy diffusers load (GPU-only at runtime).

Image-conditioned generation; also the M7c KeyframeBridge entry point
(T2I keyframe -> Film I2V). TI2V-5B handles both T2V and I2V with one
transformer, so the dev path uses the same weights for both tasks.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from theama.inference.config import (
    DEFAULT_NEGATIVE_PROMPT,
    MODELS,
    ModelSpec,
)


def load_i2v_pipeline(spec_name: str = "ti2v-5b", offload: bool = True):
    """Load a WanImageToVideoPipeline for an I2V-capable spec."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    spec = MODELS[spec_name]
    if spec.task not in ("i2v", "ti2v"):
        raise ValueError(f"{spec_name} is not I2V-capable (task={spec.task})")
    vae = AutoencoderKLWan.from_pretrained(
        spec.repo_id, subfolder="vae", torch_dtype=torch.float32
    )
    pipe = WanImageToVideoPipeline.from_pretrained(
        spec.repo_id, vae=vae, torch_dtype=torch.bfloat16
    )
    if offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe, spec


def generate_frames_i2v(
    pipe,
    spec: ModelSpec,
    image: str | Path | Image.Image,
    prompt: str,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    seed: int = 42,
    num_frames: int | None = None,
) -> list[Image.Image]:
    """Run one I2V generation from a conditioning frame."""
    import torch

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    kwargs = dict(
        image=image,
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=spec.height,
        width=spec.width,
        num_frames=num_frames or spec.num_frames,
        num_inference_steps=spec.steps,
        guidance_scale=spec.guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if spec.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = spec.guidance_scale_2
    return pipe(**kwargs).frames[0]

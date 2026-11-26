# SPDX-License-Identifier: Apache-2.0
"""Real-weight M2 smoke runners, isolated from the CPU test fixtures.

The CLI in ``scripts/smoke_m2_real_weights.py`` loads official user-side
weights. These functions accept already-loaded adapters so shape/update logic
can also be tested with tiny fakes without importing DAC or Diffusers.
"""

from __future__ import annotations

import math

import torch

from theama.model.audio_codec import decode_stereo, encode_stereo
from theama.model.audio_dit import AudioDiT
from theama.model.wan_adapter import select_wan_bridge_blocks
from theama.train.train_joint_av import JointAVTrainer

__all__ = [
    "cuda_memory_gib",
    "reset_cuda_peak",
    "run_dac_smoke",
    "run_wan_smoke",
]


def reset_cuda_peak(device: str | torch.device) -> None:
    """Reset CUDA peak counters before loading a real checkpoint."""
    dev = torch.device(device)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def cuda_memory_gib(device: str | torch.device) -> dict[str, float]:
    """Return process-local CUDA allocator peaks for the smoke report."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return {}
    torch.cuda.synchronize(dev)
    return {
        "cuda_max_allocated_gib": round(torch.cuda.max_memory_allocated(dev) / 2**30, 4),
        "cuda_max_reserved_gib": round(torch.cuda.max_memory_reserved(dev) / 2**30, 4),
    }


def run_dac_smoke(
    codec,
    *,
    device: str | torch.device = "cuda",
    seconds: float = 1.0,
) -> tuple[dict, torch.Tensor, torch.Tensor]:
    """Run official-codec stereo encode/decode and return report plus audio.

    This validates tensor layouts and finite reconstruction, not perceptual
    quality. The relative L2 error is reported for regression tracking only.
    """
    if seconds <= 0:
        raise ValueError("seconds must be > 0")
    dev = torch.device(device)
    spec = codec.spec
    frames = spec.frames_for_seconds(seconds)
    samples = frames * spec.hop_length
    time = torch.arange(samples, device=dev, dtype=torch.float32) / spec.sample_rate
    left = 0.6 * torch.sin(2 * math.pi * 440.0 * time)
    right = 0.6 * torch.sin(2 * math.pi * 660.0 * time)
    waveform = torch.stack((left, right), dim=0).unsqueeze(0)

    latents = encode_stereo(codec, waveform)
    restored = decode_stereo(codec, latents)
    expected_latents = (1, frames, spec.dit_token_dim)
    if latents.shape != expected_latents:
        raise RuntimeError(f"DAC tokens {tuple(latents.shape)} != {expected_latents}")
    if restored.shape != waveform.shape:
        raise RuntimeError(
            f"DAC reconstruction {tuple(restored.shape)} != input {tuple(waveform.shape)}"
        )
    if not torch.isfinite(latents).all() or not torch.isfinite(restored).all():
        raise RuntimeError("DAC smoke produced non-finite values")

    error = (restored - waveform).float()
    rel_l2 = error.norm() / waveform.float().norm().clamp_min(1e-12)
    report = {
        "status": "ok",
        "smoke": "dac_stereo_encode_decode",
        "codec": spec.name,
        "device": str(dev),
        "sample_rate": spec.sample_rate,
        "input_shape": list(waveform.shape),
        "latent_shape": list(latents.shape),
        "restored_shape": list(restored.shape),
        "latent_frame_rate": spec.latent_frame_rate,
        "reconstruction_relative_l2": float(rel_l2),
        "note": "codec/layout smoke only; no perceptual-quality gate is claimed",
        **cuda_memory_gib(dev),
    }
    return report, waveform, restored


def run_wan_smoke(
    video,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    audio_latent_dim: int = 2048,
    audio_dim: int = 64,
    audio_layers: int = 2,
    audio_frames: int = 8,
    text_length: int = 16,
    latent_frames: int = 1,
    latent_height: int = 2,
    latent_width: int = 2,
    learning_rate: float = 1e-4,
    bridge_location: str = "latent_pre_backbone",
    bridge_lora_rank: int = 0,
    bridge_lora_alpha: float | None = None,
    seed: int = 42,
) -> dict:
    """Run one real-Wan 5-D A/V forward/backward/optimizer step.

    Random latents and text embeddings deliberately isolate the Wan forward
    contract from VAE/text encoders. This is not a prompt-to-video quality test.
    """
    dims = (audio_latent_dim, audio_dim, audio_layers, audio_frames, text_length)
    spatial = (latent_frames, latent_height, latent_width)
    if min(dims + spatial) <= 0:
        raise ValueError("all smoke dimensions must be positive")
    dev = torch.device(device)
    spec = video.spec
    generator = torch.Generator(device=dev).manual_seed(seed)

    def random_tensor(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=generator, device=dev, dtype=dtype)

    bridge_video_dim = video.inner_dim if bridge_location == "wan_blocks" else spec.latent_channels
    effective_lora_alpha = (
        2 * bridge_lora_rank
        if bridge_lora_rank > 0 and bridge_lora_alpha is None
        else bridge_lora_alpha
    )
    audio = AudioDiT(
        dim=audio_dim,
        layers=audio_layers,
        heads=4,
        video_dim=bridge_video_dim,
        bridge_every=1,
        latent_dim=audio_latent_dim,
        # wan_blocks feeds un-normalised Wan hidden states to the bridges
        # (issue #4 red flag); the latent-grid path stays on raw latents.
        norm_video_kv=bridge_location == "wan_blocks",
    ).to(device=dev, dtype=dtype)
    video.to(dev)
    trainer = JointAVTrainer(
        video,
        audio,
        {
            "phase": "bridge",
            "lr": learning_rate,
            "bridge_location": bridge_location,
            "bridge_lora_rank": bridge_lora_rank,
            "bridge_lora_alpha": effective_lora_alpha,
        },
    )

    video_shape = (
        1,
        spec.latent_channels,
        latent_frames,
        latent_height,
        latent_width,
    )
    batch = {
        "audio_latent": random_tensor((1, audio_frames, audio_latent_dim)),
        "video_tokens": random_tensor(video_shape),
        "cond": random_tensor((1, audio_dim)),
        "video_cond": random_tensor((1, text_length, spec.text_dim)),
    }
    batch["audio_noise"] = torch.randn_like(batch["audio_latent"])
    batch["video_noise"] = torch.randn_like(batch["video_tokens"])

    before = [param.detach().clone() for param in audio.bridges.parameters()]
    metrics = trainer.train_step(batch)
    after = list(audio.bridges.parameters())
    deltas = [(new.detach() - old).abs().max().item() for old, new in zip(before, after)]
    updated = sum(delta > 0 for delta in deltas)
    grad_tensors = sum(param.grad is not None for param in after)
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"Wan smoke returned non-finite metrics: {metrics}")
    if not after or updated == 0 or grad_tensors == 0:
        raise RuntimeError("Wan smoke did not update the audio cross-modal bridges")
    if any(param.requires_grad for param in video.parameters()):
        raise RuntimeError("frozen Wan backbone unexpectedly has trainable parameters")

    return {
        "status": "ok",
        "smoke": "wan_5d_av_forward_backward",
        "adapter": spec.name,
        "device": str(dev),
        "dtype": str(dtype).removeprefix("torch."),
        "video_shape": list(video_shape),
        "audio_shape": list(batch["audio_latent"].shape),
        "text_cond_shape": list(batch["video_cond"].shape),
        "bridge_location": bridge_location,
        "bridge_video_dim": bridge_video_dim,
        "wan_bridge_blocks": (
            select_wan_bridge_blocks(len(video.transformer.blocks), len(audio.bridges))
            if bridge_location == "wan_blocks"
            else []
        ),
        "bridge_lora_rank": bridge_lora_rank,
        "bridge_lora_alpha": effective_lora_alpha,
        "trainable_parameters": sum(param.numel() for param in trainer._trainable_params),
        "bridge_grad_tensors": grad_tensors,
        "updated_bridge_tensors": updated,
        "max_bridge_delta": max(deltas, default=0.0),
        "wan_frozen": True,
        "metrics": metrics,
        "note": "random latent/text-conditioning smoke; no video or A/V quality gate is claimed",
        **cuda_memory_gib(dev),
    }

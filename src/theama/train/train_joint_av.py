# SPDX-License-Identifier: Apache-2.0
"""Joint audio-video flow-matching trainer (M2 CPU-testable core).

The video adapter must expose ``predict_velocity(video_tokens, t, cond)``.
``AudioDiT`` first updates the video tokens through its cross-modal bridges;
the frozen video adapter then predicts velocity from those joint tokens. This
keeps both audio and video losses connected to both bridge directions.

The production Wan/DAC adapter is intentionally separate: this module contains
only framework-independent training math and parameter/optimizer handling.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from theama.model.wan_adapter import flatten_video_latents, unflatten_video_tokens


def logit_normal_sample(
    batch: int,
    mu: float = 0.0,
    sigma: float = 1.0,
    device=None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample flow-matching timesteps in the open unit interval."""
    u = torch.randn(batch, device=device, generator=generator) * sigma + mu
    return torch.sigmoid(u)


def _time_view(t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return t.to(device=x.device, dtype=x.dtype).reshape(x.shape[0], *([1] * (x.ndim - 1)))


def flow_match_pair(
    x: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``x_t`` and velocity target for the Wan noise-minus-data convention.

    ``x_t = (1 - t) * x + t * noise`` and ``v_true = noise - x``.
    ``t`` has shape ``(batch,)`` and is broadcast over all remaining dims.
    """
    if x.shape != noise.shape:
        raise ValueError(f"data/noise shape mismatch: {x.shape} vs {noise.shape}")
    if t.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError(f"t must have shape ({x.shape[0]},), got {tuple(t.shape)}")
    tv = _time_view(t, x)
    return (1.0 - tv) * x + tv * noise, noise - x


def flow_matching_loss(
    model_velocity: torch.Tensor,
    x1: torch.Tensor,
    x0_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """MSE against the noise-minus-data velocity at the sampled timesteps."""
    _, target = flow_match_pair(x1, x0_noise, t)
    return F.mse_loss(model_velocity, target)


def sinusoidal_time_embedding(
    t: torch.Tensor,
    dim: int,
    scale: float = 1000.0,
) -> torch.Tensor:
    """Sinusoidal embedding for continuous ``[0, 1]`` flow timesteps."""
    if dim < 2:
        raise ValueError("time embedding dim must be >= 2")
    half = dim // 2
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / half
    )
    angles = t.detach().float()[:, None] * scale * freq[None, :]
    emb = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if emb.shape[-1] != dim:
        emb = F.pad(emb, (0, dim - emb.shape[-1]))
    return emb


class JointAVTrainer:
    """Bridge-warmup then full-audio fine-tuning with a frozen video backbone.

    Config keys: ``phase`` (``bridge`` or ``full``), ``lr``, ``weight_decay``,
    ``audio_loss_weight``, ``video_loss_weight``, ``max_grad_norm``, ``mu``,
    ``sigma`` and ``time_scale``. Changing phase rebuilds the optimizer so no
    stale optimizer state is carried across the warmup/full boundary.
    """

    def __init__(self, video_backbone, audio_dit, config: dict | None = None):
        self.video = video_backbone
        self.audio = audio_dit
        self.config = {
            "phase": "bridge",
            "lr": 1e-4,
            "weight_decay": 0.01,
            "audio_loss_weight": 1.0,
            "video_loss_weight": 1.0,
            "max_grad_norm": 1.0,
            "mu": 0.0,
            "sigma": 1.0,
            "time_scale": 1000.0,
            **(config or {}),
        }
        self.optimizer: torch.optim.Optimizer | None = None
        self._trainable_params: list[torch.nn.Parameter] = []
        self.set_phase(self.config["phase"])

    def set_phase(self, phase: str) -> None:
        """Freeze Wan always; optimize bridges only or the complete audio stream."""
        if phase not in {"bridge", "full"}:
            raise ValueError(f"unknown M2 training phase: {phase}")
        self.phase = phase
        self.config["phase"] = phase
        for param in self.video.parameters():
            param.requires_grad_(False)
        for param in self.audio.parameters():
            param.requires_grad_(False)
        params = self.audio.bridges.parameters() if phase == "bridge" else self.audio.parameters()
        self._trainable_params = list(params)
        if not self._trainable_params:
            raise ValueError(f"{phase} phase has no trainable parameters; check bridge schedule")
        for param in self._trainable_params:
            param.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(
            self._trainable_params,
            lr=float(self.config["lr"]),
            weight_decay=float(self.config["weight_decay"]),
        )

    def _condition(self, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        emb = sinusoidal_time_embedding(t, cond.shape[-1], float(self.config["time_scale"]))
        return cond + emb.to(dtype=cond.dtype)

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Run one real forward/backward/optimizer step and return finite metrics.

        Required batch keys are ``audio_latent`` (B,Ta,Da), ``video_tokens``
        (B,Tv,Dv) or Wan latents (B,C,F,H,W), and pooled audio ``cond`` (B,Da).
        Wan batches additionally supply ``video_cond`` (B,L,4096), which is
        passed directly to the video adapter. Optional ``audio_noise`` and
        ``video_noise`` make the step deterministic for tests/replays.
        """
        if self.optimizer is None:
            raise RuntimeError("optimizer is not initialized")
        missing = {"audio_latent", "video_tokens", "cond"} - batch.keys()
        if missing:
            raise KeyError(f"missing M2 batch keys: {sorted(missing)}")

        audio = batch["audio_latent"]
        video = batch["video_tokens"]
        cond = batch["cond"]
        if audio.shape[0] != video.shape[0] or audio.shape[0] != cond.shape[0]:
            raise ValueError("audio, video and cond batch sizes must match")
        if audio.ndim != 3 or video.ndim not in {3, 5} or cond.ndim != 2:
            raise ValueError(
                "expected audio tokens to be 3-D, video to be 3-D/5-D and cond to be 2-D"
            )
        video_cond = batch.get("video_cond")
        if video_cond is not None and (
            video_cond.ndim != 3 or video_cond.shape[0] != audio.shape[0]
        ):
            raise ValueError("expected video_cond to have shape (B,L,D)")

        device = audio.device
        batch_size = audio.shape[0]
        t_a = logit_normal_sample(
            batch_size,
            mu=float(self.config["mu"]),
            sigma=float(self.config["sigma"]),
            device=device,
        )
        t_v = logit_normal_sample(
            batch_size,
            mu=float(self.config["mu"]),
            sigma=float(self.config["sigma"]),
            device=device,
        )
        noise_a = batch["audio_noise"] if "audio_noise" in batch else torch.randn_like(audio)
        noise_v = batch["video_noise"] if "video_noise" in batch else torch.randn_like(video)
        audio_t, audio_target = flow_match_pair(audio, noise_a, t_a)
        video_t, video_target = flow_match_pair(video, noise_v, t_v)

        video_is_5d = video_t.ndim == 5
        bridge_tokens = flatten_video_latents(video_t) if video_is_5d else video_t
        audio_pred, joint_tokens = self.audio(
            audio_t, bridge_tokens, self._condition(cond, t_a)
        )
        joint_video = (
            unflatten_video_tokens(joint_tokens, tuple(video_t.shape))
            if video_is_5d
            else joint_tokens
        )
        if not hasattr(self.video, "predict_velocity"):
            raise TypeError("video backbone must implement predict_velocity(tokens, t, cond)")
        if video_cond is None:
            video_cond = self._condition(cond, t_v)
        video_pred = self.video.predict_velocity(joint_video, t_v, video_cond)
        audio_loss = F.mse_loss(audio_pred, audio_target)
        video_loss = F.mse_loss(video_pred, video_target)
        loss = (
            float(self.config["audio_loss_weight"]) * audio_loss
            + float(self.config["video_loss_weight"]) * video_loss
        )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._trainable_params, float(self.config["max_grad_norm"])
        )
        self.optimizer.step()
        return {
            "loss": float(loss.detach()),
            "audio_loss": float(audio_loss.detach()),
            "video_loss": float(video_loss.detach()),
            "grad_norm": float(grad_norm.detach()),
            "t_audio_mean": float(t_a.mean()),
            "t_video_mean": float(t_v.mean()),
        }

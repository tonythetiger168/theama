# SPDX-License-Identifier: Apache-2.0
"""Joint A/V training runner: joined latent caches -> JointAVTrainer loop.

``scripts/train_joint_av.py`` is a thin CLI shell over this module; everything
here is CPU-testable. The runner owns what the trainer deliberately does not:

  * config plumbing from ``configs/m2_joint_av.yaml`` (model dims, trainer
    kwargs, text-embedding width);
  * the text->audio conditioning pooler (``cond_pooler``), attached to the
    :class:`~theama.model.audio_dit.AudioDiT` *before* trainer construction so
    phase handling covers it: frozen during bridge warmup, trained in the full
    phase, and carried inside the audio ``state_dict`` so checkpoints need no
    extra payload;
  * the step loop: infinite cycling over the dataloader, device/dtype moves,
    on-device ``cond`` computation (pooled conditioning keeps gradients on the
    training device instead of inside DataLoader collation), scheduled
    bridge->full phase switch, JSONL metrics, periodic checkpoints;
  * fail-closed guards: non-finite metrics abort the run with an emergency
    checkpoint — a NaN loss must never silently consume GPU hours;
  * checkpoints that resume exactly: phase is restored (re-injecting or
    merging bridge LoRA as needed) *before* state dicts load, and optimizer
    state is moved onto the parameter device.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch import nn

from theama.data.joint_latent_loader import masked_mean_pool
from theama.model.audio_dit import AudioDiT
from theama.train.train_joint_av import JointAVTrainer

__all__ = [
    "JointAVRunnerError",
    "RunnerConfig",
    "StubVideoBackbone",
    "audio_dit_kwargs",
    "trainer_config",
    "text_dim_from_config",
    "build_cond_pooler",
    "save_checkpoint",
    "load_checkpoint",
    "latest_checkpoint",
    "JointAVRunner",
]

CHECKPOINT_FORMAT = 1


class JointAVRunnerError(RuntimeError):
    """Non-finite metrics, missing pooler, or a corrupt checkpoint."""


# ---- config plumbing ---------------------------------------------------------


def audio_dit_kwargs(audio_cfg: dict) -> dict:
    """Map the ``model.audio`` YAML section onto ``AudioDiT`` kwargs."""
    return {
        "dim": int(audio_cfg["dim"]),
        "layers": int(audio_cfg["layers"]),
        "heads": int(audio_cfg["heads"]),
        "video_dim": int(audio_cfg["video_dim"]),
        "bridge_every": int(audio_cfg["bridge_every"]),
        "latent_dim": int(audio_cfg["latent_dim"]),
    }


def trainer_config(cfg: dict) -> dict:
    """Merge the ``training`` YAML section with the model's bridge location."""
    bridge_location = cfg.get("model", {}).get("video", {}).get("bridge_location")
    config = dict(cfg["training"])
    if bridge_location is not None:
        config["bridge_location"] = bridge_location
    return config


def text_dim_from_config(cfg: dict) -> int:
    """umT5 embedding width the text cache was encoded with (e.g. 4096)."""
    return int(cfg["model"]["video"]["text_dim"])


# ---- conditioning pooler -----------------------------------------------------


def build_cond_pooler(audio_dit: AudioDiT, text_dim: int) -> nn.Linear:
    """Attach the text->audio-dim conditioning projection to the AudioDiT.

    Must be called *before* ``JointAVTrainer`` construction so phase handling
    freezes it during bridge warmup and trains it (via ``audio.parameters()``)
    in the full phase. The projection rides the audio ``state_dict``, so
    checkpoints cover it with no extra payload.
    """
    if hasattr(audio_dit, "cond_pooler"):
        raise JointAVRunnerError("audio DiT already has a cond_pooler attached")
    pooler = nn.Linear(int(text_dim), audio_dit.in_proj.out_features)
    audio_dit.add_module("cond_pooler", pooler)
    return pooler


# ---- stub video backbone (smoke/tests; real runs use WanVideoAdapter) --------


class StubVideoBackbone(nn.Module):
    """Shape-agnostic stand-in for the trainer's video-backbone contract.

    ``predict_velocity`` works on both token ``(B,Tv,Dv)`` and Wan latent
    ``(B,C,F,H,W)`` inputs so smoke runs can exercise the real 5-D
    flatten/unflatten path without Wan weights. Frozen by the trainer like
    any video backbone.
    """

    def __init__(self, scale: float = 0.9):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def predict_velocity(
        self, tokens: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        del cond
        view = t.to(device=tokens.device, dtype=tokens.dtype).reshape(
            tokens.shape[0], *([1] * (tokens.ndim - 1))
        )
        return tokens * self.scale.to(tokens.dtype) + view


# ---- checkpoints -------------------------------------------------------------


def save_checkpoint(
    trainer: JointAVTrainer,
    out_dir: str | Path,
    step: int,
    *,
    tag: str | None = None,
) -> Path:
    """Persist trainer state for exact resume; also refreshes ``ckpt_latest.pt``."""
    if trainer.optimizer is None:
        raise JointAVRunnerError("cannot checkpoint before the optimizer exists")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = f"ckpt_{tag}_step_{step:06d}.pt" if tag else f"ckpt_step_{step:06d}.pt"
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "phase": trainer.phase,
        "trainer_config": dict(trainer.config),
        "audio_state": trainer.audio.state_dict(),
        "optimizer_state": trainer.optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path = out / name
    torch.save(payload, path)
    latest = out / "ckpt_latest.pt"
    if path != latest:
        torch.save(payload, latest)
    return path


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(trainer: JointAVTrainer, path: str | Path) -> int:
    """Restore phase, weights, optimizer and RNG; returns the saved step.

    Phase is restored first: a bridge-phase checkpoint re-injects bridge LoRA
    so its ``lora_a``/``lora_b`` keys exist for the strict load, and a
    full-phase checkpoint merges any fresh LoRA back out before loading.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise JointAVRunnerError(
            f"unsupported checkpoint format in {path}: {payload.get('format')}"
        )
    trainer.set_phase(payload["phase"])
    trainer.audio.load_state_dict(payload["audio_state"])
    trainer.optimizer.load_state_dict(payload["optimizer_state"])
    device = next(iter(trainer._trainable_params)).device
    _move_optimizer_state(trainer.optimizer, device)
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    cuda_state = payload.get("cuda_rng_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(cuda_state.cpu())
    return int(payload["step"])


def latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Return the newest checkpoint in ``out_dir`` (``ckpt_latest.pt`` preferred)."""
    out = Path(out_dir)
    latest = out / "ckpt_latest.pt"
    if latest.is_file():
        return latest
    candidates = sorted(out.glob("ckpt_step_*.pt"))
    return candidates[-1] if candidates else None


# ---- runner ------------------------------------------------------------------


@dataclass
class RunnerConfig:
    """Loop settings for :class:`JointAVRunner`."""

    out_dir: str
    max_steps: int = 1000
    bridge_steps: int | None = None  # None = never leave the config phase
    log_every: int = 10
    save_every: int = 100
    device: str = "cpu"
    dtype: str = "fp32"  # "fp32" | "bf16"
    resume: str | None = None  # "auto" (latest in out_dir) or a checkpoint path


class JointAVRunner:
    """Step loop over joined-latent batches with phase schedule + checkpoints.

    ``batches`` is any iterable of batch dicts — the production DataLoader
    from :func:`theama.data.joint_latent_loader.build_joint_latent_dataloader`
    (pass ``cond_pooler=None`` there; pooling happens on-device here) or a
    plain list of synthetic batches for smoke runs. ``cond`` is computed per
    step as ``audio.cond_pooler(masked_mean_pool(video_cond, text_mask))`` so
    the pooler trains in the full phase and gradients stay on-device.
    """

    def __init__(
        self,
        trainer: JointAVTrainer,
        batches: Iterable[dict],
        config: RunnerConfig,
    ):
        pooler = getattr(trainer.audio, "cond_pooler", None)
        if pooler is None:
            raise JointAVRunnerError(
                "build_cond_pooler must attach audio.cond_pooler before the runner starts"
            )
        if config.dtype not in {"fp32", "bf16"}:
            raise JointAVRunnerError(f"unsupported dtype: {config.dtype}")
        self.trainer = trainer
        self.pooler = pooler
        self.config = config
        self.device = torch.device(config.device)
        self.dtype = torch.bfloat16 if config.dtype == "bf16" else torch.float32
        self.trainer.audio.to(self.device, dtype=self.dtype)
        self.trainer.video.to(self.device)
        self._batches = batches
        self._epoch = 0
        out = Path(config.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._metrics_path = out / "metrics.jsonl"
        self._t0 = time.monotonic()

    # -- data ------------------------------------------------------------------

    def _cycle(self) -> Iterator[dict]:
        while True:
            for batch in self._batches:
                yield batch
            self._epoch += 1

    def _to_device(self, batch: dict) -> dict:
        moved = {}
        for key, value in batch.items():
            moved[key] = (
                value.to(self.device, dtype=self.dtype)
                if torch.is_tensor(value) and value.is_floating_point()
                else value.to(self.device)
                if torch.is_tensor(value)
                else value
            )
        return moved

    def _add_cond(self, batch: dict) -> None:
        if "video_cond" not in batch or "text_mask" not in batch:
            raise JointAVRunnerError(
                "runner batches need video_cond + text_mask to derive cond "
                "(build the dataloader with cond_pooler=None)"
            )
        pooled = masked_mean_pool(batch["video_cond"], batch["text_mask"])
        batch["cond"] = self.pooler(pooled.to(next(self.pooler.parameters()).dtype))

    # -- logging/checkpoints ----------------------------------------------------

    def _log(self, record: dict) -> None:
        with self._metrics_path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")

    def _save(self, step: int, *, tag: str | None = None) -> str:
        return str(save_checkpoint(self.trainer, self.config.out_dir, step, tag=tag))

    # -- main loop ----------------------------------------------------------------

    def train(self) -> dict:
        """Run ``max_steps`` optimizer steps (plus resumed steps) and summarize."""
        cfg = self.config
        start_step = 0
        if cfg.resume:
            path = (
                latest_checkpoint(cfg.out_dir) if cfg.resume == "auto" else Path(cfg.resume)
            )
            if path is None:
                raise JointAVRunnerError(
                    f"resume requested but no checkpoint found in {cfg.out_dir}"
                )
            start_step = load_checkpoint(self.trainer, path)
            self._log({"event": "resume", "step": start_step, "path": str(path)})
        if start_step >= cfg.max_steps:
            raise JointAVRunnerError(
                f"checkpoint step {start_step} already >= max_steps {cfg.max_steps}"
            )

        checkpoints: list[str] = []
        data = self._cycle()
        last: dict[str, float] | None = None
        for step in range(start_step, cfg.max_steps):
            if (
                cfg.bridge_steps is not None
                and step == cfg.bridge_steps
                and self.trainer.phase == "bridge"
            ):
                self.trainer.set_phase("full")
                self._log({"event": "phase_switch", "step": step, "phase": "full"})
            batch = self._to_device(next(data))
            self._add_cond(batch)
            metrics = self.trainer.train_step(batch)
            last = metrics
            if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
                path = self._save(step, tag="nonfinite")
                raise JointAVRunnerError(
                    f"non-finite metrics at step {step}: {metrics} — "
                    f"emergency checkpoint at {path}"
                )
            if (step + 1) % cfg.log_every == 0 or step == start_step:
                self._log(
                    {
                        "step": step,
                        "epoch": self._epoch,
                        "phase": self.trainer.phase,
                        "elapsed_sec": round(time.monotonic() - self._t0, 3),
                        **metrics,
                    }
                )
            if (step + 1) % cfg.save_every == 0 or step + 1 == cfg.max_steps:
                checkpoints.append(self._save(step + 1))
        return {
            "steps_completed": cfg.max_steps,
            "resumed_from": start_step,
            "final_phase": self.trainer.phase,
            "final_metrics": last,
            "checkpoints": checkpoints,
            "metrics_log": str(self._metrics_path),
        }

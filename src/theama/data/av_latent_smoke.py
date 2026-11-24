# SPDX-License-Identifier: Apache-2.0
"""Straight-through decode -> VAE/DAC encode smoke for the M2 pipeline.

The M2 data path already has both ends: ``AVDataset`` decodes manifest-gated
media into latents-ready tensors, and ``build_av_dataloader`` feeds them with
training-grade skip handling. What was missing is the wire to the encoders —
this module runs decoded batches through the video VAE and the audio codec
*once*, saves every latent via :func:`~theama.data.latent_cache.save_latent_atomic`,
and proves the shapes the joint trainer consumes actually come out of real
media end to end.

This is a contract smoke, not the P1b production cache:

- no resume logic (``scripts/encode_pairs.py`` owns that pattern for DPO
  pairs; the AV cache lands with the P1b data engine);
- no quality gate — but failures are loud: a non-finite or wrongly shaped
  encoder output raises instead of being cached, and with
  ``skip_broken=False`` a broken clip aborts the run;
- per-sample .pt files keep the batch dim (video ``(1,C,F,H',W')``, audio
  ``(1,T_lat,2*latent_dim)``), matching the encode_pairs convention so
  :func:`~theama.data.latent_cache.is_valid_latent` stays reusable.

The module is CPU-safe: both encoders are injected callables, so tests drive
it with deterministic stand-ins while ``scripts/smoke_decode_to_latents.py``
wires the real Wan VAE + DAC on GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from theama.data.dataset import AVDataset
from theama.data.latent_cache import save_latent_atomic
from theama.data.loader import build_av_dataloader

MANIFEST_NAME = "av_latents.jsonl"
SUMMARY_NAME = "av_latents.summary.json"

__all__ = [
    "MANIFEST_NAME",
    "SUMMARY_NAME",
    "AVLatentSmokeReport",
    "latent_paths",
    "pad_to_hop",
    "run_decode_encode_smoke",
]


def pad_to_hop(waveform: torch.Tensor, hop_length: int) -> torch.Tensor:
    """Zero-pad the tail of ``(..., N)`` up to a multiple of ``hop_length``.

    DAC's encoder consumes whole hop windows; ``decode_audio`` emits exactly
    ``sr * seconds`` samples, which is generally not hop-aligned (e.g. 5 s at
    44.1 kHz = 220500 = 430.66 hops). Padding the tail with silence adds one
    partial window instead of crashing the codec contract.
    """
    if hop_length < 1:
        raise ValueError(f"hop_length must be >= 1, got {hop_length}")
    samples = waveform.shape[-1]
    remainder = samples % hop_length
    if remainder == 0:
        return waveform
    return F.pad(waveform, (0, hop_length - remainder))


def latent_paths(out_dir: str | Path, index: int) -> tuple[Path, Path]:
    """Canonical (video, audio) latent paths for one decoded sample."""
    out = Path(out_dir)
    return out / f"sample{index:04d}_video.pt", out / f"sample{index:04d}_audio.pt"


@dataclass
class AVLatentSmokeReport:
    """Accumulated result of a decode->encode smoke run."""

    records: list[dict] = field(default_factory=list)
    batches: int = 0
    skipped: int = 0

    def summary(self) -> dict:
        """Compact, JSON-serializable roll-up for the sidecar file."""
        video_shapes = sorted({tuple(r["video_shape"]) for r in self.records})
        audio_shapes = sorted({tuple(r["audio_shape"]) for r in self.records})
        return {
            "encoded": len(self.records),
            "skipped": self.skipped,
            "batches": self.batches,
            "video_shapes": [list(s) for s in video_shapes],
            "audio_shapes": [list(s) for s in audio_shapes],
        }


def _check_latent(name: str, latent: torch.Tensor, ndim: int, batch: int) -> None:
    """Fail loudly on a contract violation before anything hits the cache."""
    if not torch.is_tensor(latent) or latent.ndim != ndim or latent.shape[0] != batch:
        shape = tuple(latent.shape) if torch.is_tensor(latent) else type(latent).__name__
        raise ValueError(
            f"{name} encoder must return ({batch},...) with {ndim} dims, got {shape}"
        )
    if not bool(torch.isfinite(latent).all()):
        raise ValueError(f"{name} encoder returned non-finite values — refusing to cache")


def run_decode_encode_smoke(
    dataset: AVDataset,
    video_encode_fn,
    audio_encode_fn,
    out_dir: str | Path,
    *,
    limit: int | None = None,
    batch_size: int = 1,
    skip_broken: bool = True,
    progress=print,
) -> dict:
    """Feed decoded batches through both encoders and cache every latent.

    ``video_encode_fn`` maps a decoded batch ``(B,3,T,H,W)`` in ``[-1,1]`` to
    Wan-normalized latents ``(B,C,F,H',W')``; ``audio_encode_fn`` maps
    ``(B,2,N)`` to DiT-ready mid+side tokens ``(B,T_lat,2*latent_dim)``
    (callers pad with :func:`pad_to_hop` first when the codec requires
    hop-aligned input). Both are called exactly once per batch.

    Iteration is deterministic (``shuffle=False``, ``drop_last=False``) so
    sample indices are stable across re-runs. With ``skip_broken=True``
    (default) the loader's :class:`SkipDecodeErrorDataset` swaps a broken
    clip for the next healthy one and the skip is counted in the summary; a
    replacement sample may therefore appear twice — acceptable for a smoke,
    and visible in the manifest rows. Pass ``skip_broken=False`` to make any
    decode failure abort the run instead.

    Rows are appended + flushed per sample (crash-safe), and a summary
    sidecar is written on completion. Returns the summary dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    loader = build_av_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        skip_decode_errors=skip_broken,
        drop_last=False,
    )

    report = AVLatentSmokeReport()
    index = 0
    with (out / MANIFEST_NAME).open("w") as fh:
        for batch in loader:
            if limit is not None and index >= limit:
                break
            report.batches += 1
            video_latents = video_encode_fn(batch["video"])
            audio_latents = audio_encode_fn(batch["audio"])
            bs = batch["video"].shape[0]
            _check_latent("video", video_latents, 5, bs)
            _check_latent("audio", audio_latents, 3, bs)
            for i in range(bs):
                video_path, audio_path = latent_paths(out, index)
                save_latent_atomic(video_latents[i : i + 1].cpu(), video_path)
                save_latent_atomic(audio_latents[i : i + 1].cpu(), audio_path)
                row = {
                    "index": index,
                    "caption": batch["caption"][i],
                    "source": batch["source"][i],
                    "video_latent": str(video_path),
                    "audio_latent": str(audio_path),
                    "video_shape": list(video_latents[i : i + 1].shape),
                    "audio_shape": list(audio_latents[i : i + 1].shape),
                }
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                report.records.append(row)
                progress(
                    f"[{index}] {row['caption'][:40]} "
                    f"video{row['video_shape']} audio{row['audio_shape']}"
                )
                index += 1

    wrapped = getattr(loader.dataset, "skipped", None)
    report.skipped = wrapped if isinstance(wrapped, int) else 0
    summary = report.summary()
    (out / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    return summary

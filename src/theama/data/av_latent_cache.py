# SPDX-License-Identifier: Apache-2.0
"""Plan/resume AV latent cache for the P1b data engine (pure logic).

This upgrades the round-8 decode->encode contract smoke
(:mod:`theama.data.av_latent_smoke`) into the production cache: same output
directory layout (``sample{index:04d}_video.pt`` / ``sample{index:04d}_audio.pt``,
``av_latents.jsonl``, ``av_latents.summary.json``), plus the resume machinery
``scripts/encode_pairs.py`` already proved for DPO pairs
(:mod:`theama.data.latent_cache`).

Manifest schema (av_latents.jsonl), one row per sample index:

    success: {index, caption, source, video, audio, model,
              video_latent, audio_latent, video_shape, audio_shape, reused}
    failure: {index, caption, source, video, audio, model, error}

Resume rules (default on), mirroring the pair cache:

  * rows carrying an "error" are dropped so a re-run retries them;
  * rows whose model differs from the requested one are dropped (latent
    normalization is encoder-config-specific, cross-model reuse invalid);
  * drift guard: a manifest row is only reused when the sample at that
    index still has the same (video, audio) path pair — caption edits do
    NOT invalidate latents (text encoding lives outside the dataset);
  * both .pt files must validate: video 5D finite, audio 3D finite
    (:func:`~theama.data.latent_cache.is_valid_latent` with ``ndim=5`` /
    ``ndim=3``) — a truncated write from a killed run is re-encoded;
  * legacy fallback: with NO manifest row for an index, a sample is reused
    when both ``sample{index:04d}_{video,audio}.pt`` exist and validate.
    This is what lets a smoke-run directory seed the production cache
    unchanged (assumes manifest ordering is unchanged — pass --no-resume
    after reordering). The fallback NEVER applies to an index that has a
    manifest row: a row rejected by the drift guard means the media at
    that index changed and the indexed .pt files are stale — serving them
    would silently cache the wrong clip, so the sample is re-encoded;
  * out_dir/meta.json records the encoding model; a mismatch invalidates
    the whole cache (defence against silently mixing normalizations).

Module import is CPU-safe: torch is only imported lazily inside the helpers
that touch .pt files. Samples are plain dicts with keys
``video``/``audio``/``caption``/``source`` (media paths as strings), so the
planner never needs the dataset or ffmpeg.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from theama.data.latent_cache import (
    META_NAME,  # noqa: F401  (re-exported for callers)
    is_valid_latent,
    load_reusable_rows,
    read_cache_meta,
    summarize_encoding,
    write_cache_meta,
)

MANIFEST_NAME = "av_latents.jsonl"
SUMMARY_NAME = "av_latents.summary.json"

REQUIRED_SAMPLE_KEYS = ("video", "audio", "caption", "source")

__all__ = [
    "MANIFEST_NAME",
    "SUMMARY_NAME",
    "REQUIRED_SAMPLE_KEYS",
    "AVEncodePlan",
    "av_latent_paths",
    "is_valid_video_latent",
    "is_valid_audio_latent",
    "latent_shape",
    "plan_av_encoding",
    "run_av_encoding",
]


# ---- .pt file helpers ---------------------------------------------------------


def av_latent_paths(out_dir: str | Path, index: int) -> tuple[Path, Path]:
    """Canonical (video, audio) latent paths for one sample index.

    Same naming as theama.data.av_latent_smoke so a smoke directory is a
    valid seed for this cache via the legacy fallback.
    """
    out = Path(out_dir)
    return out / f"sample{index:04d}_video.pt", out / f"sample{index:04d}_audio.pt"


def latent_shape(path: str | Path, ndim: int) -> list[int] | None:
    """``list(shape)`` when `path` holds a finite `ndim` tensor, else None."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import torch

        t = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return None
    if not torch.is_tensor(t) or t.ndim != ndim or not bool(torch.isfinite(t).all()):
        return None
    return list(t.shape)


def is_valid_video_latent(path: str | Path) -> bool:
    """True for a loadable, finite, 5D (1,C,F,H',W') video latent."""
    return is_valid_latent(path, ndim=5)


def is_valid_audio_latent(path: str | Path) -> bool:
    """True for a loadable, finite, 3D (1,T_lat,C) audio token latent."""
    return is_valid_latent(path, ndim=3)


# ---- planning / execution ------------------------------------------------------


@dataclass
class AVEncodePlan:
    """Which sample indices can be served from cache and which need encoders."""

    samples: list[dict]
    out_dir: Path
    model: str
    reusable: dict[int, dict] = field(default_factory=dict)
    todo: list[int] = field(default_factory=list)
    cache_invalidated: bool = False  # meta.json model mismatch

    @property
    def manifest_path(self) -> Path:
        return self.out_dir / MANIFEST_NAME

    @property
    def summary_path(self) -> Path:
        return self.out_dir / SUMMARY_NAME


def _drifted(sample: dict, row: dict) -> bool:
    """True when the manifest row no longer matches the sample's media."""
    return (
        str(sample["video"]) != row.get("video")
        or str(sample["audio"]) != row.get("audio")
    )


def _reusable_row(
    sample: dict, index: int, out_dir: Path, model: str
) -> dict | None:
    """Build a reused row when both latent files validate, else None."""
    video_path, audio_path = av_latent_paths(out_dir, index)
    video_shape = latent_shape(video_path, 5)
    audio_shape = latent_shape(audio_path, 3)
    if video_shape is None or audio_shape is None:
        return None
    return {
        "index": index,
        "caption": sample["caption"],
        "source": sample["source"],
        "video": str(sample["video"]),
        "audio": str(sample["audio"]),
        "model": model,
        "video_latent": str(video_path),
        "audio_latent": str(audio_path),
        "video_shape": video_shape,
        "audio_shape": audio_shape,
        "reused": True,
    }


def plan_av_encoding(
    samples: list[dict],
    out_dir: str | Path,
    model: str,
    resume: bool = True,
) -> AVEncodePlan:
    """Decide reuse vs re-encode for every sample index (validates .pt files).

    `samples` are plain dicts with the REQUIRED_SAMPLE_KEYS; `video` /
    `audio` are the drift-guard keys, so they must be stable across runs
    (absolute-resolved paths from theama.data.dataset.load_manifest qualify).
    """
    plan = AVEncodePlan(samples=list(samples), out_dir=Path(out_dir), model=model)
    candidates: dict[int, dict] = {}
    if resume:
        meta = read_cache_meta(plan.out_dir)
        if meta and meta.get("model") != model:
            plan.cache_invalidated = True
        else:
            candidates = load_reusable_rows(plan.manifest_path, model=model)

    fallback_blocked: set[int] = set()  # indices the manifest has spoken for
    for index, row in candidates.items():
        if index >= len(plan.samples):
            continue
        fallback_blocked.add(index)
        sample = plan.samples[index]
        if _drifted(sample, row):
            continue  # media changed under this index — indexed .pt is stale
        reused = _reusable_row(sample, index, plan.out_dir, plan.model)
        if reused is not None:
            plan.reusable[index] = reused

    for index, sample in enumerate(plan.samples):
        if index in plan.reusable:
            continue
        if resume and not plan.cache_invalidated and index not in fallback_blocked:
            # Legacy fallback: .pt files from a pre-manifest (smoke) run.
            reused = _reusable_row(sample, index, plan.out_dir, plan.model)
            if reused is not None:
                plan.reusable[index] = reused
                continue
        plan.todo.append(index)
    return plan


def run_av_encoding(
    plan: AVEncodePlan,
    encode_fn,
    strict: bool = False,
    progress=print,
) -> dict:
    """Execute an AVEncodePlan crash-safely and write the summary sidecar.

    ``encode_fn(sample, video_dest, audio_dest)`` must decode + encode one
    sample and atomically save both latent tensors (video 5D, audio 3D).
    It is only called for ``plan.todo`` indices, so callers can skip loading
    the VAE/DAC entirely when everything is cached. After each encode the
    files are re-validated — an encoder that leaves missing/corrupt latents
    is recorded as a failure row, never manifested as success. With
    ``strict=False`` a failing sample is retried on the next run; ``strict``
    re-raises on the first failure. Returns the summary dict.
    """
    rows: dict[int, dict] = dict(plan.reusable)
    with plan.manifest_path.open("w") as fh:
        for index in sorted(plan.reusable):
            fh.write(json.dumps(plan.reusable[index]) + "\n")
        fh.flush()

        for index in plan.todo:
            sample = plan.samples[index]
            video_path, audio_path = av_latent_paths(plan.out_dir, index)
            base = {
                "index": index,
                "caption": sample["caption"],
                "source": sample["source"],
                "video": str(sample["video"]),
                "audio": str(sample["audio"]),
                "model": plan.model,
            }
            try:
                if encode_fn is None:  # pragma: no cover - defensive
                    raise RuntimeError("no encoder available for pending samples")
                encode_fn(sample, video_path, audio_path)
                video_shape = latent_shape(video_path, 5)
                audio_shape = latent_shape(audio_path, 3)
                if video_shape is None or audio_shape is None:
                    raise RuntimeError(
                        "encoder left missing/corrupt latents — refusing to cache"
                    )
                row = {
                    **base,
                    "video_latent": str(video_path),
                    "audio_latent": str(audio_path),
                    "video_shape": video_shape,
                    "audio_shape": audio_shape,
                    "reused": False,
                }
                progress(
                    f"[{index + 1}/{len(plan.samples)}] {sample['caption'][:40]} "
                    f"video{video_shape} audio{audio_shape}"
                )
            except Exception as exc:
                if strict:
                    raise
                row = {**base, "error": f"{type(exc).__name__}: {exc}"}
                progress(
                    f"[{index + 1}/{len(plan.samples)}] {sample['caption'][:40]}"
                    f" FAILED ({row['error']})"
                )
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows[index] = row

    write_cache_meta(plan.out_dir, plan.model)
    summary = {
        "model": plan.model,
        **summarize_encoding([rows[i] for i in sorted(rows)]),
    }
    plan.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary

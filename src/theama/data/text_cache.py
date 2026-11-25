# SPDX-License-Identifier: Apache-2.0
"""Content-addressed text-embedding cache for the P1b data engine.

Wan 2.2 conditions on umT5 text embeddings; like the VAE/DAC latents they
belong outside the joint-training hot loop. This module caches one
``.pt`` per *caption*, not per sample index — the filename is derived
from the caption text itself::

    text_{sha256(caption utf-8)[:16]}.pt

so caption drift can never silently reuse a stale embedding: editing a
caption yields a different path, and the old file is simply orphaned
(garbage, never served). This is the same guarantee the pair/AV caches
get from explicit drift guards
(:mod:`theama.data.latent_cache` / :mod:`theama.data.av_latent_cache`),
but here it is structural — there is no index-keyed fallback to bypass.

Manifest schema (text_latents.jsonl), one row per sample index:

    success: {index, caption, model, text_latent, shape, reused, seq_len?}
    failure: {index, caption, model, error}

Resume rules (default on):

  * a sample is reusable iff ``text_latent_path(out_dir, caption)``
    holds a finite 2D (L, D) embedding tensor
    (:func:`~theama.data.latent_cache.is_valid_latent` with ``ndim=2``) —
    a truncated write from a killed run is re-encoded;
  * error rows are retried automatically: a failed encode leaves no valid
    .pt, so the sample lands back in ``todo``;
  * the manifest is read to recover ``seq_len`` for reused rows (the .pt
    stores the padded tensor only); a row whose caption no longer matches
    the sample at its index is ignored;
  * out_dir/meta.json records the encoding model (text encoder id); a
    mismatch invalidates the whole cache — embeddings from different
    encoders must never mix;
  * duplicate captions within one run are encoded once: later samples
    with an identical caption reuse the freshly written file (their rows
    are marked ``reused`` — no encoder call was made for them).

Module import is CPU-safe: torch is only imported lazily inside the
helpers that touch .pt files. Samples are plain dicts that must carry a
non-empty string ``caption``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from theama.data.av_latent_cache import latent_shape
from theama.data.latent_cache import (
    META_NAME,  # noqa: F401  (re-exported for callers)
    is_valid_latent,
    load_reusable_rows,
    read_cache_meta,
    summarize_encoding,
    write_cache_meta,
)

MANIFEST_NAME = "text_latents.jsonl"
SUMMARY_NAME = "text_latents.summary.json"

__all__ = [
    "MANIFEST_NAME",
    "SUMMARY_NAME",
    "TextEncodePlan",
    "text_latent_path",
    "is_valid_text_latent",
    "plan_text_encoding",
    "run_text_encoding",
]


# ---- .pt file helpers ---------------------------------------------------------


def text_latent_path(out_dir: str | Path, caption: str) -> Path:
    """Canonical embedding path for a caption (content-addressed).

    Raises ValueError for a missing/empty/non-string caption — the AV
    manifest schema already rejects those, and silently hashing them
    would collide distinct samples onto one file.
    """
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must be a non-empty string")
    digest = hashlib.sha256(caption.encode("utf-8")).hexdigest()[:16]
    return Path(out_dir) / f"text_{digest}.pt"


def is_valid_text_latent(path: str | Path) -> bool:
    """True for a loadable, finite, 2D (L, D) text embedding."""
    return is_valid_latent(path, ndim=2)


# ---- planning / execution ------------------------------------------------------


@dataclass
class TextEncodePlan:
    """Which sample indices can be served from cache and which need encoder."""

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


def plan_text_encoding(
    samples: list[dict],
    out_dir: str | Path,
    model: str,
    resume: bool = True,
) -> TextEncodePlan:
    """Decide reuse vs re-encode for every sample index (validates .pt files).

    `samples` are plain dicts with a non-empty ``caption`` (absolute media
    paths are irrelevant here — text embeddings depend only on the caption
    and the encoder). Caption edits invalidate automatically via the
    content-addressed filename; no index-keyed drift guard is needed.
    """
    plan = TextEncodePlan(samples=list(samples), out_dir=Path(out_dir), model=model)
    rows_by_index: dict[int, dict] = {}
    if resume:
        meta = read_cache_meta(plan.out_dir)
        if meta and meta.get("model") != model:
            plan.cache_invalidated = True
        else:
            rows_by_index = load_reusable_rows(plan.manifest_path, model=model)

    for index, sample in enumerate(plan.samples):
        dest = text_latent_path(plan.out_dir, sample["caption"])
        if resume and not plan.cache_invalidated and is_valid_text_latent(dest):
            row = {
                "index": index,
                "caption": sample["caption"],
                "model": model,
                "text_latent": str(dest),
                "shape": latent_shape(dest, 2),
                "reused": True,
            }
            # Recover seq_len from a previous manifest row for this index
            # when it still describes the same caption.
            old = rows_by_index.get(index)
            if old and old.get("caption") == sample["caption"] and "seq_len" in old:
                row["seq_len"] = old["seq_len"]
            plan.reusable[index] = row
        else:
            plan.todo.append(index)
    return plan


def run_text_encoding(
    plan: TextEncodePlan,
    encode_fn,
    strict: bool = False,
    progress=print,
) -> dict:
    """Execute a TextEncodePlan crash-safely and write the summary sidecar.

    ``encode_fn(sample, dest_path)`` must encode one caption and atomically
    save the 2D embedding tensor (e.g. via
    :func:`~theama.data.latent_cache.save_latent_atomic`). It may return the
    unpadded sequence length (int), recorded as ``seq_len`` so training can
    rebuild the attention mask from the padded (L, D) tensor. It is only
    called for captions with no valid cache entry, so callers can skip
    loading the text encoder entirely when everything is cached. After each
    encode the file is re-validated — an encoder that leaves a
    missing/corrupt/wrong-ndim embedding is recorded as a failure row,
    never manifested as success. With ``strict=False`` a failing sample is
    retried on the next run; ``strict`` re-raises on the first failure.
    Returns the summary dict.
    """
    rows: dict[int, dict] = dict(plan.reusable)
    encoded_this_run: dict[Path, dict] = {}  # caption dedup within the run
    with plan.manifest_path.open("w") as fh:
        for index in sorted(plan.reusable):
            fh.write(json.dumps(plan.reusable[index]) + "\n")
        fh.flush()

        for index in plan.todo:
            sample = plan.samples[index]
            dest = text_latent_path(plan.out_dir, sample["caption"])
            base = {
                "index": index,
                "caption": sample["caption"],
                "model": plan.model,
            }
            try:
                if dest in encoded_this_run:
                    # Same caption already encoded this run — share the file.
                    row = {**base, **{k: v for k, v in encoded_this_run[dest].items()
                                      if k not in ("index", "caption")},
                           "reused": True}
                    progress(
                        f"[{index + 1}/{len(plan.samples)}] {sample['caption'][:40]}"
                        " (shared with earlier sample)"
                    )
                else:
                    if encode_fn is None:  # pragma: no cover - defensive
                        raise RuntimeError("no encoder available for pending samples")
                    seq_len = encode_fn(sample, dest)
                    shape = latent_shape(dest, 2)
                    if shape is None:
                        raise RuntimeError(
                            "encoder left missing/corrupt embedding — "
                            "refusing to cache"
                        )
                    row = {
                        **base,
                        "text_latent": str(dest),
                        "shape": shape,
                        "reused": False,
                    }
                    if seq_len is not None:
                        if not isinstance(seq_len, int) or isinstance(seq_len, bool):
                            raise TypeError(
                                f"encode_fn seq_len must be int or None, got {type(seq_len).__name__}"
                            )
                        row["seq_len"] = seq_len
                    encoded_this_run[dest] = row
                    progress(
                        f"[{index + 1}/{len(plan.samples)}] "
                        f"{sample['caption'][:40]} text{shape}"
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

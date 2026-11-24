# SPDX-License-Identifier: Apache-2.0
"""Latent-cache planning / resume logic for scripts/encode_pairs.py (pure logic).

Module import is CPU-safe: torch is only imported lazily inside the helpers
that touch .pt files, so the sandbox test-suite can exercise everything.

Manifest schema (pairs_latents.jsonl), one row per pair index:

    success: {index, prompt_id, prompt, model, win_latent, lose_latent, reused}
    failure: {index, prompt_id, prompt, model, error}

Resume rules (default on):
  * rows carrying an "error" are dropped so a re-run retries them;
  * rows whose model differs from the requested one are dropped (latent
    normalization is VAE-config-specific, so cross-model reuse is invalid);
  * rows are keyed by integer "index" — rows without one (legacy schema)
    are not reusable via the manifest, but their .pt files can still be
    reused through the file-existence fallback below;
  * a manifest row is only reused when the pair at that index still has the
    same prompt_id (guards against pairs.jsonl edits) and both .pt files
    validate;
  * fallback: with no usable manifest row, a pair is reused when both
    pair{index}_{win,lose}.pt exist and validate (legacy behaviour, now
    with corruption checks). This assumes pairs.jsonl ordering is unchanged
    — pass --no-resume after reordering.
  * out_dir/meta.json records the encoding model; a mismatch invalidates
    the whole cache (defence against silently mixing VAE normalizations).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

META_NAME = "meta.json"
MANIFEST_NAME = "pairs_latents.jsonl"
SUMMARY_NAME = "pairs_latents.summary.json"


# ---- .pt file helpers (lazy torch) ------------------------------------------


def latent_pair_paths(out_dir: str | Path, index: int) -> tuple[Path, Path]:
    """Canonical (win, lose) latent paths for a pair index."""
    out = Path(out_dir)
    return out / f"pair{index:04d}_win.pt", out / f"pair{index:04d}_lose.pt"


def is_valid_latent(path: str | Path, *, ndim: int = 5) -> bool:
    """True when `path` holds a loadable, finite tensor with `ndim` dims.

    The DPO pair cache stores 5D (1,C,F,H,W) video latents (the default);
    the AV cache (theama.data.av_latent_cache) reuses this with ``ndim=3``
    for (1,T_lat,C) audio token latents. A truncated write from a killed
    run must fail this check so the sample is re-encoded instead of
    poisoning training.
    """
    p = Path(path)
    if not p.exists():
        return False
    try:
        import torch

        t = torch.load(p, map_location="cpu", weights_only=True)
    except Exception:
        return False
    return (
        torch.is_tensor(t)
        and t.ndim == ndim
        and bool(torch.isfinite(t).all())
    )


def save_latent_atomic(tensor, path: str | Path) -> None:
    """Write via a temp file + os.replace so a crash never leaves a
    half-written .pt that would pass a naive existence check."""
    import torch

    dest = Path(path)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    torch.save(tensor, tmp)
    os.replace(tmp, dest)


# ---- cache meta --------------------------------------------------------------


def read_cache_meta(out_dir: str | Path) -> dict:
    """Read out_dir/meta.json ({} when missing or malformed)."""
    p = Path(out_dir) / META_NAME
    if not p.exists():
        return {}
    try:
        meta = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return meta if isinstance(meta, dict) else {}


def write_cache_meta(out_dir: str | Path, model: str) -> None:
    p = Path(out_dir) / META_NAME
    p.write_text(json.dumps({"model": model}, indent=2) + "\n")


# ---- manifest ---------------------------------------------------------------


def load_reusable_rows(manifest_path: str | Path, model: str) -> dict[int, dict]:
    """Read a manifest into {index: row} of reusable candidates.

    Keeps only rows without "error", with an integer "index", and a matching
    "model". On duplicate indices the later row wins. Missing file and
    malformed lines are skipped.
    """
    kept: dict[int, dict] = {}
    p = Path(manifest_path)
    if not p.exists():
        return kept
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in row:
            continue
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        if row.get("model") != model:
            continue
        kept[index] = row
    return kept


def summarize_encoding(rows: list[dict]) -> dict:
    """Aggregate manifest rows: encoded (fresh) / reused (cache) / failed."""
    failed = sum(1 for r in rows if "error" in r)
    reused = sum(1 for r in rows if "error" not in r and r.get("reused"))
    encoded = sum(1 for r in rows if "error" not in r and not r.get("reused"))
    return {
        "total": len(rows),
        "encoded": encoded,
        "reused": reused,
        "failed": failed,
    }


# ---- planning / execution ----------------------------------------------------


@dataclass
class EncodePlan:
    """Which pair indices can be served from cache and which need the VAE."""

    pairs: list[dict]
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


def plan_encoding(
    pairs: list[dict],
    out_dir: str | Path,
    model: str,
    resume: bool = True,
) -> EncodePlan:
    """Decide reuse vs re-encode for every pair index (validates .pt files)."""
    plan = EncodePlan(pairs=pairs, out_dir=Path(out_dir), model=model)
    candidates: dict[int, dict] = {}
    if resume:
        meta = read_cache_meta(plan.out_dir)
        if meta and meta.get("model") != model:
            plan.cache_invalidated = True
        else:
            candidates = load_reusable_rows(plan.manifest_path, model=model)

    for index, row in candidates.items():
        if index >= len(pairs):
            continue
        if pairs[index].get("prompt_id") != row.get("prompt_id"):
            continue  # pairs.jsonl changed under this index
        win, lose = latent_pair_paths(plan.out_dir, index)
        if is_valid_latent(win) and is_valid_latent(lose):
            row = dict(row)
            row["reused"] = True
            plan.reusable[index] = row

    for index, pair in enumerate(pairs):
        if index in plan.reusable:
            continue
        if resume and not plan.cache_invalidated:
            win, lose = latent_pair_paths(plan.out_dir, index)
            # Legacy fallback: .pt files from a pre-manifest run.
            if is_valid_latent(win) and is_valid_latent(lose):
                plan.reusable[index] = {
                    "index": index,
                    "prompt_id": pair.get("prompt_id"),
                    "prompt": pair.get("prompt"),
                    "model": model,
                    "win_latent": str(win),
                    "lose_latent": str(lose),
                    "reused": True,
                }
                continue
        plan.todo.append(index)
    return plan


def run_encoding(
    plan: EncodePlan,
    encode_fn,
    strict: bool = False,
    progress=print,
) -> dict:
    """Execute an EncodePlan: compact the manifest to reusable rows, encode
    the rest one pair at a time (append + flush per row, crash-safe), then
    write the summary sidecar and cache meta.

    encode_fn(video_path, dest_path) must save the normalized latent tensor
    for one video. It is only called for plan.todo indices, so callers can
    skip loading the VAE entirely when todo is empty. With strict=False a
    failing pair is recorded with an "error" row and retried on the next run.
    Returns the summary dict.
    """
    rows: dict[int, dict] = dict(plan.reusable)
    with plan.manifest_path.open("w") as fh:
        for index in sorted(plan.reusable):
            fh.write(json.dumps(plan.reusable[index]) + "\n")
        fh.flush()

        for index in plan.todo:
            pair = plan.pairs[index]
            win, lose = latent_pair_paths(plan.out_dir, index)
            base = {
                "index": index,
                "prompt_id": pair.get("prompt_id"),
                "prompt": pair.get("prompt"),
                "model": plan.model,
            }
            try:
                if encode_fn is None:  # pragma: no cover - defensive
                    raise RuntimeError("no encoder available for pending pairs")
                encode_fn(pair["win_video"], win)
                encode_fn(pair["lose_video"], lose)
                row = {
                    **base,
                    "win_latent": str(win),
                    "lose_latent": str(lose),
                    "reused": False,
                }
                progress(f"[{index + 1}/{len(plan.pairs)}] {pair.get('prompt_id')}")
            except Exception as exc:
                if strict:
                    raise
                row = {**base, "error": f"{type(exc).__name__}: {exc}"}
                progress(
                    f"[{index + 1}/{len(plan.pairs)}] {pair.get('prompt_id')}"
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

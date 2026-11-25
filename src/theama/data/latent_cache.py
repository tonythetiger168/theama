# SPDX-License-Identifier: Apache-2.0
"""Latent-cache planning / resume logic for scripts/encode_pairs.py (pure logic).

Module import is CPU-safe: torch is only imported lazily inside the helpers
that touch .pt files, so the sandbox test-suite can exercise everything.

Manifest schema (pairs_latents.jsonl), one row per pair index:

    success: {index, prompt_id, prompt, model, win_latent, lose_latent, reused}
    failure: {index, prompt_id, prompt, model, error}
    skipped: {index, prompt_id, prompt, model, skipped: true, retries, last_error}

Retry budget (opt-in via ``max_retries``), same design as the text cache
(:mod:`theama.data.text_cache`):

  * every failed encode is recorded in ``pair_failures.json`` (a small
    ledger keyed by the cache filename stem ``pair{index:04d}`` so operators
    can grep the cache dir, storing prompt_id/prompt, the accumulated
    attempt count and the last error);
  * on the next run a pair whose count has reached ``max_retries`` is
    planned as *skipped* — the encoder is never called for it and the
    manifest records the exhausted budget, so a pathological pair (e.g. one
    that OOMs the VAE deterministically) cannot burn an unbounded retry
    loop; clear the ledger (CLI ``--reset-failures``) to give such pairs
    another chance;
  * a pair that eventually encodes successfully is removed from the ledger,
    so transient failures do not accumulate against it;
  * ``max_retries=None`` (the default) keeps the pre-ledger behaviour
    exactly: unlimited retries and no ledger file is ever written.

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
  * fallback: with NO manifest row for an index, a pair is reused when both
    pair{index}_{win,lose}.pt exist and validate (legacy behaviour, now
    with corruption checks). This assumes pairs.jsonl ordering is unchanged
    — pass --no-resume after reordering. The fallback NEVER applies to an
    index that has a manifest row: a row rejected by the prompt_id drift
    guard means the .pt files at that index are stale and must be
    re-encoded, not silently re-adopted.
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
FAILURES_NAME = "pair_failures.json"


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


# ---- failure ledger (retry budget) ---------------------------------------------
#
# Generic core bound to FAILURES_NAME by the pair-cache wrappers below and to
# its own ledger filename by theama.data.av_latent_cache, so every cache
# shares one atomic-write / corrupt-tolerant implementation. Ledger upkeep is
# best-effort: callers catch OSError so it can never fail a good encode.


def _check_max_retries(max_retries: int | None) -> None:
    if max_retries is not None and (
        not isinstance(max_retries, int)
        or isinstance(max_retries, bool)
        or max_retries < 1
    ):
        raise ValueError(f"max_retries must be a positive int or None, got {max_retries!r}")


def _ledger_path(out_dir: str | Path, name: str) -> Path:
    return Path(out_dir) / name


def _ledger_load(out_dir: str | Path, name: str) -> dict[str, dict]:
    """Read a failure ledger ({key: {..., count, error}}); {} on any problem."""
    p = _ledger_path(out_dir, name)
    if not p.exists():
        return {}
    try:
        ledger = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return ledger if isinstance(ledger, dict) else {}


def _ledger_write(out_dir: str | Path, name: str, ledger: dict[str, dict]) -> None:
    """Write via temp file + os.replace so a crash cannot half-write the ledger."""
    tmp = _ledger_path(out_dir, name).with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, _ledger_path(out_dir, name))


def _ledger_record(
    out_dir: str | Path, name: str, key: str, fields: dict, error: str
) -> int:
    """Increment the failure count for `key` and persist; returns the new count."""
    ledger = _ledger_load(out_dir, name)
    entry = ledger.get(key, {})
    count = int(entry.get("count", 0)) + 1
    ledger[key] = {**fields, "count": count, "error": error}
    _ledger_write(out_dir, name, ledger)
    return count


def _ledger_clear(out_dir: str | Path, name: str, key: str) -> None:
    p = _ledger_path(out_dir, name)
    if not p.exists():
        return
    ledger = _ledger_load(out_dir, name)
    if ledger.pop(key, None) is not None:
        _ledger_write(out_dir, name, ledger)


def _ledger_reset(out_dir: str | Path, name: str) -> bool:
    p = _ledger_path(out_dir, name)
    if p.exists():
        p.unlink()
        return True
    return False


def pair_failure_key(index: int) -> str:
    """Ledger key: the cache filename stem, so operators can grep the cache dir."""
    return f"pair{index:04d}"


def load_failure_ledger(out_dir: str | Path) -> dict[str, dict]:
    """Read pair_failures.json; {} on any problem."""
    return _ledger_load(out_dir, FAILURES_NAME)


def record_failure(out_dir: str | Path, index: int, pair: dict, error: str) -> int:
    """Increment the failure count for a pair index and persist the ledger."""
    return _ledger_record(
        out_dir,
        FAILURES_NAME,
        pair_failure_key(index),
        {"prompt_id": pair.get("prompt_id"), "prompt": pair.get("prompt")},
        error,
    )


def clear_failure(out_dir: str | Path, index: int) -> None:
    """Drop a pair index from the ledger (it encoded successfully)."""
    _ledger_clear(out_dir, FAILURES_NAME, pair_failure_key(index))


def reset_failure_ledger(out_dir: str | Path) -> bool:
    """Delete the ledger file; True when one was deleted (CLI --reset-failures)."""
    return _ledger_reset(out_dir, FAILURES_NAME)


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
    skipped: dict[int, dict] = field(default_factory=dict)  # retry budget exhausted
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
    max_retries: int | None = None,
) -> EncodePlan:
    """Decide reuse vs re-encode for every pair index (validates .pt files).

    With ``max_retries`` set, the failure ledger is consulted: pairs whose
    recorded failure count has reached the budget are planned as *skipped*
    (never handed to the encoder); the rest land in ``todo`` as usual.
    ``None`` keeps the pre-ledger behaviour (no ledger file is read).
    """
    _check_max_retries(max_retries)
    plan = EncodePlan(pairs=pairs, out_dir=Path(out_dir), model=model)
    candidates: dict[int, dict] = {}
    ledger: dict[str, dict] = {}
    if resume:
        meta = read_cache_meta(plan.out_dir)
        if meta and meta.get("model") != model:
            plan.cache_invalidated = True
        else:
            candidates = load_reusable_rows(plan.manifest_path, model=model)
            if max_retries is not None:
                ledger = _ledger_load(plan.out_dir, FAILURES_NAME)

    fallback_blocked: set[int] = set()  # indices the manifest has spoken for
    for index, row in candidates.items():
        if index >= len(pairs):
            continue
        fallback_blocked.add(index)
        if pairs[index].get("prompt_id") != row.get("prompt_id"):
            continue  # pairs.jsonl changed under this index — .pt is stale
        win, lose = latent_pair_paths(plan.out_dir, index)
        if is_valid_latent(win) and is_valid_latent(lose):
            row = dict(row)
            row["reused"] = True
            plan.reusable[index] = row

    for index, pair in enumerate(pairs):
        if index in plan.reusable:
            continue
        if resume and not plan.cache_invalidated and index not in fallback_blocked:
            win, lose = latent_pair_paths(plan.out_dir, index)
            # Legacy fallback: .pt files from a pre-manifest run. Never
            # applies where the manifest has a row — a drift-rejected row
            # marks those files stale, and the pair must be re-encoded.
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
        entry = ledger.get(pair_failure_key(index))
        if entry and int(entry.get("count", 0)) >= max_retries:
            plan.skipped[index] = {
                "index": index,
                "prompt_id": pair.get("prompt_id"),
                "prompt": pair.get("prompt"),
                "model": model,
                "skipped": True,
                "retries": int(entry["count"]),
                "last_error": str(entry.get("error", "")),
            }
        else:
            plan.todo.append(index)
    return plan


def run_encoding(
    plan: EncodePlan,
    encode_fn,
    strict: bool = False,
    progress=print,
    max_retries: int | None = None,
) -> dict:
    """Execute an EncodePlan: compact the manifest to reusable rows, encode
    the rest one pair at a time (append + flush per row, crash-safe), then
    write the summary sidecar and cache meta.

    encode_fn(video_path, dest_path) must save the normalized latent tensor
    for one video. It is only called for plan.todo indices, so callers can
    skip loading the VAE entirely when todo is empty. With strict=False a
    failing pair is recorded with an "error" row and retried on the next run.
    Returns the summary dict.

    With ``max_retries`` set, every failure is counted in the persistent
    ledger; a pair that succeeds is removed from it, and pairs the planner
    marked *skipped* (budget exhausted) are written to the manifest without
    ever touching the encoder. ``None`` (default) performs no ledger I/O at
    all.
    """
    _check_max_retries(max_retries)
    rows: dict[int, dict] = dict(plan.reusable)
    with plan.manifest_path.open("w") as fh:
        for index in sorted(plan.reusable):
            fh.write(json.dumps(plan.reusable[index]) + "\n")
        for index in sorted(plan.skipped):
            fh.write(json.dumps(plan.skipped[index]) + "\n")
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
                if max_retries is not None:
                    try:
                        clear_failure(plan.out_dir, index)
                    except OSError:  # ledger upkeep must never fail a good encode
                        pass
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                if max_retries is not None:
                    try:
                        count = record_failure(plan.out_dir, index, pair, error)
                    except OSError:
                        count = 0  # ledger is best-effort; the failure row stands
                    if strict:
                        raise  # ledger already updated for observability
                    exhausted = count >= max_retries
                    row = {**base, "error": error}
                    progress(
                        f"[{index + 1}/{len(plan.pairs)}] {pair.get('prompt_id')}"
                        f" FAILED ({error})"
                        + (f" — retry budget exhausted ({count} attempts),"
                           " skipping on later runs" if exhausted else "")
                    )
                else:
                    if strict:
                        raise
                    row = {**base, "error": error}
                    progress(
                        f"[{index + 1}/{len(plan.pairs)}] {pair.get('prompt_id')}"
                        f" FAILED ({row['error']})"
                    )
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            rows[index] = row

    write_cache_meta(plan.out_dir, plan.model)
    summary_rows = [rows[i] for i in sorted(rows) if i not in plan.skipped]
    summary = {
        "model": plan.model,
        **summarize_encoding(summary_rows),
    }
    if max_retries is not None:
        summary["skipped"] = len(plan.skipped)
    plan.summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary

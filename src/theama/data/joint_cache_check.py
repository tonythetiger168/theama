# SPDX-License-Identifier: Apache-2.0
"""Preflight profiling for the P1b joint latent caches (CPU, stdlib only).

scripts/check_joint_latents.py combines these helpers with
theama.data.joint_latent_loader.load_joint_latent_records into the pre-GPU
gate for joint training (runbook §5.1's ``joint cache: N samples
(dropped: 0)`` acceptance, runnable without launching the trainer).

Why this exists: the loader's join report only sees rows that pass
load_reusable_rows — encode-error rows, bad JSON lines, non-integer
indices, and model-mismatch rows are filtered out *before* the join, so
they vanish silently and surface (if at all) as a misleading
``missing_*_row`` count or a quietly smaller training set.
profile_manifest makes each of those categories visible per manifest so a
corrupt or stale cache fails the gate with its true cause.
"""

from __future__ import annotations

import json

__all__ = [
    "PROFILE_KEYS",
    "JOIN_DROP_REASONS",
    "profile_manifest",
    "zero_filled_dropped",
]

# Fixed key order for the per-manifest profile (deterministic reports).
PROFILE_KEYS = (
    "total_rows",
    "bad_json_lines",
    "error_rows",
    "invalid_index_rows",
    "model_mismatch_rows",
    "kept_rows",
    "skipped_rows",
)

# The six reasons load_joint_latent_records can drop an index, sorted.
JOIN_DROP_REASONS = (
    "caption_mismatch",
    "missing_av_row",
    "missing_keys",
    "missing_text_row",
    "skipped_av_row",
    "skipped_text_row",
)


def profile_manifest(text: str, model: str) -> dict[str, int]:
    """Classify every line of a cache manifest for `model` (pure).

    Precedence mirrors latent_cache.load_reusable_rows so the counts line
    up with what the join actually sees: unparseable / non-object lines are
    ``bad_json_lines``; then ``error`` rows (the encoder ran and failed);
    then non-integer (or bool) indices; then rows recorded under a
    different model; everything left is ``kept_rows`` (what the join can
    see), of which rows the planner marked ``skipped`` (retry budget
    exhausted) are counted separately. ``model_mismatch_rows`` is reported
    for visibility but never fails the gate — re-encoding a directory with
    a newer encoder legitimately leaves old-model rows behind.
    """
    counts = {key: 0 for key in PROFILE_KEYS}
    for line in text.splitlines():
        if not line.strip():
            continue
        counts["total_rows"] += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            counts["bad_json_lines"] += 1
            continue
        if not isinstance(row, dict):
            counts["bad_json_lines"] += 1
            continue
        if "error" in row:
            counts["error_rows"] += 1
            continue
        index = row.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            counts["invalid_index_rows"] += 1
            continue
        if row.get("model") != model:
            counts["model_mismatch_rows"] += 1
            continue
        counts["kept_rows"] += 1
        if row.get("skipped"):
            counts["skipped_rows"] += 1
    return counts


def zero_filled_dropped(dropped: dict[str, int]) -> dict[str, int]:
    """The join's dropped dict with all six reasons present (zero-filled).

    load_joint_latent_records only creates keys for reasons that occurred;
    the gate report always carries the full set so downstream tooling (and
    humans diffing two runs) never has to handle missing keys.
    """
    return {reason: int(dropped.get(reason, 0)) for reason in JOIN_DROP_REASONS}

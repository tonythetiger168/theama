# SPDX-License-Identifier: Apache-2.0
"""P1a: preflight validation for ``pairs_latents.jsonl`` (CPU, no torch).

``scripts/encode_pairs.py`` writes one manifest row per preference pair:

  success: {index, prompt_id, prompt, model, win_latent, lose_latent, reused}
  failure: {index, prompt_id, prompt, model, error}
  skipped: {index, prompt_id, prompt, model, skipped: true, retries, last_error}

``scripts/train_dpo.py`` can only train on success rows. Before this module
the entry point parsed the file with a bare comprehension, so a single
failure/skipped row raised KeyError at startup and a missing .pt raised
FileNotFoundError deep inside the GPU training loop. The validator here
classifies every row so callers drop unusable rows *with reasons* —
fail-closed with visibility, the same contract as
``data/joint_latent_loader.py`` — or gate on them
(``scripts/check_pair_latents.py``).

Drop reasons are checked root-cause first:

  1. ``encode_failed`` — the row carries ``error`` (the encoder recorded a
     failure; rerun encode_pairs to retry it).
  2. ``skipped_budget_exhausted`` — the row carries ``skipped: true``
     (retry budget exhausted; --reset-failures to retry).
  3. ``missing_keys`` — success-row keys absent or of the wrong type
     (stale or hand-made file).
  4. ``missing_latent_file`` — the win/lose .pt is not on disk (cache moved
     or pruned after encoding).

Pure: no imports beyond the stdlib and no file reads of its own — callers
inject ``exists`` (default ``Path.is_file``) so CPU tests stay disk-free.
"""

from __future__ import annotations

import json
from pathlib import Path

REQUIRED_KEYS = ("prompt", "win_latent", "lose_latent")

DROP_REASONS = (
    "encode_failed",
    "skipped_budget_exhausted",
    "missing_keys",
    "missing_latent_file",
)


def classify_row(row, exists) -> tuple[str | None, str]:
    """Classify one manifest row. Returns (reason, detail); None = usable."""
    if not isinstance(row, dict):
        return "missing_keys", "row is not a JSON object"
    if "error" in row:
        return "encode_failed", str(row.get("error", ""))
    if row.get("skipped"):
        return "skipped_budget_exhausted", str(row.get("last_error", ""))
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        return "missing_keys", "missing: " + ",".join(missing)
    for key in ("win_latent", "lose_latent"):
        if not isinstance(row[key], str) or not row[key]:
            return "missing_keys", f"{key} is not a path string"
    for key in ("win_latent", "lose_latent"):
        if not exists(row[key]):
            return "missing_latent_file", f"{key}: {row[key]}"
    return None, ""


def validate_pair_latents(text: str, exists=None) -> dict:
    """Validate the whole manifest text. Deterministic, pure, file-free.

    Returns a report: ``rows`` (non-blank lines), ``bad_json_lines``,
    ``usable`` count, ``usable_rows`` (success rows in file order — what
    train_dpo consumes), ``dropped`` (every reason key, zero-filled),
    ``dropped_rows`` (line/index/reason/detail per rejected row).
    """
    if exists is None:
        exists = lambda p: Path(p).is_file()  # noqa: E731
    usable_rows: list[dict] = []
    dropped_rows: list[dict] = []
    counts = dict.fromkeys(DROP_REASONS, 0)
    rows = 0
    bad_json_lines = 0
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        rows += 1
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad_json_lines += 1
            continue
        reason, detail = classify_row(row, exists)
        if reason is None:
            usable_rows.append(row)
        else:
            counts[reason] += 1
            dropped_rows.append(
                {
                    "line": lineno,
                    "index": row.get("index") if isinstance(row, dict) else None,
                    "reason": reason,
                    "detail": detail,
                }
            )
    return {
        "rows": rows,
        "usable": len(usable_rows),
        "usable_rows": usable_rows,
        "dropped": counts,
        "dropped_rows": dropped_rows,
        "bad_json_lines": bad_json_lines,
    }


def summarize(report: dict) -> dict:
    """Compact summary for CLI output / logs (drops the bulky row lists)."""
    return {
        "rows": report["rows"],
        "usable": report["usable"],
        "dropped": report["dropped"],
        "bad_json_lines": report["bad_json_lines"],
    }

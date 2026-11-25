# SPDX-License-Identifier: Apache-2.0
"""P1a: preflight validation for ``pairs.jsonl`` (CPU, no torch).

``scripts/make_pref_pairs.py`` writes one row per preference pair:

  {prompt_id, prompt, win_video, lose_video, win_score, lose_score, margin}

``scripts/encode_pairs.py`` consumes that file positionally: the latent
cache is keyed by pair *index* (``latent_pair_paths(out_dir, index)`` and
the manifest's ``index`` column), and resume reuses a cached .pt when the
row at the same index still carries the same ``prompt_id``. Before this
module the entry point parsed the file with a bare comprehension — one
malformed line raised JSONDecodeError, and rows with missing keys,
NaN/±Inf scores (which Python's json parser accepts literally), inverted
margins, self-pairs, or videos that are not on disk sailed through
planning and failed only inside the GPU encode loop (or burned retry
budget on rows that can never succeed).

The validator here classifies every row so the entry point can refuse to
start *with reasons* — fail-closed with visibility, the same contract as
``reward/score_check.py`` and ``data/pair_latent_check.py`` — or gate on
them (``scripts/check_pairs.py``).

Unlike the scores/latents gates, callers must NOT drop unusable rows and
continue: dropping shifts every later pair index and misaligns the
positional latent cache (the prompt_id drift guard cannot catch a shift
between two pairs of the same prompt). pairs.jsonl is a small CPU-side
artifact — regenerate it with make_pref_pairs instead.

Drop reasons are checked root-cause first:

  1. ``missing_keys`` — required keys absent or of the wrong type (stale
     or hand-made file). Scores must be real numbers (bool is rejected
     even though bool is an int subclass).
  2. ``non_finite_score`` — win/lose score (or the optional ``margin``)
     is NaN or ±Inf (passes isinstance but poisons encoding order and
     downstream margin statistics).
  3. ``self_pair`` — win_video == lose_video; a pair of a clip against
     itself carries no DPO signal (make_pref_pairs never emits one).
  4. ``non_positive_margin`` — win_score <= lose_score; build_pairs
     guarantees a positive margin above threshold, so this means a stale
     or hand-edited row.
  5. ``missing_video_file`` — the win/lose clip is not on disk (the
     candidates directory moved or was pruned after pairing).

Pure: no imports beyond the stdlib and no file reads of its own —
callers inject ``exists`` (default ``Path.is_file``) so CPU tests stay
disk-free.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

REQUIRED_KEYS = (
    "prompt_id",
    "prompt",
    "win_video",
    "lose_video",
    "win_score",
    "lose_score",
)

DROP_REASONS = (
    "missing_keys",
    "non_finite_score",
    "self_pair",
    "non_positive_margin",
    "missing_video_file",
)


def classify_row(row, exists) -> tuple[str | None, str]:
    """Classify one pairs row. Returns (reason, detail); None = usable."""
    if not isinstance(row, dict):
        return "missing_keys", "row is not a JSON object"
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        return "missing_keys", "missing: " + ",".join(missing)
    for key in ("prompt_id", "win_video", "lose_video"):
        if not isinstance(row[key], str) or not row[key]:
            return "missing_keys", f"{key} is not a non-empty string"
    if not isinstance(row["prompt"], str):
        return "missing_keys", "prompt is not a string"
    for key in ("win_score", "lose_score"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "missing_keys", f"{key} is not a number"
        if not math.isfinite(value):
            return "non_finite_score", f"{key}={value}"
    if "margin" in row:
        margin = row["margin"]
        if (
            isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not math.isfinite(margin)
        ):
            return "non_finite_score", f"margin={margin}"
    if row["win_video"] == row["lose_video"]:
        return "self_pair", row["win_video"]
    if row["win_score"] <= row["lose_score"]:
        return (
            "non_positive_margin",
            f"win={row['win_score']} lose={row['lose_score']}",
        )
    for key in ("win_video", "lose_video"):
        if not exists(row[key]):
            return "missing_video_file", f"{key}: {row[key]}"
    return None, ""


def validate_pairs(text: str, exists=None) -> dict:
    """Validate the whole pairs.jsonl text. Deterministic, pure, file-free.

    Returns a report: ``rows`` (non-blank lines), ``bad_json_lines``,
    ``usable`` count, ``usable_rows`` (valid rows in file order — what
    encode_pairs consumes), ``dropped`` (every reason key, zero-filled),
    ``dropped_rows`` (line/prompt_id/reason/detail per rejected row).
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
                    "prompt_id": row.get("prompt_id")
                    if isinstance(row, dict)
                    else None,
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

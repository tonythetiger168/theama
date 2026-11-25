# SPDX-License-Identifier: Apache-2.0
"""P1a: preflight validation for ``scores.jsonl`` (CPU, no torch).

``scripts/score_eval_set.py`` writes one row per scored candidate:

  success: {prompt_id, prompt, video, reward, overall, dims}
  failure: {prompt_id, prompt, video, reward, error}

``scripts/make_pref_pairs.py`` can only pair success rows. Before this
module the entry point parsed the file with a bare comprehension, so a
single failure row raised KeyError at startup (build_pairs and
coverage_report both index ``prompt_id``/``overall`` directly), and a
NaN/Inf ``overall`` — which Python's json parser accepts literally —
silently corrupted margin filtering and candidate ordering.

The validator here classifies every row so callers drop unusable rows
*with reasons* — fail-closed with visibility, the same contract as
``data/pair_latent_check.py`` — or gate on them
(``scripts/check_scores.py``).

Drop reasons are checked root-cause first:

  1. ``score_failed`` — the row carries ``error`` (the scorer recorded a
     failure; rerun score_eval_set to retry it — resume drops error rows).
  2. ``missing_keys`` — success-row keys absent or of the wrong type
     (stale or hand-made file). ``overall`` must be a real number
     (bool is rejected even though bool is an int subclass).
  3. ``non_finite_score`` — ``overall`` is NaN or ±Inf (passes isinstance
     but poisons margin comparisons and summary means).
  4. ``reward_mismatch`` — only when the caller passes ``reward``: the row
     was scored by a different reward model. Rows with no ``reward`` key
     predate per-row tagging and count as the historical default
     "videoreward" (same rule as ``reward/score.load_successful_scores``).
     Mixing reward models makes cross-row margins meaningless, so the
     gate can enforce a single scorer; with ``reward=None`` the check is
     off and the ``rewards`` histogram only makes mixing visible.

Pure: no imports beyond the stdlib and no file reads of its own —
scores rows only *reference* video paths (make_pref_pairs never opens
them; encode_pairs has its own retry budget for missing media), so
unlike the pair-latent validator there is no on-disk check.
"""

from __future__ import annotations

import json
import math

REQUIRED_KEYS = ("prompt_id", "prompt", "video", "overall")

DROP_REASONS = (
    "score_failed",
    "missing_keys",
    "non_finite_score",
    "reward_mismatch",
)

# Rows without a per-row ``reward`` tag predate tagging; treat them as the
# historical default (mirrors score.load_successful_scores).
DEFAULT_REWARD = "videoreward"


def classify_row(row, reward: str | None = None) -> tuple[str | None, str]:
    """Classify one scores row. Returns (reason, detail); None = usable."""
    if not isinstance(row, dict):
        return "missing_keys", "row is not a JSON object"
    if "error" in row:
        return "score_failed", str(row.get("error", ""))
    missing = [k for k in REQUIRED_KEYS if k not in row]
    if missing:
        return "missing_keys", "missing: " + ",".join(missing)
    for key in ("prompt_id", "video"):
        if not isinstance(row[key], str) or not row[key]:
            return "missing_keys", f"{key} is not a non-empty string"
    if not isinstance(row["prompt"], str):
        return "missing_keys", "prompt is not a string"
    overall = row["overall"]
    if isinstance(overall, bool) or not isinstance(overall, (int, float)):
        return "missing_keys", "overall is not a number"
    if not math.isfinite(overall):
        return "non_finite_score", f"overall={overall}"
    if reward is not None and row.get("reward", DEFAULT_REWARD) != reward:
        return "reward_mismatch", f"reward={row.get('reward', DEFAULT_REWARD)}"
    return None, ""


def validate_scores(text: str, reward: str | None = None) -> dict:
    """Validate the whole scores.jsonl text. Deterministic, pure, file-free.

    Returns a report: ``rows`` (non-blank lines), ``bad_json_lines``,
    ``usable`` count, ``usable_rows`` (success rows in file order — what
    make_pref_pairs consumes), ``dropped`` (every reason key, zero-filled),
    ``dropped_rows`` (line/prompt_id/video/reason/detail per rejected row),
    and ``rewards`` (histogram of the ``reward`` tag across parsed rows,
    sorted by tag — visibility into mixed-scorer files even when the
    ``reward`` filter is off).
    """
    usable_rows: list[dict] = []
    dropped_rows: list[dict] = []
    counts = dict.fromkeys(DROP_REASONS, 0)
    rewards: dict[str, int] = {}
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
        if isinstance(row, dict):
            tag = row.get("reward", DEFAULT_REWARD)
            if isinstance(tag, str):
                rewards[tag] = rewards.get(tag, 0) + 1
        reason, detail = classify_row(row, reward=reward)
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
                    "video": row.get("video") if isinstance(row, dict) else None,
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
        "rewards": dict(sorted(rewards.items())),
    }


def summarize(report: dict) -> dict:
    """Compact summary for CLI output / logs (drops the bulky row lists)."""
    return {
        "rows": report["rows"],
        "usable": report["usable"],
        "dropped": report["dropped"],
        "bad_json_lines": report["bad_json_lines"],
        "rewards": report["rewards"],
    }

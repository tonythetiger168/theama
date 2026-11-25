#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile a P1a scores.jsonl before make_pref_pairs (CPU).

    python scripts/scores_stats.py --scores dpo_out/candidates/scores.jsonl
    python scripts/scores_stats.py --scores dpo_out/candidates/scores.jsonl --out stats.json

``check_scores.py`` is the pass/fail gate, but a gate only says *whether*
the file is usable — not what pairing will yield. Once score_eval_set has
written the file on the GPU pod there is no way to see, before building
pairs: how many prompts have ≥2 successful candidates (the only prompts
make_pref_pairs can pair), how many prompts lost every candidate to
scorer errors, whether one clip was scored twice, or how the overall /
per-dim scores are distributed. This tool is that profile — the P1a
analog of ``pairs_stats.py`` / ``manifest_stats.py``. It only reads the
scores text; video and latent files are never opened.

Exit code is 0 for any readable scores file (this tool describes, it does
not judge) and 1 when the file itself cannot be read. Output is one
deterministic JSON object on stdout (sorted keys, stable rounding), so
diffing two profiles is meaningful.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

# Success-row keys this profiler considers complete (mirrors the schema
# score_check.classify_row enforces, minus the optional reward filter).
_SCORE_KEYS = ("prompt_id", "prompt", "video", "overall")

# Rows without a per-row ``reward`` tag predate tagging; the historical
# default mirrors reward/score_check.DEFAULT_REWARD.
_DEFAULT_REWARD = "videoreward"


def _load_rows(scores_path: str | Path) -> tuple[list[dict], int]:
    """Parse jsonl tolerantly; returns (object rows, bad-line count)."""
    rows: list[dict] = []
    bad = 0
    with open(scores_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def _num_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def _distribution(values: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1
    # Deterministic: count desc, then key asc.
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _is_complete_success(row: dict) -> bool:
    if "error" in row:
        return False
    if any(k not in row for k in _SCORE_KEYS):
        return False
    if not isinstance(row["prompt_id"], str) or not row["prompt_id"]:
        return False
    if not isinstance(row["prompt"], str):
        return False
    if not isinstance(row["video"], str) or not row["video"]:
        return False
    overall = row["overall"]
    if isinstance(overall, bool) or not isinstance(overall, (int, float)):
        return False
    return math.isfinite(overall)


def profile_scores(rows: list[dict]) -> dict:
    """Pure profile of parsed scores rows (unknown keys ignored)."""
    success = [r for r in rows if _is_complete_success(r)]
    errors = [r for r in rows if "error" in r]
    incomplete = len(rows) - len(success) - len(errors)

    # Prompt coverage: success rows are the only ones make_pref_pairs can
    # pair, so per-prompt candidate counts are computed over them; prompts
    # seen only in failed/incomplete rows surface as prompts_all_failed.
    success_per_prompt = _distribution(
        [r["prompt_id"] for r in success]
    )
    all_prompts = {
        r["prompt_id"] for r in rows
        if isinstance(r.get("prompt_id"), str) and r["prompt_id"]
    }
    pairable = sum(1 for n in success_per_prompt.values() if n >= 2)
    single = sum(1 for n in success_per_prompt.values() if n == 1)
    all_failed = len(all_prompts) - len(success_per_prompt)

    # One clip scored twice for the same prompt inflates pairing weight;
    # count every row beyond the first per (prompt_id, video).
    seen: set[tuple[str, str]] = set()
    duplicates = 0
    for r in success:
        key = (r["prompt_id"], r["video"])
        if key in seen:
            duplicates += 1
        else:
            seen.add(key)

    overall = _num_stats([float(r["overall"]) for r in success]) if success else None

    dim_values: dict[str, list[float]] = {}
    for r in success:
        dims = r.get("dims")
        if not isinstance(dims, dict):
            continue
        for name, value in dims.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if not math.isfinite(value):
                continue
            dim_values.setdefault(str(name), []).append(float(value))
    dims = {
        name: {
            "count": len(vals),
            "mean": round(statistics.fmean(vals), 3),
            "min": round(min(vals), 3),
            "max": round(max(vals), 3),
        }
        for name, vals in sorted(dim_values.items())
    }

    rewards: dict[str, int] = {}
    for r in rows:
        tag = r.get("reward", _DEFAULT_REWARD)
        if isinstance(tag, str):
            rewards[tag] = rewards.get(tag, 0) + 1

    error_dist = _distribution([str(r.get("error", "")) for r in errors])
    error_kinds = {
        "distinct": len(error_dist),
        "max_single_error": max(error_dist.values()) if error_dist else 0,
        "top": dict(list(error_dist.items())[:5]),
    }

    return {
        "rows": len(rows),
        "success_rows": len(success),
        "error_rows": len(errors),
        "incomplete_rows": incomplete,
        "unique_prompts": len(all_prompts),
        "prompts_pairable": pairable,
        "prompts_single_candidate": single,
        "prompts_all_failed": all_failed,
        "candidates_per_prompt": _distribution(
            [int(n) for n in success_per_prompt.values()]
        ),
        "duplicate_scored_rows": duplicates,
        "overall": overall,
        "dims": dims,
        "rewards": dict(sorted(rewards.items())),
        "error_kinds": error_kinds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True,
                    help="scores jsonl from score_eval_set.py")
    ap.add_argument("--out", default=None,
                    help="also write the profile JSON here")
    args = ap.parse_args()

    try:
        rows, bad_lines = _load_rows(args.scores)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    profile = profile_scores(rows)
    profile["scores"] = str(args.scores)
    profile["bad_json_lines"] = bad_lines
    text = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

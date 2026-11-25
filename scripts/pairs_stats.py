#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile a DPO preference-pair file (P1a data engine, CPU).

    python scripts/pairs_stats.py --pairs dpo_out/pairs.jsonl
    python scripts/pairs_stats.py --pairs dpo_out/pairs.jsonl --out stats.json

``make_pref_pairs.py`` prints a one-shot coverage report at build time, but
once the file is on disk there is no way to re-inspect it before burning GPU
hours on ``encode_pairs.py`` / ``train_dpo.py``. This tool is that profile —
the P1a analog of ``manifest_stats.py``: row counts, prompt coverage,
pairs-per-prompt distribution, margin / win / lose score statistics, and
winner/loser concentration (one clip winning or losing many pairs is a bias
signal worth seeing before training). It only reads the pairs text; video
and latent files are never opened.

Exit code is 0 for any readable pairs file (this tool describes, it does
not judge) and 1 when the file itself cannot be read. Output is one
deterministic JSON object on stdout (sorted keys, stable rounding), so
diffing two profiles is meaningful.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Margin histogram edges (reward-score difference): <0.25, 0.25-0.5,
# 0.5-1, 1-2, >=2. Rows built with the default --margin 0.5 land in the
# last three buckets; a sub-threshold min_margin flags a stale or
# hand-edited file.
_MARGIN_BUCKETS = (
    ("lt_0.25", 0.0, 0.25),
    ("0.25_to_0.5", 0.25, 0.5),
    ("0.5_to_1", 0.5, 1.0),
    ("1_to_2", 1.0, 2.0),
    ("ge_2", 2.0, None),
)

_PAIR_KEYS = ("prompt_id", "prompt", "win_video", "lose_video",
              "win_score", "lose_score", "margin")


def _load_rows(pairs_path: str | Path) -> tuple[list[dict], int]:
    """Parse jsonl tolerantly; returns (object rows, bad-line count)."""
    rows: list[dict] = []
    bad = 0
    with open(pairs_path, encoding="utf-8") as fh:
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


def _concentration(clips: list[str]) -> dict:
    dist = _distribution(clips)
    top = dict(list(dist.items())[:5])
    return {
        "distinct": len(dist),
        "max_single_clip": max(dist.values()) if dist else 0,
        "top": top,
    }


def profile_pairs(rows: list[dict]) -> dict:
    """Pure profile of parsed pairs rows (unknown keys ignored)."""
    complete = [r for r in rows if all(k in r for k in _PAIR_KEYS)]
    prompt_ids = [
        r["prompt_id"] for r in rows if isinstance(r.get("prompt_id"), str)
    ]
    per_prompt = _distribution(prompt_ids)

    margins = [
        float(r["margin"]) for r in rows
        if isinstance(r.get("margin"), (int, float))
    ]
    win_scores = [
        float(r["win_score"]) for r in rows
        if isinstance(r.get("win_score"), (int, float))
    ]
    lose_scores = [
        float(r["lose_score"]) for r in rows
        if isinstance(r.get("lose_score"), (int, float))
    ]
    winners = [
        r["win_video"] for r in rows if isinstance(r.get("win_video"), str)
    ]
    losers = [
        r["lose_video"] for r in rows if isinstance(r.get("lose_video"), str)
    ]
    self_pairs = sum(
        1 for r in rows
        if isinstance(r.get("win_video"), str)
        and r.get("win_video") == r.get("lose_video")
    )

    profile: dict = {
        "rows": len(rows),
        "complete_rows": len(complete),
        "incomplete_rows": len(rows) - len(complete),
        "unique_prompts": len(per_prompt),
        "pairs_per_prompt": _distribution(
            [int(n) for n in per_prompt.values()]
        ),
        "self_pairs": self_pairs,
        "margin": None,
        "win_score": _num_stats(win_scores) if win_scores else None,
        "lose_score": _num_stats(lose_scores) if lose_scores else None,
        "winners": _concentration(winners),
        "losers": _concentration(losers),
    }

    if margins:
        stats = _num_stats(margins)
        buckets: dict[str, int] = {}
        for name, lo, hi in _MARGIN_BUCKETS:
            buckets[name] = sum(
                1 for m in margins
                if m >= lo and (hi is None or m < hi)
            )
        stats["buckets"] = buckets
        profile["margin"] = stats
    return profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True,
                    help="pairs jsonl from make_pref_pairs.py")
    ap.add_argument("--out", default=None,
                    help="also write the profile JSON here")
    args = ap.parse_args()

    try:
        rows, bad_lines = _load_rows(args.pairs)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    profile = profile_pairs(rows)
    profile["pairs"] = str(args.pairs)
    profile["bad_json_lines"] = bad_lines
    text = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

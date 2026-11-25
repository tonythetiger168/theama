#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight gate for a P1a scores.jsonl before make_pref_pairs (CPU).

    python scripts/check_scores.py --scores dpo_out/candidates/scores.jsonl

Exit code 0 when every row is a usable success row, 1 otherwise — scorer
failure rows, schema violations, non-finite scores, reward-model
mismatches (with ``--reward``), or bad JSON lines. Prints a JSON summary
and, with ``--show-errors``, every rejected row. scripts/make_pref_pairs.py
applies the same rules before it builds pairs, so a file this gate
rejects never reaches the pair builder — failure rows are retried by
re-running score_eval_set (resume drops error rows and re-scores them).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.score_check import summarize, validate_scores  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="scores.jsonl")
    ap.add_argument(
        "--show-errors",
        action="store_true",
        help="print every rejected row (line/prompt_id/video/reason/detail)",
    )
    ap.add_argument(
        "--reward",
        default=None,
        help="require every row to be scored by this reward model "
        "(untagged rows count as the historical default 'videoreward')",
    )
    args = ap.parse_args()

    try:
        text = Path(args.scores).read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    report = validate_scores(text, reward=args.reward)
    print(json.dumps(summarize(report), indent=2))
    if args.show_errors:
        for row in report["dropped_rows"]:
            print(json.dumps(row), file=sys.stderr)

    rejected = report["rows"] - report["usable"]
    if rejected or report["bad_json_lines"]:
        parts = [f"{rejected} unusable row(s)"]
        if report["bad_json_lines"]:
            parts.append(f"{report['bad_json_lines']} bad JSON line(s)")
        print(
            "scores rejected " + " + ".join(parts)
            + " — rerun score_eval_set to retry failures (resume re-scores "
            "error rows)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

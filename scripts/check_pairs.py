#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight gate for a P1a pairs.jsonl before latent encoding (CPU).

    python scripts/check_pairs.py --pairs dpo_out/pairs.jsonl

Exit code 0 when every row is a usable preference pair (both clips on
disk), 1 otherwise — schema violations, non-finite scores, self-pairs,
non-positive margins, missing video files, or bad JSON lines. Prints a
JSON summary and, with ``--show-errors``, every rejected row. Use
``--no-files`` to skip the on-disk clip check (schema-only, e.g. CI
without the candidates directory mounted). scripts/encode_pairs.py
applies the same rules before planning, so a pairs file this gate
rejects never reaches the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.pairs_check import (  # noqa: E402
    summarize,
    validate_pairs,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="pairs.jsonl")
    ap.add_argument(
        "--show-errors",
        action="store_true",
        help="print every rejected row (line/prompt_id/reason/detail)",
    )
    ap.add_argument(
        "--no-files",
        action="store_true",
        help="schema-only check; do not require the clips on disk",
    )
    args = ap.parse_args()

    try:
        text = Path(args.pairs).read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    exists = (lambda p: True) if args.no_files else None
    report = validate_pairs(text, exists=exists)
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
            "pairs.jsonl rejected " + " + ".join(parts)
            + " — regenerate with make_pref_pairs (rows must not be "
            "dropped in place: pair indices key the latent cache)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

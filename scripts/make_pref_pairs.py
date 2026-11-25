#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build DPO preference pairs from scores.jsonl (CPU).

    python scripts/make_pref_pairs.py --scores dpo_out/candidates/scores.jsonl \
        --margin 0.5 --out dpo_out/pairs.jsonl

The scores file is validated before pairing (reward/score_check.py —
the same rules scripts/check_scores.py gates on): scorer failure rows,
schema violations, non-finite scores and, with --reward, rows from other
reward models are dropped with reasons on stderr; the run refuses to
start when nothing usable remains. A fully-usable file behaves
byte-identically to the unvalidated pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.pairs import build_pairs, coverage_report  # noqa: E402
from theama.reward.score_check import validate_scores  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--max-per-prompt", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--reward",
        default=None,
        help="require every score row to come from this reward model "
        "(default: accept any; mixing scorers makes margins meaningless)",
    )
    args = ap.parse_args()

    try:
        text = Path(args.scores).read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    report = validate_scores(text, reward=args.reward)
    if report["dropped_rows"] or report["bad_json_lines"]:
        print(
            f"scores: {report['usable']}/{report['rows']} rows usable; "
            f"dropped {json.dumps(report['dropped'], sort_keys=True)}"
            + (
                f", {report['bad_json_lines']} bad JSON line(s)"
                if report["bad_json_lines"]
                else ""
            ),
            file=sys.stderr,
        )
    rows = report["usable_rows"]
    if not rows:
        print("no usable score rows", file=sys.stderr)
        return 1
    pairs = build_pairs(rows, margin=args.margin, max_pairs_per_prompt=args.max_per_prompt)
    cov = coverage_report(rows, pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p.as_dict()) + "\n")
    print(json.dumps(cov, indent=2))
    print(f"wrote {len(pairs)} pairs -> {out}")
    if not pairs:
        print("no pairs cleared the margin; add candidates or lower --margin",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

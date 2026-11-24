#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build DPO preference pairs from scores.jsonl (CPU).

    python scripts/make_pref_pairs.py --scores dpo_out/candidates/scores.jsonl \
        --margin 0.5 --out dpo_out/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.pairs import build_pairs, coverage_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--max-per-prompt", type=int, default=1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.scores).read_text().splitlines()
        if line.strip()
    ]
    pairs = build_pairs(rows, margin=args.margin, max_pairs_per_prompt=args.max_per_prompt)
    report = coverage_report(rows, pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for p in pairs:
            fh.write(json.dumps(p.as_dict()) + "\n")
    print(json.dumps(report, indent=2))
    print(f"wrote {len(pairs)} pairs -> {out}")
    if not pairs:
        print("no pairs cleared the margin; add candidates or lower --margin",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

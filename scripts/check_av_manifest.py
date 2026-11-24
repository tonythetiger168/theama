#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate an A/V data-card manifest against schema + license red lines (CPU).

    python scripts/check_av_manifest.py --manifest data/av_manifest.jsonl

Exit code 0 when every row is usable, 1 otherwise (schema violations, bad
JSON, or forbidden sources such as MiniMax H3 / LTX). Prints a JSON summary
and, with ``--show-errors``, every rejected row. Intended for CI gating and
pre-launch checks; the training dataset loader applies the same rules.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import load_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="jsonl data-card manifest")
    ap.add_argument(
        "--show-errors",
        action="store_true",
        help="print every rejected row (lineno/kind/message/raw)",
    )
    args = ap.parse_args()

    try:
        report = load_manifest(args.manifest, strict=False)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(report.summary(), indent=2))
    if args.show_errors:
        for err in report.errors:
            print(json.dumps(err.as_dict()), file=sys.stderr)

    if report.errors:
        forbidden = sum(1 for e in report.errors if e.kind == "forbidden_source")
        print(
            f"manifest rejected {len(report.errors)} row(s)"
            + (f" incl. {forbidden} license-red-line source(s)" if forbidden else ""),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

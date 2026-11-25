#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight gate for a P1a pairs_latents.jsonl before LoRA-DPO (CPU).

    python scripts/check_pair_latents.py \
        --manifest dpo_out/latents/pairs_latents.jsonl

Exit code 0 when every row is a usable success row (both .pt latents on
disk), 1 otherwise — encode-failure rows, retry-budget skips, schema
violations, missing latent files, or bad JSON lines. Prints a JSON
summary and, with ``--show-errors``, every rejected row. Use ``--no-files``
to skip the on-disk latent check (schema-only, e.g. CI without the cache
mounted). scripts/train_dpo.py applies the same rules before it loads the
backbone, so a manifest this gate rejects never reaches the GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.pair_latent_check import (  # noqa: E402
    summarize,
    validate_pair_latents,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="pairs_latents.jsonl")
    ap.add_argument(
        "--show-errors",
        action="store_true",
        help="print every rejected row (line/index/reason/detail)",
    )
    ap.add_argument(
        "--no-files",
        action="store_true",
        help="schema-only check; do not require the .pt latents on disk",
    )
    args = ap.parse_args()

    try:
        text = Path(args.manifest).read_text(encoding="utf-8")
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    exists = (lambda p: True) if args.no_files else None
    report = validate_pair_latents(text, exists=exists)
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
            "pairs_latents rejected " + " + ".join(parts)
            + " — rerun encode_pairs to retry failures (--reset-failures "
            "clears exhausted budgets)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

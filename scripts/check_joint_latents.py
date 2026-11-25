#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight gate joining the P1b AV/text latent caches before joint training.

    python scripts/check_joint_latents.py \
        --av-cache data/av/latents_dev --text-cache data/av/texts

This is runbook §5.1's ``joint cache: N samples (dropped: 0)`` acceptance
gate as a standalone command — the P1b counterpart of
check_pair_latents.py (P1a) — runnable on CPU without launching the
trainer or touching the backbone.

Exit 0 when the manifest join serves at least one record, no index is
dropped, and both manifests are clean (no encode-error rows, no bad JSON
lines); exit 1 otherwise. ``--allow-dropped`` downgrades dropped-row,
error-row, and bad-line counts to informational (useful while an encode
is still mid-run); zero served records always fails. ``--validate-files``
additionally loads every served .pt — the same up-front validation the
trainer's dataset construction performs — so a corrupt or non-finite
latent fails here before any GPU is touched. ``--av-model`` /
``--text-model`` pin the encoder ids (default: each cache's meta.json,
same as the trainer). Prints a deterministic JSON summary on stdout;
``--json-out`` writes the same bytes to a file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_cache import MANIFEST_NAME as AV_MANIFEST_NAME  # noqa: E402
from theama.data.joint_cache_check import (  # noqa: E402
    profile_manifest,
    zero_filled_dropped,
)
from theama.data.joint_latent_loader import (  # noqa: E402
    JointLatentCacheError,
    JointLatentDataset,
    load_joint_latent_records,
)
from theama.data.text_cache import MANIFEST_NAME as TEXT_MANIFEST_NAME  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-cache", required=True, help="AV latent cache dir")
    ap.add_argument("--text-cache", required=True, help="text latent cache dir")
    ap.add_argument("--av-model", default=None, help="pin AV encoder id")
    ap.add_argument("--text-model", default=None, help="pin text encoder id")
    ap.add_argument(
        "--allow-dropped",
        action="store_true",
        help="report dropped/error/bad-line counts without failing on them",
    )
    ap.add_argument(
        "--validate-files",
        action="store_true",
        help="also load every served .pt (catches corrupt/non-finite latents)",
    )
    ap.add_argument("--json-out", default=None, help="also write the summary here")
    args = ap.parse_args()

    av_dir, text_dir = Path(args.av_cache), Path(args.text_cache)
    manifests: dict[str, str] = {}
    for label, path in (
        ("av", av_dir / AV_MANIFEST_NAME),
        ("text", text_dir / TEXT_MANIFEST_NAME),
    ):
        try:
            manifests[label] = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    try:
        join = load_joint_latent_records(
            av_dir, text_dir, av_model=args.av_model, text_model=args.text_model
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    summary = {
        "av_cache": str(av_dir),
        "text_cache": str(text_dir),
        "av_model": join.av_model,
        "text_model": join.text_model,
        "served": len(join),
        "dropped": zero_filled_dropped(join.dropped),
        "av_manifest": profile_manifest(manifests["av"], join.av_model),
        "text_manifest": profile_manifest(manifests["text"], join.text_model),
    }

    problems: list[str] = []
    if len(join) == 0:
        problems.append(
            "zero served records — run encode_av_latents and encode_texts "
            "first, and check the model filters"
        )
    if not args.allow_dropped:
        dropped_total = sum(summary["dropped"].values())
        if dropped_total:
            reasons = ", ".join(
                f"{k}={v}" for k, v in summary["dropped"].items() if v
            )
            problems.append(f"{dropped_total} dropped row(s): {reasons}")
        for label in ("av", "text"):
            prof = summary[f"{label}_manifest"]
            if prof["error_rows"]:
                problems.append(
                    f"{label} manifest has {prof['error_rows']} encode-error "
                    "row(s) — re-run the encode script to repair them"
                )
            if prof["bad_json_lines"]:
                problems.append(
                    f"{label} manifest has {prof['bad_json_lines']} bad JSON "
                    "line(s)"
                )

    if not problems and args.validate_files:
        try:
            JointLatentDataset(join, validate=True)
        except (JointLatentCacheError, ValueError) as exc:
            problems.append(str(exc))
        else:
            summary["validated_latents"] = len(join) * 3

    payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not problems and args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

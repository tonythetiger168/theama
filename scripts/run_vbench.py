#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the VBench 6-dim subset over a generated eval dir, gate vs reference.

Local GPU usage:

    pip install vbench
    python scripts/run_vbench.py --videos eval_out/t2v-a14b --model t2v-a14b

Exit 0 when the overall score is within tolerance of eval/reference.json,
nonzero otherwise. The reference (AMD BF16 study, 0.8681) was measured at
832x480 / 81f @ 16fps / 30 steps / CFG 5.0 — match that config for an
apples-to-apples comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir of generated MP4s")
    ap.add_argument("--model", default="t2v-a14b", choices=["t2v-a14b", "ti2v-5b"])
    ap.add_argument("--dims", nargs="*", default=None, help="override dimension list")
    ap.add_argument("--full", action="store_true",
                    help="run all 16 VBench dims instead of the 6-dim subset")
    args = ap.parse_args()

    ref = json.loads((ROOT / "eval" / "reference.json").read_text(encoding="utf-8"))[
        args.model
    ]
    dims = args.dims or ([] if args.full else ref["six_dim_subset"])

    try:
        import torch
        from vbench import VBench
    except ImportError:
        print("vbench not installed: pip install vbench", file=sys.stderr)
        return 2

    videos = sorted(Path(args.videos).glob("*.mp4"))
    if not videos:
        print(f"no MP4s in {args.videos}", file=sys.stderr)
        return 2
    print(f"evaluating {len(videos)} videos, dims={dims or 'ALL 16'}")

    bench = VBench(torch.device("cuda"), str(ROOT / "eval" / "vbench_cache"),
                   str(ROOT / "eval" / "vbench_out"))
    result = bench.evaluate(
        videos_path=str(args.videos),
        name=f"{args.model}-m1",
        dimension_list=dims or None,
        mode="custom_input",
    )
    out_path = ROOT / "eval" / "vbench_out" / f"{args.model}-m1_results.json"
    print(f"raw results: {out_path}")

    overall = None
    if isinstance(result, dict):
        for key in ("overall", "total_score", "vbench_score"):
            if key in result:
                overall = float(result[key])
                break
    target = ref["vbench_overall_bf16"]
    if target is None:
        print("no reference score for this model; record the printed scores "
              "as our own baseline in eval/reference.json")
        return 0
    if overall is None:
        print("could not parse an overall score from VBench output; "
              "inspect the results JSON manually", file=sys.stderr)
        return 2
    # VBench reports 0-100 in some versions, 0-1 in others; normalize.
    if overall > 1.5:
        overall /= 100.0
    delta = overall - target
    tol = ref["tolerance"]
    print(f"overall={overall:.4f}  reference={target:.4f}  "
          f"delta={delta:+.4f}  tolerance=±{tol}")
    if abs(delta) <= tol:
        print("PASS: within ±2% of the BF16 reference")
        return 0
    print("FAIL: outside tolerance — check eval config parity first", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

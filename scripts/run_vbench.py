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

The gate itself (overall-score extraction, 0-100 vs 0-1 normalization,
tolerance verdict) is pure and CPU-tested in tests/test_m1_vbench_gate.py,
so a parsing bug surfaces before a GPU VBench run, not after one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_OVERALL_KEYS = ("overall", "total_score", "vbench_score")


def extract_overall(result: object) -> float | None:
    """Pull an overall score out of a VBench evaluate() return value.

    Returns None when the result is not a dict, carries none of the known
    keys, or the first known key holds a non-numeric value (clean exit 2
    upstream instead of a traceback).
    """
    if not isinstance(result, dict):
        return None
    for key in _OVERALL_KEYS:
        if key in result:
            try:
                return float(result[key])
            except (TypeError, ValueError):
                return None
    return None


def normalize_overall(score: float) -> float:
    """VBench reports 0-100 in some versions, 0-1 in others; normalize to 0-1."""
    return score / 100.0 if score > 1.5 else score


def gate_passed(overall: float, target: float, tolerance: float) -> bool:
    """True when |overall - target| <= tolerance (boundary inclusive)."""
    return abs(overall - target) <= tolerance


def dimension_scores(eval_results: object) -> dict[str, float]:
    """Per-dimension aggregate scores from a VBench `*_eval_results.json`.

    VBench 0.1.5 `evaluate()` returns None and writes
    `{dim: [aggregate_score, [per-video records]]}`; non-numeric entries are
    skipped rather than raising.
    """
    scores: dict[str, float] = {}
    if not isinstance(eval_results, dict):
        return scores
    for dim, value in eval_results.items():
        head = value[0] if isinstance(value, (list, tuple)) and value else value
        try:
            scores[dim] = float(head)
        except (TypeError, ValueError):
            continue
    return scores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="dir of generated MP4s")
    ap.add_argument("--model", default="t2v-a14b", choices=["t2v-a14b", "ti2v-5b"])
    ap.add_argument("--dims", nargs="*", default=None, help="override dimension list")
    ap.add_argument("--full", action="store_true",
                    help="run all 16 VBench dims instead of the 6-dim subset")
    ap.add_argument("--out-dir", default=str(ROOT / "eval" / "vbench_out"),
                    help="VBench output dir (raw results + summary JSON)")
    ap.add_argument("--cache-dir", default=str(ROOT / "eval" / "vbench_cache"))
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

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    bench = VBench(torch.device("cuda"), str(args.cache_dir), str(out_dir))
    result = bench.evaluate(
        videos_path=str(args.videos),
        name=f"{args.model}-m1",
        dimension_list=dims or None,
        mode="custom_input",
    )
    out_path = out_dir / f"{args.model}-m1_eval_results.json"
    print(f"raw results: {out_path}")
    if result is None and out_path.is_file():
        result = json.loads(out_path.read_text(encoding="utf-8"))
    scores = dimension_scores(result)
    summary = {
        "model": args.model,
        "videos": len(videos),
        "dimensions": scores,
        "six_dim_mean": (round(sum(scores.values()) / len(scores), 6)
                         if scores else None),
        "note": "unweighted mean of the evaluated dims; not VBench's weighted Total Score",
    }
    (out_dir / f"{args.model}-m1_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("VBENCH_SUMMARY " + json.dumps(summary))

    overall = extract_overall(result)
    target = ref["vbench_overall_bf16"]
    if target is None:
        print("no reference score for this model; record the printed scores "
              "as our own baseline in eval/reference.json")
        return 0
    if overall is None:
        print("could not parse an overall score from VBench output; "
              "inspect the results JSON manually", file=sys.stderr)
        return 2
    overall = normalize_overall(overall)
    delta = overall - target
    tol = ref["tolerance"]
    print(f"overall={overall:.4f}  reference={target:.4f}  "
          f"delta={delta:+.4f}  tolerance=±{tol}")
    if gate_passed(overall, target, tol):
        print("PASS: within ±2% of the BF16 reference")
        return 0
    print("FAIL: outside tolerance — check eval config parity first", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

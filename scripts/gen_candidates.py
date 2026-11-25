#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate N candidate clips per prompt for DPO pair mining (GPU, resumable).

Layout produced:

    <out>/<prompt_id>/cand_<k>.mp4        (k = 0..N-1, seed = seed_base + k)
    <out>/<prompt_id>/manifest.jsonl

Existing probe-valid MP4s are skipped, so reruns resume cheaply.

    python scripts/gen_candidates.py --model ti2v-5b --out dpo_out/candidates \
        --candidates 4 --limit 5        # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.inference.config import MODELS, frames_for_seconds  # noqa: E402
from theama.inference.mux import probe_mp4  # noqa: E402


def probe_ok(path: Path, spec) -> bool:
    try:
        info = probe_mp4(str(path))
    except Exception:
        return False
    return (
        info["video_codec"] == "h264"
        and abs(info["fps"] - spec.fps) <= 0.5
        and info["duration"] >= spec.native_seconds - 0.5
        and info["has_audio"]
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ti2v-5b", choices=sorted(MODELS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "eval" / "t2v_v1.jsonl"))
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    spec = MODELS[args.model]
    prompts = [
        json.loads(line)
        for line in Path(args.prompts).read_text().splitlines()
        if line.strip()
    ]
    if args.limit:
        prompts = prompts[: args.limit]

    from theama.inference.t2v import generate_frames, load_pipeline, save_frames
    from theama.inference.mux import mux_mp4

    pipe, _ = load_pipeline(args.model)  # returns (pipe, spec)
    out_root = Path(args.out)
    failures, made = [], 0
    for item in prompts:
        pid, text = item["id"], item["prompt"]
        pdir = out_root / pid
        pdir.mkdir(parents=True, exist_ok=True)
        for k in range(args.candidates):
            mp4 = pdir / f"cand_{k}.mp4"
            if mp4.exists() and probe_ok(mp4, spec):
                continue
            seed = args.seed_base + k
            t0 = time.time()
            try:
                frames = generate_frames(
                    pipe, spec, text, seed=seed,
                    num_frames=frames_for_seconds(spec, spec.native_seconds),
                )
                fdir = pdir / f"frames_{k}"
                save_frames(frames, str(fdir))
                mux_mp4(str(fdir), str(mp4), fps=spec.fps)
                for f in sorted(fdir.glob("*.png")):
                    f.unlink()
                fdir.rmdir()
                if not probe_ok(mp4, spec):
                    raise RuntimeError("probe failed after mux")
                made += 1
                with (pdir / "manifest.jsonl").open("a") as fh:
                    fh.write(json.dumps({
                        "id": pid, "candidate": k, "seed": seed,
                        "seconds": round(time.time() - t0, 1),
                    }) + "\n")
            except Exception as exc:  # keep mining; report at the end
                failures.append({"id": pid, "candidate": k, "error": str(exc)})
                print(f"[FAIL] {pid} cand {k}: {exc}", flush=True)

    summary = {
        "model": args.model,
        "candidates_per_prompt": args.candidates,
        "prompts": len(prompts),
        "clips_made_this_run": made,
        "failures": failures,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "candidates_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

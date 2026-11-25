#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M1 GPU one-click: generate the fixed 50-prompt eval set to MP4.

Local GPU usage:

    pip install -e ".[dev]"
    python scripts/gen_eval_set.py --model ti2v-5b --out eval_out/ti2v-5b
    python scripts/gen_eval_set.py --model t2v-a14b --out eval_out/t2v-a14b

Resumable: existing probe-valid MP4s in --out are skipped, so a crashed run
continues where it stopped. Exit code is nonzero unless every prompt in the
eval set produced a probe-valid MP4.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.inference.config import MODELS  # noqa: E402
from theama.inference.mux import mux_mp4, probe_mp4  # noqa: E402


def load_eval_set(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def probe_ok(info: dict, fps: float, min_seconds: float) -> bool:
    return (
        info["video_codec"] == "h264"
        and abs(info["fps"] - fps) < 0.5
        and info["duration"] >= min_seconds
        and info["has_audio"]
        and info["audio_codec"] == "aac"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ti2v-5b", choices=sorted(MODELS))
    ap.add_argument("--eval-set", default=str(ROOT / "eval" / "t2v_v1.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only first N prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-path", default=None,
                    help="local Diffusers checkout; default resolves the HF repo id")
    ap.add_argument("--no-offload", action="store_true",
                    help="keep the whole pipeline on GPU (fast, needs 40GB+ for A14B)")
    args = ap.parse_args()

    rows = load_eval_set(Path(args.eval_set))
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    expected_fps = spec.fps
    min_seconds = spec.native_seconds - 0.5

    pending = []
    for row in rows:
        target = out_dir / f"{row['id']}.mp4"
        if target.exists():
            try:
                if probe_ok(probe_mp4(target), expected_fps, min_seconds):
                    print(f"[skip] {row['id']} already OK")
                    continue
            except Exception:
                pass
            target.unlink()
        pending.append(row)

    pipe = None
    if pending:
        from theama.inference.i2v import generate_frames_i2v, load_i2v_pipeline
        from theama.inference.t2v import generate_frames, load_pipeline, save_frames

        is_i2v = spec.task == "i2v"
        if is_i2v:
            pipe, _ = load_i2v_pipeline(args.model, offload=not args.no_offload)
        else:
            pipe, _ = load_pipeline(
                args.model, offload=not args.no_offload, model_path=args.model_path
            )

    manifest_path = out_dir / "manifest.jsonl"
    for row in pending:
        t0 = time.time()
        print(f"[gen ] {row['id']} ({row['category']}): {row['prompt'][:60]}...")
        if spec.task == "i2v":
            raise SystemExit(
                "I2V eval needs conditioning frames; T2V eval set is text-only. "
                "Use --model ti2v-5b (its TI2V weights run text-only) for now."
            )
        frames = generate_frames(pipe, spec, row["prompt"], seed=args.seed)
        with tempfile.TemporaryDirectory() as tmp:
            save_frames(frames, tmp)
            target = out_dir / f"{row['id']}.mp4"
            mux_mp4(tmp, target, fps=spec.fps)
        rec = {
            "id": row["id"],
            "model": args.model,
            "seed": args.seed,
            "seconds_wall": round(time.time() - t0, 1),
            "probe": probe_mp4(target),
        }
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[ ok ] {row['id']} -> {target.name} "
              f"({rec['seconds_wall']}s, {rec['probe']['duration']:.2f}s)")

    # Final gate: every row must have a probe-valid MP4.
    failed = []
    for row in rows:
        target = out_dir / f"{row['id']}.mp4"
        try:
            if not probe_ok(probe_mp4(target), expected_fps, min_seconds):
                failed.append(row["id"])
        except Exception:
            failed.append(row["id"])
    summary = {
        "model": args.model,
        "total": len(rows),
        "ok": len(rows) - len(failed),
        "failed": failed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

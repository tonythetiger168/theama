#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M7a GPU one-click: generate the CJK T2I eval seed set to PNG.

Local GPU usage:

    pip install -e ".[dev]"
    python scripts/gen_cjk_images.py --model qwen-image-2512 --out eval_out/m7a/qwen
    python scripts/gen_cjk_images.py --model z-image-turbo --out eval_out/m7a/zimage

Resumable: existing readable PNGs in --out are skipped, so a crashed run
continues where it stopped. Exit code is nonzero unless every prompt in the
eval set produced a readable PNG. Rows and the final gate use the strict
loader in ``theama.image`` (the same schema CI validates).

Both defaults are the Apache-2.0 M7a bases from configs/theama_base.yaml;
keep the mapping in sync with that file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.image import load_t2i_eval  # noqa: E402

# Keep in sync with configs/theama_base.yaml (image_backbone / image_backbone_fast).
IMAGE_MODELS = {
    "qwen-image-2512": "Qwen/Qwen-Image-2512",       # quality path
    "z-image-turbo": "Tongyi-MAI/Z-Image-Turbo",     # fast dev/draft path
}


def png_ok(path: Path) -> bool:
    """A generated image counts only if PIL can fully decode it."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen-image-2512", choices=sorted(IMAGE_MODELS))
    ap.add_argument("--eval-set", default=str(ROOT / "eval" / "cjk_t2i_v1.jsonl"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="only first N prompts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=None,
                    help="default: 30 (quality) / 8 (turbo)")
    ap.add_argument("--guidance", type=float, default=None,
                    help="default: 3.5 (quality) / 0.0 (turbo)")
    ap.add_argument("--no-offload", action="store_true",
                    help="keep the whole pipeline on GPU (fast, needs more VRAM)")
    args = ap.parse_args()

    rows = load_t2i_eval(args.eval_set)
    if args.limit:
        rows = rows[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_id = IMAGE_MODELS[args.model]
    is_turbo = args.model == "z-image-turbo"
    steps = args.steps if args.steps is not None else (8 if is_turbo else 30)
    guidance = args.guidance if args.guidance is not None else (0.0 if is_turbo else 3.5)

    pending = []
    for row in rows:
        target = out_dir / f"{row.id}.png"
        if png_ok(target):
            print(f"[skip] {row.id} already OK")
            continue
        if target.exists():
            target.unlink()  # partial write from a crashed run
        pending.append(row)

    pipe = None
    if pending:
        import torch
        from diffusers import AutoPipelineForText2Image

        pipe = AutoPipelineForText2Image.from_pretrained(
            repo_id, torch_dtype=torch.bfloat16
        )
        if args.no_offload:
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()

    manifest_path = out_dir / "manifest.jsonl"
    for row in pending:
        t0 = time.time()
        print(f"[gen ] {row.id} ({row.category}): {row.prompt[:60]}...")
        kwargs = dict(
            prompt=row.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=steps,
            generator=None if args.seed is None else _generator(args.seed, row.id),
        )
        # Guidance-distilled turbo models take guidance_scale=0; Qwen-Image
        # uses a positive CFG. diffusers rejects unexpected kwargs on some
        # pipelines, so only pass what the loaded pipeline accepts.
        if "guidance_scale" in pipe_signature(pipe):
            kwargs["guidance_scale"] = guidance
        image = pipe(**kwargs).images[0]
        target = out_dir / f"{row.id}.png"
        image.save(target)
        rec = {
            "id": row.id,
            "model": args.model,
            "repo_id": repo_id,
            "seed": args.seed,
            "steps": steps,
            "guidance": guidance,
            "width": args.width,
            "height": args.height,
            "seconds_wall": round(time.time() - t0, 1),
        }
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[ ok ] {row.id} -> {target.name} ({rec['seconds_wall']}s)")

    # Final gate: every row must have a readable PNG.
    failed = [row.id for row in rows if not png_ok(out_dir / f"{row.id}.png")]
    summary = {
        "model": args.model,
        "repo_id": repo_id,
        "total": len(rows),
        "ok": len(rows) - len(failed),
        "failed": failed,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if failed else 0


def _generator(seed: int, prompt_id: str):
    import torch

    # Per-prompt derived seed keeps the run deterministic but avoids one
    # identical noise field across all 24 prompts.
    return torch.Generator(device="cuda").manual_seed(seed + abs(hash(prompt_id)) % 1000)


def pipe_signature(pipe) -> set[str]:
    """Parameter names accepted by the loaded pipeline's __call__."""
    import inspect

    try:
        return set(inspect.signature(pipe.__call__).parameters)
    except (TypeError, ValueError):
        return set()


if __name__ == "__main__":
    raise SystemExit(main())

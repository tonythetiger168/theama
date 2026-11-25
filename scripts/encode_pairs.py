#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-encode DPO pair videos into Wan VAE latents (GPU).

Keeps the VAE out of the DPO hot loop. Applies the Wan 2.2 TI2V-5B latent
normalization (per-channel latents_mean / latents_std from the pipeline) so
the saved tensors are exactly what the DiT expects as x_0.

Resume is on by default: pairs whose win/lose .pt files already exist and
validate (loadable, finite, 5D) are skipped, corrupt half-written files are
re-encoded, and pairs that previously failed (manifest "error" rows) are
retried. The cache is tied to the model via out_dir/meta.json — switching
--model invalidates it automatically (latent normalization is VAE-specific).
Use --no-resume to force a full re-encode and --strict to abort on the
first failure. A pairs_latents.summary.json sidecar (encoded / reused /
failed counts) is written on every run. When everything is cached the VAE
is never loaded, so resume checks are cheap.

With --max-retries N, failed pairs are counted in a persistent
pair_failures.json ledger: a pair that has failed N times is skipped
("retry budget exhausted") until you clear the ledger with
--reset-failures. Without the flag, retries stay unlimited and no ledger
file is written (previous behaviour).

    python scripts/encode_pairs.py --pairs dpo_out/pairs.jsonl \
        --model ti2v-5b --out dpo_out/latents
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def encode_video(vae, video_path: str):
    """MP4 -> normalized latent tensor (1, C, F, H//8, W//8), float32.

    `vae` must already be on cuda/float32 (hoisted out of the pair loop).
    """
    import numpy as np
    import torch
    from diffusers.utils import load_video

    frames = load_video(video_path)  # list[PIL.Image]
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # 1,C,F,H,W
    x = (x / 127.5 - 1.0).to("cuda", torch.float32)

    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample()
    mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
    latent = (latent - mean.to(latent.device)) * (1.0 / std).to(latent.device)
    return latent.cpu()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", default="ti2v-5b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="encode only 2 pairs")
    ap.add_argument("--limit", type=int, default=None,
                    help="encode at most N pairs")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the existing cache and re-encode everything")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first pair that fails to encode")
    ap.add_argument("--max-retries", type=int, default=None, metavar="N",
                    help="cap the per-pair encode retry budget; pairs "
                         "that failed N times are skipped until the ledger "
                         "is cleared (default: unlimited, no ledger file)")
    ap.add_argument("--reset-failures", action="store_true",
                    help="clear the pair_failures.json ledger before planning, "
                         "giving budget-exhausted pairs another chance")
    args = ap.parse_args()

    pairs = [
        json.loads(line)
        for line in Path(args.pairs).read_text().splitlines()
        if line.strip()
    ]
    if args.smoke:
        pairs = pairs[:2]
    if args.limit is not None:
        pairs = pairs[: args.limit]
    if not pairs:
        print("no pairs to encode", file=sys.stderr)
        return 1

    from theama.data.latent_cache import (
        plan_encoding,
        reset_failure_ledger,
        run_encoding,
        save_latent_atomic,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reset_failures and reset_failure_ledger(out):
        print(f"cleared failure ledger in {out}")

    plan = plan_encoding(
        pairs, out, args.model,
        resume=not args.no_resume, max_retries=args.max_retries,
    )
    if plan.cache_invalidated:
        print(f"cache model mismatch in {out} — re-encoding everything")
    print(
        f"{len(pairs)} pairs: {len(plan.reusable)} cached, {len(plan.todo)} to encode"
        + (f", {len(plan.skipped)} skipped (retry budget exhausted)"
           if plan.skipped else "")
    )

    encode_fn = None
    if plan.todo:
        from diffusers import WanPipeline
        import torch

        from theama.inference.config import MODELS

        spec = MODELS[args.model]
        pipe = WanPipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16)
        vae = pipe.vae.to("cuda", torch.float32)

        def encode_fn(video_path, dest):  # noqa: F811
            save_latent_atomic(encode_video(vae, video_path), dest)

    def progress(msg):
        print(msg, flush=True)

    summary = run_encoding(
        plan, encode_fn, strict=args.strict, progress=progress,
        max_retries=args.max_retries,
    )
    if summary["failed"]:
        hint = "rerun to retry"
        if args.max_retries is not None:
            hint = ("rerun to retry — pairs that exhaust the budget are "
                    "skipped until --reset-failures")
        print(
            f"{summary['failed']} pair(s) failed to encode ({hint})",
            file=sys.stderr,
        )
    print(f"wrote manifest -> {plan.manifest_path}")
    print(f"summary: {json.dumps(summary)} -> {plan.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

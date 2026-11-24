#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-encode DPO pair videos into Wan VAE latents (GPU).

Keeps the VAE out of the DPO hot loop. Applies the Wan 2.2 TI2V-5B latent
normalization (per-channel latents_mean / latents_std from the pipeline) so
the saved tensors are exactly what the DiT expects as x_0.

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


def encode_video(pipe, video_path: str):
    """MP4 -> normalized latent tensor (1, C, F, H//8, W//8), float32."""
    import numpy as np
    import torch
    from diffusers.utils import load_video

    frames = load_video(video_path)  # list[PIL.Image]
    arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
    x = torch.from_numpy(arr).permute(3, 0, 1, 2).unsqueeze(0)  # 1,C,F,H,W
    x = (x / 127.5 - 1.0).to("cuda", torch.float32)

    vae = pipe.vae.to("cuda", torch.float32)
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
    args = ap.parse_args()

    from diffusers import WanPipeline
    import torch

    from theama.inference.config import MODELS

    spec = MODELS[args.model]
    pipe = WanPipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16)

    pairs = [
        json.loads(line)
        for line in Path(args.pairs).read_text().splitlines()
        if line.strip()
    ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, pair in enumerate(pairs):
        win = out / f"pair{i:04d}_win.pt"
        lose = out / f"pair{i:04d}_lose.pt"
        if not win.exists():
            torch.save(encode_video(pipe, pair["win_video"]), win)
        if not lose.exists():
            torch.save(encode_video(pipe, pair["lose_video"]), lose)
        rows.append({
            "prompt": pair["prompt"],
            "win_latent": str(win),
            "lose_latent": str(lose),
        })
        print(f"[{i + 1}/{len(pairs)}] {pair['prompt_id']}", flush=True)

    manifest = out / "pairs_latents.jsonl"
    with manifest.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} encoded pairs -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

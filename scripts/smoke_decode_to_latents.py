#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Straight-through decode -> VAE/DAC encode smoke (GPU).

One pass over the M2 data path end to end: manifest -> AVDataset ffmpeg
decode -> build_av_dataloader -> Wan VAE (video) + DAC (audio) -> per-sample
.pt latents + av_latents.summary.json. This is a contract smoke (shapes and
finiteness), not a quality gate and not the P1b cache — no resume, small
--limit runs are the intended usage.

    python scripts/smoke_decode_to_latents.py \
        --manifest data/av/manifest.jsonl --out smoke_out/av_latents \
        --limit 4 --batch-size 2

The video leg applies the same Wan latent normalization as
scripts/encode_pairs.py (per-channel latents_mean / latents_std), so saved
tensors are directly comparable with the DPO pair cache. The audio leg pads
to the DAC hop and encodes stereo as mid+side (latent_dim 2048 tokens).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="AV manifest (jsonl)")
    ap.add_argument("--out", required=True, help="output dir for latents + summary")
    ap.add_argument("--model", default="ti2v-5b", help="Wan model key (video VAE)")
    ap.add_argument("--codec", default="dac_44k", help="audio codec registry name")
    ap.add_argument("--seconds", type=int, default=5)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--width", type=int, default=432)
    ap.add_argument("--sample-rate", type=int, default=44100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="encode at most N samples")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first clip that fails to decode")
    args = ap.parse_args()

    import torch

    from theama.data.av_latent_smoke import pad_to_hop, run_decode_encode_smoke
    from theama.data.dataset import AVDataset
    from theama.inference.config import MODELS
    from theama.model.audio_codec import encode_stereo, get_codec_spec

    dataset = AVDataset(
        args.manifest,
        seconds=args.seconds,
        fps=args.fps,
        sample_rate=args.sample_rate,
        height=args.height,
        width=args.width,
    )
    if not len(dataset):
        print("manifest yielded zero usable samples", file=sys.stderr)
        return 1

    from diffusers import WanPipeline

    spec = MODELS[args.model]
    pipe = WanPipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16)
    vae = pipe.vae.to("cuda", torch.float32)

    def video_encode_fn(video):
        """(B,3,T,H,W) in [-1,1] -> Wan-normalized latents (B,C,F,H',W')."""
        x = video.to("cuda", torch.float32)
        with torch.no_grad():
            latent = vae.encode(x).latent_dist.sample()
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1)
        return (latent - mean.to(latent.device)) * (1.0 / std).to(latent.device)

    from theama.model.dac_codec import DACAudioCodec

    codec_spec = get_codec_spec(args.codec)
    codec = DACAudioCodec.from_pretrained(device="cuda")

    def audio_encode_fn(audio):
        """(B,2,N) stereo -> mid+side DiT tokens (B,T_lat,2*latent_dim)."""
        wave = pad_to_hop(audio, codec_spec.hop_length).to("cuda", torch.float32)
        return encode_stereo(codec, wave)

    summary = run_decode_encode_smoke(
        dataset,
        video_encode_fn,
        audio_encode_fn,
        args.out,
        limit=args.limit,
        batch_size=args.batch_size,
        skip_broken=not args.strict,
    )
    print(f"summary: {json.dumps(summary)} -> {Path(args.out) / 'av_latents.summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

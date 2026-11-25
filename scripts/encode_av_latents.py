#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-encode AV manifest media into Wan VAE + DAC latents (GPU).

P1b production cache: keeps both encoders out of the joint-training hot
loop. Video latents get the same Wan 2.2 latent normalization as
scripts/encode_pairs.py (per-channel latents_mean / latents_std); audio is
padded to the DAC hop and encoded stereo as mid+side tokens
(1, T_lat, 2*latent_dim).

Resume is on by default (theama.data.av_latent_cache): samples whose
sample{index:04d}_{video,audio}.pt files already validate (video 5D / audio
3D, finite) are skipped, error rows are retried, corrupt half-written
files are re-encoded, and a directory produced by
scripts/smoke_decode_to_latents.py seeds this cache unchanged. The cache
is tied to --model via out_dir/meta.json — switching models invalidates it.
A sample is only reused while the manifest row at its index still points at
the same (video, audio) files. Use --no-resume to force a full re-encode
and --strict to abort on the first failure. An av_latents.summary.json
sidecar (encoded / reused / failed) is written on every run. When
everything is cached neither encoder is loaded, so resume checks are cheap.

With --max-retries N, failed samples are counted in a persistent
av_failures.json ledger: a sample that has failed N times is skipped
("retry budget exhausted") until you clear the ledger with
--reset-failures. Without the flag, retries stay unlimited and no ledger
file is written (previous behaviour).

    python scripts/encode_av_latents.py \
        --manifest data/av/manifest.jsonl --out data/av/latents \
        --model ti2v-5b --codec dac_44k
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
    ap.add_argument("--smoke", action="store_true", help="encode only 2 samples")
    ap.add_argument("--limit", type=int, default=None, help="encode at most N samples")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the existing cache and re-encode everything")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first sample that fails to encode")
    ap.add_argument("--max-retries", type=int, default=None, metavar="N",
                    help="cap the per-sample encode retry budget; samples "
                         "that failed N times are skipped until the ledger "
                         "is cleared (default: unlimited, no ledger file)")
    ap.add_argument("--reset-failures", action="store_true",
                    help="clear the av_failures.json ledger before planning, "
                         "giving budget-exhausted samples another chance")
    args = ap.parse_args()

    from theama.data.dataset import load_manifest

    report = load_manifest(args.manifest)  # tolerant: errors collected, not fatal
    samples = [
        {
            "video": str(s.video_path),
            "audio": str(s.audio_path),
            "caption": s.caption,
            "source": s.source,
        }
        for s in report.samples
    ]
    print(f"manifest: {report.summary()}")
    if args.smoke:
        samples = samples[:2]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print("manifest yielded zero usable samples", file=sys.stderr)
        return 1

    from theama.data.av_latent_cache import (
        plan_av_encoding,
        reset_failure_ledger,
        run_av_encoding,
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.reset_failures and reset_failure_ledger(out):
        print(f"cleared failure ledger in {out}")

    plan = plan_av_encoding(
        samples, out, args.model,
        resume=not args.no_resume, max_retries=args.max_retries,
    )
    if plan.cache_invalidated:
        print(f"cache model mismatch in {out} — re-encoding everything")
    print(
        f"{len(samples)} samples: {len(plan.reusable)} cached, "
        f"{len(plan.todo)} to encode"
        + (f", {len(plan.skipped)} skipped (retry budget exhausted)"
           if plan.skipped else "")
    )

    encode_fn = None
    if plan.todo:
        import torch

        from theama.data.av_latent_smoke import pad_to_hop
        from theama.data.decode import decode_audio, decode_video
        from theama.data.latent_cache import save_latent_atomic
        from theama.inference.config import MODELS
        from theama.model.audio_codec import encode_stereo, get_codec_spec

        from diffusers import WanPipeline

        from theama.model.dac_codec import DACAudioCodec

        spec = MODELS[args.model]
        pipe = WanPipeline.from_pretrained(spec.repo_id, torch_dtype=torch.bfloat16)
        vae = pipe.vae.to("cuda", torch.float32)
        codec_spec = get_codec_spec(args.codec)
        codec = DACAudioCodec.from_pretrained(device="cuda")

        def encode_fn(sample, video_dest, audio_dest):  # noqa: F811
            """Decode one sample and atomically save both latents."""
            frames = decode_video(
                sample["video"],
                fps=args.fps,
                seconds=args.seconds,
                height=args.height,
                width=args.width,
            )
            x = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
            x = (x.float() / 127.5 - 1.0).to("cuda", torch.float32)
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.sample()
            mean = torch.tensor(vae.config.latents_mean).view(
                1, vae.config.z_dim, 1, 1, 1
            )
            std = torch.tensor(vae.config.latents_std).view(
                1, vae.config.z_dim, 1, 1, 1
            )
            latent = (latent - mean.to(latent.device)) * (
                1.0 / std
            ).to(latent.device)
            save_latent_atomic(latent.cpu(), video_dest)

            wave = decode_audio(
                sample["audio"],
                sample_rate=args.sample_rate,
                seconds=args.seconds,
            )
            wave_t = pad_to_hop(
                torch.from_numpy(wave).unsqueeze(0), codec_spec.hop_length
            ).to("cuda", torch.float32)
            tokens = encode_stereo(codec, wave_t)  # (1, T_lat, 2*latent_dim)
            save_latent_atomic(tokens.cpu(), audio_dest)

    def progress(msg):
        print(msg, flush=True)

    summary = run_av_encoding(
        plan, encode_fn, strict=args.strict, progress=progress,
        max_retries=args.max_retries,
    )
    if summary["failed"]:
        hint = "rerun to retry"
        if args.max_retries is not None:
            hint = ("rerun to retry — samples that exhaust the budget are "
                    "skipped until --reset-failures")
        print(
            f"{summary['failed']} sample(s) failed to encode ({hint})",
            file=sys.stderr,
        )
    print(f"wrote manifest -> {plan.manifest_path}")
    print(f"summary: {json.dumps(summary)} -> {plan.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

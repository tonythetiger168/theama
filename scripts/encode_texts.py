#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pre-encode AV manifest captions into umT5 text embeddings (GPU).

Companion to scripts/encode_av_latents.py: that one caches Wan VAE + DAC
media latents, this one caches the text condition so the joint-training
hot loop loads neither the media encoders nor the text encoder. Each
caption is stored once under a content-addressed name
(text_{sha256[:16]}.pt) as a padded (max_length, D) bf16 tensor; the
manifest rows carry seq_len so training can rebuild the attention mask.

Resume is on by default (theama.data.text_cache): captions whose .pt
already validates (2D finite) are skipped — caption edits land on a new
filename automatically, so stale embeddings are never served; failed
captions are retried; duplicate captions encode once per run. The cache
is tied to --encoder via out_dir/meta.json — switching text encoders
invalidates it. Use --no-resume to force a full re-encode and --strict
to abort on the first failure. A text_latents.summary.json sidecar
(encoded / reused / failed) is written on every run. When everything is
cached the text encoder is not loaded, so resume checks are cheap.

    python scripts/encode_texts.py \
        --manifest data/av/manifest.jsonl --out data/av/texts \
        --model ti2v-5b
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
    ap.add_argument("--out", required=True, help="output dir for embeddings + summary")
    ap.add_argument("--model", default="ti2v-5b",
                    help="Wan model key (supplies the umT5 text encoder repo)")
    ap.add_argument("--encoder", default=None,
                    help="override text encoder repo id (default: --model repo)")
    ap.add_argument("--max-length", type=int, default=512,
                    help="umT5 padded sequence length (Wan default 512)")
    ap.add_argument("--smoke", action="store_true", help="encode only 2 samples")
    ap.add_argument("--limit", type=int, default=None, help="encode at most N samples")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore the existing cache and re-encode everything")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first sample that fails to encode")
    args = ap.parse_args()

    from theama.data.dataset import load_manifest

    report = load_manifest(args.manifest)  # tolerant: errors collected, not fatal
    samples = [{"caption": s.caption} for s in report.samples]
    print(f"manifest: {report.summary()}")
    if args.smoke:
        samples = samples[:2]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        print("manifest yielded zero usable samples", file=sys.stderr)
        return 1

    from theama.data.text_cache import plan_text_encoding, run_text_encoding
    from theama.inference.config import MODELS

    encoder_id = args.encoder or MODELS[args.model].repo_id
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    plan = plan_text_encoding(samples, out, encoder_id, resume=not args.no_resume)
    if plan.cache_invalidated:
        print(f"cache model mismatch in {out} — re-encoding everything")
    print(
        f"{len(samples)} samples: {len(plan.reusable)} cached, "
        f"{len(plan.todo)} to encode"
    )

    encode_fn = None
    if plan.todo:
        import torch
        from transformers import AutoTokenizer, UMT5EncoderModel

        from theama.data.latent_cache import save_latent_atomic

        tokenizer = AutoTokenizer.from_pretrained(encoder_id, subfolder="tokenizer")
        text_encoder = UMT5EncoderModel.from_pretrained(
            encoder_id, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).to("cuda")
        text_encoder.eval()

        def encode_fn(sample, dest):  # noqa: F811
            """Encode one caption; save padded (L, D) embedding; return seq_len."""
            tokens = tokenizer(
                sample["caption"],
                max_length=args.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                emb = text_encoder(**tokens).last_hidden_state[0]  # (L, D)
            seq_len = int(tokens.attention_mask[0].sum().item())
            save_latent_atomic(emb.cpu(), dest)
            return seq_len

    def progress(msg):
        print(msg, flush=True)

    summary = run_text_encoding(plan, encode_fn, strict=args.strict, progress=progress)
    if summary["failed"]:
        print(
            f"{summary['failed']} sample(s) failed to encode (rerun to retry)",
            file=sys.stderr,
        )
    print(f"wrote manifest -> {plan.manifest_path}")
    print(f"summary: {json.dumps(summary)} -> {plan.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

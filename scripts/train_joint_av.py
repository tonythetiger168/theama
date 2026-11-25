#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""P1b/M2 entry point: joint A/V training from the two latent caches.

Prerequisites (GPU, see docs/gpu-acceptance-runbook.md):
  1. scripts/encode_av_latents.py  -> AV cache (av_latents.jsonl)
  2. scripts/encode_texts.py       -> text cache (text_latents.jsonl)

Real run:

    python scripts/train_joint_av.py --config configs/m2_joint_av.yaml \
        --av-cache av_cache/ --text-cache text_cache/ --out runs/m2_run1 \
        --steps 20000 --bridge-steps 2000 --batch-size 2

Quick GPU validation of the whole path (no caches, no Wan weights, ~1 min):

    python scripts/train_joint_av.py --smoke --out runs/smoke

Success prints TRAIN_JOINT_AV_SMOKE_OK. Resume an interrupted run with
``--resume auto`` (latest checkpoint in --out) or ``--resume <path>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _synthetic_batches(
    *, batch_size: int, channels: int, frames: int, hw: int, ta: int, latent: int, lt: int, text_dim: int
) -> list[dict]:
    """Fixed synthetic joint batches exercising the real 5-D Wan-latent path."""
    import torch

    gen = torch.Generator().manual_seed(0)
    batches = []
    for _ in range(4):
        seq_lens = torch.randint(1, lt + 1, (batch_size,), generator=gen)
        mask = torch.arange(lt)[None, :] < seq_lens[:, None]
        batches.append(
            {
                "video_tokens": torch.randn(batch_size, channels, frames, hw, hw, generator=gen),
                "audio_latent": torch.randn(batch_size, ta, latent, generator=gen),
                "video_cond": torch.randn(batch_size, lt, text_dim, generator=gen),
                "text_mask": mask,
            }
        )
    return batches


def _smoke(args: argparse.Namespace) -> int:
    import torch

    from theama.model.audio_dit import AudioDiT
    from theama.train.joint_av_runner import (
        JointAVRunner,
        RunnerConfig,
        StubVideoBackbone,
        build_cond_pooler,
    )
    from theama.train.train_joint_av import JointAVTrainer

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    dims = dict(dim=64, layers=4, heads=4, video_dim=4, bridge_every=2, latent_dim=32)
    text_dim = 128
    audio = AudioDiT(**dims)
    build_cond_pooler(audio, text_dim)
    trainer = JointAVTrainer(
        StubVideoBackbone(),
        audio,
        {"phase": "bridge", "lr": 1e-3, "bridge_lora_rank": 4, "bridge_lora_alpha": 8},
    )
    batches = _synthetic_batches(
        batch_size=2, channels=dims["video_dim"], frames=2, hw=4,
        ta=6, latent=dims["latent_dim"], lt=8, text_dim=text_dim,
    )
    cfg = RunnerConfig(
        out_dir=args.out,
        max_steps=4,
        bridge_steps=2,  # exercises the scheduled bridge -> full switch
        log_every=1,
        save_every=2,
        device=device,
        dtype=args.dtype,
    )
    summary = JointAVRunner(trainer, batches, cfg).train()
    assert summary["final_phase"] == "full", summary
    # Resume path: +2 steps from ckpt_latest.
    audio2 = AudioDiT(**dims)
    build_cond_pooler(audio2, text_dim)
    trainer2 = JointAVTrainer(
        StubVideoBackbone(),
        audio2,
        {"phase": "bridge", "lr": 1e-3, "bridge_lora_rank": 4, "bridge_lora_alpha": 8},
    )
    cfg2 = RunnerConfig(
        out_dir=args.out, max_steps=6, log_every=1, save_every=2,
        device=device, dtype=args.dtype, resume="auto",
    )
    summary2 = JointAVRunner(trainer2, batches, cfg2).train()
    assert summary2["resumed_from"] == 4 and summary2["steps_completed"] == 6, summary2
    print(json.dumps({"smoke": summary["final_metrics"], "resume": summary2["resumed_from"]}))
    print("TRAIN_JOINT_AV_SMOKE_OK")
    return 0


def _real(args: argparse.Namespace) -> int:
    import torch
    import yaml

    from theama.data.joint_latent_loader import (
        build_joint_latent_dataloader,
        load_joint_latent_records,
    )
    from theama.model.audio_dit import AudioDiT
    from theama.model.wan_adapter import WanVideoAdapter
    from theama.train.joint_av_runner import (
        JointAVRunner,
        RunnerConfig,
        audio_dit_kwargs,
        build_cond_pooler,
        text_dim_from_config,
        trainer_config,
    )
    from theama.train.train_joint_av import JointAVTrainer

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    video_cfg = cfg["model"]["video"]
    print(f"loading frozen Wan backbone: {video_cfg['adapter']}", flush=True)
    video = WanVideoAdapter.from_pretrained(
        video_cfg["adapter"],
        model_path=args.video_weights,
        device=device,
        dtype=torch.bfloat16,
    )
    audio = AudioDiT(**audio_dit_kwargs(cfg["model"]["audio"]))
    build_cond_pooler(audio, text_dim_from_config(cfg))
    trainer = JointAVTrainer(video, audio, trainer_config(cfg))

    records = load_joint_latent_records(args.av_cache, args.text_cache)
    print(f"joint cache: {len(records)} samples (dropped: {records.dropped})", flush=True)
    loader = build_joint_latent_dataloader(
        records,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cond_pooler=None,  # cond is pooled on-device inside the runner
    )
    run_cfg = RunnerConfig(
        out_dir=args.out,
        max_steps=args.steps,
        bridge_steps=args.bridge_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        device=device,
        dtype=args.dtype,
        resume=args.resume,
    )
    summary = JointAVRunner(trainer, loader, run_cfg).train()
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs" / "m2_joint_av.yaml"))
    ap.add_argument("--av-cache", help="AV latent cache dir (av_latents.jsonl)")
    ap.add_argument("--text-cache", help="text latent cache dir (text_latents.jsonl)")
    ap.add_argument("--out", required=True, help="run dir (checkpoints + metrics.jsonl)")
    ap.add_argument("--steps", type=int, default=1000, help="max optimizer steps")
    ap.add_argument("--bridge-steps", type=int, default=None,
                    help="switch bridge -> full at this step (default: stay in config phase)")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--device", default=None, help="default: cuda if available else cpu")
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None, help='"auto" or checkpoint path')
    ap.add_argument("--video-weights", default=None,
                    help="local Wan Diffusers path (default: resolve the registered repo id)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny synthetic run, no caches/Wan weights; prints TRAIN_JOINT_AV_SMOKE_OK")
    args = ap.parse_args()

    if args.smoke:
        return _smoke(args)
    if not args.av_cache or not args.text_cache:
        ap.error("--av-cache and --text-cache are required without --smoke")
    return _real(args)


if __name__ == "__main__":
    raise SystemExit(main())

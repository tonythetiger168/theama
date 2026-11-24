#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M2 real-weight smoke: official DAC and frozen Wan TI2V-5B.

Examples (weights stay user-side):

    python scripts/smoke_m2_real_weights.py dac \
      --weights checkpoints/wan2.2/dac_44khz/weights.pth --out smoke_out/dac

    python scripts/smoke_m2_real_weights.py wan \
      --model-path checkpoints/wan2.2/ti2v-5b --out smoke_out/wan_ti2v_5b

Each command writes a JSON report and exits nonzero on a failed contract. DAC
also saves input.wav and reconstructed.wav for an optional listening check.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.train.real_weight_smoke import (  # noqa: E402
    reset_cuda_peak,
    run_dac_smoke,
    run_wan_smoke,
)


def _write_report(out: Path, name: str, report: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    target = out / name
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def _dac(args: argparse.Namespace) -> int:
    from theama.model.dac_codec import DACAudioCodec

    reset_cuda_peak(args.device)
    codec = DACAudioCodec.from_pretrained(args.weights, device=args.device)
    report, waveform, restored = run_dac_smoke(
        codec, device=args.device, seconds=args.seconds
    )

    import soundfile as sf

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / "input.wav", waveform[0].detach().cpu().numpy().T, codec.spec.sample_rate)
    sf.write(out / "reconstructed.wav", restored[0].detach().cpu().numpy().T, codec.spec.sample_rate)
    report["artifacts"] = ["input.wav", "reconstructed.wav"]
    target = _write_report(out, "m2_dac_smoke.json", report)
    print(f"M2_DAC_SMOKE_OK report={target}")
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _wan(args: argparse.Namespace) -> int:
    from theama.model.wan_adapter import WanVideoAdapter

    dtype = getattr(torch, args.dtype)
    reset_cuda_peak(args.device)
    video = WanVideoAdapter.from_pretrained(
        "ti2v-5b",
        model_path=args.model_path,
        device=args.device,
        dtype=dtype,
    )
    report = run_wan_smoke(
        video,
        device=args.device,
        dtype=dtype,
        audio_frames=args.audio_frames,
        text_length=args.text_length,
        latent_frames=args.latent_frames,
        latent_height=args.latent_height,
        latent_width=args.latent_width,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    target = _write_report(Path(args.out), "m2_wan_smoke.json", report)
    print(f"M2_WAN_SMOKE_OK report={target}")
    print(json.dumps(report, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    dac = commands.add_parser("dac", help="official DAC stereo encode/decode")
    dac.add_argument("--weights", help="local DAC weights.pth; omit to resolve user-side")
    dac.add_argument("--out", required=True)
    dac.add_argument("--device", default="cuda")
    dac.add_argument("--seconds", type=float, default=1.0)
    dac.set_defaults(handler=_dac)

    wan = commands.add_parser("wan", help="TI2V-5B tiny 5-D A/V training step")
    wan.add_argument("--model-path", help="local Diffusers repo; omit to resolve user-side")
    wan.add_argument("--out", required=True)
    wan.add_argument("--device", default="cuda")
    wan.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    wan.add_argument("--audio-frames", type=int, default=8)
    wan.add_argument("--text-length", type=int, default=16)
    wan.add_argument("--latent-frames", type=int, default=1)
    wan.add_argument("--latent-height", type=int, default=2)
    wan.add_argument("--latent-width", type=int, default=2)
    wan.add_argument("--learning-rate", type=float, default=1e-4)
    wan.add_argument("--seed", type=int, default=42)
    wan.set_defaults(handler=_wan)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())

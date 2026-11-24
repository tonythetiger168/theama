# SPDX-License-Identifier: Apache-2.0
"""M1 inference CLI: prompt -> frames -> MP4 (H.264 + silent AAC).

    python -m theama.infer.pipeline --prompt "夜市霓虹" --out out.mp4
    python -m theama.infer.pipeline --model t2v-a14b --prompt "..." --out out.mp4 --no-offload
    python -m theama.infer.pipeline --image keyframe.png --prompt "..." --out out.mp4
"""

from __future__ import annotations

import argparse
import sys
import tempfile

from theama.infer.config import MODELS, frames_for_seconds
from theama.infer.mux import mux_mp4, probe_mp4


def main() -> int:
    ap = argparse.ArgumentParser(prog="theama.infer.pipeline")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=None,
                    help="clip length; default = model native (5s)")
    ap.add_argument("--model", default="ti2v-5b", choices=sorted(MODELS),
                    help="ti2v-5b = 24GB dev path; t2v-a14b/i2v-a14b = full backbone")
    ap.add_argument("--image", default=None, help="conditioning frame -> I2V mode")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-offload", action="store_true")
    args = ap.parse_args()

    spec = MODELS[args.model]
    num_frames = frames_for_seconds(spec, args.seconds) if args.seconds else spec.num_frames

    if args.image:
        from theama.infer.i2v import generate_frames_i2v, load_i2v_pipeline

        pipe, _ = load_i2v_pipeline(args.model, offload=not args.no_offload)
        frames = generate_frames_i2v(pipe, spec, args.image, args.prompt,
                                     seed=args.seed, num_frames=num_frames)
    else:
        from theama.infer.t2v import generate_frames, load_pipeline

        pipe, _ = load_pipeline(args.model, offload=not args.no_offload)
        frames = generate_frames(pipe, spec, args.prompt,
                                 seed=args.seed, num_frames=num_frames)

    from theama.infer.t2v import save_frames

    with tempfile.TemporaryDirectory() as tmp:
        save_frames(frames, tmp)
        mux_mp4(tmp, args.out, fps=spec.fps)
    info = probe_mp4(args.out)
    print(f"wrote {args.out}: {info['width']}x{info['height']} "
          f"{info['duration']:.2f}s @ {info['fps']:.0f}fps, audio={info['audio_codec']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

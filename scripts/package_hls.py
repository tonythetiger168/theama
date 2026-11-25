#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Package a finished film MP4 into an HLS fMP4/CMAF playback bundle (M4).

Usage:
    python scripts/package_hls.py master.mp4 --out-dir dist/film42 [--hdr hdr10]

Produces <out-dir>/<rung>/{init.mp4,seg_*.m4s,index.m3u8} per ladder rung
(4K HDR AV1 -> 1080p HEVC -> 720p/480p H.264; rungs above the source are
dropped, never upscaled) plus master.m3u8, then ffprobe-verifies every
variant. Exit 0 only when every rung verifies; deterministic JSON summary
via --json-out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theama.inference.hls import package_hls, summary_json  # noqa: E402
from theama.inference.mux import MuxError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="master MP4 to package")
    parser.add_argument("--out-dir", required=True, help="output bundle directory")
    parser.add_argument("--hdr", choices=("hdr10", "hlg"), default=None,
                        help="tag the top rung as HDR10/HLG (source must be 10-bit)")
    parser.add_argument("--hls-time", type=int, default=4,
                        help="target segment duration in seconds (default 4)")
    parser.add_argument("--json-out", help="also write the JSON summary here")
    args = parser.parse_args(argv)

    try:
        summary = package_hls(args.input, args.out_dir, hdr=args.hdr,
                              hls_time=args.hls_time)
    except MuxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = summary_json(summary)
    for variant in summary["variants"]:
        print(f"  {variant['name']}: {variant['width']}x{variant['height']} "
              f"{variant['vcodec']} bw={variant['bandwidth']} "
              f"segments={variant['segments']}")
    print(f"master: {summary['master']}  verified: {summary['verified']}")
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

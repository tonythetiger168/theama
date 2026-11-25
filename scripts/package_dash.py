#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Add a DASH manifest.mpd to an existing HLS fMP4/CMAF bundle (M4).

Usage:
    python scripts/package_dash.py dist/film42 [--json-out dash.json]

Reads a package_hls.py output directory (master.m3u8 + per-rung fMP4
variants), cross-checks every variant playlist against the master
declarations and the files on disk, writes a deterministic manifest.mpd
referencing the SAME segments (no re-encode — the 2026-11-26 decision is
"同批 segment 兼出 DASH"), then ffprobe-verifies the manifest itself.
Exit 0 only when verification passes; deterministic JSON summary via
--json-out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from theama.inference.dash import package_dash  # noqa: E402
from theama.inference.hls import summary_json  # noqa: E402
from theama.inference.mux import MuxError  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("hls_dir", help="existing package_hls output directory")
    parser.add_argument("--json-out", help="also write the JSON summary here")
    args = parser.parse_args(argv)

    try:
        summary = package_dash(args.hls_dir)
    except MuxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = summary_json(summary)
    for rep in summary["representations"]:
        print(f"  {rep['name']}: {rep['width']}x{rep['height']} "
              f"{rep['vcodec']} bw={rep['bandwidth']} "
              f"segments={rep['segments']}")
    print(f"manifest: {summary['manifest']}  verified: {summary['verified']}")
    if args.json_out:
        Path(args.json_out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

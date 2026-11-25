# SPDX-License-Identifier: Apache-2.0
"""HLS distribution packaging — fMP4/CMAF segments + adaptive ladder (M4).

Decided 2026-11-26 (training-plan M4 / roadmap): every finished film ships
an HLS playback package next to the master MP4 — fMP4 segments (AV1/HEVC in
HLS never ride MPEG-TS) plus a master .m3u8 adaptive ladder:

    4K HDR AV1  ->  1080p HEVC (Safari fallback, mandatory)  ->  720p/480p H.264

Rungs above the source resolution are dropped (never upscale); a source
below 480p gets a single H.264 rung clamped at the source height. HDR10/HLG
tags (from mux.HDR_FORMATS) apply to the top rung only, and only when the
caller says the master is HDR.

Avatar realtime (M6b) deliberately does NOT use this: it goes LL-HLS or
WebRTC. This module is for film VOD distribution only.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from .mux import HDR_FORMATS, VCODECS, MuxError, _require, probe_mp4


# Canonical ladder, high to low. video_kbps/audio_kbps feed both the
# encoder (-b:v) and the master-playlist BANDWIDTH declaration.
LADDER = (
    {"name": "2160p", "height": 2160, "vcodec": "av1", "video_kbps": 12000, "audio_kbps": 192},
    {"name": "1080p", "height": 1080, "vcodec": "hevc", "video_kbps": 6000, "audio_kbps": 192},
    {"name": "720p", "height": 720, "vcodec": "h264", "video_kbps": 3000, "audio_kbps": 128},
    {"name": "480p", "height": 480, "vcodec": "h264", "video_kbps": 1200, "audio_kbps": 128},
)

# RFC 6381 codec strings for EXT-X-STREAM-INF. Fixed per codec: the ladder
# rungs share one profile/level per codec (h264 High, hevc Main10-compatible
# tag for the HDR-capable fallback path, av1 Main 10-bit).
_CODECS_TAG = {
    "h264": "avc1.640028",
    "hevc": "hvc1.1.6.L123.90",
    "av1": "av01.0.13M.10",
}
_AUDIO_TAG = "mp4a.40.2"  # AAC-LC


def build_ladder(width: int, height: int, hdr: str | None = None) -> list[dict]:
    """Pick the rungs for a source of `width`x`height`.

    Rungs taller than the source are dropped (never upscale). If the source
    is below the smallest rung, a single H.264 rung clamped to the source
    height is returned. `hdr` (None / "hdr10" / "hlg") tags the top rung
    only. Widths are derived proportionally and forced even (yuv420).
    """
    if hdr is not None and hdr not in HDR_FORMATS:
        raise MuxError(f"unknown hdr {hdr!r}; expected one of {sorted(HDR_FORMATS)}")
    if width <= 0 or height <= 0:
        raise MuxError(f"invalid source geometry {width}x{height}")
    rungs = [dict(r) for r in LADDER if r["height"] <= height]
    if not rungs:
        rungs = [dict(LADDER[-1], name=f"{height}p", height=height)]
    for i, rung in enumerate(rungs):
        rung["width"] = 2 * round(width * rung["height"] / height / 2)
        rung["bandwidth"] = (rung["video_kbps"] + rung["audio_kbps"]) * 1000
        rung["hdr"] = hdr if i == 0 else None
    return rungs


def codecs_string(rung: dict) -> str:
    """CODECS attribute for one rung: video tag + AAC-LC."""
    return f"{_CODECS_TAG[rung['vcodec']]},{_AUDIO_TAG}"


def render_master_playlist(rungs: list[dict]) -> str:
    """Deterministic master .m3u8 text for the given rungs (high to low)."""
    lines = ["#EXTM3U", "#EXT-X-VERSION:7"]
    for rung in rungs:
        lines.append(
            "#EXT-X-STREAM-INF:"
            f"BANDWIDTH={rung['bandwidth']},"
            f"AVERAGE-BANDWIDTH={rung['bandwidth']},"
            f"RESOLUTION={rung['width']}x{rung['height']},"
            f'CODECS="{codecs_string(rung)}"'
        )
        lines.append(f"{rung['name']}/index.m3u8")
    return "\n".join(lines) + "\n"


def _encode_rung(ffmpeg: str, src: Path, rung: dict, variant_dir: Path,
                 fps: float, hls_time: int) -> None:
    gop = max(1, int(round((fps or 24.0) * 2)))
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-i", str(src),
        "-vf", f"scale=-2:{rung['height']}",
        *VCODECS[rung["vcodec"]],
        "-pix_fmt", "yuv420p10le" if rung["hdr"] else "yuv420p",
        "-b:v", f"{rung['video_kbps']}k",
        "-maxrate", f"{int(rung['video_kbps'] * 1.5)}k",
        "-bufsize", f"{rung['video_kbps'] * 2}k",
        "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
    ]
    if rung["hdr"]:
        cmd += HDR_FORMATS[rung["hdr"]]
    cmd += [
        "-c:a", "aac", "-b:a", f"{rung['audio_kbps']}k",
        "-f", "hls",
        "-hls_time", str(hls_time),
        "-hls_playlist_type", "vod",
        "-hls_segment_type", "fmp4",
        # Relative names on purpose: the playlist must reference portable
        # relative URIs, and ffmpeg resolves a relative segment pattern
        # against the CWD — so the encode runs with cwd=variant_dir.
        "-hls_fmp4_init_filename", "init.mp4",
        "-hls_segment_filename", "seg_%05d.m4s",
        "-hls_flags", "independent_segments",
        "index.m3u8",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=variant_dir)
    if proc.returncode != 0:
        raise MuxError(f"ffmpeg failed for rung {rung['name']} ({proc.returncode}): {proc.stderr[-2000:]}")


def _verify_rung(rung: dict, variant_dir: Path) -> dict:
    """ffprobe the packaged variant playlist: codec + geometry must match."""
    probe = probe_mp4(variant_dir / "index.m3u8")
    expected_codec = {"h264": "h264", "hevc": "hevc", "av1": "av1"}[rung["vcodec"]]
    problems = []
    if probe["video_codec"] != expected_codec:
        problems.append(f"codec {probe['video_codec']} != {expected_codec}")
    if probe["height"] != rung["height"]:
        problems.append(f"height {probe['height']} != {rung['height']}")
    if not probe["has_audio"]:
        problems.append("audio stream missing")
    segments = sorted(variant_dir.glob("seg_*.m4s"))
    if not (variant_dir / "init.mp4").exists() or not segments:
        problems.append("fMP4 init/segments missing")
    if problems:
        raise MuxError(f"rung {rung['name']} failed verification: {'; '.join(problems)}")
    return {
        "name": rung["name"],
        "vcodec": rung["vcodec"],
        "width": probe["width"],
        "height": probe["height"],
        "bandwidth": rung["bandwidth"],
        "hdr": rung["hdr"],
        "segments": len(segments),
        "duration": round(probe["duration"], 3),
    }


def package_hls(input_path: str | Path, out_dir: str | Path,
                hdr: str | None = None, hls_time: int = 4) -> dict:
    """Encode + segment + master-playlist + verify. Returns the summary dict.

    Raises MuxError on unreadable input, ffmpeg failure, or a verification
    mismatch — a package that does not verify is not a package.
    """
    ffmpeg = _require("ffmpeg")
    src = Path(input_path)
    if not src.is_file():
        raise MuxError(f"input not found: {src}")
    src = src.resolve()  # rung encodes run with cwd=variant_dir
    probe = probe_mp4(src)
    if not probe["width"] or not probe["height"]:
        raise MuxError(f"no video stream in {src}")
    rungs = build_ladder(probe["width"], probe["height"], hdr=hdr)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    variants = []
    for rung in rungs:
        variant_dir = out / rung["name"]
        variant_dir.mkdir(parents=True, exist_ok=True)
        _encode_rung(ffmpeg, src, rung, variant_dir, probe["fps"], hls_time)
        variants.append(_verify_rung(rung, variant_dir))

    master = out / "master.m3u8"
    master.write_text(render_master_playlist(rungs), encoding="utf-8")
    return {
        "input": str(src),
        "source": {"width": probe["width"], "height": probe["height"],
                   "fps": round(probe["fps"], 3), "duration": round(probe["duration"], 3)},
        "hdr": hdr,
        "hls_time": hls_time,
        "master": str(master),
        "variants": variants,
        "verified": True,
    }


def summary_json(summary: dict) -> str:
    """Deterministic JSON (sorted keys) for --json-out / stdout."""
    return json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n"

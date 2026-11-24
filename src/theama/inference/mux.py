# SPDX-License-Identifier: Apache-2.0
"""MP4 mux pipeline — H.264 / HEVC / AV1 video + AAC audio via system ffmpeg.

M1 ships the silent-AAC-track interface only: the audio DiT lands in M2,
but every MP4 Theama produces already carries a stereo AAC stream so
downstream players, NLEs, and the eval harness always see one.

Delivery matrix:
- vcodec="h264"  — default, universal playback, up to 4K in practice
- vcodec="hevc"  — 4K/8K delivery (M4/M4b), tagged hvc1 for player compat
- vcodec="av1"   — 4K/8K web delivery (SVT-AV1), best bitrate/quality
- hdr="hdr10"    — BT.2020 gamut + PQ (smpte2084), 10-bit
- hdr="hlg"      — BT.2020 gamut + HLG (arib-std-b67), 10-bit
HDR pairs with hevc/av1 (h264 10-bit playback support is poor).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MuxError(RuntimeError):
    """ffmpeg/ffprobe failure with the tool's stderr attached."""


VCODECS = {
    "h264": ["-c:v", "libx264"],
    "hevc": ["-c:v", "libx265", "-tag:v", "hvc1"],
    "av1": ["-c:v", "libsvtav1"],
}

HDR_FORMATS = {
    # pix_fmt stays yuv420p10le for both; the tags carry the HDR signal.
    "hdr10": [
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
    ],
    "hlg": [
        "-color_primaries", "bt2020",
        "-color_trc", "arib-std-b67",
        "-colorspace", "bt2020nc",
    ],
}


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise MuxError(f"{tool} not found on PATH")
    return path


def mux_mp4(
    frames_dir: str | Path,
    out_path: str | Path,
    fps: int = 24,
    audio_path: str | Path | None = None,
    crf: int = 18,
    preset: str = "medium",
    vcodec: str = "h264",
    hdr: str | None = None,
) -> Path:
    """Mux `%04d.png` frames (starting at 0001) into an MP4.

    audio_path=None muxes a silent 44.1kHz stereo AAC track (M1 default).
    vcodec: "h264" (default), "hevc" (4K/8K), "av1" (4K/8K web).
    hdr: None (SDR), "hdr10" or "hlg" (forces 10-bit yuv420p10le).
    Note: preset is passed through to the encoder — x264/x265 take named
    presets, SVT-AV1 takes its 0-13 numeric scale.
    """
    if vcodec not in VCODECS:
        raise MuxError(f"unknown vcodec {vcodec!r}; expected one of {sorted(VCODECS)}")
    if hdr is not None and hdr not in HDR_FORMATS:
        raise MuxError(f"unknown hdr {hdr!r}; expected one of {sorted(HDR_FORMATS)}")

    ffmpeg = _require("ffmpeg")
    frames_dir = Path(frames_dir)
    out_path = Path(out_path)
    if not list(frames_dir.glob("*.png")):
        raise MuxError(f"no PNG frames in {frames_dir}")
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", str(frames_dir / "%04d.png"),
    ]
    if audio_path is None:
        cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"]
    else:
        cmd += ["-i", str(audio_path)]
    cmd += [
        *VCODECS[vcodec],
        "-pix_fmt", "yuv420p10le" if hdr else "yuv420p",
        "-crf", str(crf), "-preset", preset,
    ]
    if hdr:
        cmd += HDR_FORMATS[hdr]
    cmd += [
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise MuxError(f"ffmpeg failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return out_path


def probe_mp4(path: str | Path) -> dict:
    """ffprobe an MP4: codecs, geometry, fps, HDR tags, audio presence."""
    ffprobe = _require("ffprobe")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise MuxError(f"ffprobe failed ({proc.returncode}): {proc.stderr[-2000:]}")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    num, _, den = str(video.get("avg_frame_rate", "0/1")).partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    return {
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "fps": fps,
        "pix_fmt": video.get("pix_fmt"),
        "color_space": video.get("color_space"),
        "color_transfer": video.get("color_transfer"),
        "color_primaries": video.get("color_primaries"),
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "has_audio": bool(audio),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": audio.get("channels", 0),
    }

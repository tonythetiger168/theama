# SPDX-License-Identifier: Apache-2.0
"""MP4 mux pipeline — H.264 video + AAC audio via system ffmpeg.

M1 ships the silent-AAC-track interface only: the audio DiT lands in M2,
but every MP4 Theama produces already carries a stereo AAC stream so
downstream players, NLEs, and the eval harness always see one.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MuxError(RuntimeError):
    """ffmpeg/ffprobe failure with the tool's stderr attached."""


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
) -> Path:
    """Mux `%04d.png` frames (starting at 0001) into an H.264/AAC MP4.

    audio_path=None muxes a silent 44.1kHz stereo AAC track (M1 default).
    """
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
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-preset", preset,
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
    """ffprobe an MP4: codecs, geometry, fps, duration, audio presence."""
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
        "duration": float(data.get("format", {}).get("duration", 0.0)),
        "has_audio": bool(audio),
        "audio_codec": audio.get("codec_name"),
        "audio_sample_rate": int(audio.get("sample_rate", 0) or 0),
        "audio_channels": audio.get("channels", 0),
    }

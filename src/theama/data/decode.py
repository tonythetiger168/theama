# SPDX-License-Identifier: Apache-2.0
"""FFmpeg-backed media decoding for the M2 audio-video data pipeline.

Route decision (torchcodec vs ffmpeg): we shell out to the system
``ffmpeg``/``ffprobe`` binaries instead of linking torchcodec. The system
FFmpeg is present on every target box (sandbox, RunPod images, CI runners),
needs no extra Python dependency or matching torch build, and keeps this
module importable without torch so CI can exercise the full decode path on
CPU-only runners. Tensors are materialised by the caller (``AVDataset``),
not here.

Contracts (latents-ready):
- :func:`decode_audio` returns float32 ``(2, N)`` stereo in ``[-1, 1]`` with
  exactly ``sample_rate * seconds`` samples (mid/side direct-concat, matching
  the codec yaml ``latent_dim: 2048`` decision, happens later in the codec).
- :func:`decode_video` returns uint8 ``(T, H, W, 3)`` RGB with exactly
  ``fps * seconds`` frames. Short inputs loop-pad the last frame; long
  inputs are centre-cropped in time after fps resampling.

Every failure raises :class:`MediaDecodeError` with the ffmpeg stderr tail —
silently returning zeros would poison training batches.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STDERR_TAIL = 400
_AUDIO_CHANNELS = 2


class MediaDecodeError(RuntimeError):
    """Raised when probing or decoding a media file fails."""


def _require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        raise MediaDecodeError(
            f"{name} not found on PATH; install FFmpeg (the M2 decode route)"
        )
    return tool


def _stderr_tail(proc: subprocess.CompletedProcess) -> str:
    tail = (proc.stderr or "").strip()[-_STDERR_TAIL:]
    return tail or f"exit code {proc.returncode}"


@dataclass(frozen=True)
class MediaInfo:
    """Probe result for one media file."""

    path: Path
    duration_seconds: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    sample_rate: int | None = None

    def as_dict(self) -> dict:
        return {
            "path": str(self.path),
            "duration_seconds": self.duration_seconds,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "sample_rate": self.sample_rate,
        }


def probe_media(path: str | Path) -> MediaInfo:
    """Probe one media file with ffprobe (streams + duration)."""
    path = Path(path)
    if not path.is_file():
        raise MediaDecodeError(f"media file not found: {path}")
    ffprobe = _require_tool("ffprobe")
    proc = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise MediaDecodeError(f"ffprobe failed for {path}: {_stderr_tail(proc)}")
    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise MediaDecodeError(f"ffprobe returned invalid JSON for {path}: {exc}") from exc

    streams = info.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = None
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    if duration is None or not np.isfinite(duration) or duration <= 0:
        raise MediaDecodeError(f"no usable duration for {path}")

    fps = None
    if video is not None:
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/0"
        try:
            num, den = (float(part) for part in rate.split("/"))
            fps = num / den if den else None
        except (ValueError, ZeroDivisionError):
            fps = None

    sample_rate = None
    if audio is not None:
        try:
            sample_rate = int(audio.get("sample_rate"))
        except (TypeError, ValueError):
            sample_rate = None

    return MediaInfo(
        path=path,
        duration_seconds=duration,
        has_video=video is not None,
        has_audio=audio is not None,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        fps=fps,
        sample_rate=sample_rate,
    )


def decode_audio(
    path: str | Path,
    *,
    sample_rate: int = 44100,
    seconds: float = 5.0,
) -> np.ndarray:
    """Decode audio to float32 stereo ``(2, N)`` in ``[-1, 1]``.

    Resampled to ``sample_rate`` and forced to exactly
    ``round(sample_rate * seconds)`` samples: short files are zero-padded,
    long files are trimmed from the start of the file (training clips are
    cut from the head, matching how the manifest was curated).
    """
    if sample_rate < 1 or seconds <= 0:
        raise ValueError(f"bad audio shape request: sr={sample_rate}, seconds={seconds}")
    path = Path(path)
    if not path.is_file():
        raise MediaDecodeError(f"media file not found: {path}")
    ffmpeg = _require_tool("ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            str(_AUDIO_CHANNELS),
            "-ar",
            str(sample_rate),
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise MediaDecodeError(f"ffmpeg audio decode failed for {path}: {_stderr_tail(proc)}")
    pcm = np.frombuffer(proc.stdout, dtype=np.float32)
    if pcm.size % _AUDIO_CHANNELS:
        pcm = pcm[: pcm.size - (pcm.size % _AUDIO_CHANNELS)]
    wave = pcm.reshape(-1, _AUDIO_CHANNELS).T.copy()  # (2, N)

    target = int(round(sample_rate * seconds))
    if wave.shape[1] < target:
        pad = np.zeros((_AUDIO_CHANNELS, target - wave.shape[1]), dtype=np.float32)
        wave = np.concatenate([wave, pad], axis=1)
    elif wave.shape[1] > target:
        wave = wave[:, :target]
    return np.clip(wave, -1.0, 1.0)


def decode_video(
    path: str | Path,
    *,
    fps: int = 24,
    seconds: float = 5.0,
    height: int = 240,
    width: int = 432,
) -> np.ndarray:
    """Decode video to uint8 RGB ``(T, H, W, 3)`` with exactly ``fps*seconds`` frames.

    Frames are resampled to ``fps`` and scaled to ``(width, height)`` by
    ffmpeg. Short clips loop-pad the final decoded frame; long clips keep
    the centre ``T`` frames so both ends' ramp-up/ramp-down are dropped.
    An empty decode (corrupt stream, zero frames) raises.
    """
    if fps < 1 or seconds <= 0 or height < 1 or width < 1:
        raise ValueError(
            f"bad video shape request: fps={fps}, seconds={seconds}, size={width}x{height}"
        )
    path = Path(path)
    if not path.is_file():
        raise MediaDecodeError(f"media file not found: {path}")
    ffmpeg = _require_tool("ffmpeg")
    proc = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"fps={fps},scale={width}:{height}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise MediaDecodeError(f"ffmpeg video decode failed for {path}: {_stderr_tail(proc)}")
    frame_size = height * width * 3
    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frames = raw[: raw.size - (raw.size % frame_size)].reshape(-1, height, width, 3)
    if frames.shape[0] == 0:
        raise MediaDecodeError(f"no decodable frames in {path}")

    target = int(round(fps * seconds))
    if frames.shape[0] < target:
        pad = np.repeat(frames[-1:], target - frames.shape[0], axis=0)
        frames = np.concatenate([frames, pad], axis=0)
    elif frames.shape[0] > target:
        start = (frames.shape[0] - target) // 2
        frames = frames[start : start + target]
    return frames.copy()

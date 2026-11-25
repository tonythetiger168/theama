# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the FFmpeg decode path (M2 data pipeline decode half).

Synthetic media is generated with the system ffmpeg at runtime, so these
tests need no fixtures checked into the repo. They skip cleanly on boxes
without FFmpeg (the decode route requires it, so CI installs it).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import AVDataset
from theama.data.decode import MediaDecodeError, decode_audio, decode_video, probe_media

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    ffmpeg is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe not installed; decode route requires it",
)

W, H, FPS = 48, 36, 8
SR = 8000


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def _color_clip(path: Path, color: str, duration: float, *, with_audio: bool = False) -> None:
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color={color}:s={W}x{H}:r={FPS}:d={duration}",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={SR}:d={duration}"]
    cmd += ["-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    _run(cmd)


def _wav(path: Path, duration: float, *, freq: int = 440) -> None:
    _run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"sine=frequency={freq}:sample_rate={SR}:d={duration}",
        "-ac", "1", str(path),
    ])


@pytest.fixture
def red_blue_clip(tmp_path: Path) -> Path:
    """0.5 s red then 0.5 s blue, 8 frames total — makes crop order visible."""
    path = tmp_path / "red_blue.mp4"
    _run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color=red:s={W}x{H}:r={FPS}:d=0.5",
        "-f", "lavfi", "-i", f"color=blue:s={W}x{H}:r={FPS}:d=0.5",
        "-filter_complex", "concat=n=2:v=1",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(path),
    ])
    return path


def test_probe_media_reports_streams_and_duration(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _color_clip(clip, "green", 0.5, with_audio=True)
    info = probe_media(clip)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (W, H)
    assert info.fps == pytest.approx(FPS, abs=0.5)
    assert info.sample_rate == SR
    assert info.duration_seconds == pytest.approx(0.5, abs=0.2)

    wav = tmp_path / "tone.wav"
    _wav(wav, 1.0)
    winfo = probe_media(wav)
    assert winfo.has_audio and not winfo.has_video
    assert winfo.sample_rate == SR

    with pytest.raises(MediaDecodeError, match="not found"):
        probe_media(tmp_path / "missing.mp4")
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a media file")
    with pytest.raises(MediaDecodeError, match="ffprobe failed"):
        probe_media(junk)


def test_decode_audio_exact_length_pad_and_trim(tmp_path: Path):
    short = tmp_path / "short.wav"
    _wav(short, 0.25)
    wave = decode_audio(short, sample_rate=SR, seconds=1.0)
    assert wave.shape == (2, SR)  # zero-padded to the full second
    assert wave.dtype == np.float32
    assert np.abs(wave[:, : SR // 4]).max() > 0.05  # real signal up front
    assert np.abs(wave[:, SR // 2 :]).max() == 0.0  # silence in the padded tail
    assert wave.min() >= -1.0 and wave.max() <= 1.0

    long = tmp_path / "long.wav"
    _wav(long, 1.0)
    half = decode_audio(long, sample_rate=SR, seconds=0.5)
    assert half.shape == (2, SR // 2)

    with pytest.raises(MediaDecodeError, match="not found"):
        decode_audio(tmp_path / "nope.wav", sample_rate=SR, seconds=1.0)
    with pytest.raises(ValueError, match="bad audio shape"):
        decode_audio(long, sample_rate=SR, seconds=0)


def test_decode_video_exact_frames_pad_loop_and_centre_crop(tmp_path: Path, red_blue_clip: Path):
    frames = decode_video(red_blue_clip, fps=FPS, seconds=1.0, height=H, width=W)
    assert frames.shape == (FPS, H, W, 3)
    assert frames.dtype == np.uint8
    # First half red, second half blue.
    assert frames[0, :, :, 0].mean() > frames[0, :, :, 2].mean() + 40
    assert frames[-1, :, :, 2].mean() > frames[-1, :, :, 0].mean() + 40

    # Centre crop: 8 frames -> 4 keeps frames 2..5, i.e. ends mixed red/blue.
    cropped = decode_video(red_blue_clip, fps=FPS, seconds=0.5, height=H, width=W)
    assert cropped.shape == (4, H, W, 3)
    assert cropped[0, :, :, 0].mean() > cropped[0, :, :, 2].mean() + 40
    assert cropped[-1, :, :, 2].mean() > cropped[-1, :, :, 0].mean() + 40

    # Short clip loop-pads the final frame: all blue.
    short = tmp_path / "short.mp4"
    _color_clip(short, "blue", 0.25)
    padded = decode_video(short, fps=FPS, seconds=0.5, height=H, width=W)
    assert padded.shape == (4, H, W, 3)
    assert padded[-1, :, :, 2].mean() > padded[-1, :, :, 0].mean() + 40

    with pytest.raises(ValueError, match="bad video shape"):
        decode_video(short, fps=0, seconds=1.0, height=H, width=W)
    with pytest.raises(MediaDecodeError, match="not found"):
        decode_video(tmp_path / "nope.mp4", fps=FPS, seconds=1.0, height=H, width=W)


def test_decode_video_rejects_audio_only_input(tmp_path: Path):
    wav = tmp_path / "tone.wav"
    _wav(wav, 1.0)
    with pytest.raises(MediaDecodeError, match="video decode failed"):
        decode_video(wav, fps=FPS, seconds=1.0, height=H, width=W)


def test_av_dataset_getitem_decodes_latents_ready_tensors(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _color_clip(clip, "teal", 0.5)
    wav = tmp_path / "tone.wav"
    _wav(wav, 0.5)
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        json.dumps(
            {"video": "clip.mp4", "audio": "tone.wav", "caption": "潮水聲", "source": "pexels:1"}
        )
        + "\n",
        encoding="utf-8",
    )
    ds = AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)
    item = ds[0]
    assert tuple(item["video"].shape) == (3, FPS, H, W)
    assert tuple(item["audio"].shape) == (2, SR)
    assert item["video"].min() >= -1.0 and item["video"].max() <= 1.0
    # Teal is mid-bright: normalised mean sits well above the -1 floor.
    assert item["video"].mean() > -0.5
    assert item["caption"] == "潮水聲"
    assert item["source"] == "pexels:1"

    # Decode failure surfaces loudly (missing media on disk).
    broken = tmp_path / "broken.jsonl"
    broken.write_text(
        json.dumps(
            {"video": "gone.mp4", "audio": "tone.wav", "caption": "x", "source": "pexels:2"}
        )
        + "\n",
        encoding="utf-8",
    )
    bad = AVDataset(broken, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)
    with pytest.raises(MediaDecodeError, match="not found"):
        bad[0]

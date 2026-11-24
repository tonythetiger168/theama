# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the M2 DataLoader layer (collate + skip-on-decode-error).

Reuses the runtime-generated ffmpeg synthetic media approach from
test_m2_decode; skips cleanly on boxes without FFmpeg.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import AVDataset
from theama.data.decode import MediaDecodeError
from theama.data.loader import (
    SkipDecodeErrorDataset,
    build_av_dataloader,
    collate_av,
)

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


def _make_clip(path: Path, color: str, duration: float = 1.0) -> None:
    _run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color={color}:s={W}x{H}:r={FPS}:d={duration}",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", str(path),
    ])


def _make_wav(path: Path, duration: float = 1.0) -> None:
    _run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={SR}:d={duration}",
        "-ac", "1", str(path),
    ])


def _write_manifest(tmp_path: Path, rows: list[dict]) -> Path:
    manifest = tmp_path / "data.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    return manifest


def _good_row(video: str, audio: str, n: int) -> dict:
    return {
        "video": video,
        "audio": audio,
        "caption": f"第 {n} 段",
        "source": f"pexels:{n}",
    }


@pytest.fixture
def two_clip_dataset(tmp_path: Path) -> AVDataset:
    _make_clip(tmp_path / "a.mp4", "red")
    _make_clip(tmp_path / "b.mp4", "blue")
    _make_wav(tmp_path / "a.wav")
    _make_wav(tmp_path / "b.wav")
    manifest = _write_manifest(tmp_path, [
        _good_row("a.mp4", "a.wav", 1),
        _good_row("b.mp4", "b.wav", 2),
    ])
    return AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)


def test_collate_stacks_fixed_shape_batches(two_clip_dataset: AVDataset):
    items = [two_clip_dataset[0], two_clip_dataset[1]]
    batch = collate_av(items)
    assert tuple(batch["video"].shape) == (2, 3, FPS, H, W)
    assert tuple(batch["audio"].shape) == (2, 2, SR)
    assert batch["video"].dtype == torch.float32
    assert batch["caption"] == ["第 1 段", "第 2 段"]
    assert batch["source"] == ["pexels:1", "pexels:2"]
    # Distinct clips stay distinct after stacking.
    assert not torch.equal(batch["video"][0], batch["video"][1])

    with pytest.raises(ValueError, match="empty batch"):
        collate_av([])


def test_collate_raises_on_shape_mismatch():
    a = {"video": torch.zeros(3, 8, 4, 4), "audio": torch.zeros(2, 16),
         "caption": "x", "source": "pexels:1"}
    b = {"video": torch.zeros(3, 8, 4, 5), "audio": torch.zeros(2, 16),
         "caption": "y", "source": "pexels:2"}
    with pytest.raises(ValueError, match="shape mismatch"):
        collate_av([a, b])


def test_skip_wrapper_walks_past_broken_samples(tmp_path: Path):
    _make_clip(tmp_path / "ok.mp4", "green")
    _make_wav(tmp_path / "ok.wav")
    manifest = _write_manifest(tmp_path, [
        _good_row("missing.mp4", "ok.wav", 1),  # broken video
        _good_row("ok.mp4", "ok.wav", 2),
    ])
    ds = AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)
    safe = SkipDecodeErrorDataset(ds)

    item = safe[0]  # index 0 is broken -> walks to index 1
    assert item["source"] == "pexels:2"
    assert safe.skipped == 1
    assert safe.skip_log[0].index == 0
    assert "not found" in safe.skip_log[0].message

    # The healthy index still serves directly without extra skips.
    item = safe[1]
    assert item["source"] == "pexels:2"
    assert safe.skipped == 1


def test_skip_wrapper_raises_when_everything_is_broken(tmp_path: Path):
    manifest = _write_manifest(tmp_path, [
        _good_row("gone1.mp4", "gone1.wav", 1),
        _good_row("gone2.mp4", "gone2.wav", 2),
    ])
    ds = AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)
    safe = SkipDecodeErrorDataset(ds)
    with pytest.raises(MediaDecodeError, match="every sample failed"):
        safe[0]
    assert safe.skipped == len(ds)  # bounded: one full pass, no infinite loop


def test_skip_wrapper_rejects_empty_dataset(tmp_path: Path):
    manifest = _write_manifest(tmp_path, [])
    ds = AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)
    with pytest.raises(ValueError, match="empty dataset"):
        SkipDecodeErrorDataset(ds)


def test_build_av_dataloader_end_to_end(two_clip_dataset: AVDataset):
    loader = build_av_dataloader(
        two_clip_dataset, batch_size=2, shuffle=False, skip_decode_errors=True
    )
    assert isinstance(loader.dataset, SkipDecodeErrorDataset)
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]
    assert tuple(batch["video"].shape) == (2, 3, FPS, H, W)
    assert tuple(batch["audio"].shape) == (2, 2, SR)
    assert sorted(batch["source"]) == ["pexels:1", "pexels:2"]
    assert loader.dataset.skipped == 0

    # Without the wrapper the raw dataset is used directly.
    raw_loader = build_av_dataloader(
        two_clip_dataset, batch_size=2, shuffle=False, skip_decode_errors=False
    )
    assert raw_loader.dataset is two_clip_dataset


def test_dataloader_skips_broken_rows_and_still_yields(tmp_path: Path):
    _make_clip(tmp_path / "ok.mp4", "yellow")
    _make_wav(tmp_path / "ok.wav")
    manifest = _write_manifest(tmp_path, [
        _good_row("gone.mp4", "gone.wav", 1),
        _good_row("ok.mp4", "ok.wav", 2),
    ])
    ds = AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)
    loader = build_av_dataloader(ds, batch_size=1, shuffle=False)
    batches = list(loader)
    assert len(batches) == 2  # broken index still yields the healthy sample
    assert all(b["source"] == ["pexels:2"] for b in batches)
    assert loader.dataset.skipped == 1  # only the pass over index 0 skips

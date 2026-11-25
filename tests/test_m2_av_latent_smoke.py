# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the decode -> encode straight-through smoke (M2 data path).

Synthetic media is generated with the system ffmpeg at runtime, so these
tests need no fixtures checked into the repo. They skip cleanly on boxes
without FFmpeg. Encoders are deterministic stand-ins: the video leg is a
fixed linear projection + downsample, the audio leg is the real
``MockAudioCodec`` mid/side path (hop-aligned via ``pad_to_hop``).
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_smoke import (
    MANIFEST_NAME,
    SUMMARY_NAME,
    pad_to_hop,
    run_decode_encode_smoke,
)
from theama.data.dataset import AVDataset
from theama.data.decode import MediaDecodeError
from theama.data.latent_cache import is_valid_latent
from theama.model.audio_codec import DAC_44K, MockAudioCodec, encode_stereo

ffmpeg = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(
    ffmpeg is None or shutil.which("ffprobe") is None,
    reason="FFmpeg/ffprobe not installed; decode route requires it",
)

W, H, FPS = 48, 36, 8
SR = 8000

_PROJ = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0
_CODEC = MockAudioCodec(DAC_44K)


def _mock_video_encode(video: torch.Tensor) -> torch.Tensor:
    """Deterministic (B,3,T,H,W) -> (B,4,T,H//8,W//8) stand-in for the Wan VAE."""
    b, c, t, h, w = video.shape
    lat = torch.einsum("bcthw,cd->bdthw", video, _PROJ.to(video.dtype))
    return F.interpolate(lat, size=(t, h // 8, w // 8), mode="trilinear", align_corners=False)


def _mock_audio_encode(waveform: torch.Tensor) -> torch.Tensor:
    """Real mid/side contract: pad to the DAC hop, then MockAudioCodec encode."""
    return encode_stereo(_CODEC, pad_to_hop(waveform, DAC_44K.hop_length))


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def _clip_with_audio(path: Path, color: str, duration: float = 1.0) -> None:
    _run([
        ffmpeg, "-y", "-v", "error",
        "-f", "lavfi", "-i", f"color={color}:s={W}x{H}:r={FPS}:d={duration}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={SR}:d={duration}",
        "-c:v", "mpeg4", "-pix_fmt", "yuv420p", "-shortest", str(path),
    ])


def _dataset(tmp_path: Path, rows: list[dict]) -> AVDataset:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return AVDataset(manifest, seconds=1, fps=FPS, sample_rate=SR, height=H, width=W)


def _three_sample_dataset(tmp_path: Path) -> AVDataset:
    rows = []
    for i, color in enumerate(("red", "green", "blue")):
        clip = tmp_path / f"clip{i}.mp4"
        _clip_with_audio(clip, color)
        rows.append({
            "video": clip.name,
            "audio": clip.name,
            "caption": f"合成測試 clip {i}",
            "source": "synthetic-fixture",
        })
    return _dataset(tmp_path, rows)


def _smoke(dataset: AVDataset, out: Path, **kw) -> dict:
    return run_decode_encode_smoke(
        dataset,
        _mock_video_encode,
        _mock_audio_encode,
        out,
        progress=lambda *_: None,
        **kw,
    )


def test_pad_to_hop_pads_tail_and_preserves_values():
    wave = torch.randn(1, 2, 1000)
    padded = pad_to_hop(wave, 512)
    assert padded.shape == (1, 2, 1024)
    assert torch.equal(padded[..., :1000], wave)
    assert torch.count_nonzero(padded[..., 1000:]) == 0

    aligned = torch.randn(2, 2, 1024)
    assert pad_to_hop(aligned, 512) is aligned  # no copy when already aligned

    with pytest.raises(ValueError, match="hop_length"):
        pad_to_hop(wave, 0)


def test_smoke_end_to_end(tmp_path: Path):
    dataset = _three_sample_dataset(tmp_path)
    out = tmp_path / "latents"
    summary = _smoke(dataset, out, batch_size=2)

    assert summary["encoded"] == 3
    assert summary["batches"] == 2  # drop_last=False keeps the ragged tail
    assert summary["skipped"] == 0
    assert summary["video_shapes"] == [[1, 4, FPS, H // 8, W // 8]]
    assert summary["audio_shapes"] == [[1, 16, DAC_44K.dit_token_dim]]  # 8000 -> 16 hops

    rows = [json.loads(l) for l in (out / MANIFEST_NAME).read_text().splitlines()]
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert row["index"] == i
        assert row["source"] == "synthetic-fixture"
        assert "合成測試" in row["caption"]
        # Video .pt honors the encode_pairs contract (5D, finite, loadable).
        assert is_valid_latent(row["video_latent"])
        audio = torch.load(row["audio_latent"], weights_only=True)
        assert audio.shape == (1, 16, DAC_44K.dit_token_dim)
        assert torch.isfinite(audio).all()
    assert json.loads((out / SUMMARY_NAME).read_text())["encoded"] == 3


def test_smoke_limit_caps_encoded_samples(tmp_path: Path):
    dataset = _three_sample_dataset(tmp_path)
    out = tmp_path / "latents"
    summary = _smoke(dataset, out, batch_size=2, limit=2)
    assert summary["encoded"] == 2
    rows = (out / MANIFEST_NAME).read_text().splitlines()
    assert len(rows) == 2
    assert not (out / "sample0002_video.pt").exists()


def test_smoke_skips_broken_clip_and_reports(tmp_path: Path):
    dataset = _three_sample_dataset(tmp_path)
    (tmp_path / "clip1.mp4").write_bytes(b"not a media file")
    out = tmp_path / "latents"
    summary = _smoke(dataset, out)

    assert summary["skipped"] == 1
    assert summary["encoded"] == 3  # replacement keeps the smoke going
    captions = [r["caption"] for r in
                map(json.loads, (out / MANIFEST_NAME).read_text().splitlines())]
    assert "合成測試 clip 1" not in captions  # the broken clip never encodes


def test_smoke_strict_aborts_on_broken_clip(tmp_path: Path):
    dataset = _three_sample_dataset(tmp_path)
    (tmp_path / "clip1.mp4").write_bytes(b"junk")
    with pytest.raises(MediaDecodeError):
        _smoke(dataset, tmp_path / "latents", skip_broken=False)


def test_non_finite_latent_raises_and_caches_nothing(tmp_path: Path):
    dataset = _three_sample_dataset(tmp_path)

    def nan_video_encode(video):
        return torch.full((video.shape[0], 4, FPS, H // 8, W // 8), float("nan"))

    out = tmp_path / "latents"
    with pytest.raises(ValueError, match="non-finite"):
        run_decode_encode_smoke(
            dataset, nan_video_encode, _mock_audio_encode, out,
            progress=lambda *_: None,
        )
    assert not list(out.glob("*.pt"))


def test_encoder_contract_violation_raises(tmp_path: Path):
    dataset = _three_sample_dataset(tmp_path)

    def bad_audio_encode(waveform):
        return torch.zeros(waveform.shape[0], 16)  # 2D, not (B,T_lat,D)

    with pytest.raises(ValueError, match="audio encoder"):
        run_decode_encode_smoke(
            dataset, _mock_video_encode, bad_audio_encode, tmp_path / "latents",
            progress=lambda *_: None,
        )

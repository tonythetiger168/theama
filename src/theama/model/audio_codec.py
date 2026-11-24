# SPDX-License-Identifier: Apache-2.0
"""Audio codec latent interface for M2 (DAC 44.1 kHz, MIT license).

This module pins the contract between the audio codec and the parallel audio
DiT *before* GPU wiring, so the trainer, the data pipeline and the mux all
agree on shapes and rates:

- Codec: Descript Audio Codec (DAC) 44.1 kHz model, MIT license
  (`descriptinc/descript-audio-codec`). Verified against upstream
  ``conf/base.yml``: encoder rates [2, 4, 8, 8] (hop 512), encoder dim 64
  doubling per stage -> 1024-dim continuous latents, 9 codebooks of 1024
  entries. Latent frame rate = 44100 / 512 = 86.1328125 Hz.
- DAC is a mono codec. Theama targets 44.1 kHz stereo, so stereo is carried
  as mid/side: two mono DAC encodes per clip (see ``stereo_to_mid_side``).
- The DiT consumes *continuous* (pre-quantization) latents of shape
  ``(B, T_lat, 2 * latent_dim)`` — mid and side channels concatenated on the
  feature axis. ``AudioDiT(latent_dim=...)`` must therefore be set to
  ``2 * DAC_44K.latent_dim`` (2048) or use a learned compressor (M2 wiring
  decision; ``configs/m2_joint_av.yaml`` records the codec name).

Only the interface and a deterministic CPU mock live here; the real DAC
encode/decode path lands with the GPU adapter (weights are downloaded
user-side, like Wan 2.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import torch

__all__ = [
    "AudioCodecSpec",
    "AudioCodec",
    "DAC_44K",
    "CODEC_REGISTRY",
    "get_codec_spec",
    "MockAudioCodec",
    "stereo_to_mid_side",
    "mid_side_to_stereo",
]


@dataclass(frozen=True)
class AudioCodecSpec:
    """Immutable description of an audio codec's latent interface."""

    name: str
    repo_id: str  # Hugging Face repo id (weights downloaded user-side)
    license: str
    sample_rate: int  # waveform samples per second
    hop_length: int  # waveform samples per latent frame
    latent_dim: int  # continuous (pre-quantization) latent width, per mono channel
    n_codebooks: int  # RVQ codebooks (informational; DiT uses continuous latents)
    codebook_size: int
    stereo_layout: str  # "mid_side" — stereo carried as two mono encodes

    @property
    def latent_frame_rate(self) -> float:
        """Latent frames per second."""
        return self.sample_rate / self.hop_length

    @property
    def dit_token_dim(self) -> int:
        """Feature width of one DiT audio token (mid + side concatenated)."""
        return 2 * self.latent_dim

    def frames_for_seconds(self, seconds: float) -> int:
        """Latent frames needed for ``seconds`` of audio, padding up to a full hop."""
        samples = seconds * self.sample_rate
        if samples != int(samples):
            raise ValueError(f"{seconds}s is not an integer number of samples")
        return math.ceil(int(samples) / self.hop_length)

    def seconds_for_frames(self, frames: int) -> float:
        """Seconds spanned by ``frames`` latent frames."""
        if frames < 0:
            raise ValueError("frames must be >= 0")
        return frames * self.hop_length / self.sample_rate


# Verified against descript-audio-codec conf/base.yml (MIT):
# sample_rate 44100, encoder_rates [2, 4, 8, 8] -> hop 512,
# encoder_dim 64 with 4 rate stages -> latent_dim 1024,
# n_codebooks 9, codebook_size 1024.
DAC_44K = AudioCodecSpec(
    name="dac_44k",
    repo_id="descript/dac_44khz",
    license="MIT",
    sample_rate=44100,
    hop_length=512,
    latent_dim=1024,
    n_codebooks=9,
    codebook_size=1024,
    stereo_layout="mid_side",
)

CODEC_REGISTRY: dict[str, AudioCodecSpec] = {spec.name: spec for spec in (DAC_44K,)}


def get_codec_spec(name: str) -> AudioCodecSpec:
    """Look up a codec spec by registry name."""
    try:
        return CODEC_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown audio codec {name!r}; registered: {sorted(CODEC_REGISTRY)}"
        ) from None


def stereo_to_mid_side(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split stereo ``(B, 2, T)`` into mid/side mono signals (each ``(B, 1, T)``)."""
    if waveform.ndim != 3 or waveform.shape[1] != 2:
        raise ValueError(f"expected stereo waveform (B, 2, T), got {tuple(waveform.shape)}")
    left, right = waveform[:, 0:1, :], waveform[:, 1:2, :]
    return (left + right) * 0.5, (left - right) * 0.5


def mid_side_to_stereo(mid: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """Reconstruct stereo ``(B, 2, T)`` from mid/side mono signals."""
    if mid.shape != side.shape:
        raise ValueError(f"mid/side shape mismatch: {tuple(mid.shape)} vs {tuple(side.shape)}")
    return torch.cat((mid + side, mid - side), dim=1)


class AudioCodec(Protocol):
    """Codec contract consumed by the M2 trainer and data pipeline.

    ``encode`` maps a mono waveform ``(B, 1, T)`` to continuous latents
    ``(B, T_lat, latent_dim)`` with ``T_lat = T // hop_length``; ``decode`` is
    the approximate inverse. Stereo is handled by the caller via mid/side.
    """

    spec: AudioCodecSpec

    def encode(self, waveform: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latents: torch.Tensor) -> torch.Tensor: ...


class MockAudioCodec:
    """Deterministic CPU codec honoring a spec's shapes, for tests and dry-runs.

    Encode average-pools each hop window and projects it into ``latent_dim``
    with a seeded fixed matrix; decode applies the transpose and repeats each
    frame across its hop window. Not a generative model — just a shape-true,
    reproducible stand-in so CPU tests never download weights.
    """

    def __init__(self, spec: AudioCodecSpec = DAC_44K, seed: int = 0):
        self.spec = spec
        gen = torch.Generator().manual_seed(seed)
        self._proj = torch.randn(1, spec.hop_length, spec.latent_dim, generator=gen)
        self._proj /= math.sqrt(spec.hop_length)

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError(
                f"expected mono waveform (B, 1, T), got {tuple(waveform.shape)}"
            )
        batch, _, samples = waveform.shape
        if samples % self.spec.hop_length != 0:
            raise ValueError(
                f"waveform length {samples} is not a multiple of hop {self.spec.hop_length}"
            )
        frames = samples // self.spec.hop_length
        windows = waveform.reshape(batch, frames, self.spec.hop_length)
        return torch.einsum("bfh,hd->bfd", windows, self._proj[0])

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3 or latents.shape[-1] != self.spec.latent_dim:
            raise ValueError(
                f"expected latents (B, T_lat, {self.spec.latent_dim}), "
                f"got {tuple(latents.shape)}"
            )
        batch, frames, _ = latents.shape
        windows = torch.einsum("bfd,hd->bfh", latents, self._proj[0])
        return windows.reshape(batch, 1, frames * self.spec.hop_length)

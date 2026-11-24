# SPDX-License-Identifier: Apache-2.0
"""Audio-video dataset skeleton.

Hard rule (license-risks.md): no MiniMax H3 / LTX weights, code, or generated
outputs may enter the training set. Every sample must carry a data-card
source record; the CI source-check script enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

FORBIDDEN_SOURCES = ("minimax", "hailuo", "h3-", "ltx", "lightricks")


@dataclass
class AVSample:
    video_path: Path
    audio_path: Path
    caption: str
    source: str  # data-card source id, e.g. "pexels:12345"


class AVDataset(Dataset):
    """Reads (video, audio, caption) triples with a mandatory data-card manifest."""

    def __init__(self, manifest_path: str | Path, seconds: int = 5, fps: int = 24,
                 sample_rate: int = 44100):
        self.manifest_path = Path(manifest_path)
        self.seconds, self.fps, self.sample_rate = seconds, fps, sample_rate
        self.samples: list[AVSample] = []
        # TODO(M1): load manifest (jsonl), build AVSample list, validate sources
        # via check_source() before accepting any row.

    @staticmethod
    def check_source(source: str) -> None:
        low = source.lower()
        if any(bad in low for bad in FORBIDDEN_SOURCES):
            raise ValueError(f"forbidden training source (license red line): {source}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        self.check_source(sample.source)
        # TODO(M1/M2): decode video -> (T,H,W,C) tensor; decode audio -> stereo
        # waveform; encode text; return latents-ready tensors.
        raise NotImplementedError("M1: decode + tokenize path lands with the eval harness")

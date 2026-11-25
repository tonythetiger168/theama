# SPDX-License-Identifier: Apache-2.0
"""DataLoader layer for the M2 audio-video pipeline (collate + safe skip).

``AVDataset.__getitem__`` deliberately raises on decode failure — a corrupt
clip must never slip zeros into a batch. That is the right contract for
validation, but a multi-hour training launch should not die because one clip
in ten thousand is broken. This module provides the training-side layer:

- :func:`collate_av` — stacks fixed-shape samples into batch tensors.
  ``decode_video``/``decode_audio`` already guarantee exact shapes
  (``(3, fps*seconds, H, W)`` and ``(2, sr*seconds)``), so collation is a
  plain stack; a shape mismatch means a bug and raises loudly.
- :class:`SkipDecodeErrorDataset` — wraps an ``AVDataset`` and, on
  :class:`~theama.data.decode.MediaDecodeError`, advances to the next index
  (wrapping around, bounded by ``len(dataset)`` attempts so an all-broken
  dataset raises instead of spinning forever). Skips are counted on
  ``skipped`` / ``skip_log`` for the training summary.
- :func:`build_av_dataloader` — one-call ``DataLoader`` wiring the two
  together with sane defaults (``num_workers=0`` so the skip counter lives
  in the main process; raise it and counts stay per-worker).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset

from theama.data.dataset import AVDataset
from theama.data.decode import MediaDecodeError

_MAX_SKIP_MSG = "every sample failed to decode"


def collate_av(samples: list[dict[str, object]]) -> dict[str, object]:
    """Stack a list of ``AVDataset`` items into one batch.

    Video and audio shapes are exact by construction (decode pads/trims to
    ``fps*seconds`` frames and ``sr*seconds`` samples at a fixed resolution),
    so a mismatch here is a bug, not data variance — raise on it rather than
    silently padding. ``caption``/``source`` stay as per-sample lists.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    videos = [s["video"] for s in samples]
    audios = [s["audio"] for s in samples]
    first_v, first_a = videos[0].shape, audios[0].shape
    for i, (v, a) in enumerate(zip(videos, audios)):
        if v.shape != first_v or a.shape != first_a:
            raise ValueError(
                f"shape mismatch in batch item {i}: video {tuple(v.shape)} vs "
                f"{tuple(first_v)}, audio {tuple(a.shape)} vs {tuple(first_a)}"
            )
    return {
        "video": torch.stack(videos),
        "audio": torch.stack(audios),
        "caption": [s["caption"] for s in samples],
        "source": [s["source"] for s in samples],
    }


@dataclass
class SkipRecord:
    """One skipped sample: dataset index and the decode error message."""

    index: int
    message: str


class SkipDecodeErrorDataset(Dataset):
    """Wraps an :class:`AVDataset`, skipping samples that fail to decode.

    On :class:`MediaDecodeError` the wrapper advances ``(idx + 1) % len`` and
    retries, up to ``len(dataset)`` attempts per ``__getitem__``; if every
    sample is broken it re-raises the last error with ``_MAX_SKIP_MSG``
    context instead of looping forever. Any other exception (schema bug,
    OOM, ...) propagates untouched — only decode failures are skippable.
    Skips are recorded on ``skipped`` (count) and ``skip_log`` (details).
    """

    def __init__(self, dataset: AVDataset):
        if len(dataset) == 0:
            raise ValueError("cannot wrap an empty dataset")
        self.dataset = dataset
        self.skipped: int = 0
        self.skip_log: list[SkipRecord] = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, object]:
        last_error: MediaDecodeError | None = None
        for _ in range(len(self.dataset)):
            try:
                return self.dataset[idx]
            except MediaDecodeError as exc:
                last_error = exc
                self.skipped += 1
                self.skip_log.append(SkipRecord(index=idx, message=str(exc)))
                idx = (idx + 1) % len(self.dataset)
        raise MediaDecodeError(
            f"{_MAX_SKIP_MSG} (tried all {len(self.dataset)} samples); "
            f"last error: {last_error}"
        ) from last_error


def build_av_dataloader(
    dataset: AVDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    skip_decode_errors: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Build a training ``DataLoader`` over an :class:`AVDataset`.

    With ``skip_decode_errors=True`` (default) the dataset is wrapped in
    :class:`SkipDecodeErrorDataset`; the wrapper is reachable as
    ``loader.dataset`` for skip stats after the run. ``drop_last`` defaults
    to True: fixed-shape collation means a ragged final batch is only a
    wasted partial step. Keep ``num_workers=0`` unless you accept per-worker
    (unaggregated) skip counters.
    """
    ds: Dataset = SkipDecodeErrorDataset(dataset) if skip_decode_errors else dataset
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_av,
        drop_last=drop_last,
    )

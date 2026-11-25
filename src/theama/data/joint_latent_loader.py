# SPDX-License-Identifier: Apache-2.0
"""Training-side joint latent loader: cache manifests -> JointAVTrainer batches.

The P1b data engine produces two caches *ahead* of training so the hot loop
never touches ffmpeg or the encoders:

  * AV latents (:mod:`theama.data.av_latent_cache`):
    ``sample{index:04d}_video.pt`` (1,C,F,H,W),
    ``sample{index:04d}_audio.pt`` (1,Ta,Da), manifest ``av_latents.jsonl``;
  * text embeddings (:mod:`theama.data.text_cache`):
    ``text_{sha256(caption)[:16]}.pt`` (L,Dt), manifest ``text_latents.jsonl``.

This module joins the two manifests on sample index and assembles the batch
dict that :meth:`theama.train.train_joint_av.JointAVTrainer.train_step`
consumes::

    video_tokens (B,C,F,H,W)  <- stacked video latents (leading 1 squeezed)
    audio_latent (B,Ta,Da)    <- stacked audio latents (leading 1 squeezed)
    video_cond   (B,L,Dt)     <- stacked umT5 text embeddings
    text_mask    (B,L) bool   <- True on real tokens (from seq_len; all True
                                 when the row predates seq_len recording)
    cond         (B,D)        <- only when a cond_pooler is supplied:
                                 cond_pooler(masked_mean_pool(video_cond, mask))

Join safety rules (fail-closed, mirroring the caches' own guards):

  * only rows that are successful in BOTH manifests are served — an encode
    failure on either side excludes the sample until the next encode run;
  * rows are model-filtered (defaults: each cache's own meta.json model), so
    latents from different encoder configs never mix in one batch;
  * the AV row's caption must equal the text row's caption — the AV drift
    guard intentionally ignores caption edits, so this cross-check is the
    loader-side backstop against pairing a video with the previous caption's
    embedding;
  * every served .pt is validated (ndim + finite) at dataset construction
    and re-checked for finiteness on every __getitem__ — NaN latents must
    never reach a loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from theama.data.av_latent_cache import MANIFEST_NAME as AV_MANIFEST_NAME
from theama.data.latent_cache import (
    is_valid_latent,
    load_reusable_rows,
    read_cache_meta,
)
from theama.data.text_cache import MANIFEST_NAME as TEXT_MANIFEST_NAME

__all__ = [
    "JointLatentCacheError",
    "JointLatentJoin",
    "load_joint_latent_records",
    "JointLatentDataset",
    "masked_mean_pool",
    "collate_joint_latent",
    "build_joint_latent_dataloader",
]


class JointLatentCacheError(RuntimeError):
    """A served cache entry is missing, corrupt, or non-finite."""


# ---- manifest join -----------------------------------------------------------


@dataclass
class JointLatentJoin:
    """Result of joining the AV and text manifests on sample index."""

    records: list[dict]
    dropped: dict[str, int] = field(default_factory=dict)
    av_model: str = ""
    text_model: str = ""

    def __len__(self) -> int:
        return len(self.records)


def _resolve_model(cache_dir: Path, model: str | None, which: str) -> str:
    if model:
        return model
    meta_model = read_cache_meta(cache_dir).get("model")
    if not meta_model:
        raise ValueError(
            f"pass {which}_model explicitly or run the {which} cache first "
            f"(no meta.json model in {cache_dir})"
        )
    return str(meta_model)


def load_joint_latent_records(
    av_dir: str | Path,
    text_dir: str | Path,
    *,
    av_model: str | None = None,
    text_model: str | None = None,
) -> JointLatentJoin:
    """Join both cache manifests into per-sample latent records.

    ``av_model`` / ``text_model`` default to each cache's own meta.json model
    (the value the encoder run recorded); pass them explicitly to pin a
    specific encoder config. Only indices with a successful, model-matching
    row in both manifests — and agreeing captions — produce a record.
    """
    av_dir, text_dir = Path(av_dir), Path(text_dir)
    av_model = _resolve_model(av_dir, av_model, "av")
    text_model = _resolve_model(text_dir, text_model, "text")
    av_rows = load_reusable_rows(av_dir / AV_MANIFEST_NAME, model=av_model)
    text_rows = load_reusable_rows(text_dir / TEXT_MANIFEST_NAME, model=text_model)

    join = JointLatentJoin(records=[], av_model=av_model, text_model=text_model)

    def _drop(reason: str) -> None:
        join.dropped[reason] = join.dropped.get(reason, 0) + 1

    for index in sorted(set(av_rows) | set(text_rows)):
        av_row = av_rows.get(index)
        text_row = text_rows.get(index)
        if av_row is None:
            _drop("missing_av_row")
            continue
        if text_row is None:
            _drop("missing_text_row")
            continue
        if av_row.get("caption") != text_row.get("caption"):
            _drop("caption_mismatch")
            continue
        keys = ("video_latent", "audio_latent")
        if any(not av_row.get(k) for k in keys) or not text_row.get("text_latent"):
            _drop("missing_keys")
            continue
        record = {
            "index": index,
            "caption": av_row["caption"],
            "source": av_row.get("source", ""),
            "video_latent": av_row["video_latent"],
            "audio_latent": av_row["audio_latent"],
            "text_latent": text_row["text_latent"],
        }
        seq_len = text_row.get("seq_len")
        if isinstance(seq_len, int) and not isinstance(seq_len, bool):
            record["seq_len"] = seq_len
        join.records.append(record)
    return join


# ---- dataset -----------------------------------------------------------------


def _load_finite(path: str | Path, *, index: int, kind: str) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise JointLatentCacheError(
            f"sample {index}: {kind} latent unreadable ({path}): {exc}"
        ) from exc
    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all()):
        raise JointLatentCacheError(
            f"sample {index}: {kind} latent is not a finite tensor ({path})"
        )
    return tensor


class JointLatentDataset(Dataset):
    """Serves joined latent records as per-sample tensors for collation.

    With ``validate=True`` (default) every referenced .pt is checked up front
    (loadable, finite, expected ndim) and the first failures abort
    construction with a :class:`JointLatentCacheError` naming the offending
    indices — a multi-hour training launch must fail fast on a corrupt cache,
    not mid-epoch. Regardless of ``validate``, ``__getitem__`` re-checks
    finiteness on every load.
    """

    def __init__(self, records: JointLatentJoin | list[dict], *, validate: bool = True):
        if isinstance(records, JointLatentJoin):
            self.join = records
            records = records.records
        else:
            self.join = None
        if not records:
            raise ValueError(
                "joint latent join produced zero records — run encode_av_latents "
                "and encode_texts first, and check the model filters"
            )
        self.records = list(records)
        if validate:
            bad: list[str] = []
            for record in self.records:
                index = record["index"]
                if not is_valid_latent(record["video_latent"], ndim=5):
                    bad.append(f"{index}:video")
                if not is_valid_latent(record["audio_latent"], ndim=3):
                    bad.append(f"{index}:audio")
                if not is_valid_latent(record["text_latent"], ndim=2):
                    bad.append(f"{index}:text")
            if bad:
                preview = ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
                raise JointLatentCacheError(
                    f"{len(bad)} cached latent(s) failed validation: {preview} — "
                    "re-run the encode scripts to repair them"
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        index = record["index"]
        video = _load_finite(record["video_latent"], index=index, kind="video")
        audio = _load_finite(record["audio_latent"], index=index, kind="audio")
        text = _load_finite(record["text_latent"], index=index, kind="text")
        if video.ndim != 5 or video.shape[0] != 1:
            raise JointLatentCacheError(
                f"sample {index}: video latent must be (1,C,F,H,W), got {tuple(video.shape)}"
            )
        if audio.ndim != 3 or audio.shape[0] != 1:
            raise JointLatentCacheError(
                f"sample {index}: audio latent must be (1,Ta,Da), got {tuple(audio.shape)}"
            )
        if text.ndim != 2:
            raise JointLatentCacheError(
                f"sample {index}: text latent must be (L,D), got {tuple(text.shape)}"
            )
        return {
            "index": index,
            "caption": record["caption"],
            "source": record["source"],
            "video_latent": video[0],
            "audio_latent": audio[0],
            "text_latent": text,
            # Rows predating seq_len recording treat the full length as real.
            "seq_len": int(record.get("seq_len", text.shape[0])),
        }


# ---- collation ---------------------------------------------------------------


def masked_mean_pool(text_latents: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool (B,L,D) embeddings over real tokens only; mask is (B,L) bool.

    Padded positions contribute exactly zero, so pooled conditioning is
    invariant to the cache's padding length. Rows with seq_len 0 divide by
    one (clamp) and pool to zeros rather than NaN.
    """
    if text_latents.ndim != 3 or mask.ndim != 2:
        raise ValueError("expected text_latents (B,L,D) and mask (B,L)")
    weights = mask.to(dtype=text_latents.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp(min=1.0)
    return (text_latents * weights).sum(dim=1) / denom


def collate_joint_latent(
    samples: list[dict[str, object]],
    *,
    cond_pooler=None,
) -> dict[str, object]:
    """Stack dataset items into one JointAVTrainer batch.

    Per-sample shapes are exact by construction (fixed encode config), so a
    mismatch here is a bug and raises loudly — never pad silently. ``seq_len``
    is clamped to [0, L]. With ``cond_pooler`` (e.g. the training script's
    text->audio-dim projection) the batch gains ``cond``; without it the key
    is absent and ``train_step`` will name it in its missing-keys error.
    """
    if not samples:
        raise ValueError("cannot collate an empty batch")
    videos = [s["video_latent"] for s in samples]
    audios = [s["audio_latent"] for s in samples]
    texts = [s["text_latent"] for s in samples]
    first = (videos[0].shape, audios[0].shape, texts[0].shape)
    for i, (v, a, t) in enumerate(zip(videos, audios, texts)):
        if (v.shape, a.shape, t.shape) != first:
            raise ValueError(
                f"shape mismatch in batch item {i}: video {tuple(v.shape)} vs "
                f"{tuple(first[0])}, audio {tuple(a.shape)} vs {tuple(first[1])}, "
                f"text {tuple(t.shape)} vs {tuple(first[2])}"
            )
    seq_len = texts[0].shape[0]
    seq_lens = torch.tensor(
        [max(0, min(int(s["seq_len"]), seq_len)) for s in samples],
        dtype=torch.long,
    )
    mask = torch.arange(seq_len)[None, :] < seq_lens[:, None]
    video_cond = torch.stack(texts)
    batch: dict[str, object] = {
        "video_tokens": torch.stack(videos),
        "audio_latent": torch.stack(audios),
        "video_cond": video_cond,
        "text_mask": mask,
        "index": [s["index"] for s in samples],
        "caption": [s["caption"] for s in samples],
        "source": [s["source"] for s in samples],
    }
    if cond_pooler is not None:
        batch["cond"] = cond_pooler(masked_mean_pool(video_cond, mask))
    return batch


def build_joint_latent_dataloader(
    records: JointLatentJoin | list[dict] | JointLatentDataset,
    *,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    cond_pooler=None,
    validate: bool = True,
) -> DataLoader:
    """One-call DataLoader over the joined latent caches.

    ``drop_last`` defaults to True: fixed-shape collation means a ragged
    final batch is only a wasted partial step. Keep ``num_workers=0`` unless
    the cache lives on fast local storage — latent loads are the bottleneck,
    not decode.
    """
    dataset = (
        records
        if isinstance(records, JointLatentDataset)
        else JointLatentDataset(records, validate=validate)
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=partial(collate_joint_latent, cond_pooler=cond_pooler),
        drop_last=drop_last,
    )

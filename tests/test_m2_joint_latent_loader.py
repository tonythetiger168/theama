# SPDX-License-Identifier: Apache-2.0
"""CPU tests for theama.data.joint_latent_loader (P1b training-side loader).

Builds real AV/text caches via the production planners with deterministic
stand-in encoders (small tensors, no ffmpeg/GPU), then exercises the join
(manifest intersection, model filters, caption-mismatch backstop), dataset
validation and per-load finiteness checks, collation (stacking, seq_len
mask, cond pooling), and the money test: a DataLoader batch fed straight
into JointAVTrainer.train_step.
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_cache import (  # noqa: E402
    plan_av_encoding,
    run_av_encoding,
)
from theama.data.joint_latent_loader import (  # noqa: E402
    JointLatentCacheError,
    JointLatentDataset,
    build_joint_latent_dataloader,
    collate_joint_latent,
    load_joint_latent_records,
    masked_mean_pool,
)
from theama.data.latent_cache import save_latent_atomic  # noqa: E402
from theama.data.text_cache import plan_text_encoding, run_text_encoding  # noqa: E402
from theama.model.audio_dit import AudioDiT  # noqa: E402
from theama.train.train_joint_av import JointAVTrainer  # noqa: E402

# (1,C,F,H,W) video / (1,Ta,Da) audio / (L,Dt) text stand-ins.
VID_SHAPE = (1, 4, 2, 3, 3)
AUD_SHAPE = (1, 6, 16)
TXT_SHAPE = (8, 16)
SEQ_LEN = 5
AV_MODEL = "wan2.2-vae+dac"
TXT_MODEL = "umT5-xxl"
DIM = 32  # AudioDiT hidden dim for the E2E test


def _samples(n: int = 4, *, caption_edit: dict[int, str] | None = None) -> list[dict]:
    edits = caption_edit or {}
    return [
        {
            "video": f"/media/v{i}.mp4",
            "audio": f"/media/a{i}.wav",
            "caption": edits.get(i, f"第 {i} 段"),
            "source": f"pexels:{i}",
        }
        for i in range(n)
    ]


def _av_encode_fn(fail_index: int | None = None):
    def encode(sample, video_dest, audio_dest):
        if fail_index is not None and sample["video"].endswith(f"v{fail_index}.mp4"):
            raise RuntimeError("simulated VAE failure")
        save_latent_atomic(torch.randn(*VID_SHAPE), video_dest)
        save_latent_atomic(torch.randn(*AUD_SHAPE), audio_dest)

    return encode


def _text_encode_fn(fail_index: int | None = None, seq_len: int | None = SEQ_LEN):
    def encode(sample, dest):
        if fail_index is not None and sample["caption"] == f"第 {fail_index} 段":
            raise RuntimeError("simulated encoder OOM")
        save_latent_atomic(torch.randn(*TXT_SHAPE), dest)
        return seq_len

    return encode


def _run_caches(
    tmp_path: Path,
    samples: list[dict],
    *,
    av_fail: int | None = None,
    text_fail: int | None = None,
    text_seq_len: int | None = SEQ_LEN,
) -> tuple[Path, Path]:
    av_dir, text_dir = tmp_path / "av", tmp_path / "text"
    av_dir.mkdir()
    text_dir.mkdir()
    run_av_encoding(
        plan_av_encoding(samples, av_dir, AV_MODEL),
        _av_encode_fn(av_fail),
        progress=lambda *_: None,
    )
    run_text_encoding(
        plan_text_encoding(samples, text_dir, TXT_MODEL),
        _text_encode_fn(text_fail, text_seq_len),
        progress=lambda *_: None,
    )
    return av_dir, text_dir


@pytest.fixture
def caches(tmp_path: Path) -> tuple[Path, Path]:
    """AV cache fails on sample 2; text cache fails on sample 3."""
    return _run_caches(tmp_path, _samples(), av_fail=2, text_fail=3)


# ---- join --------------------------------------------------------------------


def test_join_intersects_success_rows_and_counts_drops(caches):
    av_dir, text_dir = caches
    join = load_joint_latent_records(av_dir, text_dir)
    assert [r["index"] for r in join.records] == [0, 1]
    assert join.dropped == {"missing_av_row": 1, "missing_text_row": 1}
    assert join.av_model == AV_MODEL and join.text_model == TXT_MODEL
    record = join.records[0]
    assert record["caption"] == "第 0 段" and record["source"] == "pexels:0"
    assert record["seq_len"] == SEQ_LEN


def test_join_caption_mismatch_excludes_sample(tmp_path):
    # Caption of sample 1 edited after the AV encode: the AV drift guard
    # ignores caption edits by design, so the loader's cross-check must
    # refuse to pair that video with the edited caption's embedding.
    samples = _samples()
    av_dir, text_dir = tmp_path / "av", tmp_path / "text"
    av_dir.mkdir()
    text_dir.mkdir()
    run_av_encoding(
        plan_av_encoding(samples, av_dir, AV_MODEL),
        _av_encode_fn(),
        progress=lambda *_: None,
    )
    edited = _samples(caption_edit={1: "改過的 caption"})
    run_text_encoding(
        plan_text_encoding(edited, text_dir, TXT_MODEL),
        _text_encode_fn(),
        progress=lambda *_: None,
    )
    join = load_joint_latent_records(av_dir, text_dir)
    assert [r["index"] for r in join.records] == [0, 2, 3]
    assert join.dropped == {"caption_mismatch": 1}


def test_join_model_filters_and_missing_meta(tmp_path):
    av_dir, text_dir = _run_caches(tmp_path, _samples())
    assert load_joint_latent_records(av_dir, text_dir, av_model="other").records == []
    no_meta = tmp_path / "bare"
    no_meta.mkdir()
    with pytest.raises(ValueError, match="av_model"):
        load_joint_latent_records(no_meta, text_dir)
    with pytest.raises(ValueError, match="text_model"):
        load_joint_latent_records(av_dir, no_meta)


def test_join_without_seq_len_records_full_length(tmp_path):
    av_dir, text_dir = _run_caches(tmp_path, _samples(2), text_seq_len=None)
    join = load_joint_latent_records(av_dir, text_dir)
    assert "seq_len" not in join.records[0]
    dataset = JointLatentDataset(join)
    assert dataset[0]["seq_len"] == TXT_SHAPE[0]


# ---- dataset ------------------------------------------------------------------


def test_dataset_validate_fails_fast_on_corrupt_cache(caches):
    av_dir, text_dir = caches
    join = load_joint_latent_records(av_dir, text_dir)
    save_latent_atomic(torch.full(VID_SHAPE, float("nan")), join.records[0]["video_latent"])
    with pytest.raises(JointLatentCacheError, match="0:video"):
        JointLatentDataset(join)


def test_dataset_rejects_empty_join(tmp_path):
    with pytest.raises(ValueError, match="zero records"):
        JointLatentDataset([])


def test_getitem_squeezes_batch_dim_and_carries_seq_len(caches):
    dataset = JointLatentDataset(load_joint_latent_records(*caches))
    item = dataset[0]
    assert item["video_latent"].shape == VID_SHAPE[1:]
    assert item["audio_latent"].shape == AUD_SHAPE[1:]
    assert item["text_latent"].shape == TXT_SHAPE
    assert item["seq_len"] == SEQ_LEN and item["index"] == 0


def test_getitem_rechecks_finiteness_after_validation(caches):
    join = load_joint_latent_records(*caches)
    dataset = JointLatentDataset(join)
    save_latent_atomic(torch.full(TXT_SHAPE, float("inf")), join.records[1]["text_latent"])
    with pytest.raises(JointLatentCacheError, match="finite"):
        dataset[1]


# ---- collation -----------------------------------------------------------------


def test_collate_stacks_and_builds_seq_len_mask(caches):
    dataset = JointLatentDataset(load_joint_latent_records(*caches))
    batch = collate_joint_latent([dataset[0], dataset[1]])
    assert batch["video_tokens"].shape == (2, *VID_SHAPE[1:])
    assert batch["audio_latent"].shape == (2, *AUD_SHAPE[1:])
    assert batch["video_cond"].shape == (2, *TXT_SHAPE)
    assert batch["text_mask"].dtype == torch.bool
    assert batch["text_mask"].sum(dim=1).tolist() == [SEQ_LEN, SEQ_LEN]
    assert batch["text_mask"][0, SEQ_LEN:].eq(False).all()
    assert batch["caption"] == ["第 0 段", "第 1 段"]
    assert "cond" not in batch  # no pooler supplied


def test_collate_rejects_shape_mismatch_and_empty(caches):
    dataset = JointLatentDataset(load_joint_latent_records(*caches))
    good = dataset[0]
    bad = {**good, "audio_latent": torch.randn(7, AUD_SHAPE[2])}
    with pytest.raises(ValueError, match="shape mismatch"):
        collate_joint_latent([good, bad])
    with pytest.raises(ValueError, match="empty"):
        collate_joint_latent([])


def test_collate_clamps_seq_len_to_padding_length(caches):
    dataset = JointLatentDataset(load_joint_latent_records(*caches))
    item = {**dataset[0], "seq_len": 99}
    batch = collate_joint_latent([item])
    assert batch["text_mask"].all()  # clamped to L: every position real
    item["seq_len"] = -3
    batch = collate_joint_latent([item])
    assert not batch["text_mask"].any()  # clamped to 0: pools to zeros, not NaN
    pooled = masked_mean_pool(batch["video_cond"], batch["text_mask"])
    assert bool(torch.isfinite(pooled).all()) and pooled.abs().sum() == 0


def test_masked_mean_pool_ignores_padding():
    text = torch.randn(2, TXT_SHAPE[0], TXT_SHAPE[1])
    mask = torch.zeros(2, TXT_SHAPE[0], dtype=torch.bool)
    mask[:, :SEQ_LEN] = True
    poisoned = text.clone()
    poisoned[:, SEQ_LEN:] = 1e6
    assert torch.allclose(masked_mean_pool(text, mask), masked_mean_pool(poisoned, mask))


def test_collate_cond_pooler_receives_masked_pooled_text(caches):
    torch.manual_seed(3)
    pooler = torch.nn.Linear(TXT_SHAPE[1], DIM)
    dataset = JointLatentDataset(load_joint_latent_records(*caches))
    items = [dataset[0], dataset[1]]
    batch = collate_joint_latent(items, cond_pooler=pooler)
    expected = pooler(
        masked_mean_pool(
            torch.stack([i["text_latent"] for i in items]),
            batch["text_mask"],
        )
    )
    assert batch["cond"].shape == (2, DIM)
    assert torch.allclose(batch["cond"], expected)


# ---- E2E: cache -> DataLoader -> JointAVTrainer.train_step ---------------------


class _TinyVideoBackbone(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.proj = torch.nn.Linear(channels, channels)

    def predict_velocity(self, tokens, t, cond):
        del cond
        view = t.reshape(-1, *([1] * (tokens.ndim - 1))).to(tokens.dtype)
        if tokens.ndim == 5:  # Wan latents (B,C,F,H,W): channels are dim 1
            out = self.proj(tokens.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        else:
            out = self.proj(tokens)
        return out + view


def test_dataloader_batch_feeds_train_step_end_to_end(caches):
    torch.manual_seed(11)
    pooler = torch.nn.Linear(TXT_SHAPE[1], DIM)
    loader = build_joint_latent_dataloader(
        load_joint_latent_records(*caches),
        batch_size=2,
        shuffle=False,
        cond_pooler=pooler,
    )
    batch = next(iter(loader))
    audio = AudioDiT(
        dim=DIM, layers=2, heads=4,
        video_dim=VID_SHAPE[1], bridge_every=1, latent_dim=AUD_SHAPE[2],
    )
    trainer = JointAVTrainer(
        _TinyVideoBackbone(VID_SHAPE[1]), audio, {"phase": "full", "lr": 1e-3}
    )
    metrics = trainer.train_step(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert metrics["loss"] > 0.0


def test_batch_without_cond_pooler_fails_train_step_cleanly(caches):
    dataset = JointLatentDataset(load_joint_latent_records(*caches))
    batch = collate_joint_latent([dataset[0]])
    trainer = JointAVTrainer(
        _TinyVideoBackbone(VID_SHAPE[1]),
        AudioDiT(dim=DIM, layers=2, heads=4, video_dim=VID_SHAPE[1],
                 bridge_every=1, latent_dim=AUD_SHAPE[2]),
    )
    with pytest.raises(KeyError, match="cond"):
        trainer.train_step(batch)

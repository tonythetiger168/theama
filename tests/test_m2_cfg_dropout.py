# SPDX-License-Identifier: Apache-2.0
"""CPU tests for M2 classifier-free-guidance text-conditioning dropout.

Covers the pure mask/apply helpers (edge probabilities, RNG determinism,
shape/type fail-fast), the JointAVTrainer wiring (config validation, sampled
vs explicit ``cfg_drop_mask``, metric reporting, gradient flow on null rows),
and the YAML contract that both M2 configs carry the same ``cfg_dropout``.
"""

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_dit import AudioDiT  # noqa: E402
from theama.train.joint_av_runner import StubVideoBackbone, trainer_config  # noqa: E402
from theama.train.train_joint_av import (  # noqa: E402
    JointAVTrainer,
    apply_cfg_dropout,
    sample_cfg_drop_mask,
)

DIMS = dict(dim=32, layers=4, heads=4, video_dim=4, bridge_every=2, latent_dim=16)


def _trainer(cfg: dict | None = None) -> JointAVTrainer:
    return JointAVTrainer(
        StubVideoBackbone(), AudioDiT(**DIMS), {"phase": "bridge", "lr": 1e-3, **(cfg or {})}
    )


def _batch(batch_size: int = 4) -> dict:
    return {
        "video_tokens": torch.randn(batch_size, 6, DIMS["video_dim"]),
        "audio_latent": torch.randn(batch_size, 5, DIMS["latent_dim"]),
        "cond": torch.randn(batch_size, DIMS["dim"]),
    }


# ---- pure helpers -------------------------------------------------------------


def test_mask_edge_probabilities_short_circuit_without_rng():
    assert not sample_cfg_drop_mask(8, 0.0).any()
    assert sample_cfg_drop_mask(8, 1.0).all()
    assert sample_cfg_drop_mask(8, 0.0).dtype == torch.bool


def test_mask_rejects_invalid_probability_and_batch():
    with pytest.raises(ValueError):
        sample_cfg_drop_mask(4, -0.1)
    with pytest.raises(ValueError):
        sample_cfg_drop_mask(4, 1.1)
    with pytest.raises(ValueError):
        sample_cfg_drop_mask(0, 0.5)


def test_mask_generator_determinism():
    a = sample_cfg_drop_mask(64, 0.3, generator=torch.Generator().manual_seed(7))
    b = sample_cfg_drop_mask(64, 0.3, generator=torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert 0 < int(a.sum()) < 64


def test_apply_replaces_masked_rows_with_zero_null():
    cond = torch.randn(4, 8)
    mask = torch.tensor([True, False, True, False])
    out = apply_cfg_dropout(cond, mask)
    assert torch.equal(out[0], torch.zeros(8))
    assert torch.equal(out[2], torch.zeros(8))
    assert torch.equal(out[1], cond[1]) and torch.equal(out[3], cond[3])
    assert cond[0].abs().sum() > 0  # input not mutated


def test_apply_all_false_returns_input_unchanged():
    cond = torch.randn(4, 8)
    out = apply_cfg_dropout(cond, torch.zeros(4, dtype=torch.bool))
    assert out is cond


def test_apply_custom_null_cond():
    cond = torch.randn(3, 8)
    null = torch.full((8,), 5.0)
    out = apply_cfg_dropout(cond, torch.tensor([True, True, False]), null_cond=null)
    assert torch.equal(out[0], null) and torch.equal(out[1], null)
    out2 = apply_cfg_dropout(cond, torch.tensor([True, False, False]), null_cond=null[None, :])
    assert torch.equal(out2[0], null)


def test_apply_fail_fast_on_bad_shapes_and_dtype():
    cond = torch.randn(4, 8)
    with pytest.raises(ValueError):
        apply_cfg_dropout(torch.randn(4, 8, 2), torch.zeros(4, dtype=torch.bool))
    with pytest.raises(ValueError):
        apply_cfg_dropout(cond, torch.zeros(3, dtype=torch.bool))
    with pytest.raises(TypeError):
        apply_cfg_dropout(cond, torch.tensor([0, 1, 0, 1], dtype=torch.int64))
    with pytest.raises(ValueError):
        apply_cfg_dropout(cond, torch.tensor([True] * 4), null_cond=torch.zeros(7))


# ---- trainer wiring -----------------------------------------------------------


def test_trainer_rejects_out_of_range_cfg_dropout():
    with pytest.raises(ValueError):
        _trainer({"cfg_dropout": 1.5})


def test_default_dropout_rate_leaves_step_untouched():
    torch.manual_seed(0)
    metrics = _trainer().train_step(_batch())
    assert metrics["cfg_dropped"] == 0.0
    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())


def test_explicit_mask_overrides_sampled_rate_and_flows_gradients():
    trainer = _trainer({"cfg_dropout": 1.0})  # sampled rate would drop all rows
    batch = _batch()
    batch["cfg_drop_mask"] = torch.zeros(4, dtype=torch.bool)  # explicit override wins
    metrics = trainer.train_step(batch)
    assert metrics["cfg_dropped"] == 0.0

    trainer2 = _trainer()
    batch2 = _batch()
    batch2["cfg_drop_mask"] = torch.tensor([True, True, False, False])
    metrics2 = trainer2.train_step(batch2)
    assert metrics2["cfg_dropped"] == 2.0
    grads = [p.grad for p in trainer2.audio.bridges.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_sampled_full_dropout_matches_all_true_mask():
    torch.manual_seed(0)
    metrics = _trainer({"cfg_dropout": 1.0}).train_step(_batch())
    assert metrics["cfg_dropped"] == 4.0


def test_dropout_changes_the_loss_landscape():
    batch = _batch()
    batch["audio_noise"] = torch.randn_like(batch["audio_latent"])
    batch["video_noise"] = torch.randn_like(batch["video_tokens"])
    torch.manual_seed(0)
    clean = _trainer().train_step(batch)
    torch.manual_seed(0)
    dropped = _trainer().train_step(batch | {"cfg_drop_mask": torch.ones(4, dtype=torch.bool)})
    assert clean["audio_loss"] != pytest.approx(dropped["audio_loss"])


# ---- YAML contract ------------------------------------------------------------


def test_m2_yamls_carry_matching_cfg_dropout():
    prod = yaml.safe_load((ROOT / "configs" / "m2_joint_av.yaml").read_text())
    dev = yaml.safe_load((ROOT / "configs" / "m2_joint_av_dev.yaml").read_text())
    assert trainer_config(prod)["cfg_dropout"] == trainer_config(dev)["cfg_dropout"]
    assert 0.0 < float(trainer_config(prod)["cfg_dropout"]) < 1.0
    assert trainer_config(dev) == trainer_config(prod)  # dev/prod training parity preserved

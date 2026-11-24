# SPDX-License-Identifier: Apache-2.0
"""CPU tests for M2 bridge-warmup LoRA injection (theama.model.lora)."""

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_dit import AudioDiT
from theama.model.lora import (
    LoRALinear,
    inject_lora_into_audio_dit,
    iter_lora_modules,
    lora_parameters,
    lora_state_dict,
    merge_lora,
)
from theama.train.train_joint_av import JointAVTrainer

DIM, VID_DIM, LATENT, LAYERS, BRIDGE_EVERY = 64, 96, 16, 4, 2
N_BRIDGES = LAYERS // BRIDGE_EVERY


def _audio_dit() -> AudioDiT:
    return AudioDiT(dim=DIM, layers=LAYERS, heads=4, video_dim=VID_DIM,
                    bridge_every=BRIDGE_EVERY, latent_dim=LATENT)


class _TinyVideoBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(VID_DIM, VID_DIM)

    def predict_velocity(self, tokens, t, cond):
        del cond
        return self.proj(tokens) + t[:, None, None].to(tokens.dtype)


def _joint_batch(batch_size: int = 2) -> dict[str, torch.Tensor]:
    return {
        "audio_latent": torch.randn(batch_size, 6, LATENT),
        "video_tokens": torch.randn(batch_size, 4, VID_DIM),
        "cond": torch.randn(batch_size, DIM),
        "audio_noise": torch.randn(batch_size, 6, LATENT),
        "video_noise": torch.randn(batch_size, 4, VID_DIM),
    }


# ---- LoRALinear ---------------------------------------------------------------

def test_lora_linear_identity_at_init():
    base = torch.nn.Linear(32, 16)
    lora = LoRALinear(base, rank=4, alpha=8)
    x = torch.randn(3, 5, 32)
    assert torch.equal(lora(x), base(x)), "zero-init B must make injection an identity"


def test_lora_linear_freezes_base_and_trains_ab():
    lora = LoRALinear(torch.nn.Linear(32, 16), rank=4, alpha=8)
    assert all(not p.requires_grad for p in lora.base.parameters())
    assert lora.lora_a.requires_grad and lora.lora_b.requires_grad


def test_lora_linear_preserves_mha_contract_and_base_device():
    base = torch.nn.Linear(8, 8)
    lora = LoRALinear(base, rank=2, alpha=4)
    assert torch.equal(lora.weight, base.weight)  # zero-init B, fused MHA path
    assert lora.bias is base.bias
    assert lora.lora_a.device == base.weight.device
    assert lora.lora_b.device == base.weight.device
    merged = lora.merged_linear()
    assert merged.weight.device == base.weight.device
    assert merged.bias.device == base.bias.device


def test_lora_linear_scaling_math():
    base = torch.nn.Linear(8, 8)
    lora = LoRALinear(base, rank=2, alpha=6)  # scale = 3
    with torch.no_grad():
        lora.lora_a.copy_(torch.ones(2, 8) * 0.1)
        lora.lora_b.copy_(torch.ones(8, 2) * 0.2)
    x = torch.randn(4, 8)
    expected = base(x) + (x @ lora.lora_a.T @ lora.lora_b.T) * 3.0
    assert torch.allclose(lora(x), expected)


def test_lora_linear_validates_rank_and_alpha():
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(torch.nn.Linear(8, 8), rank=0, alpha=1)
    with pytest.raises(ValueError, match="alpha"):
        LoRALinear(torch.nn.Linear(8, 8), rank=1, alpha=0)


def test_lora_merged_linear_matches_lora_output():
    base = torch.nn.Linear(16, 8)
    lora = LoRALinear(base, rank=4, alpha=4)
    with torch.no_grad():
        lora.lora_b.copy_(torch.randn_like(lora.lora_b))
    merged = lora.merged_linear()
    x = torch.randn(2, 3, 16)
    assert torch.allclose(lora(x), merged(x), atol=1e-6)


# ---- injection into AudioDiT ---------------------------------------------------

def test_inject_wraps_every_bridge_out_proj():
    model = _audio_dit()
    n = inject_lora_into_audio_dit(model, rank=4, alpha=8)
    assert n == 2 * N_BRIDGES
    for bridge in model.bridges:
        assert isinstance(bridge.v2a.out_proj, LoRALinear)
        assert isinstance(bridge.a2v.out_proj, LoRALinear)


def test_inject_is_idempotent_and_identity_forward():
    model = _audio_dit().eval()
    a_lat = torch.randn(1, 6, LATENT)
    v_tok = torch.randn(1, 4, VID_DIM)
    cond = torch.randn(1, DIM)
    with torch.no_grad():
        ref = model(a_lat, v_tok, cond)
        n = inject_lora_into_audio_dit(model, rank=4)
        out = model(a_lat, v_tok, cond)
    assert n == 2 * N_BRIDGES
    assert inject_lora_into_audio_dit(model, rank=4) == 0, "re-injection must be a no-op"
    assert torch.equal(ref[0], out[0]) and torch.equal(ref[1], out[1])


def test_lora_parameters_and_state_dict_cover_only_ab():
    model = _audio_dit()
    inject_lora_into_audio_dit(model, rank=4, alpha=8)
    params = lora_parameters(model)
    assert len(params) == 2 * 2 * N_BRIDGES  # A + B per projection
    state = lora_state_dict(model)
    assert len(state) == len(params)
    assert all(name.endswith((".lora_a", ".lora_b")) for name in state)
    assert "bridges.0.v2a.out_proj.lora_a" in state


def test_merge_lora_restores_plain_linear_and_preserves_output():
    model = _audio_dit().eval()
    inject_lora_into_audio_dit(model, rank=4, alpha=8)
    for _, lora in iter_lora_modules(model):
        with torch.no_grad():
            lora.lora_b.copy_(torch.randn_like(lora.lora_b))
    a_lat = torch.randn(1, 6, LATENT)
    v_tok = torch.randn(1, 4, VID_DIM)
    cond = torch.randn(1, DIM)
    with torch.no_grad():
        ref = model(a_lat, v_tok, cond)
        n = merge_lora(model)
        out = model(a_lat, v_tok, cond)
    assert n == 2 * N_BRIDGES
    assert iter_lora_modules(model) == []
    assert torch.allclose(ref[0], out[0], atol=1e-5)
    assert torch.allclose(ref[1], out[1], atol=1e-5)


# ---- trainer integration -------------------------------------------------------

def test_trainer_bridge_phase_with_lora_trains_only_low_rank():
    torch.manual_seed(7)
    audio = _audio_dit()
    trainer = JointAVTrainer(
        _TinyVideoBackbone(),
        audio,
        {"phase": "bridge", "lr": 0.05, "bridge_lora_rank": 4, "bridge_lora_alpha": 8},
    )
    trainable = set(trainer._trainable_params)
    assert trainable == set(lora_parameters(audio.bridges))
    base_before = audio.bridges[0].v2a.out_proj.base.weight.detach().clone()
    block_before = audio.blocks[0].adaln.weight.detach().clone()

    metrics = trainer.train_step(_joint_batch())

    assert all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
    assert torch.equal(base_before, audio.bridges[0].v2a.out_proj.base.weight)
    assert torch.equal(block_before, audio.blocks[0].adaln.weight)
    assert audio.bridges[0].v2a.out_proj.lora_b.abs().sum() > 0, "LoRA B must move off zero"


def test_trainer_full_phase_merges_lora_then_unfreezes_audio():
    audio = _audio_dit()
    trainer = JointAVTrainer(
        _TinyVideoBackbone(),
        audio,
        {"phase": "bridge", "lr": 0.05, "bridge_lora_rank": 4},
    )
    trainer.train_step(_joint_batch())
    assert iter_lora_modules(audio), "LoRA must be active during bridge warmup"

    trainer.set_phase("full")

    assert iter_lora_modules(audio) == [], "full phase must merge LoRA back"
    assert all(p.requires_grad for p in audio.parameters())
    assert all(not p.requires_grad for p in trainer.video.parameters())
    metrics = trainer.train_step(_joint_batch(batch_size=1))
    assert metrics["loss"] > 0.0


def test_trainer_default_rank_zero_keeps_dense_warmup():
    audio = _audio_dit()
    trainer = JointAVTrainer(_TinyVideoBackbone(), audio, {"phase": "bridge"})
    assert iter_lora_modules(audio) == []
    assert set(trainer._trainable_params) == set(audio.bridges.parameters())


def test_m2_yaml_wires_bridge_lora():
    cfg = yaml.safe_load((ROOT / "configs" / "m2_joint_av.yaml").read_text())
    training = cfg["training"]
    assert training["bridge_lora_rank"] == 8
    assert training["bridge_lora_alpha"] == 16
    trainer = JointAVTrainer(
        _TinyVideoBackbone(), _audio_dit(),
        {"phase": training["phase"], "bridge_lora_rank": training["bridge_lora_rank"],
         "bridge_lora_alpha": training["bridge_lora_alpha"]},
    )
    assert len(trainer._trainable_params) == 2 * 2 * N_BRIDGES

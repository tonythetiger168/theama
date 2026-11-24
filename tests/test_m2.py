# SPDX-License-Identifier: Apache-2.0
"""CPU tests for M2: parallel audio DiT skeleton + joint A/V training utils."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_dit import AudioDiT, AudioDiTBlock, CrossModalBridge
from theama.train.train_joint_av import (
    JointAVTrainer,
    flow_match_pair,
    flow_matching_loss,
    logit_normal_sample,
    sinusoidal_time_embedding,
)

DIM, VID_DIM, LATENT, LAYERS, BRIDGE_EVERY = 64, 96, 16, 4, 2


def _audio_dit() -> AudioDiT:
    return AudioDiT(dim=DIM, layers=LAYERS, heads=4, video_dim=VID_DIM,
                    bridge_every=BRIDGE_EVERY, latent_dim=LATENT)


# ---- CrossModalBridge --------------------------------------------------------

def test_bridge_shapes_and_residual():
    bridge = CrossModalBridge(video_dim=VID_DIM, audio_dim=DIM, heads=4)
    v = torch.randn(2, 7, VID_DIM)
    a = torch.randn(2, 11, DIM)
    v2, a2 = bridge(v, a)
    assert v2.shape == v.shape and a2.shape == a.shape
    assert not torch.allclose(v2, v), "a2v cross-attention must update video tokens"
    assert not torch.allclose(a2, a), "v2a cross-attention must update audio tokens"


def test_bridge_gradient_flows_both_directions():
    bridge = CrossModalBridge(video_dim=VID_DIM, audio_dim=DIM, heads=4)
    v = torch.randn(1, 5, VID_DIM)
    a = torch.randn(1, 9, DIM, requires_grad=True)
    _, a2 = bridge(v, a)
    a2.sum().backward()
    assert a.grad is not None and a.grad.abs().sum() > 0


# ---- AudioDiTBlock -----------------------------------------------------------

def test_block_adaLN_conditioning_changes_output():
    blk = AudioDiTBlock(DIM, heads=4)
    x = torch.randn(2, 6, DIM)
    out_a = blk(x, torch.zeros(2, DIM))
    out_b = blk(x, torch.ones(2, DIM))
    assert not torch.allclose(out_a, out_b), "timestep/text conditioning must modulate"


# ---- AudioDiT ----------------------------------------------------------------

def test_audio_dit_forward_shapes():
    model = _audio_dit()
    a_lat = torch.randn(2, 11, LATENT)
    v_tok = torch.randn(2, 7, VID_DIM)
    cond = torch.randn(2, DIM)
    eps, v_out = model(a_lat, v_tok, cond)
    assert eps.shape == a_lat.shape, "audio eps must match latent shape"
    assert v_out.shape == v_tok.shape


def test_audio_dit_bridge_count():
    model = _audio_dit()
    assert len(model.bridges) == LAYERS // BRIDGE_EVERY
    assert len(model.blocks) == LAYERS


def test_audio_dit_requires_divisible_bridge_schedule():
    model = AudioDiT(dim=DIM, layers=5, heads=4, video_dim=VID_DIM,
                     bridge_every=2, latent_dim=LATENT)
    # 5 layers, bridge every 2 -> bridges at blocks 2 and 4 only (floor schedule)
    assert len(model.bridges) == 2
    a_lat = torch.randn(1, 4, LATENT)
    v_tok = torch.randn(1, 3, VID_DIM)
    eps, _ = model(a_lat, v_tok, torch.randn(1, DIM))
    assert eps.shape == a_lat.shape


def test_audio_dit_gradient_reaches_bridges_and_blocks():
    model = _audio_dit()
    a_lat = torch.randn(1, 6, LATENT)
    v_tok = torch.randn(1, 4, VID_DIM)
    # Joint A/V loss over both streams (as in train_joint_av): the audio-only
    # path cannot reach the final bridge's a2v half — the video loss can.
    eps, v_out = model(a_lat, v_tok, torch.randn(1, DIM))
    (eps.square().mean() + v_out.square().mean()).backward()
    for p in model.bridges.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0
    for p in model.blocks.parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0


def test_audio_dit_deterministic_in_eval():
    model = _audio_dit().eval()
    a_lat = torch.randn(1, 6, LATENT)
    v_tok = torch.randn(1, 4, VID_DIM)
    cond = torch.randn(1, DIM)
    with torch.no_grad():
        e1, _ = model(a_lat, v_tok, cond)
        e2, _ = model(a_lat, v_tok, cond)
    assert torch.allclose(e1, e2)


# ---- joint A/V training utils -------------------------------------------------

def test_logit_normal_sample_unit_interval():
    torch.manual_seed(0)
    t = logit_normal_sample(4096)
    assert t.shape == (4096,)
    assert float(t.min()) > 0.0 and float(t.max()) < 1.0


def test_logit_normal_sample_mu_shifts_median():
    torch.manual_seed(0)
    lo = logit_normal_sample(8192, mu=-2.0).median()
    hi = logit_normal_sample(8192, mu=2.0).median()
    assert lo < 0.5 < hi


def test_flow_matching_loss_zero_on_perfect_velocity():
    x1 = torch.randn(2, 8, LATENT)
    x0 = torch.randn(2, 8, LATENT)
    t = torch.rand(2)
    assert float(flow_matching_loss(x0 - x1, x1, x0, t)) == pytest.approx(0.0)


def test_flow_match_pair_endpoints_and_target_sign():
    x = torch.randn(2, 5, LATENT)
    noise = torch.randn_like(x)
    x_t0, target0 = flow_match_pair(x, noise, torch.zeros(2))
    x_t1, target1 = flow_match_pair(x, noise, torch.ones(2))
    assert torch.equal(x_t0, x)
    assert torch.equal(x_t1, noise)
    assert torch.equal(target0, noise - x)
    assert torch.equal(target1, noise - x)


def test_flow_match_pair_rejects_bad_time_shape():
    x = torch.randn(2, 5, LATENT)
    with pytest.raises(ValueError, match="t must have shape"):
        flow_match_pair(x, torch.randn_like(x), torch.rand(2, 1))


def test_sinusoidal_time_embedding_shape_and_variation():
    t = torch.tensor([0.0, 0.5, 1.0])
    emb = sinusoidal_time_embedding(t, DIM)
    assert emb.shape == (3, DIM)
    assert not torch.allclose(emb[0], emb[1])
    assert not torch.allclose(emb[1], emb[2])
    assert sinusoidal_time_embedding(t, DIM + 1).shape == (3, DIM + 1)


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


def test_joint_trainer_bridge_step_updates_only_bridges():
    torch.manual_seed(7)
    video = _TinyVideoBackbone()
    audio = _audio_dit()
    trainer = JointAVTrainer(video, audio, {"phase": "bridge", "lr": 0.01})
    bridge_before = audio.bridges[0].v_norm.weight.detach().clone()
    block_before = audio.blocks[0].adaln.weight.detach().clone()
    video_before = video.proj.weight.detach().clone()

    metrics = trainer.train_step(_joint_batch())

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(bridge_before, audio.bridges[0].v_norm.weight)
    assert torch.equal(block_before, audio.blocks[0].adaln.weight)
    assert torch.equal(video_before, video.proj.weight)
    assert all(not p.requires_grad for p in video.parameters())
    assert all(not p.requires_grad for p in audio.blocks.parameters())
    assert all(p.requires_grad for p in audio.bridges.parameters())


def test_joint_trainer_full_phase_unfreezes_audio_not_video():
    video = _TinyVideoBackbone()
    audio = _audio_dit()
    trainer = JointAVTrainer(video, audio, {"phase": "full"})
    assert all(p.requires_grad for p in audio.parameters())
    assert all(not p.requires_grad for p in video.parameters())
    metrics = trainer.train_step(_joint_batch(batch_size=1))
    assert metrics["loss"] > 0.0
    with pytest.raises(ValueError, match="unknown M2 training phase"):
        trainer.set_phase("video")


def test_joint_trainer_validates_batch_and_video_adapter():
    trainer = JointAVTrainer(_TinyVideoBackbone(), _audio_dit())
    with pytest.raises(KeyError, match="missing M2 batch keys"):
        trainer.train_step({"cond": torch.randn(1, DIM)})
    trainer.video = torch.nn.Identity()
    with pytest.raises(TypeError, match="predict_velocity"):
        trainer.train_step(_joint_batch(batch_size=1))


def test_m2_yaml_matches_trainer_contract():
    cfg = yaml.safe_load((ROOT / "configs" / "m2_joint_av.yaml").read_text())
    train = cfg["training"]
    assert train["phase"] in {"bridge", "full"}
    assert train["audio_loss_weight"] > 0 and train["video_loss_weight"] > 0
    assert cfg["model"]["video"]["frozen"] is True
    assert cfg["acceptance"]["av_sync_offset_ms_max"] == 40


def test_train_joint_av_compiles():
    subprocess.run(
        [sys.executable, "-m", "py_compile",
         str(ROOT / "src" / "theama" / "train" / "train_joint_av.py")],
        check=True,
    )

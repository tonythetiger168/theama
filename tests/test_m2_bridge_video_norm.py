# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for the issue #4 bridge-input scale red flag.

RunPod smoke (2026-11-26, A40, commit 2e16595): internal ``wan_blocks``
bridging fed un-normalised Wan block hidden states (~40x latent scale) into
the v2a cross-attention keys/values — audio loss 79.0 and grad norm 94.0 vs
~2.0 / ~2.35 on the latent-grid path. ``CrossModalBridge.norm_video_kv``
applies a parameter-free per-token LayerNorm to the video tokens used as v2a
keys/values, and ``WanVideoAdapter.predict_joint_velocity`` refuses internal
bridging without it (fail-closed before GPU time is burned).
"""

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_dit import AudioDiT, CrossModalBridge
from theama.model.wan_adapter import WanAdapterSpec, WanVideoAdapter
from theama.train.joint_av_runner import audio_dit_kwargs
from theama.train.real_weight_smoke import run_wan_smoke
from theama.train.train_joint_av import JointAVTrainer

LATENT_CHANNELS = 8
INNER_DIM = 12
TEXT_DIM = 12
AUDIO_DIM = 32
AUDIO_LATENT_DIM = 16
WAN_SCALE = 40.0  # issue #4: internal audio loss / grad norm were ~40x the latent path


class _TinyWanBlock(torch.nn.Module):
    def __init__(self, gain: float = 1.0):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(gain))

    def forward(self, hidden_states, encoder_hidden_states, timestep, rotary_emb):
        del encoder_hidden_states, timestep, rotary_emb
        return hidden_states * self.gain


class _TinyInternalWan(torch.nn.Module):
    """Diffusers-shaped fake whose blocks can emit Wan-scale hidden states."""

    def __init__(self, block_gain: float = 1.0):
        super().__init__()
        self.patch_embedding = torch.nn.Conv3d(LATENT_CHANNELS, INNER_DIM, kernel_size=1)
        self.blocks = torch.nn.ModuleList(_TinyWanBlock(block_gain) for _ in range(4))
        self.proj_out = torch.nn.Linear(INNER_DIM, LATENT_CHANNELS)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
    ):
        assert return_dict is False
        tokens = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()
        for block in self.blocks:
            tokens = block(tokens, encoder_hidden_states, timestep, None)
        sample = self.proj_out(tokens).transpose(1, 2).reshape_as(hidden_states)
        tv = timestep.to(hidden_states.dtype)[:, None, None, None, None] / 1000.0
        return (sample + tv,)


def _video(block_gain: float = 1.0) -> WanVideoAdapter:
    spec = WanAdapterSpec(
        "tiny-internal-wan",
        "example/tiny-internal-wan",
        latent_channels=LATENT_CHANNELS,
        text_dim=TEXT_DIM,
    )
    return WanVideoAdapter(_TinyInternalWan(block_gain), spec)


def _audio(*, norm_video_kv: bool = True) -> AudioDiT:
    return AudioDiT(
        dim=AUDIO_DIM,
        layers=2,
        heads=4,
        video_dim=INNER_DIM,
        bridge_every=1,
        latent_dim=AUDIO_LATENT_DIM,
        norm_video_kv=norm_video_kv,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "audio_latent": torch.randn(1, 5, AUDIO_LATENT_DIM),
        "video_tokens": torch.randn(1, LATENT_CHANNELS, 1, 2, 2),
        "cond": torch.randn(1, AUDIO_DIM),
        "video_cond": torch.randn(1, 3, TEXT_DIM),
        "audio_noise": torch.randn(1, 5, AUDIO_LATENT_DIM),
        "video_noise": torch.randn(1, LATENT_CHANNELS, 1, 2, 2),
    }


# ---- CrossModalBridge flag semantics ----------------------------------------


def test_flag_defaults_off_and_threads_to_all_bridges():
    assert CrossModalBridge(video_dim=8, audio_dim=4, heads=2).norm_video_kv is False
    dit = AudioDiT(dim=AUDIO_DIM, layers=2, heads=4, video_dim=INNER_DIM,
                   bridge_every=1, latent_dim=AUDIO_LATENT_DIM, norm_video_kv=True)
    assert dit.bridges and all(bridge.norm_video_kv for bridge in dit.bridges)
    assert not any(param.requires_grad for param in dit.buffers()), "norm is state-free"
    # no new parameters: flag on adds zero tensors vs flag off
    dit_off = AudioDiT(dim=AUDIO_DIM, layers=2, heads=4, video_dim=INNER_DIM,
                       bridge_every=1, latent_dim=AUDIO_LATENT_DIM)
    assert sum(p.numel() for p in dit.parameters()) == sum(
        p.numel() for p in dit_off.parameters()
    )


def test_flag_on_makes_v2a_update_scale_invariant():
    torch.manual_seed(7)
    bridge = CrossModalBridge(video_dim=INNER_DIM, audio_dim=AUDIO_DIM, heads=4,
                              norm_video_kv=True)
    v = torch.randn(2, 6, INNER_DIM)
    a = torch.randn(2, 5, AUDIO_DIM)
    _, a_unit = bridge(v, a)
    _, a_scaled = bridge(v * WAN_SCALE, a)
    assert torch.allclose(a_unit, a_scaled, rtol=1e-4, atol=1e-5), (
        "positive per-token scaling must not change the v2a output once the "
        "video keys/values are normalised (the issue #4 defect)"
    )


def test_flag_off_stays_raw_and_scale_sensitive():
    torch.manual_seed(7)
    bridge = CrossModalBridge(video_dim=INNER_DIM, audio_dim=AUDIO_DIM, heads=4,
                              norm_video_kv=False)
    v = torch.randn(2, 6, INNER_DIM)
    a = torch.randn(2, 5, AUDIO_DIM)
    _, a_unit = bridge(v, a)
    _, a_scaled = bridge(v * WAN_SCALE, a)
    delta_unit = (a_unit - a).norm()
    delta_scaled = (a_scaled - a).norm()
    assert delta_scaled > 3 * delta_unit, (
        "without the flag the v2a update keeps tracking raw token scale; "
        "this documents the pre-fix behaviour the guard protects"
    )


def test_flag_on_bounds_v2a_update_at_wan_scale():
    torch.manual_seed(7)
    bridge = CrossModalBridge(video_dim=INNER_DIM, audio_dim=AUDIO_DIM, heads=4,
                              norm_video_kv=True)
    v = torch.randn(2, 6, INNER_DIM)
    a = torch.randn(2, 5, AUDIO_DIM)
    _, a2 = bridge(v * WAN_SCALE, a)
    delta = (a2 - a).norm()
    assert delta < 10 * a.norm(), "v2a update stays bounded at Wan hidden-state scale"


def test_video_residual_stream_untouched_by_flag():
    torch.manual_seed(7)
    bridge = CrossModalBridge(video_dim=INNER_DIM, audio_dim=AUDIO_DIM, heads=4,
                              norm_video_kv=True)
    v = torch.randn(2, 6, INNER_DIM)
    a = torch.randn(2, 5, AUDIO_DIM)
    v2, _ = bridge(v, a)
    # a2v residual must ride on the raw hidden states, not their normalised copy,
    # so the frozen Wan stream keeps its own activation statistics
    assert not torch.allclose(v2, torch.nn.functional.layer_norm(v, (INNER_DIM,)))
    assert v2.shape == v.shape


# ---- fail-closed guard -------------------------------------------------------


def test_wan_blocks_rejects_bridges_without_norm_flag():
    video = _video()
    trainer = JointAVTrainer(
        video,
        _audio(norm_video_kv=False),
        {"phase": "bridge", "bridge_location": "wan_blocks"},
    )
    with pytest.raises(ValueError, match="norm_video_kv"):
        trainer.train_step(_batch())
    assert all(not block._forward_hooks for block in video.transformer.blocks)


# ---- issue #4 regression: Wan-scale hidden states, bounded audio loss --------


def test_joint_step_audio_loss_bounded_with_wan_scale_hidden_states():
    torch.manual_seed(17)
    # gain 8 compounds over the four tiny blocks (x64 at the first bridge
    # hook) — far above latent scale, matching the issue #4 defect regime
    video = _video(block_gain=8.0)
    audio = _audio()
    trainer = JointAVTrainer(
        video,
        audio,
        {"phase": "bridge", "bridge_location": "wan_blocks", "lr": 0.0},
    )
    metrics = trainer.train_step(_batch())
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["audio_loss"] < 10.0, (
        f"audio loss {metrics['audio_loss']:.3f} must stay in the single digits "
        "with normalised video keys/values (issue #4 measured 79.0 without them)"
    )


def test_run_wan_smoke_enables_flag_only_for_wan_blocks():
    report = run_wan_smoke(
        _video(block_gain=8.0),
        device="cpu",
        dtype=torch.float32,
        audio_latent_dim=AUDIO_LATENT_DIM,
        audio_dim=AUDIO_DIM,
        audio_frames=5,
        text_length=3,
        bridge_location="wan_blocks",
    )
    assert report["status"] == "ok"
    assert report["bridge_location"] == "wan_blocks"
    assert report["metrics"]["audio_loss"] < 10.0, (
        f"smoke audio loss {report['metrics']['audio_loss']:.3f} must not "
        "reproduce the issue #4 red flag (79.0) at Wan hidden-state scale"
    )
    latent_report = run_wan_smoke(
        _video(),
        device="cpu",
        dtype=torch.float32,
        audio_latent_dim=AUDIO_LATENT_DIM,
        audio_dim=AUDIO_DIM,
        audio_frames=5,
        text_length=3,
        bridge_location="latent_pre_backbone",
    )
    assert latent_report["status"] == "ok"
    assert latent_report["bridge_video_dim"] == LATENT_CHANNELS


# ---- YAML wiring -------------------------------------------------------------


def test_audio_dit_kwargs_passes_norm_video_kv_with_default_off():
    base = {
        "dim": 64,
        "layers": 2,
        "heads": 4,
        "video_dim": 48,
        "bridge_every": 1,
        "latent_dim": 16,
    }
    assert audio_dit_kwargs(base)["norm_video_kv"] is False
    assert audio_dit_kwargs({**base, "norm_video_kv": True})["norm_video_kv"] is True

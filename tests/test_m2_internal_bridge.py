# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests for bridges inserted inside Wan transformer blocks."""

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_dit import AudioDiT
from theama.model.wan_adapter import (
    WanAdapterSpec,
    WanVideoAdapter,
    select_wan_bridge_blocks,
)
from theama.train.real_weight_smoke import run_wan_smoke
from theama.train.train_joint_av import JointAVTrainer

LATENT_CHANNELS = 8
INNER_DIM = 12
TEXT_DIM = 12
AUDIO_DIM = 32
AUDIO_LATENT_DIM = 16


class _TinyWanBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_states, encoder_hidden_states, timestep, rotary_emb):
        del encoder_hidden_states, timestep, rotary_emb
        return hidden_states * self.gain


class _TinyInternalWan(torch.nn.Module):
    """Diffusers-shaped fake with real patch tokens and forward-hook support."""

    def __init__(self, *, fail: bool = False):
        super().__init__()
        self.patch_embedding = torch.nn.Conv3d(LATENT_CHANNELS, INNER_DIM, kernel_size=1)
        self.blocks = torch.nn.ModuleList(_TinyWanBlock() for _ in range(4))
        self.proj_out = torch.nn.Linear(INNER_DIM, LATENT_CHANNELS)
        self.gradient_checkpointing = False
        self.fail = fail
        self.seen = {}

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
    ):
        assert return_dict is False
        self.seen = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_image": encoder_hidden_states_image,
        }
        tokens = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2).contiguous()
        for block in self.blocks:
            tokens = block(tokens, encoder_hidden_states, timestep, None)
        if self.fail:
            raise RuntimeError("intentional Wan failure")
        sample = self.proj_out(tokens).transpose(1, 2).reshape_as(hidden_states)
        tv = timestep.to(hidden_states.dtype)[:, None, None, None, None] / 1000.0
        return (sample + tv,)


def _video(*, fail: bool = False) -> WanVideoAdapter:
    spec = WanAdapterSpec(
        "tiny-internal-wan",
        "example/tiny-internal-wan",
        latent_channels=LATENT_CHANNELS,
        text_dim=TEXT_DIM,
    )
    return WanVideoAdapter(_TinyInternalWan(fail=fail), spec)


def _audio(*, video_dim: int = INNER_DIM, norm_video_kv: bool = True) -> AudioDiT:
    return AudioDiT(
        dim=AUDIO_DIM,
        layers=2,
        heads=4,
        video_dim=video_dim,
        bridge_every=1,
        latent_dim=AUDIO_LATENT_DIM,
        # wan_blocks feeds un-normalised Wan hidden states to the bridges
        # (issue #4 red flag); the adapter rejects bridges without this flag
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


def test_wan_bridge_schedule_is_even_distinct_and_validated():
    assert select_wan_bridge_blocks(30, 4) == [6, 14, 21, 29]
    assert select_wan_bridge_blocks(4, 2) == [1, 3]
    with pytest.raises(ValueError, match="only 2 Wan blocks"):
        select_wan_bridge_blocks(2, 3)
    with pytest.raises(ValueError, match="at least one"):
        select_wan_bridge_blocks(0, 1)


def test_internal_joint_step_trains_both_directions_and_removes_hooks():
    torch.manual_seed(17)
    video = _video()
    audio = _audio()
    trainer = JointAVTrainer(
        video,
        audio,
        {"phase": "bridge", "bridge_location": "wan_blocks", "lr": 0.01},
    )
    before = [param.detach().clone() for param in audio.bridges.parameters()]

    metrics = trainer.train_step(_batch())

    assert all(math.isfinite(value) for value in metrics.values())
    assert all(not param.requires_grad for param in video.parameters())
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, audio.bridges.parameters())
    )
    for bridge in audio.bridges:
        assert bridge.v2a.out_proj.weight.grad is not None
        assert bridge.a2v.out_proj.weight.grad is not None
    assert all(not block._forward_hooks for block in video.transformer.blocks)
    assert video.transformer.seen["timestep"].dtype == torch.long


def test_internal_real_weight_smoke_supports_lora_warmup():
    video = _video()
    report = run_wan_smoke(
        video,
        device="cpu",
        dtype=torch.float32,
        audio_latent_dim=AUDIO_LATENT_DIM,
        audio_dim=AUDIO_DIM,
        audio_frames=5,
        text_length=3,
        learning_rate=0.01,
        bridge_location="wan_blocks",
        bridge_lora_rank=2,
        bridge_lora_alpha=4,
    )
    assert report["status"] == "ok"
    assert report["bridge_location"] == "wan_blocks"
    assert report["bridge_video_dim"] == INNER_DIM
    assert report["wan_bridge_blocks"] == [1, 3]
    assert report["bridge_lora_rank"] == 2
    assert report["trainable_parameters"] == 352
    assert report["updated_bridge_tensors"] > 0
    assert report["wan_frozen"] is True


def test_internal_bridge_rejects_latent_dim_audio_and_checkpointing():
    video = _video()
    mismatched = JointAVTrainer(
        video,
        _audio(video_dim=LATENT_CHANNELS),
        {"bridge_location": "wan_blocks"},
    )
    with pytest.raises(ValueError, match="Wan inner dim"):
        mismatched.train_step(_batch())

    checkpointed = _video()
    checkpointed.transformer.gradient_checkpointing = True
    trainer = JointAVTrainer(
        checkpointed,
        _audio(),
        {"bridge_location": "wan_blocks"},
    )
    with pytest.raises(NotImplementedError, match="gradient checkpointing"):
        trainer.train_step(_batch())
    assert all(not block._forward_hooks for block in checkpointed.transformer.blocks)


def test_internal_bridge_removes_hooks_after_wan_failure():
    video = _video(fail=True)
    audio = _audio()
    with pytest.raises(RuntimeError, match="intentional Wan failure"):
        video.predict_joint_velocity(
            torch.randn(1, LATENT_CHANNELS, 1, 2, 2),
            torch.tensor([0.5]),
            torch.randn(1, 3, TEXT_DIM),
            audio_dit=audio,
            audio_latent=torch.randn(1, 5, AUDIO_LATENT_DIM),
            audio_cond=torch.randn(1, AUDIO_DIM),
        )
    assert all(not block._forward_hooks for block in video.transformer.blocks)


def test_real_weight_cli_exposes_internal_bridge_and_lora_flags():
    path = ROOT / "scripts" / "smoke_m2_real_weights.py"
    spec = importlib.util.spec_from_file_location("smoke_m2_real_weights_internal", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.build_parser().parse_args(
        [
            "wan",
            "--out",
            "smoke/wan",
            "--bridge-location",
            "wan_blocks",
            "--bridge-lora-rank",
            "8",
            "--bridge-lora-alpha",
            "16",
        ]
    )
    assert args.bridge_location == "wan_blocks"
    assert args.bridge_lora_rank == 8
    assert args.bridge_lora_alpha == 16

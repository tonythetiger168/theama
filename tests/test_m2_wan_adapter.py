# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Wan 2.2 adapter and 5-D joint A/V smoke path."""

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_codec import DAC_44K
from theama.model.audio_dit import AudioDiT
from theama.model.wan_adapter import (
    WanAdapterSpec,
    WanVideoAdapter,
    flatten_video_latents,
    get_wan_adapter_spec,
    scale_wan_timestep,
    unflatten_video_tokens,
)
from theama.train.train_joint_av import JointAVTrainer


def _tiny_spec(image_conditioned: bool = False) -> WanAdapterSpec:
    return WanAdapterSpec(
        "tiny-wan",
        "example/tiny-wan",
        latent_channels=8,
        text_dim=12,
        image_conditioned=image_conditioned,
    )


class _FakeWanTransformer(torch.nn.Module):
    """Shape-true Wan double; no diffusers import or weight download."""

    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.25))
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
        tv = timestep.to(hidden_states.dtype)[:, None, None, None, None] / 1000.0
        return (hidden_states * self.gain + tv,)


def test_wan_registry_matches_official_transformer_configs():
    a14b = get_wan_adapter_spec("t2v-a14b")
    dev = get_wan_adapter_spec("ti2v-5b")
    assert a14b.latent_channels == 16
    assert dev.latent_channels == 48
    assert a14b.text_dim == dev.text_dim == 4096
    assert a14b.repo_id.endswith("-Diffusers")
    assert dev.repo_id.endswith("-Diffusers")
    with pytest.raises(ValueError, match="unknown Wan adapter"):
        get_wan_adapter_spec("wan-3-api")


def test_latent_token_layout_roundtrip_preserves_values_and_grad():
    latent = torch.randn(2, 8, 3, 2, 2, requires_grad=True)
    tokens = flatten_video_latents(latent)
    assert tokens.shape == (2, 12, 8)
    restored = unflatten_video_tokens(tokens, tuple(latent.shape))
    assert torch.equal(restored, latent)
    restored.square().mean().backward()
    assert latent.grad is not None and torch.all(latent.grad != 0)
    with pytest.raises(ValueError, match="do not match"):
        unflatten_video_tokens(tokens[:, :-1], tuple(latent.shape))


def test_wan_timestep_scaling_and_validation():
    t = scale_wan_timestep(torch.tensor([0.0, 0.5, 1.0]))
    assert t.dtype == torch.long
    assert t.tolist() == [0, 500, 1000]
    with pytest.raises(ValueError, match="shape"):
        scale_wan_timestep(torch.zeros(2, 1))


def test_wan_adapter_real_forward_contract_and_frozen_backbone():
    transformer = _FakeWanTransformer()
    adapter = WanVideoAdapter(transformer, _tiny_spec())
    hidden = torch.randn(3, 8, 2, 2, 2)
    text = torch.randn(3, 5, 12)
    out = adapter.predict_velocity(hidden, torch.tensor([0.0, 0.5, 1.0]), text)
    assert out.shape == hidden.shape
    assert transformer.seen["timestep"].tolist() == [0, 500, 1000]
    assert transformer.seen["encoder_hidden_states"] is text
    assert all(not p.requires_grad for p in adapter.parameters())
    adapter.train()
    assert not transformer.training


def test_wan_adapter_rejects_bad_shapes_and_missing_image_cond():
    adapter = WanVideoAdapter(_FakeWanTransformer(), _tiny_spec())
    with pytest.raises(ValueError, match="latent channels"):
        adapter.predict_velocity(
            torch.randn(1, 4, 2, 2, 2), torch.zeros(1), torch.randn(1, 2, 12)
        )
    with pytest.raises(ValueError, match="text embeddings"):
        adapter.predict_velocity(
            torch.randn(1, 8, 2, 2, 2), torch.zeros(1), torch.randn(1, 12)
        )
    image_adapter = WanVideoAdapter(
        _FakeWanTransformer(), _tiny_spec(image_conditioned=True)
    )
    with pytest.raises(ValueError, match="requires image conditioning"):
        image_adapter.predict_velocity(
            torch.randn(1, 8, 2, 2, 2), torch.zeros(1), torch.randn(1, 2, 12)
        )


def test_joint_trainer_5d_wan_smoke_uses_sequence_text_cond():
    torch.manual_seed(11)
    transformer = _FakeWanTransformer()
    video = WanVideoAdapter(transformer, _tiny_spec())
    audio = AudioDiT(
        dim=32,
        layers=2,
        heads=4,
        video_dim=8,
        bridge_every=1,
        latent_dim=16,
    )
    trainer = JointAVTrainer(video, audio, {"phase": "bridge", "lr": 0.01})
    batch = {
        "audio_latent": torch.randn(1, 5, 16),
        "video_tokens": torch.randn(1, 8, 2, 2, 2),
        "cond": torch.randn(1, 32),
        "video_cond": torch.randn(1, 3, 12),
        "audio_noise": torch.randn(1, 5, 16),
        "video_noise": torch.randn(1, 8, 2, 2, 2),
    }

    metrics = trainer.train_step(batch)

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert transformer.seen["hidden_states"].shape == batch["video_tokens"].shape
    assert transformer.seen["encoder_hidden_states"] is batch["video_cond"]
    assert transformer.seen["timestep"].dtype == torch.long
    assert all(not p.requires_grad for p in video.parameters())
    assert all(p.requires_grad for p in audio.bridges.parameters())


def test_joint_trainer_rejects_empty_bridge_schedule():
    video = WanVideoAdapter(_FakeWanTransformer(), _tiny_spec())
    audio = AudioDiT(dim=32, layers=2, heads=4, video_dim=8, latent_dim=16)
    with pytest.raises(ValueError, match="no trainable parameters"):
        JointAVTrainer(video, audio)


def test_joint_trainer_rejects_bad_video_cond():
    video = WanVideoAdapter(_FakeWanTransformer(), _tiny_spec())
    audio = AudioDiT(dim=32, layers=2, heads=4, video_dim=8, bridge_every=1, latent_dim=16)
    trainer = JointAVTrainer(video, audio)
    with pytest.raises(ValueError, match="video_cond"):
        trainer.train_step(
            {
                "audio_latent": torch.randn(1, 5, 16),
                "video_tokens": torch.randn(1, 8, 2, 2, 2),
                "cond": torch.randn(1, 32),
                "video_cond": torch.randn(1, 12),
            }
        )


def test_m2_yaml_wan_and_dac_dimensions_agree():
    cfg = yaml.safe_load((ROOT / "configs" / "m2_joint_av.yaml").read_text())
    audio = cfg["model"]["audio"]
    video = cfg["model"]["video"]
    spec = get_wan_adapter_spec(video["adapter"])
    dev_spec = get_wan_adapter_spec(video["dev_adapter"])
    assert audio["latent_dim"] == DAC_44K.dit_token_dim
    assert audio["video_dim"] == video["latent_channels"] == spec.latent_channels
    assert video["dev_latent_channels"] == dev_spec.latent_channels
    assert video["text_dim"] == spec.text_dim == 4096
    assert video["bridge_location"] == "latent_pre_backbone"

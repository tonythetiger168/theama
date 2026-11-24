# SPDX-License-Identifier: Apache-2.0
"""CPU tests for real-weight smoke orchestration; no checkpoint downloads."""

import importlib.util
import math
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_codec import AudioCodecSpec, MockAudioCodec
from theama.model.wan_adapter import WanAdapterSpec, WanVideoAdapter
from theama.train.real_weight_smoke import run_dac_smoke, run_wan_smoke


class _FakeWanTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1.25))

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        return_dict=True,
    ):
        assert return_dict is False
        tv = timestep.to(hidden_states.dtype)[:, None, None, None, None] / 1000.0
        return (hidden_states * self.gain + tv,)


def _tiny_wan() -> WanVideoAdapter:
    spec = WanAdapterSpec("tiny-wan", "example/tiny-wan", latent_channels=8, text_dim=12)
    return WanVideoAdapter(_FakeWanTransformer(), spec)


def _tiny_codec() -> MockAudioCodec:
    spec = AudioCodecSpec(
        name="tiny-dac",
        repo_id="example/tiny-dac",
        license="MIT",
        sample_rate=8,
        hop_length=2,
        latent_dim=4,
        n_codebooks=1,
        codebook_size=8,
        stereo_layout="mid_side",
    )
    return MockAudioCodec(spec)


def test_dac_smoke_checks_stereo_layout_and_reports_audio():
    report, waveform, restored = run_dac_smoke(_tiny_codec(), device="cpu", seconds=1.0)
    assert report["status"] == "ok"
    assert report["input_shape"] == [1, 2, 8]
    assert report["latent_shape"] == [1, 4, 8]
    assert report["restored_shape"] == [1, 2, 8]
    assert waveform.shape == restored.shape
    assert math.isfinite(report["reconstruction_relative_l2"])
    assert "no perceptual-quality gate" in report["note"]


def test_wan_smoke_runs_real_backward_and_updates_only_audio_bridges():
    video = _tiny_wan()
    report = run_wan_smoke(
        video,
        device="cpu",
        dtype=torch.float32,
        audio_latent_dim=16,
        audio_dim=32,
        audio_frames=5,
        text_length=3,
        learning_rate=0.01,
    )
    assert report["status"] == "ok"
    assert report["video_shape"] == [1, 8, 1, 2, 2]
    assert report["text_cond_shape"] == [1, 3, 12]
    assert report["bridge_grad_tensors"] > 0
    assert report["updated_bridge_tensors"] > 0
    assert report["max_bridge_delta"] > 0
    assert report["wan_frozen"] is True
    assert all(math.isfinite(value) for value in report["metrics"].values())


def test_wan_loader_accepts_local_diffusers_path_without_download(monkeypatch):
    calls = {}

    class _Loader:
        @classmethod
        def from_pretrained(cls, source, subfolder, torch_dtype):
            calls.update(source=source, subfolder=subfolder, torch_dtype=torch_dtype)
            return _FakeWanTransformer()

    fake_diffusers = types.ModuleType("diffusers")
    fake_diffusers.WanTransformer3DModel = _Loader
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)

    adapter = WanVideoAdapter.from_pretrained(
        "ti2v-5b",
        model_path="/models/ti2v-5b",
        device="cpu",
        dtype=torch.float32,
    )
    assert adapter.spec.latent_channels == 48
    assert calls == {
        "source": "/models/ti2v-5b",
        "subfolder": "transformer",
        "torch_dtype": torch.float32,
    }


def test_real_weight_cli_exposes_dac_and_wan_commands():
    path = ROOT / "scripts" / "smoke_m2_real_weights.py"
    spec = importlib.util.spec_from_file_location("smoke_m2_real_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module.build_parser()
    dac = parser.parse_args(["dac", "--out", "smoke/dac", "--device", "cpu"])
    wan = parser.parse_args(
        ["wan", "--model-path", "/models/ti2v", "--out", "smoke/wan", "--dtype", "float32"]
    )
    assert dac.command == "dac" and dac.handler is module._dac
    assert wan.command == "wan" and wan.handler is module._wan
    assert wan.model_path == "/models/ti2v"

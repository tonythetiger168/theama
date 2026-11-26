# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the Wan 2.2 DiT backbone wrapper (model/dit.py).

Last un-pinned module under src/theama/model/: wan_adapter, audio_dit,
lora, codecs and rope3d all have coverage; dit.py is the thin lazy-loading
wrapper that keeps the rest of Theama away from backbone internals. These
tests pin the decided contracts so the wrapper cannot silently drift:

- Repo id constants: A14B full + TI2V-5B dev (Apache-2.0 Wan-AI repos).
- BackboneConfig defaults: A14B primary, TI2V-5B dev, bf16, MoE expert
  boundary 0.875 (from the official Wan 2.2 model card).
- VideoBackbone wiring: None -> default config, dev flag stored, _pipe
  starts as None (lazy).
- load_t2v: dev flag selects dev_repo vs repo_id; VAE forced to float32
  from the "vae" subfolder; pipeline gets config.dtype; _pipe cached.
  (diffusers is faked via sys.modules -- no GPU, no download.)
- generate(): Wan calling convention -- size (W, H) maps to width/height
  kwargs, num_frames = seconds * fps + 1 (Wan 8n+1 convention), 40
  inference steps, negative_prompt default empty, seed -> deterministic
  torch.Generator, runs under torch.no_grad, returns out.frames[0].
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from theama.model.dit import (
    A14B_REPO,
    TI2V5B_REPO,
    BackboneConfig,
    VideoBackbone,
)


# ---------------------------------------------------------------------------
# Repo id constants + BackboneConfig
# ---------------------------------------------------------------------------


class TestRepoConstants:
    def test_a14b_repo_constant(self):
        assert A14B_REPO == "Wan-AI/Wan2.2-T2V-A14B"

    def test_ti2v5b_repo_constant(self):
        assert TI2V5B_REPO == "Wan-AI/Wan2.2-TI2V-5B"


class TestBackboneConfig:
    def test_defaults(self):
        c = BackboneConfig()
        assert c.repo_id == A14B_REPO
        assert c.dev_repo == TI2V5B_REPO
        assert c.dtype == torch.bfloat16
        assert c.expert_boundary == pytest.approx(0.875)

    def test_custom_values_stored(self):
        c = BackboneConfig(repo_id="r", dev_repo="d",
                           dtype=torch.float16, expert_boundary=0.5)
        assert (c.repo_id, c.dev_repo) == ("r", "d")
        assert c.dtype == torch.float16
        assert c.expert_boundary == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# VideoBackbone wiring
# ---------------------------------------------------------------------------


class TestVideoBackboneInit:
    def test_default_config_when_none(self):
        bb = VideoBackbone()
        assert isinstance(bb.config, BackboneConfig)
        assert bb.config.repo_id == A14B_REPO
        assert bb.dev is False

    def test_custom_config_and_dev_stored(self):
        cfg = BackboneConfig(repo_id="x")
        bb = VideoBackbone(cfg, dev=True)
        assert bb.config is cfg
        assert bb.dev is True

    def test_pipe_lazy_starts_none(self):
        assert VideoBackbone()._pipe is None


# ---------------------------------------------------------------------------
# load_t2v (diffusers faked via sys.modules)
# ---------------------------------------------------------------------------


def _install_fake_diffusers(monkeypatch):
    """Register a fake diffusers module; returns a dict recording calls."""
    calls = {}
    vae_obj = object()

    class FakeVAE:
        @staticmethod
        def from_pretrained(repo, subfolder=None, torch_dtype=None):
            calls["vae"] = {"repo": repo, "subfolder": subfolder,
                            "torch_dtype": torch_dtype}
            return vae_obj

    class FakePipe:
        @staticmethod
        def from_pretrained(repo, vae=None, torch_dtype=None):
            calls["pipe"] = {"repo": repo, "vae": vae,
                             "torch_dtype": torch_dtype}
            return "PIPE"

    mod = types.ModuleType("diffusers")
    mod.AutoencoderKLWan = FakeVAE
    mod.WanPipeline = FakePipe
    monkeypatch.setitem(sys.modules, "diffusers", mod)
    calls["vae_obj"] = vae_obj
    return calls


class TestLoadT2V:
    def test_dev_selects_ti2v5b_repo(self, monkeypatch):
        calls = _install_fake_diffusers(monkeypatch)
        bb = VideoBackbone(dev=True)
        pipe = bb.load_t2v()
        assert calls["vae"]["repo"] == TI2V5B_REPO
        assert calls["pipe"]["repo"] == TI2V5B_REPO
        assert pipe == "PIPE"
        assert bb._pipe == "PIPE"

    def test_full_selects_a14b_repo(self, monkeypatch):
        calls = _install_fake_diffusers(monkeypatch)
        VideoBackbone().load_t2v()
        assert calls["vae"]["repo"] == A14B_REPO
        assert calls["pipe"]["repo"] == A14B_REPO

    def test_vae_float32_from_vae_subfolder(self, monkeypatch):
        calls = _install_fake_diffusers(monkeypatch)
        VideoBackbone().load_t2v()
        assert calls["vae"]["subfolder"] == "vae"
        assert calls["vae"]["torch_dtype"] == torch.float32

    def test_pipe_gets_config_dtype_and_vae(self, monkeypatch):
        calls = _install_fake_diffusers(monkeypatch)
        cfg = BackboneConfig(dtype=torch.float16)
        VideoBackbone(cfg).load_t2v()
        assert calls["pipe"]["torch_dtype"] == torch.float16
        assert calls["pipe"]["vae"] is calls["vae_obj"]


# ---------------------------------------------------------------------------
# generate() calling convention (fake pipeline object)
# ---------------------------------------------------------------------------


class _FakePipe:
    def __init__(self):
        self.kwargs = None
        self.grad_enabled = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        self.grad_enabled = torch.is_grad_enabled()
        return SimpleNamespace(frames=["frames0", "frames1"])


class TestGenerate:
    def test_default_call_contract(self):
        pipe = _FakePipe()
        VideoBackbone().generate(pipe, "a cat")
        k = pipe.kwargs
        assert k["prompt"] == "a cat"
        assert k["negative_prompt"] == ""
        assert k["width"] == 1280
        assert k["height"] == 720
        assert k["num_frames"] == 5 * 24 + 1  # 121, Wan 8n+1 convention
        assert k["num_inference_steps"] == 40
        assert k["generator"] is None

    def test_size_maps_width_height(self):
        pipe = _FakePipe()
        VideoBackbone().generate(pipe, "p", size=(640, 360))
        assert pipe.kwargs["width"] == 640
        assert pipe.kwargs["height"] == 360

    def test_num_frames_custom_seconds_and_fps(self):
        pipe = _FakePipe()
        VideoBackbone().generate(pipe, "p", seconds=2, fps=8)
        assert pipe.kwargs["num_frames"] == 17

    def test_negative_prompt_passthrough(self):
        pipe = _FakePipe()
        VideoBackbone().generate(pipe, "p", negative_prompt="blur")
        assert pipe.kwargs["negative_prompt"] == "blur"

    def test_seed_builds_deterministic_generator(self):
        pipe = _FakePipe()
        VideoBackbone().generate(pipe, "p", seed=7)
        g = pipe.kwargs["generator"]
        assert isinstance(g, torch.Generator)
        expected = torch.Generator().manual_seed(7)
        assert torch.equal(torch.rand(4, generator=g),
                           torch.rand(4, generator=expected))

    def test_returns_first_frames_batch(self):
        pipe = _FakePipe()
        out = VideoBackbone().generate(pipe, "p")
        assert out == "frames0"

    def test_runs_under_no_grad(self):
        pipe = _FakePipe()
        VideoBackbone().generate(pipe, "p")
        assert pipe.grad_enabled is False

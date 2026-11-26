# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/train_joint_av.py CLI wiring (P1b/M2 train entry).

The runner core (step loop / checkpoints / resume / non-finite guard) is owned
by test_m2_joint_av_runner.py, and the --smoke path by its subprocess test.
What had no coverage is the ``_real`` CLI path — the exact code that burns GPU
hours on the bridge-warmup run once issue #4's red flag clears: argument
gating (--smoke vs --av-cache/--text-cache), YAML config loading, frozen Wan
backbone construction args, cond-pooler text_dim wiring, the joint-cache
loader calls, and the args -> RunnerConfig mapping. A wiring bug here (dropped
resume flag, swapped cache paths, wrong device) only surfaces on the pod.

Strategy: real torch/yaml, real AudioDiT (tiny dims), real JointAVTrainer,
real config helpers (audio_dit_kwargs / trainer_config / text_dim_from_config)
and the real RunnerConfig dataclass; fakes only the GPU/IO edges —
WanVideoAdapter.from_pretrained (no Wan weights download), the joint-cache
loader functions (no .pt fixtures), and JointAVRunner (no step loop). The
script does its imports inside ``_real()``, so monkeypatched module attributes
are picked up at call time.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "train_joint_av_cli", ROOT / "scripts" / "train_joint_av.py"
)
tja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tja)  # inserts ROOT/src into sys.path itself

from theama.data import joint_latent_loader  # noqa: E402
from theama.model import wan_adapter  # noqa: E402
from theama.model.audio_dit import AudioDiT  # noqa: E402
from theama.train import joint_av_runner  # noqa: E402
from theama.train.joint_av_runner import RunnerConfig, StubVideoBackbone  # noqa: E402
from theama.train.train_joint_av import JointAVTrainer  # noqa: E402

TINY_CFG = {
    "model": {
        "audio": {
            "dim": 32, "layers": 2, "heads": 4, "video_dim": 4,
            "bridge_every": 1, "latent_dim": 16,
        },
        "video": {
            "adapter": "t2v-a14b", "text_dim": 24,
            "bridge_location": "latent_pre_backbone",
        },
    },
    "training": {
        "phase": "bridge", "lr": 0.001,
        "bridge_lora_rank": 4, "bridge_lora_alpha": 8,
    },
}
SUMMARY = {"final_phase": "full", "steps_completed": 42, "resumed_from": None}


class _FakeRecords:
    """Stand-in for JointLatentRecords: len() + dropped are all _real reads."""

    def __init__(self, n: int = 7, dropped: int = 1):
        self._n = n
        self.dropped = dropped

    def __len__(self) -> int:
        return self._n


def _write_config(tmp_path: Path, cfg: dict | None = None) -> Path:
    path = tmp_path / "m2_tiny.yaml"
    path.write_text(yaml.safe_dump(cfg or TINY_CFG), encoding="utf-8")
    return path


def _install_fakes(monkeypatch, *, records_n=7, dropped=1, summary=None):
    """Patch the GPU/IO edges; return the call log."""
    calls: dict = {
        "from_pretrained": [], "load_records": [], "build_loader": [],
        "runner_init": [], "runner_train": 0, "seeds": [],
    }
    stub = StubVideoBackbone()
    calls["stub"] = stub
    sentinel_loader = object()
    calls["loader"] = sentinel_loader

    @classmethod
    def fake_from_pretrained(cls, name, model_path=None, device=None, dtype=None):
        calls["from_pretrained"].append(
            {"name": name, "model_path": model_path, "device": device, "dtype": dtype}
        )
        return stub

    def fake_load_records(av_cache, text_cache):
        calls["load_records"].append((av_cache, text_cache))
        return _FakeRecords(records_n, dropped)

    def fake_build_loader(records, batch_size, num_workers, cond_pooler):
        calls["build_loader"].append(
            {"records": records, "batch_size": batch_size,
             "num_workers": num_workers, "cond_pooler": cond_pooler}
        )
        return sentinel_loader

    class FakeRunner:
        def __init__(self, trainer, batches, cfg):
            calls["runner_init"].append({"trainer": trainer, "batches": batches, "cfg": cfg})

        def train(self):
            calls["runner_train"] += 1
            return summary if summary is not None else SUMMARY

    monkeypatch.setattr(
        wan_adapter.WanVideoAdapter, "from_pretrained", fake_from_pretrained
    )
    monkeypatch.setattr(joint_latent_loader, "load_joint_latent_records", fake_load_records)
    monkeypatch.setattr(
        joint_latent_loader, "build_joint_latent_dataloader", fake_build_loader
    )
    monkeypatch.setattr(joint_av_runner, "JointAVRunner", FakeRunner)
    monkeypatch.setattr(torch, "manual_seed", lambda s: calls["seeds"].append(s))
    return calls


def _argv(tmp_path: Path, extra=(), cfg: dict | None = None) -> list[str]:
    return [
        "train_joint_av.py",
        "--config", str(_write_config(tmp_path, cfg)),
        "--av-cache", str(tmp_path / "av_cache"),
        "--text-cache", str(tmp_path / "text_cache"),
        "--out", str(tmp_path / "run"),
        *extra,
    ]


def _run(monkeypatch, tmp_path, extra=(), cfg=None):
    """Run main() on the _real path with fakes; return (exit_code, calls)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, extra, cfg))
    code = tja.main()
    return code, calls


# ------------------------------------------------------------------ gating


@pytest.mark.parametrize("drop", ["both", "text"])
def test_real_requires_caches_without_smoke(monkeypatch, tmp_path, capsys, drop):
    argv = [
        "train_joint_av.py",
        "--config", str(_write_config(tmp_path)),
        "--out", str(tmp_path / "run"),
    ]
    if drop == "text":
        argv += ["--av-cache", str(tmp_path / "av_cache")]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as excinfo:
        tja.main()
    assert excinfo.value.code == 2
    assert "--av-cache and --text-cache are required" in capsys.readouterr().err


# ------------------------------------------------------------- happy path


def test_real_happy_path_wiring(monkeypatch, tmp_path, capsys):
    code, calls = _run(monkeypatch, tmp_path)
    assert code == 0

    # Frozen Wan backbone: adapter id from the config's video section, no
    # local weights, resolved device, hardcoded bf16 (runner dtype is a
    # separate knob, see the arg-mapping test).
    assert calls["from_pretrained"] == [
        {"name": "t2v-a14b", "model_path": None, "device": "cpu",
         "dtype": torch.bfloat16}
    ]
    assert calls["seeds"] == [0]  # default --seed 0

    # Real trainer over the real tiny AudioDiT, config-mapped.
    init = calls["runner_init"]
    assert len(init) == 1
    trainer, batches, run_cfg = (init[0][k] for k in ("trainer", "batches", "cfg"))
    assert isinstance(trainer, JointAVTrainer)
    assert trainer.video is calls["stub"]
    assert isinstance(trainer.audio, AudioDiT)
    assert trainer.config["phase"] == "bridge"
    assert trainer.config["lr"] == 0.001
    assert trainer.config["bridge_location"] == "latent_pre_backbone"
    # cond pooler rides the config's video.text_dim onto the audio dim
    assert trainer.audio.cond_pooler.in_features == 24
    assert trainer.audio.cond_pooler.out_features == 32

    # Joint caches: exact CLI paths, cond pooling deferred to the runner.
    av = str(tmp_path / "av_cache")
    txt = str(tmp_path / "text_cache")
    assert calls["load_records"] == [(av, txt)]
    loader_call = calls["build_loader"]
    assert len(loader_call) == 1
    assert isinstance(loader_call[0]["records"], _FakeRecords)
    assert loader_call[0]["batch_size"] == 1
    assert loader_call[0]["num_workers"] == 0
    assert loader_call[0]["cond_pooler"] is None
    # the runner trains on the loader built from those records
    assert batches is calls["loader"]
    assert calls["runner_train"] == 1

    # RunnerConfig defaults straight from the argparse defaults.
    assert isinstance(run_cfg, RunnerConfig)
    assert run_cfg.out_dir == str(tmp_path / "run")
    assert run_cfg.max_steps == 1000
    assert run_cfg.bridge_steps is None
    assert run_cfg.log_every == 10
    assert run_cfg.save_every == 100
    assert run_cfg.device == "cpu"
    assert run_cfg.dtype == "fp32"
    assert run_cfg.resume is None

    out = capsys.readouterr().out
    assert "loading frozen Wan backbone: t2v-a14b" in out
    assert "joint cache: 7 samples (dropped: 1)" in out
    assert json.dumps(SUMMARY, indent=2) in out


# --------------------------------------------------------- args -> RunnerConfig


def test_real_runner_config_maps_all_flags(monkeypatch, tmp_path):
    code, calls = _run(
        monkeypatch, tmp_path,
        extra=[
            "--steps", "55", "--bridge-steps", "7", "--batch-size", "4",
            "--num-workers", "2", "--log-every", "3", "--save-every", "11",
            "--resume", "auto", "--dtype", "bf16", "--seed", "123",
        ],
    )
    assert code == 0
    run_cfg = calls["runner_init"][0]["cfg"]
    assert run_cfg.max_steps == 55
    assert run_cfg.bridge_steps == 7
    assert run_cfg.log_every == 3
    assert run_cfg.save_every == 11
    assert run_cfg.dtype == "bf16"
    assert run_cfg.resume == "auto"
    assert calls["build_loader"][0]["batch_size"] == 4
    assert calls["build_loader"][0]["num_workers"] == 2
    assert calls["seeds"] == [123]
    # --dtype drives the runner only; the frozen backbone stays bf16.
    assert calls["from_pretrained"][0]["dtype"] == torch.bfloat16


# ------------------------------------------------------------------- device


def test_real_device_defaults_to_cpu_without_cuda(monkeypatch, tmp_path):
    code, calls = _run(monkeypatch, tmp_path)  # is_available patched False
    assert code == 0
    assert calls["runner_init"][0]["cfg"].device == "cpu"
    assert calls["from_pretrained"][0]["device"] == "cpu"


def test_real_device_prefers_cuda_when_available(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = _install_fakes(monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path))
    assert tja.main() == 0
    assert calls["runner_init"][0]["cfg"].device == "cuda"
    assert calls["from_pretrained"][0]["device"] == "cuda"


def test_real_device_flag_overrides_detection(monkeypatch, tmp_path):
    code, calls = _run(monkeypatch, tmp_path, extra=["--device", "cuda"])
    assert code == 0  # flag wins even though is_available was patched False
    assert calls["runner_init"][0]["cfg"].device == "cuda"
    assert calls["from_pretrained"][0]["device"] == "cuda"


# ------------------------------------------------------------ config plumbing


def test_real_video_weights_pass_through(monkeypatch, tmp_path):
    code, calls = _run(
        monkeypatch, tmp_path, extra=["--video-weights", "/tmp/wan-local"]
    )
    assert code == 0
    assert calls["from_pretrained"][0]["model_path"] == "/tmp/wan-local"


def test_real_config_override_drives_adapter_dims_and_text_dim(
    monkeypatch, tmp_path
):
    alt = {
        "model": {
            "audio": {
                "dim": 64, "layers": 3, "heads": 8, "video_dim": 8,
                "bridge_every": 2, "latent_dim": 8, "norm_video_kv": True,
            },
            "video": {
                "adapter": "ti2v-5b", "text_dim": 48,
                "bridge_location": "wan_blocks",
            },
        },
        "training": {"phase": "full", "lr": 0.0001},
    }
    code, calls = _run(monkeypatch, tmp_path, cfg=alt)
    assert code == 0
    assert calls["from_pretrained"][0]["name"] == "ti2v-5b"
    trainer = calls["runner_init"][0]["trainer"]
    assert trainer.config["phase"] == "full"
    # trainer_config merges the video section's bridge_location
    assert trainer.config["bridge_location"] == "wan_blocks"
    # audio_dit_kwargs maps the audio section verbatim (norm_video_kv incl.)
    assert trainer.audio.cond_pooler.in_features == 48
    assert trainer.audio.cond_pooler.out_features == 64

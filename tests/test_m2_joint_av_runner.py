# SPDX-License-Identifier: Apache-2.0
"""CPU tests for theama.train.joint_av_runner (P1b joint training runner).

Covers YAML config plumbing, cond-pooler wiring across phases, the step loop
(metrics JSONL, checkpoints, bridge->full schedule), exact checkpoint resume
(phase/weights/optimizer/RNG, LoRA bridge and merged full variants), the
non-finite fail-closed guard, and the train_joint_av.py CLI smoke path.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_cache import plan_av_encoding, run_av_encoding  # noqa: E402
from theama.data.joint_latent_loader import (  # noqa: E402
    build_joint_latent_dataloader,
    load_joint_latent_records,
)
from theama.data.latent_cache import save_latent_atomic  # noqa: E402
from theama.data.text_cache import plan_text_encoding, run_text_encoding  # noqa: E402
from theama.model.audio_dit import AudioDiT  # noqa: E402
from theama.model.lora import iter_lora_modules  # noqa: E402
from theama.train.joint_av_runner import (  # noqa: E402
    JointAVRunner,
    JointAVRunnerError,
    RunnerConfig,
    StubVideoBackbone,
    audio_dit_kwargs,
    build_cond_pooler,
    latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
    text_dim_from_config,
    trainer_config,
)
from theama.train.train_joint_av import JointAVTrainer  # noqa: E402

DIMS = dict(dim=32, layers=4, heads=4, video_dim=4, bridge_every=2, latent_dim=16)
TEXT_DIM = 16
LORA = {"bridge_lora_rank": 4, "bridge_lora_alpha": 8}


def _audio() -> AudioDiT:
    return AudioDiT(**DIMS)


def _trainer(cfg: dict | None = None) -> JointAVTrainer:
    audio = _audio()
    build_cond_pooler(audio, TEXT_DIM)
    return JointAVTrainer(StubVideoBackbone(), audio, {"phase": "bridge", "lr": 1e-3, **(cfg or {})})


def _batch(batch_size: int = 2, *, nan: bool = False) -> dict:
    lt = 8
    seq_lens = torch.randint(1, lt + 1, (batch_size,))
    audio = torch.randn(batch_size, 6, DIMS["latent_dim"])
    if nan:
        audio[0, 0, 0] = float("nan")
    return {
        "video_tokens": torch.randn(batch_size, DIMS["video_dim"], 2, 3, 3),
        "audio_latent": audio,
        "video_cond": torch.randn(batch_size, lt, TEXT_DIM),
        "text_mask": torch.arange(lt)[None, :] < seq_lens[:, None],
    }


def _batches(n: int = 4) -> list[dict]:
    return [_batch() for _ in range(n)]


def _run_cfg(out: Path, **kw) -> RunnerConfig:
    defaults = dict(out_dir=str(out), max_steps=4, log_every=1, save_every=2)
    return RunnerConfig(**(defaults | kw))


def _read_metrics(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]


def _step_records(out: Path) -> list[dict]:
    """Per-step metric records only (excludes resume/phase_switch events)."""
    return [r for r in _read_metrics(out) if "loss" in r]


# ---- config plumbing ----------------------------------------------------------


def test_yaml_config_maps_to_runner_builders():
    cfg = yaml.safe_load((ROOT / "configs" / "m2_joint_av.yaml").read_text())
    kwargs = audio_dit_kwargs(cfg["model"]["audio"])
    assert kwargs == {
        "dim": 1024, "layers": 12, "heads": 16,
        "video_dim": 16, "bridge_every": 3, "latent_dim": 2048,
    }
    train = trainer_config(cfg)
    assert train["phase"] == "bridge" and train["bridge_location"] == "latent_pre_backbone"
    assert train["bridge_lora_rank"] == 8 and train["lr"] == 0.0001
    assert text_dim_from_config(cfg) == 4096


# ---- cond pooler ---------------------------------------------------------------


def test_cond_pooler_follows_phase_handling():
    audio = _audio()
    pooler = build_cond_pooler(audio, TEXT_DIM)
    assert pooler.out_features == DIMS["dim"]
    assert any(k.startswith("cond_pooler.") for k in audio.state_dict())
    trainer = JointAVTrainer(StubVideoBackbone(), audio, {"phase": "bridge"})
    assert all(not p.requires_grad for p in pooler.parameters())
    assert all(p not in set(trainer._trainable_params) for p in pooler.parameters())
    trainer.set_phase("full")
    assert all(p.requires_grad for p in pooler.parameters())
    assert all(p in set(trainer._trainable_params) for p in pooler.parameters())


def test_cond_pooler_rejects_double_attach():
    audio = _audio()
    build_cond_pooler(audio, TEXT_DIM)
    with pytest.raises(JointAVRunnerError, match="already has a cond_pooler"):
        build_cond_pooler(audio, TEXT_DIM)


def test_runner_requires_pooler(tmp_path):
    trainer = JointAVTrainer(StubVideoBackbone(), _audio(), {"phase": "bridge"})
    with pytest.raises(JointAVRunnerError, match="cond_pooler"):
        JointAVRunner(trainer, _batches(), _run_cfg(tmp_path))


# ---- step loop ------------------------------------------------------------------


def test_runner_trains_logs_and_checkpoints(tmp_path):
    trainer = _trainer()
    summary = JointAVRunner(trainer, _batches(), _run_cfg(tmp_path)).train()
    assert summary["steps_completed"] == 4 and summary["resumed_from"] == 0
    assert summary["final_phase"] == "bridge"
    assert all(torch.isfinite(torch.tensor(v)) for v in summary["final_metrics"].values())
    lines = _step_records(tmp_path)
    assert [r["step"] for r in lines] == [0, 1, 2, 3]
    assert {r["phase"] for r in lines} == {"bridge"}
    assert (tmp_path / "ckpt_step_000002.pt").is_file()
    assert (tmp_path / "ckpt_step_000004.pt").is_file()
    assert (tmp_path / "ckpt_latest.pt").is_file()


def test_runner_scheduled_phase_switch_bridge_to_full(tmp_path):
    trainer = _trainer(LORA)
    assert iter_lora_modules(trainer.audio), "bridge phase should inject LoRA"
    summary = JointAVRunner(
        trainer, _batches(), _run_cfg(tmp_path, bridge_steps=2)
    ).train()
    assert summary["final_phase"] == "full"
    lines = _read_metrics(tmp_path)
    phases = {r["step"]: r["phase"] for r in _step_records(tmp_path)}
    assert phases[0] == phases[1] == "bridge"
    assert phases[2] == phases[3] == "full"
    assert any(r.get("event") == "phase_switch" and r["step"] == 2 for r in lines)
    assert not iter_lora_modules(trainer.audio), "full phase must merge bridge LoRA"


def test_runner_without_bridge_steps_stays_in_config_phase(tmp_path):
    trainer = _trainer(LORA)
    summary = JointAVRunner(trainer, _batches(), _run_cfg(tmp_path)).train()
    assert summary["final_phase"] == "bridge"
    assert iter_lora_modules(trainer.audio), "no schedule -> LoRA stays unmerged"


# ---- checkpoint resume ------------------------------------------------------------


def test_checkpoint_resume_restores_state_exactly(tmp_path):
    torch.manual_seed(0)
    trainer = _trainer()
    runner = JointAVRunner(trainer, _batches(), _run_cfg(tmp_path, max_steps=2))
    runner.train()
    ckpt = tmp_path / "ckpt_latest.pt"
    payload_rng = torch.load(ckpt, map_location="cpu", weights_only=True)["torch_rng_state"]

    fresh = _trainer()
    step = load_checkpoint(fresh, ckpt)
    assert step == 2
    for (name, p), (_, q) in zip(
        trainer.audio.state_dict().items(), fresh.audio.state_dict().items()
    ):
        assert torch.equal(p, q), f"weight mismatch at {name}"
    assert fresh.optimizer.state_dict()["state"], "optimizer state must be restored"
    assert torch.equal(torch.get_rng_state(), payload_rng)


def test_runner_auto_resume_continues_numbering(tmp_path):
    JointAVRunner(_trainer(), _batches(), _run_cfg(tmp_path, max_steps=3)).train()
    summary = JointAVRunner(
        _trainer(), _batches(), _run_cfg(tmp_path, max_steps=5, resume="auto")
    ).train()
    assert summary["resumed_from"] == 3 and summary["steps_completed"] == 5
    steps = [r["step"] for r in _step_records(tmp_path)]
    assert steps == [0, 1, 2, 3, 4]


def test_resume_lora_bridge_checkpoint_roundtrip(tmp_path):
    trainer = _trainer(LORA)
    JointAVRunner(trainer, _batches(), _run_cfg(tmp_path, max_steps=2)).train()
    fresh = _trainer(LORA)
    load_checkpoint(fresh, tmp_path / "ckpt_latest.pt")
    assert fresh.phase == "bridge"
    assert iter_lora_modules(fresh.audio)
    for (name, p), (_, q) in zip(
        trainer.audio.state_dict().items(), fresh.audio.state_dict().items()
    ):
        assert torch.equal(p, q), f"weight mismatch at {name}"


def test_resume_full_phase_checkpoint_into_bridge_config(tmp_path):
    trainer = _trainer(LORA)
    trainer.set_phase("full")  # merges LoRA, trains everything
    path = save_checkpoint(trainer, tmp_path, 7)
    fresh = _trainer(LORA)  # fresh bridge-phase model has unmerged LoRA keys
    step = load_checkpoint(fresh, path)
    assert step == 7 and fresh.phase == "full"
    assert not iter_lora_modules(fresh.audio)
    for (name, p), (_, q) in zip(
        trainer.audio.state_dict().items(), fresh.audio.state_dict().items()
    ):
        assert torch.equal(p, q), f"weight mismatch at {name}"


# ---- fail-closed guards ---------------------------------------------------------


def test_nonfinite_metrics_abort_with_emergency_checkpoint(tmp_path):
    trainer = _trainer()
    batches = [_batch(), _batch(nan=True)] + _batches(2)
    with pytest.raises(JointAVRunnerError, match="non-finite metrics"):
        JointAVRunner(trainer, batches, _run_cfg(tmp_path, save_every=99)).train()
    emergencies = list(tmp_path.glob("ckpt_nonfinite_step_*.pt"))
    assert emergencies, "non-finite abort must leave an emergency checkpoint"


def test_batch_without_cond_inputs_rejected(tmp_path):
    trainer = _trainer()
    bad = [{"video_tokens": torch.randn(1, 4, 2, 3, 3), "audio_latent": torch.randn(1, 6, 16)}]
    with pytest.raises(JointAVRunnerError, match="video_cond"):
        JointAVRunner(trainer, bad, _run_cfg(tmp_path)).train()


# ---- real-cache dataloader integration ------------------------------------------


def _build_caches(tmp_path: Path) -> tuple[Path, Path]:
    samples = [
        {"video": f"/media/v{i}.mp4", "audio": f"/media/a{i}.wav",
         "caption": f"第 {i} 段", "source": f"pexels:{i}"}
        for i in range(4)
    ]
    av_dir, text_dir = tmp_path / "av", tmp_path / "text"
    av_dir.mkdir()
    text_dir.mkdir()

    def av_encode(sample, video_dest, audio_dest):
        save_latent_atomic(torch.randn(1, DIMS["video_dim"], 2, 3, 3), video_dest)
        save_latent_atomic(torch.randn(1, 6, DIMS["latent_dim"]), audio_dest)

    def text_encode(sample, dest):
        save_latent_atomic(torch.randn(8, TEXT_DIM), dest)
        return 5

    run_av_encoding(plan_av_encoding(samples, av_dir, model="wan2.2-vae+dac"), av_encode)
    run_text_encoding(plan_text_encoding(samples, text_dir, model="umT5-xxl"), text_encode)
    return av_dir, text_dir


def test_runner_consumes_production_dataloader(tmp_path):
    av_dir, text_dir = _build_caches(tmp_path)
    records = load_joint_latent_records(av_dir, text_dir)
    assert len(records) == 4
    loader = build_joint_latent_dataloader(
        records, batch_size=2, shuffle=False, cond_pooler=None
    )
    trainer = _trainer()
    summary = JointAVRunner(
        trainer, loader, _run_cfg(tmp_path / "run", max_steps=3)
    ).train()
    assert summary["steps_completed"] == 3
    assert all(torch.isfinite(torch.tensor(v)) for v in summary["final_metrics"].values())


# ---- CLI ----------------------------------------------------------------------


def test_train_joint_av_script_compiles_and_smoke_passes(tmp_path):
    script = ROOT / "scripts" / "train_joint_av.py"
    subprocess.run([sys.executable, "-m", "py_compile", str(script)], check=True)
    proc = subprocess.run(
        [sys.executable, str(script), "--smoke", "--out", str(tmp_path / "smoke"),
         "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert "TRAIN_JOINT_AV_SMOKE_OK" in proc.stdout
    assert (tmp_path / "smoke" / "ckpt_latest.pt").is_file()

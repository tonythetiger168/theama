# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/gen_eval_set.py (M1 eval-set generation resume gate).

The resume contract — existing probe-valid MP4s are skipped, invalid or
corrupt ones are deleted and regenerated, exit code is nonzero unless every
prompt has a probe-valid MP4 — decides whether a crashed GPU run continues
where it stopped or silently ships broken MP4s into the VBench gate. A bug
here costs GPU hours, so it is covered on CPU with fake probe/mux/pipeline
hooks. No GPU, torch, ffmpeg, or real pipeline required.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "gen_eval_set", ROOT / "scripts" / "gen_eval_set.py"
)
ge = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ge)

# ti2v-5b spec: fps=24, native_seconds=5 → expected_fps 24, min_seconds 4.5.
GOOD_INFO = {
    "video_codec": "h264",
    "fps": 24.0,
    "duration": 5.0,
    "has_audio": True,
    "audio_codec": "aac",
}

ROWS = [
    {"id": "p01", "category": "nature", "prompt": "a calm lake at dawn"},
    {"id": "p02", "category": "urban", "prompt": "busy street timelapse"},
]


def _write_eval_set(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS),
        encoding="utf-8",
    )


def _run_main(monkeypatch, tmp_path, probe_map, extra_args=(), regen_map=None):
    """Run ge.main() with fake probe/mux/pipeline. probe_map: filename →
    info dict or Exception instance. regen_map: filename → info the fake
    mux leaves behind (default: a probe-valid file)."""
    regen_map = regen_map or {}
    calls = {"load": 0, "gen": [], "mux": [], "probe": []}

    def fake_probe(path):
        name = Path(path).name
        calls["probe"].append(name)
        info = probe_map[name]
        if isinstance(info, Exception):
            raise info
        return info

    def fake_mux(frames_dir, target, fps):
        name = Path(target).name
        calls["mux"].append(name)
        Path(target).write_bytes(b"\x00fake")
        probe_map[name] = regen_map.get(name, dict(GOOD_INFO))

    t2v = types.ModuleType("theama.inference.t2v")

    def load_pipeline(model, offload=True, model_path=None):
        calls["load"] += 1
        return object(), None

    def generate_frames(pipe, spec, prompt, seed):
        calls["gen"].append(prompt)
        return ["frame"]

    def save_frames(frames, out_dir):
        pass

    t2v.load_pipeline = load_pipeline
    t2v.generate_frames = generate_frames
    t2v.save_frames = save_frames

    i2v = types.ModuleType("theama.inference.i2v")
    i2v.generate_frames_i2v = lambda *a, **k: pytest.fail("i2v path not expected")
    i2v.load_i2v_pipeline = lambda *a, **k: (object(), None)

    monkeypatch.setitem(sys.modules, "theama.inference.t2v", t2v)
    monkeypatch.setitem(sys.modules, "theama.inference.i2v", i2v)
    monkeypatch.setattr(ge, "probe_mp4", fake_probe)
    monkeypatch.setattr(ge, "mux_mp4", fake_mux)

    eval_set = tmp_path / "eval.jsonl"
    _write_eval_set(eval_set)
    out_dir = tmp_path / "out"
    argv = [
        "gen_eval_set.py", "--model", "ti2v-5b",
        "--eval-set", str(eval_set), "--out", str(out_dir),
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    code = ge.main()
    return code, calls, out_dir


def _touch(out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    p.write_bytes(b"\x00old")
    return p


# ---------------------------------------------------------------- pure fns


def test_probe_ok_accepts_valid():
    assert ge.probe_ok(dict(GOOD_INFO), fps=24.0, min_seconds=4.5) is True


def test_probe_ok_rejects_wrong_video_codec():
    for codec in ("hevc", "av1", "vp9"):
        info = {**GOOD_INFO, "video_codec": codec}
        assert ge.probe_ok(info, fps=24.0, min_seconds=4.5) is False


def test_probe_ok_fps_tolerance_boundary():
    # |fps - expected| must be strictly < 0.5
    assert ge.probe_ok({**GOOD_INFO, "fps": 24.49}, 24.0, 4.5) is True
    assert ge.probe_ok({**GOOD_INFO, "fps": 24.5}, 24.0, 4.5) is False
    assert ge.probe_ok({**GOOD_INFO, "fps": 23.51}, 24.0, 4.5) is True
    assert ge.probe_ok({**GOOD_INFO, "fps": 23.5}, 24.0, 4.5) is False


def test_probe_ok_duration_boundary_inclusive():
    assert ge.probe_ok({**GOOD_INFO, "duration": 4.5}, 24.0, 4.5) is True
    assert ge.probe_ok({**GOOD_INFO, "duration": 4.49}, 24.0, 4.5) is False


def test_probe_ok_requires_aac_audio():
    assert ge.probe_ok({**GOOD_INFO, "has_audio": False}, 24.0, 4.5) is False
    assert ge.probe_ok({**GOOD_INFO, "audio_codec": "mp3"}, 24.0, 4.5) is False


# ------------------------------------------------------------- resume gate


def test_resume_skips_all_valid_without_loading_pipeline(monkeypatch, tmp_path, capsys):
    probe_map = {}
    out_pre = tmp_path / "out"
    for row in ROWS:
        _touch(out_pre, f"{row['id']}.mp4")
        probe_map[f"{row['id']}.mp4"] = dict(GOOD_INFO)

    code, calls, out_dir = _run_main(monkeypatch, tmp_path, probe_map)

    assert code == 0
    assert calls["load"] == 0 and calls["gen"] == [] and calls["mux"] == []
    assert not (out_dir / "manifest.jsonl").exists()
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {"model": "ti2v-5b", "total": 2, "ok": 2, "failed": []}
    assert "[skip] p01" in capsys.readouterr().out


def test_resume_regenerates_only_invalid(monkeypatch, tmp_path):
    bad = {**GOOD_INFO, "fps": 12.0}
    probe_map = {"p01.mp4": dict(GOOD_INFO), "p02.mp4": dict(bad)}
    out_pre = tmp_path / "out"
    _touch(out_pre, "p01.mp4")
    _touch(out_pre, "p02.mp4")

    code, calls, out_dir = _run_main(monkeypatch, tmp_path, probe_map)

    assert code == 0
    assert calls["load"] == 1
    assert calls["gen"] == [ROWS[1]["prompt"]]
    assert calls["mux"] == ["p02.mp4"]
    # invalid file was unlinked and replaced by the regenerated one
    assert (out_dir / "p02.mp4").read_bytes() == b"\x00fake"
    lines = (out_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    assert len(lines) == 1 and rec["id"] == "p02"
    assert rec["model"] == "ti2v-5b" and rec["seed"] == 42
    assert rec["probe"]["video_codec"] == "h264"


def test_resume_probe_exception_treated_as_corrupt(monkeypatch, tmp_path):
    probe_map = {"p01.mp4": dict(GOOD_INFO),
                 "p02.mp4": RuntimeError("ffprobe: moov atom not found")}
    out_pre = tmp_path / "out"
    _touch(out_pre, "p01.mp4")
    _touch(out_pre, "p02.mp4")

    code, calls, _ = _run_main(monkeypatch, tmp_path, probe_map)

    assert code == 0
    assert calls["mux"] == ["p02.mp4"]


def test_final_gate_fails_when_regen_still_invalid(monkeypatch, tmp_path):
    probe_map = {"p01.mp4": dict(GOOD_INFO),
                 "p02.mp4": RuntimeError("corrupt")}
    _touch(tmp_path / "out", "p01.mp4")
    _touch(tmp_path / "out", "p02.mp4")
    regen_bad = {**GOOD_INFO, "duration": 1.0}  # regenerated but too short

    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path, probe_map, regen_map={"p02.mp4": regen_bad}
    )

    assert code == 1
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] == 1 and summary["failed"] == ["p02"]


def test_fresh_run_generates_every_row(monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(monkeypatch, tmp_path, {})

    assert code == 0
    assert calls["load"] == 1
    assert calls["gen"] == [r["prompt"] for r in ROWS]
    assert calls["mux"] == ["p01.mp4", "p02.mp4"]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 2 and summary["ok"] == 2


def test_limit_truncates_eval_set(monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path, {}, extra_args=("--limit", "1")
    )

    assert code == 0
    assert calls["gen"] == [ROWS[0]["prompt"]]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 1 and summary["ok"] == 1


def test_i2v_model_aborts_with_clear_message(monkeypatch, tmp_path):
    eval_set = tmp_path / "eval.jsonl"
    _write_eval_set(eval_set)
    argv = [
        "gen_eval_set.py", "--model", "i2v-a14b",
        "--eval-set", str(eval_set), "--out", str(tmp_path / "out"),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    i2v = types.ModuleType("theama.inference.i2v")
    i2v.load_i2v_pipeline = lambda *a, **k: (object(), None)
    i2v.generate_frames_i2v = lambda *a, **k: []
    t2v = types.ModuleType("theama.inference.t2v")
    t2v.load_pipeline = lambda *a, **k: pytest.fail("t2v path not expected")
    t2v.generate_frames = lambda *a, **k: []
    t2v.save_frames = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "theama.inference.i2v", i2v)
    monkeypatch.setitem(sys.modules, "theama.inference.t2v", t2v)

    with pytest.raises(SystemExit, match="TI2V"):
        ge.main()

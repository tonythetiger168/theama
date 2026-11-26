# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/gen_candidates.py (P1a DPO candidate mining).

The resume contract — existing probe-valid MP4s are skipped, invalid or
corrupt ones are (re)generated, the pipeline is loaded lazily (a fully
cached rerun never pays the model-load cost), and the exit code is nonzero
if any candidate failed — decides whether a crashed GPU mining run
continues where it stopped or silently ships broken clips into
score_eval_set / make_pref_pairs. A bug here costs GPU hours, so it is
covered on CPU with fake probe/mux/pipeline hooks, mirroring
tests/test_m1_gen_eval_set.py. No GPU, torch, ffmpeg, or real pipeline
required.

Note: gen_candidates.probe_ok deliberately uses an *inclusive* fps
tolerance (|fps - spec.fps| <= 0.5), unlike gen_eval_set's strict bound —
both contracts are pinned by their own tests.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "gen_candidates", ROOT / "scripts" / "gen_candidates.py"
)
gc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gc)

SPEC = gc.MODELS["ti2v-5b"]  # fps=24, native_seconds=5 → min duration 4.5

GOOD_INFO = {
    "video_codec": "h264",
    "fps": 24.0,
    "duration": 5.0,
    "has_audio": True,
    "audio_codec": "aac",
}

ROWS = [
    {"id": "p01", "prompt": "a calm lake at dawn"},
    {"id": "p02", "prompt": "busy street timelapse"},
]


def _write_prompts(path: Path) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS),
        encoding="utf-8",
    )


def _touch_mp4(out_root: Path, pid: str, k: int) -> Path:
    pdir = out_root / pid
    pdir.mkdir(parents=True, exist_ok=True)
    p = pdir / f"cand_{k}.mp4"
    p.write_bytes(b"\x00old")
    return p


def _manifest_rows(pdir: Path):
    f = pdir / "manifest.jsonl"
    if not f.exists():
        return []
    return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]


def _run_main(monkeypatch, tmp_path, probe_map, extra_args=(), regen_map=None,
              gen_fail=()):
    """Run gc.main() with fake probe/mux/pipeline.

    probe_map: "<prompt_id>/<mp4 name>" → info dict or Exception instance
    (keyed by parent dir + name: every prompt has its own cand_<k>.mp4).
    regen_map: same key → info the fake mux leaves behind (default: a
    probe-valid file). gen_fail: set of (prompt, seed) tuples whose
    generate_frames raises.
    """
    regen_map = regen_map or {}
    calls = {"load": 0, "gen": [], "mux": []}

    def _key(path):
        p = Path(path)
        return f"{p.parent.name}/{p.name}"

    def fake_probe(path):
        info = probe_map[_key(path)]
        if isinstance(info, Exception):
            raise info
        return info

    def fake_mux(frames_dir, target, fps):
        key = _key(target)
        calls["mux"].append(key)
        Path(target).write_bytes(b"\x00fake")
        probe_map[key] = regen_map.get(key, dict(GOOD_INFO))

    t2v = types.ModuleType("theama.inference.t2v")

    def load_pipeline(model):
        calls["load"] += 1
        return object(), None

    def generate_frames(pipe, spec, prompt, seed=None, num_frames=None):
        calls["gen"].append((prompt, seed))
        if (prompt, seed) in gen_fail:
            raise RuntimeError("boom")
        return ["frame"]

    def save_frames(frames, out_dir):
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / "000.png").write_bytes(b"\x00png")

    t2v.load_pipeline = load_pipeline
    t2v.generate_frames = generate_frames
    t2v.save_frames = save_frames

    monkeypatch.setitem(sys.modules, "theama.inference.t2v", t2v)
    monkeypatch.setattr(gc, "probe_mp4", fake_probe)
    # mux_mp4 is imported inside main() from the already-loaded real module.
    monkeypatch.setattr(sys.modules["theama.inference.mux"], "mux_mp4", fake_mux)

    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    out_root = tmp_path / "candidates"
    argv = [
        "gen_candidates.py", "--model", "ti2v-5b",
        "--prompts", str(prompts), "--out", str(out_root),
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    code = gc.main()
    return code, calls, out_root


# ------------------------------------------------------------------ probe_ok


def test_probe_ok_accepts_valid(monkeypatch):
    monkeypatch.setattr(gc, "probe_mp4", lambda p: dict(GOOD_INFO))
    assert gc.probe_ok(Path("cand_0.mp4"), SPEC) is True


def test_probe_ok_rejects_wrong_codec_and_silence(monkeypatch):
    for info in (
        {**GOOD_INFO, "video_codec": "hevc"},
        {**GOOD_INFO, "video_codec": "vp9"},
        {**GOOD_INFO, "has_audio": False},
    ):
        monkeypatch.setattr(gc, "probe_mp4", lambda p, i=info: dict(i))
        assert gc.probe_ok(Path("cand_0.mp4"), SPEC) is False


def test_probe_ok_fps_tolerance_inclusive(monkeypatch):
    # gen_candidates uses |fps - spec.fps| <= 0.5 (inclusive), unlike
    # gen_eval_set's strictly-< 0.5 bound.
    def check(fps):
        monkeypatch.setattr(gc, "probe_mp4", lambda p: {**GOOD_INFO, "fps": fps})
        return gc.probe_ok(Path("cand_0.mp4"), SPEC)

    assert check(24.5) is True
    assert check(24.51) is False
    assert check(23.5) is True
    assert check(23.49) is False


def test_probe_ok_duration_boundary_inclusive(monkeypatch):
    def check(dur):
        monkeypatch.setattr(gc, "probe_mp4",
                            lambda p: {**GOOD_INFO, "duration": dur})
        return gc.probe_ok(Path("cand_0.mp4"), SPEC)

    assert check(4.5) is True   # native_seconds - 0.5 is accepted
    assert check(4.49) is False


def test_probe_ok_probe_exception_is_invalid(monkeypatch):
    def boom(path):
        raise RuntimeError("ffprobe died")

    monkeypatch.setattr(gc, "probe_mp4", boom)
    assert gc.probe_ok(Path("cand_0.mp4"), SPEC) is False


# ------------------------------------------------------------- resume / CLI


def test_fresh_run_generates_all_and_records_seeds(monkeypatch, tmp_path):
    code, calls, out_root = _run_main(
        monkeypatch, tmp_path, {}, extra_args=["--candidates", "2"]
    )
    assert code == 0
    assert calls["load"] == 1
    assert calls["gen"] == [
        ("a calm lake at dawn", 1000), ("a calm lake at dawn", 1001),
        ("busy street timelapse", 1000), ("busy street timelapse", 1001),
    ]
    for pid in ("p01", "p02"):
        rows = _manifest_rows(out_root / pid)
        assert [(r["id"], r["candidate"], r["seed"]) for r in rows] == [
            (pid, 0, 1000), (pid, 1, 1001),
        ]
    summary = json.loads((out_root / "candidates_summary.json").read_text())
    assert summary["clips_made_this_run"] == 4
    assert summary["prompts"] == 2
    assert summary["failures"] == []


def test_full_resume_skips_all_and_never_loads_pipeline(monkeypatch, tmp_path):
    out_root = tmp_path / "candidates"
    probe_map = {}
    for pid in ("p01", "p02"):
        for k in (0, 1):
            _touch_mp4(out_root, pid, k)
            probe_map[f"{pid}/cand_{k}.mp4"] = dict(GOOD_INFO)
    code, calls, _ = _run_main(
        monkeypatch, tmp_path, probe_map, extra_args=["--candidates", "2"]
    )
    assert code == 0
    assert calls["load"] == 0          # lazy pipeline: fully cached run is cheap
    assert calls["gen"] == []
    assert calls["mux"] == []
    assert _manifest_rows(out_root / "p01") == []  # no rows appended this run
    summary = json.loads((out_root / "candidates_summary.json").read_text())
    assert summary["clips_made_this_run"] == 0


def test_invalid_candidate_only_that_one_regenerated(monkeypatch, tmp_path):
    out_root = tmp_path / "candidates"
    probe_map = {}
    for pid in ("p01", "p02"):
        for k in (0, 1):
            _touch_mp4(out_root, pid, k)
            probe_map[f"{pid}/cand_{k}.mp4"] = dict(GOOD_INFO)
    probe_map["p02/cand_1.mp4"] = {**GOOD_INFO, "video_codec": "hevc"}
    code, calls, _ = _run_main(
        monkeypatch, tmp_path, probe_map, extra_args=["--candidates", "2"]
    )
    assert code == 0
    assert calls["load"] == 1
    assert calls["gen"] == [("busy street timelapse", 1001)]
    assert calls["mux"] == ["p02/cand_1.mp4"]
    assert _manifest_rows(out_root / "p01") == []
    rows = _manifest_rows(out_root / "p02")
    assert [(r["id"], r["candidate"], r["seed"]) for r in rows] == [
        ("p02", 1, 1001),
    ]


def test_probe_exception_treated_as_invalid_and_regenerated(
    monkeypatch, tmp_path
):
    out_root = tmp_path / "candidates"
    probe_map = {}
    for pid in ("p01", "p02"):
        for k in (0, 1):
            _touch_mp4(out_root, pid, k)
            probe_map[f"{pid}/cand_{k}.mp4"] = dict(GOOD_INFO)
    probe_map["p01/cand_0.mp4"] = RuntimeError("ffprobe died")
    code, calls, _ = _run_main(
        monkeypatch, tmp_path, probe_map, extra_args=["--candidates", "2"]
    )
    assert code == 0
    assert calls["gen"] == [("a calm lake at dawn", 1000)]
    assert [r["candidate"] for r in _manifest_rows(out_root / "p01")] == [0]


def test_regen_still_invalid_fails_run(monkeypatch, tmp_path):
    out_root = tmp_path / "candidates"
    probe_map = {}
    for pid in ("p01", "p02"):
        for k in (0, 1):
            _touch_mp4(out_root, pid, k)
            probe_map[f"{pid}/cand_{k}.mp4"] = dict(GOOD_INFO)
    probe_map["p01/cand_0.mp4"] = {**GOOD_INFO, "duration": 4.0}  # stale
    code, calls, out_root = _run_main(
        monkeypatch, tmp_path, probe_map,
        extra_args=["--candidates", "2"],
        regen_map={"p01/cand_0.mp4": {**GOOD_INFO, "duration": 4.0}},  # bad
    )
    assert code == 1
    summary = json.loads((out_root / "candidates_summary.json").read_text())
    assert summary["clips_made_this_run"] == 0
    assert summary["failures"] == [
        {"id": "p01", "candidate": 0, "error": "probe failed after mux"},
    ]
    assert _manifest_rows(out_root / "p01") == []  # no row for a bad clip


def test_generation_exception_keeps_mining_and_fails_run(monkeypatch, tmp_path):
    code, calls, out_root = _run_main(
        monkeypatch, tmp_path, {},
        extra_args=["--candidates", "2"],
        gen_fail={("a calm lake at dawn", 1001)},
    )
    assert code == 1
    assert len(calls["gen"]) == 4      # the other three were still mined
    summary = json.loads((out_root / "candidates_summary.json").read_text())
    assert summary["clips_made_this_run"] == 3
    assert summary["failures"] == [
        {"id": "p01", "candidate": 1, "error": "boom"},
    ]
    assert [r["candidate"] for r in _manifest_rows(out_root / "p01")] == [0]
    assert len(_manifest_rows(out_root / "p02")) == 2


def test_limit_truncates_prompts(monkeypatch, tmp_path):
    code, calls, out_root = _run_main(
        monkeypatch, tmp_path, {},
        extra_args=["--candidates", "2", "--limit", "1"],
    )
    assert code == 0
    assert [p for p, _ in calls["gen"]] == ["a calm lake at dawn"] * 2
    summary = json.loads((out_root / "candidates_summary.json").read_text())
    assert summary["prompts"] == 1
    assert summary["clips_made_this_run"] == 2
    assert not (out_root / "p02").exists()


def test_seed_base_override(monkeypatch, tmp_path):
    code, calls, out_root = _run_main(
        monkeypatch, tmp_path, {},
        extra_args=["--candidates", "2", "--limit", "1", "--seed-base", "5000"],
    )
    assert code == 0
    assert [s for _, s in calls["gen"]] == [5000, 5001]
    assert [r["seed"] for r in _manifest_rows(out_root / "p01")] == [5000, 5001]

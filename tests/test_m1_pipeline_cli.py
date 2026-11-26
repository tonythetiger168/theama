# SPDX-License-Identifier: Apache-2.0
"""CPU tests for src/theama/inference/pipeline.py (M1 one-shot inference CLI).

pipeline.main() is the user-facing M1 entry
(``python -m theama.inference.pipeline --prompt ... --out ...``) that wires
model-registry lookup -> frame count (``--seconds`` snapped to 4k+1) ->
T2V vs I2V branch on ``--image`` -> offload passthrough ->
save -> mux -> probe-verified summary. All heavy imports (t2v/i2v pipeline
loaders) are lazy inside main(), so the whole wiring contract is testable on
CPU with fake modules planted in sys.modules. A wiring bug here (wrong
branch, wrong fps to mux, frames override ignored) would otherwise only
surface after a multi-minute GPU pipeline load. No GPU, torch, diffusers, or
ffmpeg required.
"""

import sys
import types
from pathlib import Path

import pytest

from theama.inference import pipeline
from theama.inference.config import MODELS, frames_for_seconds

TI2V = MODELS["ti2v-5b"]   # dev/CI path: 121f @ 24fps, 1280x704
A14B = MODELS["t2v-a14b"]  # full backbone: 81f @ 16fps

PROBE_INFO = {
    "width": 1280,
    "height": 704,
    "duration": 5.04,
    "fps": 24.0,
    "audio_codec": "aac",
}


def _fake_t2v(calls, *, explode_gen=False):
    """Fake theama.inference.t2v with explicit production signatures."""
    mod = types.ModuleType("theama.inference.t2v")

    def load_pipeline(model, offload=True):
        calls["t2v_load"].append({"model": model, "offload": offload})
        return ("PIPE", None)

    def generate_frames(pipe, spec, prompt, seed=0, num_frames=1):
        calls["t2v_gen"].append(
            {"pipe": pipe, "spec": spec, "prompt": prompt,
             "seed": seed, "num_frames": num_frames}
        )
        if explode_gen:
            raise RuntimeError("boom")
        return ["f1", "f2"]

    def save_frames(frames, frames_dir):
        frames_dir = Path(frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)
        (frames_dir / "0001.png").write_bytes(b"fake")
        calls["save"].append({"frames": frames, "dir": frames_dir})
        return frames_dir

    mod.load_pipeline = load_pipeline
    mod.generate_frames = generate_frames
    mod.save_frames = save_frames
    return mod


def _fake_i2v(calls):
    """Fake theama.inference.i2v with explicit production signatures."""
    mod = types.ModuleType("theama.inference.i2v")

    def load_i2v_pipeline(spec_name="ti2v-5b", offload=True):
        calls["i2v_load"].append({"model": spec_name, "offload": offload})
        return ("I2V_PIPE", None)

    def generate_frames_i2v(pipe, spec, image, prompt, seed=0, num_frames=1):
        calls["i2v_gen"].append(
            {"pipe": pipe, "spec": spec, "image": image, "prompt": prompt,
             "seed": seed, "num_frames": num_frames}
        )
        return ["g1"]

    mod.load_i2v_pipeline = load_i2v_pipeline
    mod.generate_frames_i2v = generate_frames_i2v
    return mod


def _setup(monkeypatch, *, explode_gen=False):
    """Plant fake t2v/i2v modules + fake mux/probe. Returns recorded calls."""
    calls = {"t2v_load": [], "t2v_gen": [], "i2v_load": [], "i2v_gen": [],
             "save": [], "mux": [], "probe": []}
    monkeypatch.setitem(sys.modules, "theama.inference.t2v",
                        _fake_t2v(calls, explode_gen=explode_gen))
    monkeypatch.setitem(sys.modules, "theama.inference.i2v", _fake_i2v(calls))

    def fake_mux(frames_dir, out, fps):
        calls["mux"].append({"frames_dir": Path(frames_dir),
                             "out": str(out), "fps": fps})
        Path(out).write_bytes(b"fake-mp4")

    def fake_probe(path):
        calls["probe"].append(str(path))
        return dict(PROBE_INFO)

    monkeypatch.setattr(pipeline, "mux_mp4", fake_mux)
    monkeypatch.setattr(pipeline, "probe_mp4", fake_probe)
    return calls


def _main(monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["theama.inference.pipeline"] + argv)
    return pipeline.main()


def test_t2v_default_wiring_and_probe_summary(monkeypatch, tmp_path, capsys):
    calls = _setup(monkeypatch)
    out = tmp_path / "out.mp4"
    rc = _main(monkeypatch, ["--prompt", "night market neon",
                             "--out", str(out)])
    assert rc == 0
    # registry lookup + default model, offload on by default
    assert calls["t2v_load"] == [{"model": "ti2v-5b", "offload": True}]
    # native frame count when --seconds is omitted; prompt/seed passthrough
    assert calls["t2v_gen"] == [
        {"pipe": "PIPE", "spec": TI2V, "prompt": "night market neon",
         "seed": 42, "num_frames": TI2V.num_frames}
    ]
    assert TI2V.num_frames == 121  # registry sanity for the assertion above
    # frames object identity flows generate -> save -> mux, spec fps to mux
    assert calls["save"][0]["frames"] == ["f1", "f2"]
    assert calls["mux"] == [{"frames_dir": calls["save"][0]["dir"],
                             "out": str(out), "fps": TI2V.fps}]
    assert calls["probe"] == [str(out)]
    # I2V branch untouched on the T2V path
    assert calls["i2v_load"] == [] and calls["i2v_gen"] == []
    # probe-verified summary on stdout
    line = capsys.readouterr().out.strip()
    assert line == (f"wrote {out}: 1280x704 5.04s @ 24fps, audio=aac")


def test_seconds_override_snaps_to_4k_plus_1(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    rc = _main(monkeypatch, ["--prompt", "p", "--out", str(tmp_path / "o.mp4"),
                             "--seconds", "2"])
    assert rc == 0
    expected = frames_for_seconds(TI2V, 2.0)
    assert expected == 49  # 2s * 24fps + 1, already 4k+1
    assert expected != TI2V.num_frames
    assert calls["t2v_gen"][0]["num_frames"] == expected


def test_no_offload_passthrough(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    rc = _main(monkeypatch, ["--prompt", "p", "--out", str(tmp_path / "o.mp4"),
                             "--no-offload"])
    assert rc == 0
    assert calls["t2v_load"][0]["offload"] is False


def test_seed_passthrough(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    rc = _main(monkeypatch, ["--prompt", "p", "--out", str(tmp_path / "o.mp4"),
                             "--seed", "7"])
    assert rc == 0
    assert calls["t2v_gen"][0]["seed"] == 7


def test_image_flag_selects_i2v_branch(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    rc = _main(monkeypatch, ["--prompt", "p", "--out", str(tmp_path / "o.mp4"),
                             "--image", "key.png"])
    assert rc == 0
    assert calls["i2v_load"] == [{"model": "ti2v-5b", "offload": True}]
    assert calls["i2v_gen"] == [
        {"pipe": "I2V_PIPE", "spec": TI2V, "image": "key.png",
         "prompt": "p", "seed": 42, "num_frames": TI2V.num_frames}
    ]
    # T2V pipeline never loaded on the I2V path (lazy-import economy)
    assert calls["t2v_load"] == [] and calls["t2v_gen"] == []
    # save_frames still comes from the t2v module (shared helper)
    assert calls["save"][0]["frames"] == ["g1"]


def test_model_choice_a14b_spec_flows_to_mux(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    rc = _main(monkeypatch, ["--prompt", "p", "--out", str(tmp_path / "o.mp4"),
                             "--model", "t2v-a14b"])
    assert rc == 0
    assert calls["t2v_load"][0]["model"] == "t2v-a14b"
    assert calls["t2v_gen"][0]["spec"] is A14B
    assert calls["t2v_gen"][0]["num_frames"] == 81
    assert calls["mux"][0]["fps"] == 16


def test_unknown_model_rejected_by_argparse(monkeypatch, tmp_path):
    _setup(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        _main(monkeypatch, ["--prompt", "p",
                            "--out", str(tmp_path / "o.mp4"),
                            "--model", "wan-9x"])
    assert excinfo.value.code == 2


def test_generation_failure_propagates_without_mux(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, explode_gen=True)
    out = tmp_path / "o.mp4"
    with pytest.raises(RuntimeError, match="boom"):
        _main(monkeypatch, ["--prompt", "p", "--out", str(out)])
    # fail-fast: no partial MP4 is muxed or probed after a generation error
    assert calls["mux"] == [] and calls["probe"] == []
    assert not out.exists()


def test_frames_dir_is_temporary_and_cleaned(monkeypatch, tmp_path):
    calls = _setup(monkeypatch)
    rc = _main(monkeypatch, ["--prompt", "p", "--out", str(tmp_path / "o.mp4")])
    assert rc == 0
    frames_dir = calls["save"][0]["dir"]
    # frames were staged inside a TemporaryDirectory, not next to the output
    assert frames_dir != tmp_path
    assert not frames_dir.exists()

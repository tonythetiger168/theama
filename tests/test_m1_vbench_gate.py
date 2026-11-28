# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/run_vbench.py (M1 VBench tolerance gate).

Covers the pure gate logic (overall-score extraction, 0-100 vs 0-1
normalization, tolerance verdict) and the CLI contract end-to-end with a
fake vbench module — so a gate-parsing bug surfaces before a GPU VBench
run, not after one. No GPU, torch, or real vbench install required.
"""

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "run_vbench", ROOT / "scripts" / "run_vbench.py"
)
rv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rv)


# ---------------------------------------------------------------- pure fns


def test_extract_overall_key_priority():
    result = {"overall": 0.9, "total_score": 0.8, "vbench_score": 0.7}
    assert rv.extract_overall(result) == pytest.approx(0.9)


def test_extract_overall_fallback_order():
    assert rv.extract_overall({"total_score": 0.8, "vbench_score": 0.7}) == pytest.approx(0.8)
    assert rv.extract_overall({"vbench_score": 0.7}) == pytest.approx(0.7)
    # numeric strings are accepted (float() coercion), same as before
    assert rv.extract_overall({"overall": "0.86"}) == pytest.approx(0.86)


def test_extract_overall_missing_keys_and_non_dict():
    assert rv.extract_overall({}) is None
    assert rv.extract_overall({"subject_consistency": 0.9}) is None
    assert rv.extract_overall(None) is None
    assert rv.extract_overall([0.9]) is None
    assert rv.extract_overall("0.9") is None


def test_extract_overall_non_numeric_returns_none_not_traceback():
    assert rv.extract_overall({"overall": "not-a-number"}) is None
    assert rv.extract_overall({"overall": None}) is None
    assert rv.extract_overall({"overall": {"nested": 1}}) is None


def test_normalize_overall_scales():
    assert rv.normalize_overall(86.81) == pytest.approx(0.8681)
    assert rv.normalize_overall(0.8681) == pytest.approx(0.8681)
    # boundary: exactly 1.5 is treated as 0-1 scale, above it as 0-100
    assert rv.normalize_overall(1.5) == pytest.approx(1.5)
    assert rv.normalize_overall(1.51) == pytest.approx(0.0151)


def test_gate_passed_within_and_boundary_inclusive():
    assert rv.gate_passed(0.8681, 0.8681, 0.02)
    # boundary checks use exactly-representable binary fractions
    assert rv.gate_passed(1.0, 0.875, 0.125)      # exactly +tol
    assert rv.gate_passed(0.75, 0.875, 0.125)     # exactly -tol


def test_gate_passed_outside():
    assert not rv.gate_passed(0.8882, 0.8681, 0.02)
    assert not rv.gate_passed(0.8480, 0.8681, 0.02)


# --------------------------------------------------------------------- CLI


def _fake_vbench(monkeypatch, result):
    """Inject fake torch + vbench modules; VBench.evaluate returns `result`."""
    torch = types.ModuleType("torch")
    torch.device = lambda name: name

    vbench = types.ModuleType("vbench")

    class VBench:
        def __init__(self, device, cache_dir, out_dir):
            self.device = device

        def evaluate(self, videos_path, name, dimension_list, mode):
            return result

    vbench.VBench = VBench
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "vbench", vbench)


def _videos_dir(tmp_path: Path, n: int = 2) -> Path:
    vdir = tmp_path / "videos"
    vdir.mkdir()
    for i in range(n):
        (vdir / f"T2V-{i:03d}.mp4").write_bytes(b"fake")
    return vdir


def _run_cli(monkeypatch, capsys, argv):
    if "--out-dir" not in argv:  # keep CLI tests from writing into the repo
        scratch = Path(tempfile.mkdtemp(prefix="vbench_test_"))
        argv = argv + ["--out-dir", str(scratch / "out"),
                       "--cache-dir", str(scratch / "cache")]
    monkeypatch.setattr(sys, "argv", ["run_vbench.py"] + argv)
    code = rv.main()
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_pass_within_tolerance(monkeypatch, capsys, tmp_path):
    _fake_vbench(monkeypatch, {"overall": 0.8701})
    code, out, _ = _run_cli(
        monkeypatch, capsys, ["--videos", str(_videos_dir(tmp_path))])
    assert code == 0
    assert "PASS" in out
    assert "overall=0.8701" in out


def test_cli_pass_normalizes_0_to_100_scale(monkeypatch, capsys, tmp_path):
    _fake_vbench(monkeypatch, {"overall": 86.81})
    code, out, _ = _run_cli(
        monkeypatch, capsys, ["--videos", str(_videos_dir(tmp_path))])
    assert code == 0
    assert "PASS" in out
    assert "overall=0.8681" in out  # normalized before the delta print


def test_cli_fail_outside_tolerance(monkeypatch, capsys, tmp_path):
    _fake_vbench(monkeypatch, {"overall": 0.80})
    code, _, err = _run_cli(
        monkeypatch, capsys, ["--videos", str(_videos_dir(tmp_path))])
    assert code == 1
    assert "FAIL" in err


def test_cli_unparsable_result_exit_2(monkeypatch, capsys, tmp_path):
    _fake_vbench(monkeypatch, {"subject_consistency": [0.9, "x"]})
    code, _, err = _run_cli(
        monkeypatch, capsys, ["--videos", str(_videos_dir(tmp_path))])
    assert code == 2
    assert "could not parse an overall score" in err


def test_cli_no_reference_target_exit_0(monkeypatch, capsys, tmp_path):
    # ti2v-5b has vbench_overall_bf16: null in eval/reference.json
    _fake_vbench(monkeypatch, {"overall": 0.5})
    code, out, _ = _run_cli(
        monkeypatch, capsys,
        ["--videos", str(_videos_dir(tmp_path)), "--model", "ti2v-5b"])
    assert code == 0
    assert "no reference score for this model" in out


def test_cli_vbench_missing_exit_2(monkeypatch, capsys, tmp_path):
    # sys.modules entry None forces ImportError even if vbench is installed
    monkeypatch.setitem(sys.modules, "vbench", None)
    code, _, err = _run_cli(
        monkeypatch, capsys, ["--videos", str(_videos_dir(tmp_path))])
    assert code == 2
    assert "vbench not installed" in err


def test_cli_empty_videos_dir_exit_2(monkeypatch, capsys, tmp_path):
    _fake_vbench(monkeypatch, {"overall": 0.8681})
    empty = tmp_path / "empty"
    empty.mkdir()
    code, _, err = _run_cli(monkeypatch, capsys, ["--videos", str(empty)])
    assert code == 2
    assert "no MP4s" in err


def test_dimension_scores_reads_vbench_015_layout():
    raw = {"subject_consistency": [0.95, [{"video_path": "a"}]],
           "dynamic_degree": [0.4, []], "broken": ["x", []]}
    assert rv.dimension_scores(raw) == {
        "subject_consistency": 0.95, "dynamic_degree": 0.4}
    assert rv.dimension_scores(None) == {}


def test_cli_reads_results_file_when_evaluate_returns_none(monkeypatch, capsys, tmp_path):
    # VBench 0.1.5 evaluate() returns None and only writes the JSON file.
    out_dir = tmp_path / "vb_out"
    out_dir.mkdir()
    (out_dir / "ti2v-5b-m1_eval_results.json").write_text(json.dumps(
        {"subject_consistency": [0.9, []], "imaging_quality": [0.7, []]}))
    _fake_vbench(monkeypatch, None)
    code, out, _ = _run_cli(monkeypatch, capsys, [
        "--videos", str(_videos_dir(tmp_path)), "--model", "ti2v-5b",
        "--out-dir", str(out_dir), "--cache-dir", str(tmp_path / "c")])
    assert code == 0
    summary = json.loads((out_dir / "ti2v-5b-m1_summary.json").read_text())
    assert summary["six_dim_mean"] == 0.8
    assert summary["videos"] == 2
    assert "VBENCH_SUMMARY" in out

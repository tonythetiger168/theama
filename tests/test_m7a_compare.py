# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/compare_m7a_summaries.py (M7a eval-run comparator).

Covers the pure build_comparison function (labels, category union, spread,
diagnostic-only flagging, duplicate-label rejection) and the CLI contract
(table on stdout, deterministic --out JSON, structural-error exit codes).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "compare_m7a_summaries", ROOT / "scripts" / "compare_m7a_summaries.py"
)
cmp7a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cmp7a)


def _summary(model, exact, recall, per_cat):
    return {
        "scorer": "cjk-text-1",
        "model": model,
        "total": 100,
        "scored": 100,
        "exact_hit_rate": exact,
        "mean_char_recall": recall,
        "per_category_char_recall": per_cat,
    }


def _entries(*pairs):
    return [(Path(f"{model}.summary.json"), s) for model, s in pairs]


def test_build_comparison_labels_and_overall():
    comp = cmp7a.build_comparison(_entries(
        ("qwen-image-2512", _summary("qwen-image-2512", 0.5, 0.8, {"poster": 0.9})),
        ("z-image-turbo", _summary("z-image-turbo", 0.4, 0.7, {"poster": 0.6})),
    ))
    assert [m["label"] for m in comp["models"]] == ["qwen-image-2512", "z-image-turbo"]
    assert comp["models"][0]["exact_hit_rate"] == 0.5
    assert comp["gate_models"] == ["qwen-image-2512", "z-image-turbo"]
    assert comp["categories"]["poster"] == {"qwen-image-2512": 0.9, "z-image-turbo": 0.6}
    assert comp["per_category_spread"]["poster"] == 0.3


def test_build_comparison_category_union_and_null_fill():
    comp = cmp7a.build_comparison(_entries(
        ("a", _summary("a", 0.1, 0.2, {"poster": 0.9, "sign": 0.7})),
        ("b", _summary("b", 0.1, 0.2, {"poster": 0.8, "menu": 0.5})),
    ))
    assert sorted(comp["categories"]) == ["menu", "poster", "sign"]
    assert comp["categories"]["menu"] == {"a": None, "b": 0.5}
    assert comp["categories"]["sign"] == {"a": 0.7, "b": None}
    assert comp["per_category_spread"]["menu"] == 0.0  # single scored side


def test_build_comparison_ming_flagged_diagnostic_only():
    comp = cmp7a.build_comparison(_entries(
        ("qwen-image-2512", _summary("qwen-image-2512", 0.5, 0.8, {})),
        ("ming-image-design", _summary("ming-image-design", 0.3, 0.6, {})),
    ))
    by_label = {m["label"]: m for m in comp["models"]}
    assert by_label["ming-image-design"]["diagnostic_only"] is True
    assert by_label["qwen-image-2512"]["diagnostic_only"] is False
    assert comp["gate_models"] == ["qwen-image-2512"]


def test_build_comparison_duplicate_label_rejected():
    entries = [
        (Path("run1/out.summary.json"), _summary("qwen-image-2512", 0.5, 0.8, {})),
        (Path("run2/out.summary.json"), _summary("qwen-image-2512", 0.4, 0.7, {})),
    ]
    try:
        cmp7a.build_comparison(entries)
    except ValueError as exc:
        assert "duplicate model label" in str(exc)
    else:
        raise AssertionError("duplicate labels must be rejected")


def test_label_for_falls_back_to_file_stem():
    no_model = _summary(None, 0.1, 0.2, {})
    assert cmp7a.label_for(Path("zimage.summary.json"), no_model) == "zimage"
    assert cmp7a.label_for(Path("plain.json"), no_model) == "plain"


def test_load_summary_missing_keys_rejected(tmp_path):
    bad = tmp_path / "bad.summary.json"
    bad.write_text(json.dumps({"total": 1}), encoding="utf-8")
    try:
        cmp7a.load_summary(bad)
    except ValueError as exc:
        assert "missing keys" in str(exc)
    else:
        raise AssertionError("structurally invalid summaries must be rejected")


def _write_summary(tmp_path: Path, name: str, obj: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return path


def _run_cli(*argv):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "compare_m7a_summaries.py"), *argv],
        capture_output=True,
        text=True,
    )


def test_cli_table_and_deterministic_out(tmp_path):
    qwen = _write_summary(tmp_path, "qwen.summary.json", _summary(
        "qwen-image-2512", 0.5, 0.8, {"poster": 0.9, "sign": 0.7}))
    zimage = _write_summary(tmp_path, "zimage.summary.json", _summary(
        "z-image-turbo", 0.4, 0.7, {"poster": 0.6}))
    out = tmp_path / "compare.json"

    first = _run_cli(str(qwen), str(zimage), "--out", str(out))
    assert first.returncode == 0, first.stderr
    assert "OVERALL(char_recall)" in first.stdout
    assert "qwen-image-2512" in first.stdout and "z-image-turbo" in first.stdout
    assert "-" in first.stdout  # z-image never scored the sign category
    written = out.read_text(encoding="utf-8")
    comp = json.loads(written)
    assert comp["categories"]["sign"] == {"qwen-image-2512": 0.7, "z-image-turbo": None}

    second = _run_cli(str(qwen), str(zimage), "--out", str(out))
    assert second.returncode == 0, second.stderr
    assert out.read_text(encoding="utf-8") == written  # deterministic


def test_cli_single_summary_exits_1(tmp_path):
    qwen = _write_summary(tmp_path, "qwen.summary.json", _summary(
        "qwen-image-2512", 0.5, 0.8, {}))
    res = _run_cli(str(qwen))
    assert res.returncode == 1
    assert "at least two" in res.stderr


def test_cli_unreadable_summary_exits_1(tmp_path):
    good = _write_summary(tmp_path, "good.summary.json", _summary("a", 0.1, 0.2, {}))
    bad = tmp_path / "bad.summary.json"
    bad.write_text("not json", encoding="utf-8")
    res = _run_cli(str(good), str(bad))
    assert res.returncode == 1
    assert "unreadable summary" in res.stderr


def test_cli_missing_file_exits_1(tmp_path):
    good = _write_summary(tmp_path, "good.summary.json", _summary("a", 0.1, 0.2, {}))
    res = _run_cli(str(good), str(tmp_path / "nope.summary.json"))
    assert res.returncode == 1

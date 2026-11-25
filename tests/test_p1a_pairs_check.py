# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P1a pairs preflight (reward/pairs_check.py,
scripts/check_pairs.py, and the encode_pairs.py fail-fast wiring).

Covers the pure validator (drop-reason precedence, injected exists,
deterministic zero-filled report), the gate CLI contract (exit codes,
--show-errors, --no-files), and that encode_pairs.py refuses an unusable
pairs.jsonl before importing the latent-cache planner — rows are never
dropped in place because pair indices key the latent cache.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.pairs_check import (  # noqa: E402
    DROP_REASONS,
    summarize,
    validate_pairs,
)

_spec = importlib.util.spec_from_file_location(
    "check_pairs", ROOT / "scripts" / "check_pairs.py"
)
cp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cp)


def _ok_row(index: int = 0, win: str = "c/p0_w.mp4", lose: str = "c/p0_l.mp4") -> dict:
    return {
        "prompt_id": f"p{index}",
        "prompt": f"prompt {index}",
        "win_video": win,
        "lose_video": lose,
        "win_score": 4.5,
        "lose_score": 2.0,
        "margin": 2.5,
    }


def _to_text(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


ALL = lambda p: True  # noqa: E731
NONE = lambda p: False  # noqa: E731


# ---- pure validator ---------------------------------------------------------


def test_all_usable_passthrough_in_order():
    rows = [_ok_row(i, f"c/p{i}_w.mp4", f"c/p{i}_l.mp4") for i in range(3)]
    report = validate_pairs(_to_text(rows), exists=ALL)
    assert report["rows"] == 3
    assert report["usable"] == 3
    assert [r["prompt_id"] for r in report["usable_rows"]] == ["p0", "p1", "p2"]
    assert report["dropped"] == dict.fromkeys(DROP_REASONS, 0)
    assert report["dropped_rows"] == []
    assert report["bad_json_lines"] == 0


def test_missing_and_mistyped_keys():
    no_key = _ok_row(0)
    del no_key["lose_video"]
    bool_score = _ok_row(1)
    bool_score["win_score"] = True
    empty_path = _ok_row(2, win="")
    report = validate_pairs(
        _to_text([no_key, bool_score, empty_path]), exists=ALL
    )
    assert report["usable"] == 0
    assert report["dropped"]["missing_keys"] == 3
    assert [r["reason"] for r in report["dropped_rows"]] == ["missing_keys"] * 3
    assert report["dropped_rows"][0]["line"] == 1


def test_non_finite_scores():
    nan_score = _ok_row(0)
    nan_score["win_score"] = float("nan")
    inf_score = _ok_row(1)
    inf_score["lose_score"] = float("inf")
    bad_margin = _ok_row(2)
    bad_margin["margin"] = float("nan")
    # json.dumps emits the NaN/Infinity literals, which Python's json
    # parser accepts — this is exactly the leak under test.
    text = _to_text([nan_score, inf_score, bad_margin])
    report = validate_pairs(text, exists=ALL)
    assert report["usable"] == 0
    assert report["dropped"]["non_finite_score"] == 3


def test_self_pair_precedes_margin():
    row = _ok_row(0, win="c/same.mp4", lose="c/same.mp4")
    report = validate_pairs(_to_text([row]), exists=ALL)
    assert report["usable"] == 0
    assert report["dropped"]["self_pair"] == 1
    assert report["dropped_rows"][0]["detail"] == "c/same.mp4"


def test_non_positive_margin():
    inverted = _ok_row(0)
    inverted["win_score"], inverted["lose_score"] = 1.0, 3.0
    tied = _ok_row(1)
    tied["win_score"] = tied["lose_score"] = 2.0
    report = validate_pairs(_to_text([inverted, tied]), exists=ALL)
    assert report["usable"] == 0
    assert report["dropped"]["non_positive_margin"] == 2


def test_missing_video_file_injected_exists():
    report = validate_pairs(_to_text([_ok_row(0)]), exists=NONE)
    assert report["usable"] == 0
    assert report["dropped"]["missing_video_file"] == 1
    assert "win_video" in report["dropped_rows"][0]["detail"]


def test_bad_lines_tolerated_and_report_deterministic():
    text = _to_text([_ok_row(0)]) + "{not json\n" + _to_text([_ok_row(1)])
    first = validate_pairs(text, exists=ALL)
    second = validate_pairs(text, exists=ALL)
    assert first == second
    assert first["rows"] == 3
    assert first["usable"] == 2
    assert first["bad_json_lines"] == 1
    assert set(first["dropped"]) == set(DROP_REASONS)
    assert summarize(first) == {
        "rows": 3,
        "usable": 2,
        "dropped": dict.fromkeys(DROP_REASONS, 0),
        "bad_json_lines": 1,
    }


# ---- gate CLI ---------------------------------------------------------------


def _run_cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_pairs.py"), *argv],
        capture_output=True,
        text=True,
    )


def test_cli_clean_exit_0(tmp_path):
    win = tmp_path / "w.mp4"
    lose = tmp_path / "l.mp4"
    win.write_bytes(b"")
    lose.write_bytes(b"")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(_to_text([_ok_row(0, str(win), str(lose))]))
    proc = _run_cli("--pairs", str(pairs))
    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["usable"] == 1


def test_cli_gate_exit_1_show_errors(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(_to_text([_ok_row(0)]))  # clips not on disk
    proc = _run_cli("--pairs", str(pairs), "--show-errors")
    assert proc.returncode == 1
    assert "missing_video_file" in proc.stderr
    assert "refused" not in proc.stderr  # gate wording: rejected
    assert "regenerate with make_pref_pairs" in proc.stderr


def test_cli_missing_file_exit_1(tmp_path):
    proc = _run_cli("--pairs", str(tmp_path / "nope.jsonl"))
    assert proc.returncode == 1


def test_cli_no_files_schema_only(tmp_path):
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(_to_text([_ok_row(0)]))  # clips not on disk
    assert _run_cli("--pairs", str(pairs)).returncode == 1
    assert _run_cli("--pairs", str(pairs), "--no-files").returncode == 0


# ---- encode_pairs fail-fast wiring -------------------------------------------


def test_encode_pairs_refuses_unusable_pairs_before_cache_import(tmp_path):
    # The preflight must fire before `from theama.data.latent_cache import
    # ...` — proven here by exiting 1 quickly on a box without diffusers.
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps({"prompt_id": "p0", "prompt": "x", "win_video": "w.mp4"})
        + "\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "encode_pairs.py"),
            "--pairs",
            str(pairs),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "refusing to start" in proc.stderr
    assert "missing_keys" in proc.stderr
    assert not (tmp_path / "out").exists()

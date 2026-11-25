# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P1a scores preflight (reward/score_check.py,
scripts/check_scores.py, and the make_pref_pairs.py fail-fast wiring).

Covers the pure validator (drop-reason precedence, non-finite scores,
reward-model mismatch filter, deterministic report), the gate CLI
contract (exit codes, --show-errors, --reward), and that
make_pref_pairs.py drops unusable rows instead of KeyError-ing — and
refuses to start when nothing usable remains.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.reward.score_check import (  # noqa: E402
    DROP_REASONS,
    summarize,
    validate_scores,
)

_spec = importlib.util.spec_from_file_location(
    "check_scores", ROOT / "scripts" / "check_scores.py"
)
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)


def _ok_row(index: int = 0, overall: float = 1.0) -> dict:
    return {
        "prompt_id": f"p{index}",
        "prompt": f"prompt {index}",
        "video": f"cand/p{index}/cand_0.mp4",
        "reward": "videoreward",
        "overall": overall,
        "dims": {"visual_quality": overall},
    }


def _to_text(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


# ---- pure validator ---------------------------------------------------------


def test_all_usable_passthrough_in_order():
    rows = [_ok_row(i, float(i)) for i in range(3)]
    report = validate_scores(_to_text(rows))
    assert report["rows"] == 3
    assert report["usable"] == 3
    assert report["usable_rows"] == rows  # file order preserved
    assert report["dropped"] == dict.fromkeys(DROP_REASONS, 0)
    assert report["dropped_rows"] == []
    assert report["bad_json_lines"] == 0
    assert report["rewards"] == {"videoreward": 3}


def test_error_row_reports_score_failed_before_missing_keys():
    # Failure rows carry no overall; the root cause must still be
    # score_failed, not missing_keys (same precedence as pair_latent_check).
    row = {"prompt_id": "p0", "prompt": "x", "video": "v.mp4",
           "reward": "videoreward", "error": "RuntimeError: scorer blew up"}
    report = validate_scores(_to_text([row]))
    assert report["usable"] == 0
    assert report["dropped"]["score_failed"] == 1
    assert report["dropped"]["missing_keys"] == 0
    assert report["dropped_rows"][0]["reason"] == "score_failed"
    assert "scorer blew up" in report["dropped_rows"][0]["detail"]


def test_missing_and_mistyped_keys():
    no_overall = {"prompt_id": "p0", "prompt": "x", "video": "v.mp4"}
    str_overall = _ok_row(1)
    str_overall["overall"] = "1.0"
    bool_overall = _ok_row(2)
    bool_overall["overall"] = True  # bool is an int subclass — must reject
    empty_video = _ok_row(3)
    empty_video["video"] = ""
    report = validate_scores(
        _to_text([no_overall, str_overall, bool_overall, empty_video, [1, 2]])
    )
    assert report["dropped"]["missing_keys"] == 5
    details = [r["detail"] for r in report["dropped_rows"]]
    assert "missing: overall" in details[0]
    assert "overall is not a number" in details[1]
    assert "overall is not a number" in details[2]
    assert "video is not a non-empty string" in details[3]
    assert details[4] == "row is not a JSON object"
    assert report["dropped_rows"][4]["prompt_id"] is None


def test_non_finite_scores_rejected():
    # Python's json parser accepts NaN/Infinity literals; both pass
    # isinstance(float) but poison margin filtering in build_pairs.
    text = (
        '{"prompt_id":"p0","prompt":"x","video":"a.mp4","overall":NaN}\n'
        '{"prompt_id":"p1","prompt":"y","video":"b.mp4","overall":Infinity}\n'
        '{"prompt_id":"p2","prompt":"z","video":"c.mp4","overall":-Infinity}\n'
    )
    report = validate_scores(text)
    assert report["usable"] == 0
    assert report["dropped"]["non_finite_score"] == 3
    assert report["dropped_rows"][0]["detail"] == "overall=nan"


def test_reward_mismatch_filter():
    mixed = [
        _ok_row(0),
        {**_ok_row(1), "reward": "va-judger"},
        {k: v for k, v in _ok_row(2).items() if k != "reward"},  # untagged
    ]
    text = _to_text(mixed)
    # Filter off: all usable, histogram makes the mix visible.
    report = validate_scores(text)
    assert report["usable"] == 3
    assert report["rewards"] == {"va-judger": 1, "videoreward": 2}
    # Filter on: foreign-tagged row drops; untagged counts as videoreward.
    gated = validate_scores(text, reward="videoreward")
    assert gated["usable"] == 2
    assert gated["dropped"]["reward_mismatch"] == 1
    assert gated["dropped_rows"][0]["detail"] == "reward=va-judger"
    assert [r["prompt_id"] for r in gated["usable_rows"]] == ["p0", "p2"]


def test_bad_json_lines_counted_valid_rows_survive():
    text = json.dumps(_ok_row(0)) + "\n{bad json\n\n" + json.dumps(_ok_row(1))
    report = validate_scores(text)
    assert report["rows"] == 3
    assert report["bad_json_lines"] == 1
    assert report["usable"] == 2


def test_report_deterministic_across_runs():
    rows = [_ok_row(0), {"prompt_id": "p1", "error": "x"}, _ok_row(2)]
    text = _to_text(rows)
    a = validate_scores(text)
    b = validate_scores(text)
    assert json.dumps(summarize(a), sort_keys=True) == json.dumps(
        summarize(b), sort_keys=True
    )
    assert list(a["dropped"].keys()) == list(DROP_REASONS)


# ---- gate CLI ----------------------------------------------------------------


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_scores.py"), *cli_args],
        capture_output=True,
        text=True,
    )


def test_cli_clean_scores_exit_0(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(_to_text([_ok_row(0), _ok_row(1, 2.0)]))
    proc = _run_cli("--scores", str(scores))
    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["usable"] == 2
    assert summary["dropped"] == dict.fromkeys(DROP_REASONS, 0)
    assert summary["rewards"] == {"videoreward": 2}


def test_cli_rejects_unusable_rows_and_show_errors(tmp_path):
    rows = [
        {"prompt_id": "p0", "prompt": "x", "video": "a.mp4", "error": "boom"},
        {"prompt_id": "p1", "prompt": "y", "video": "b.mp4"},
    ]
    scores = tmp_path / "scores.jsonl"
    scores.write_text(_to_text(rows))
    proc = _run_cli("--scores", str(scores), "--show-errors")
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert summary["dropped"]["score_failed"] == 1
    assert summary["dropped"]["missing_keys"] == 1
    err_lines = [json.loads(ln) for ln in proc.stderr.splitlines()
                 if ln.startswith("{")]
    assert [r["reason"] for r in err_lines] == ["score_failed", "missing_keys"]


def test_cli_missing_file_exit_1(tmp_path):
    proc = _run_cli("--scores", str(tmp_path / "nope.jsonl"))
    assert proc.returncode == 1
    assert proc.stderr.strip()


def test_cli_reward_flag_enforces_single_scorer(tmp_path):
    scores = tmp_path / "scores.jsonl"
    scores.write_text(
        _to_text([_ok_row(0), {**_ok_row(1), "reward": "va-judger"}])
    )
    # No filter: both rows usable, exit 0.
    assert _run_cli("--scores", str(scores)).returncode == 0
    proc = _run_cli("--scores", str(scores), "--reward", "videoreward")
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert summary["dropped"]["reward_mismatch"] == 1


# ---- make_pref_pairs fail-fast wiring ----------------------------------------


def test_make_pref_pairs_drops_unusable_and_refuses_empty(tmp_path):
    script = ROOT / "scripts" / "make_pref_pairs.py"
    # (a) Mixed file: error row dropped with a reason on stderr, the two
    # usable rows still pair (margin 1.5 >= 0.5), exit 0.
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text(
        _to_text(
            [
                _ok_row(0, 2.0),
                {**_ok_row(0, 0.5), "video": "cand/p0/cand_1.mp4"},
                {"prompt_id": "p1", "prompt": "x", "video": "c.mp4",
                 "error": "boom"},
            ]
        )
    )
    out = tmp_path / "pairs.jsonl"
    proc = subprocess.run(
        [sys.executable, str(script), "--scores", str(mixed),
         "--out", str(out)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "score_failed" in proc.stderr
    pairs = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(pairs) == 1
    assert pairs[0]["prompt_id"] == "p0"
    assert pairs[0]["win_score"] == 2.0
    # (b) All-error file: refuses before building anything, no output file.
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps({"prompt_id": "p9", "prompt": "x", "video": "z.mp4",
                    "error": "boom"})
        + "\n"
    )
    out2 = tmp_path / "pairs2.jsonl"
    proc = subprocess.run(
        [sys.executable, str(script), "--scores", str(bad),
         "--out", str(out2)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "no usable score rows" in proc.stderr
    assert not out2.exists()

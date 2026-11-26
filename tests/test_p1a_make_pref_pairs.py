# SPDX-License-Identifier: Apache-2.0
"""CLI contract tests for scripts/make_pref_pairs.py main() layer.

The pure functions (build_pairs / coverage_report / validate_scores) are
covered in test_p1a.py and test_p1a_score_check.py, and the fail-fast
wiring (drop-with-reason / refuse-on-empty) by
test_p1a_score_check.py::test_make_pref_pairs_drops_unusable_and_refuses_empty.
What remained uncovered is the rest of the CLI surface: the clean-run
stdout/stderr contract, --margin / --max-per-prompt / --reward flag
pass-through, the missing-scores-file exit path, the bad-JSON suffix,
and rerun determinism. A wiring bug here would otherwise only surface
on the GPU pod between score_eval_set and encode_pairs.

All scores use binary-exact values (multiples of 0.125) so float
equality is exact. The script is CPU-only; tests run the real script
via subprocess with no injection.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_pref_pairs.py"


def _row(prompt_id, video, overall, prompt="p", reward=None):
    row = {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "video": video,
        "overall": overall,
    }
    if reward is not None:
        row["reward"] = reward
    return row


def _write(path, rows, bad_json_lines=0):
    text = "".join(json.dumps(r) + "\n" for r in rows)
    text = "{not json\n" * bad_json_lines + text
    path.write_text(text, encoding="utf-8")
    return path


def _run(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
    )


def _split_stdout(stdout):
    """Return (coverage dict, wrote line). Coverage JSON is indent-2."""
    lines = stdout.splitlines()
    idx = next(i for i, ln in enumerate(lines) if ln.startswith("wrote "))
    return json.loads("\n".join(lines[:idx])), lines[idx]


def test_happy_path_all_usable_clean_run(tmp_path):
    scores = _write(
        tmp_path / "scores.jsonl",
        [
            _row("p1", "cand/p1/cand_0.mp4", 2.0),
            _row("p1", "cand/p1/cand_1.mp4", 1.0),
            _row("p0", "cand/p0/cand_0.mp4", 1.5),
            _row("p0", "cand/p0/cand_1.mp4", 0.5),
        ],
    )
    out = tmp_path / "pairs.jsonl"
    proc = _run("--scores", str(scores), "--out", str(out))
    assert proc.returncode == 0
    # Fully-usable file: no dropped-rows report on stderr at all.
    assert proc.stderr == ""
    cov, wrote = _split_stdout(proc.stdout)
    assert cov == {
        "prompts_total": 2,
        "prompts_paired": 2,
        "pairs": 2,
        "pair_rate": 1.0,
    }
    assert wrote == f"wrote 2 pairs -> {out}"
    pairs = [json.loads(ln) for ln in out.read_text().splitlines()]
    # Sorted by prompt_id regardless of file order.
    assert [p["prompt_id"] for p in pairs] == ["p0", "p1"]
    assert pairs[1]["win_video"] == "cand/p1/cand_0.mp4"
    assert pairs[1]["lose_video"] == "cand/p1/cand_1.mp4"
    assert pairs[1]["win_score"] == 2.0
    assert pairs[1]["lose_score"] == 1.0


def test_pair_row_field_set_and_margin_value(tmp_path):
    scores = _write(
        tmp_path / "scores.jsonl",
        [
            _row("p0", "a.mp4", 2.0, prompt="the prompt"),
            _row("p0", "b.mp4", 1.25, prompt="the prompt"),
        ],
    )
    out = tmp_path / "pairs.jsonl"
    proc = _run("--scores", str(scores), "--out", str(out))
    assert proc.returncode == 0
    (pair,) = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert set(pair) == {
        "prompt_id",
        "prompt",
        "win_video",
        "lose_video",
        "win_score",
        "lose_score",
        "margin",
    }
    assert pair["prompt"] == "the prompt"
    assert pair["margin"] == 0.75  # win - lose, binary-exact


def test_margin_flag_pass_through_and_inclusive_boundary(tmp_path):
    rows = [
        _row("p0", "a.mp4", 2.0),
        _row("p0", "b.mp4", 1.25),  # gap exactly 0.75
    ]
    scores = _write(tmp_path / "scores.jsonl", rows)
    out = tmp_path / "pairs.jsonl"
    # margin above the gap: pairs file is still written (empty) but the
    # run exits 1 with the advisory on stderr.
    proc = _run("--scores", str(scores), "--out", str(out), "--margin", "1.0")
    assert proc.returncode == 1
    assert "no pairs cleared the margin" in proc.stderr
    assert out.read_text() == ""
    # Inclusive boundary: gap == margin passes.
    proc = _run("--scores", str(scores), "--out", str(out), "--margin", "0.75")
    assert proc.returncode == 0
    assert len(out.read_text().splitlines()) == 1


def test_max_per_prompt_pass_through(tmp_path):
    rows = [
        _row("p0", "a.mp4", 3.0),
        _row("p0", "b.mp4", 2.0),
        _row("p0", "c.mp4", 1.0),
    ]
    scores = _write(tmp_path / "scores.jsonl", rows)
    out = tmp_path / "pairs.jsonl"
    # Default: classic best-vs-worst only.
    proc = _run("--scores", str(scores), "--out", str(out))
    assert proc.returncode == 0
    pairs = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(pairs) == 1
    assert pairs[0]["lose_score"] == 1.0
    # --max-per-prompt 2: best against the two worst, both recorded.
    proc = _run(
        "--scores", str(scores), "--out", str(out), "--max-per-prompt", "2"
    )
    assert proc.returncode == 0
    pairs = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert len(pairs) == 2
    assert {p["lose_score"] for p in pairs} == {1.0, 2.0}
    assert all(p["win_video"] == "a.mp4" for p in pairs)


def test_reward_flag_filters_other_scorers(tmp_path):
    scores = _write(
        tmp_path / "scores.jsonl",
        [
            _row("p0", "a.mp4", 2.0, reward="videoreward"),
            _row("p0", "b.mp4", 1.0, reward="videoreward"),
            _row("p1", "c.mp4", 2.0, reward="other-rm"),
            _row("p1", "d.mp4", 1.0, reward="other-rm"),
        ],
    )
    out = tmp_path / "pairs.jsonl"
    proc = _run(
        "--scores", str(scores), "--out", str(out), "--reward", "videoreward"
    )
    assert proc.returncode == 0
    assert "reward_mismatch" in proc.stderr
    pairs = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert [p["prompt_id"] for p in pairs] == ["p0"]
    # Coverage is computed over the filtered (usable) rows only.
    cov, _ = _split_stdout(proc.stdout)
    assert cov["prompts_total"] == 1
    assert cov["pairs"] == 1


def test_missing_scores_file_exit_1_no_output(tmp_path):
    out = tmp_path / "pairs.jsonl"
    proc = _run(
        "--scores", str(tmp_path / "nope.jsonl"), "--out", str(out)
    )
    assert proc.returncode == 1
    assert proc.stderr.strip() != ""
    assert not out.exists()


def test_bad_json_lines_suffix_on_stderr(tmp_path):
    scores = _write(
        tmp_path / "scores.jsonl",
        [
            _row("p0", "a.mp4", 2.0),
            _row("p0", "b.mp4", 1.0),
        ],
        bad_json_lines=2,
    )
    out = tmp_path / "pairs.jsonl"
    proc = _run("--scores", str(scores), "--out", str(out))
    assert proc.returncode == 0
    assert "2 bad JSON line(s)" in proc.stderr
    assert len(out.read_text().splitlines()) == 1


def test_rerun_byte_identical(tmp_path):
    scores = _write(
        tmp_path / "scores.jsonl",
        [
            _row("p1", "cand/p1/cand_0.mp4", 2.0),
            _row("p1", "cand/p1/cand_1.mp4", 0.875),
            _row("p0", "cand/p0/cand_0.mp4", 1.5),
            _row("p0", "cand/p0/cand_1.mp4", 0.5),
        ],
    )
    out = tmp_path / "pairs.jsonl"
    first = _run("--scores", str(scores), "--out", str(out))
    assert first.returncode == 0
    payload = out.read_bytes()
    second = _run("--scores", str(scores), "--out", str(out))
    assert second.returncode == 0
    assert second.stdout == first.stdout
    assert second.stderr == first.stderr == ""
    assert out.read_bytes() == payload

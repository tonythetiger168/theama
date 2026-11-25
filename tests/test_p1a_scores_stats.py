# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/scores_stats.py (P1a scores.jsonl profiler).

Covers the pure profile function (prompt coverage, pairable yield,
duplicates, overall/dims stats, error concentration) and the CLI contract
(tolerant jsonl parsing, deterministic stdout, --out).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "scores_stats", ROOT / "scripts" / "scores_stats.py"
)
ss = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ss)


def _write_scores(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "scores.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _score(prompt_id: str = "p1", video: str | None = None,
           overall: float = 3.0, dims: dict | None = None,
           reward: str | None = None) -> dict:
    row = {
        "prompt_id": prompt_id,
        "prompt": f"prompt for {prompt_id}",
        "video": video or f"v/{prompt_id}_a.mp4",
        "overall": overall,
        "dims": {"visual_quality": overall} if dims is None else dims,
    }
    if reward is not None:
        row["reward"] = reward
    return row


def _error(prompt_id: str = "p1", video: str | None = None,
           error: str = "decode failed") -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt": f"prompt for {prompt_id}",
        "video": video or f"v/{prompt_id}_a.mp4",
        "reward": "videoreward",
        "error": error,
    }


def test_profile_empty_scores():
    profile = ss.profile_scores([])
    assert profile["rows"] == 0
    assert profile["success_rows"] == 0
    assert profile["error_rows"] == 0
    assert profile["incomplete_rows"] == 0
    assert profile["unique_prompts"] == 0
    assert profile["prompts_pairable"] == 0
    assert profile["prompts_single_candidate"] == 0
    assert profile["prompts_all_failed"] == 0
    assert profile["candidates_per_prompt"] == {}
    assert profile["duplicate_scored_rows"] == 0
    assert profile["overall"] is None
    assert profile["dims"] == {}
    assert profile["rewards"] == {}
    assert profile["error_kinds"] == {
        "distinct": 0, "max_single_error": 0, "top": {},
    }


def test_profile_prompt_coverage_and_pairable_yield():
    rows = [
        _score("p1", video="v/p1_a.mp4"),              # p1: 2 candidates
        _score("p1", video="v/p1_b.mp4"),
        _score("p2"),                                  # p2: 1 candidate
        _error("p3", error="oom"),                     # p3: all failed
        _error("p3", video="v/p3_b.mp4", error="oom"),
    ]
    profile = ss.profile_scores(rows)
    assert profile["rows"] == 5
    assert profile["success_rows"] == 3
    assert profile["error_rows"] == 2
    assert profile["unique_prompts"] == 3
    assert profile["prompts_pairable"] == 1
    assert profile["prompts_single_candidate"] == 1
    assert profile["prompts_all_failed"] == 1
    assert profile["candidates_per_prompt"] == {"1": 1, "2": 1}


def test_profile_overall_and_dims_stats():
    rows = [
        _score("p1", video="v/1.mp4", overall=1.0,
               dims={"visual_quality": 1.0, "motion": 2.0}),
        _score("p1", video="v/2.mp4", overall=2.0,
               dims={"visual_quality": 2.0, "motion": 4.0}),
        _score("p2", video="v/3.mp4", overall=4.0,
               dims={"visual_quality": 4.0}),
    ]
    profile = ss.profile_scores(rows)
    overall = profile["overall"]
    assert overall["count"] == 3
    assert overall["min"] == 1.0 and overall["max"] == 4.0
    assert overall["median"] == 2.0
    dims = profile["dims"]
    assert dims["visual_quality"] == {
        "count": 3, "mean": 2.333, "min": 1.0, "max": 4.0,
    }
    assert dims["motion"] == {"count": 2, "mean": 3.0, "min": 2.0, "max": 4.0}


def test_profile_error_concentration():
    rows = [
        _score("p1"),
        _error("p2", error="decode failed"),
        _error("p3", error="decode failed"),
        _error("p4", error="oom"),
    ]
    kinds = ss.profile_scores(rows)["error_kinds"]
    assert kinds["distinct"] == 2
    assert kinds["max_single_error"] == 2
    # count desc, then key asc
    assert list(kinds["top"].items()) == [("decode failed", 2), ("oom", 1)]


def test_profile_duplicates_incomplete_and_reward_tags():
    rows = [
        _score("p1", video="v/dup.mp4", reward="videoreward"),
        _score("p1", video="v/dup.mp4"),   # duplicate (prompt_id, video)
        _score("p1", video="v/other.mp4", reward="videoreward"),
        {"prompt_id": "p2", "prompt": "missing video/overall"},
        {"prompt_id": "p3", "prompt": "x", "video": "v/nan.mp4",
         "overall": float("nan")},
    ]
    profile = ss.profile_scores(rows)
    assert profile["rows"] == 5
    assert profile["success_rows"] == 3
    assert profile["incomplete_rows"] == 2
    assert profile["duplicate_scored_rows"] == 1
    # NaN row is incomplete, not a success: it must not poison the stats.
    assert profile["overall"]["count"] == 3
    # Untagged rows count as the historical default reward.
    assert profile["rewards"] == {"videoreward": 5}


def test_profile_deterministic_key_order():
    rows = [
        _score("p2", video="v/b.mp4", overall=2.0),
        _score("p1", video="v/a.mp4", overall=1.0),
        _score("p1", video="v/c.mp4", overall=3.0),
        _error("p3", error="x"),
    ]
    profile = ss.profile_scores(rows)
    again = ss.profile_scores(list(reversed(rows)))
    assert json.dumps(profile, sort_keys=True) == json.dumps(again,
                                                             sort_keys=True)


def test_cli_tolerates_bad_json_lines(tmp_path):
    scores = _write_scores(tmp_path, [
        json.dumps(_score("p1")),
        "{not json",
        json.dumps(["a", "list", "is", "not", "a", "row"]),
        "",
        json.dumps(_score("p1", video="v/p1_b.mp4")),
    ])
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scores_stats.py"),
         "--scores", str(scores)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    profile = json.loads(proc.stdout)
    assert profile["rows"] == 2
    assert profile["bad_json_lines"] == 2
    assert profile["prompts_pairable"] == 1
    assert profile["scores"] == str(scores)


def test_cli_missing_scores_exits_1(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "scores_stats.py"),
         "--scores", str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert proc.stderr.strip()


def test_cli_deterministic_stdout_and_out_file(tmp_path):
    lines = [
        json.dumps(_score("p1", video="v/a.mp4", overall=1.5)),
        json.dumps(_score("p1", video="v/b.mp4", overall=2.5)),
        json.dumps(_error("p2", error="oom")),
    ]
    scores = _write_scores(tmp_path, lines)
    cmd = [sys.executable, str(ROOT / "scripts" / "scores_stats.py"),
           "--scores", str(scores)]
    first = subprocess.run(cmd, capture_output=True, text=True)
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout  # byte-identical across runs

    out = tmp_path / "stats.json"
    proc = subprocess.run(cmd + ["--out", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == json.loads(proc.stdout)

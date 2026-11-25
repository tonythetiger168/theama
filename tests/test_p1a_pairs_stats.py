# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/pairs_stats.py (P1a preference-pair profiler).

Covers the pure profile function (prompt coverage, pairs-per-prompt,
margin buckets, winner/loser concentration, self-pair visibility) and the
CLI contract (tolerant jsonl parsing, deterministic stdout, --out).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "pairs_stats", ROOT / "scripts" / "pairs_stats.py"
)
ps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ps)


def _write_pairs(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "pairs.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _pair(prompt_id: str = "p1", win: str = "v/p1_a.mp4",
          lose: str = "v/p1_b.mp4", win_score: float = 3.0,
          lose_score: float = 2.0) -> dict:
    return {
        "prompt_id": prompt_id,
        "prompt": f"prompt for {prompt_id}",
        "win_video": win,
        "lose_video": lose,
        "win_score": win_score,
        "lose_score": lose_score,
        "margin": round(win_score - lose_score, 6),
    }


def test_profile_empty_pairs():
    profile = ps.profile_pairs([])
    assert profile["rows"] == 0
    assert profile["complete_rows"] == 0
    assert profile["incomplete_rows"] == 0
    assert profile["unique_prompts"] == 0
    assert profile["pairs_per_prompt"] == {}
    assert profile["self_pairs"] == 0
    assert profile["margin"] is None
    assert profile["win_score"] is None
    assert profile["lose_score"] is None
    assert profile["winners"] == {"distinct": 0, "max_single_clip": 0, "top": {}}
    assert profile["losers"] == {"distinct": 0, "max_single_clip": 0, "top": {}}


def test_profile_prompt_coverage_and_pairs_per_prompt():
    rows = [
        _pair("p1"),
        _pair("p2", win="v/p2_a.mp4", lose="v/p2_b.mp4"),
        _pair("p2", win="v/p2_a.mp4", lose="v/p2_c.mp4"),
        _pair("p3", win="v/p3_a.mp4", lose="v/p3_b.mp4"),
        _pair("p3", win="v/p3_a.mp4", lose="v/p3_c.mp4"),
        _pair("p3", win="v/p3_b.mp4", lose="v/p3_c.mp4"),
    ]
    profile = ps.profile_pairs(rows)
    assert profile["rows"] == 6
    assert profile["complete_rows"] == 6
    assert profile["unique_prompts"] == 3
    # p1 has 1 pair, p2 has 2, p3 has 3 -> one prompt at each depth.
    assert profile["pairs_per_prompt"] == {"1": 1, "2": 1, "3": 1}


def test_profile_margin_stats_and_buckets():
    rows = [
        _pair("a", win_score=1.2, lose_score=1.0),   # 0.2 -> lt_0.25
        _pair("b", win_score=2.25, lose_score=2.0),  # 0.25 -> 0.25_to_0.5
        _pair("c", win_score=3.0, lose_score=2.0),   # 1.0 -> 1_to_2
        _pair("d", win_score=5.5, lose_score=2.0),   # 3.5 -> ge_2
    ]
    margin = ps.profile_pairs(rows)["margin"]
    assert margin["count"] == 4
    assert margin["min"] == 0.2
    assert margin["max"] == 3.5
    assert margin["buckets"] == {
        "lt_0.25": 1,
        "0.25_to_0.5": 1,
        "0.5_to_1": 0,
        "1_to_2": 1,
        "ge_2": 1,
    }
    win = ps.profile_pairs(rows)["win_score"]
    assert win["min"] == 1.2 and win["max"] == 5.5
    lose = ps.profile_pairs(rows)["lose_score"]
    assert lose["count"] == 4


def test_profile_winner_loser_concentration():
    rows = [
        _pair("p1", win="v/champ.mp4", lose="v/p1_b.mp4"),
        _pair("p2", win="v/champ.mp4", lose="v/p2_b.mp4"),
        _pair("p3", win="v/champ.mp4", lose="v/p2_b.mp4"),
        _pair("p4", win="v/p4_a.mp4", lose="v/p2_b.mp4"),
    ]
    profile = ps.profile_pairs(rows)
    winners = profile["winners"]
    assert winners["distinct"] == 2
    assert winners["max_single_clip"] == 3
    assert winners["top"] == {"v/champ.mp4": 3, "v/p4_a.mp4": 1}
    losers = profile["losers"]
    assert losers["distinct"] == 2
    assert losers["max_single_clip"] == 3
    assert losers["top"]["v/p2_b.mp4"] == 3


def test_profile_self_pairs_and_incomplete_rows():
    rows = [
        _pair("p1", win="v/same.mp4", lose="v/same.mp4"),  # self pair
        {"prompt_id": "p2", "prompt": "missing most keys"},  # incomplete
        _pair("p3"),
    ]
    profile = ps.profile_pairs(rows)
    assert profile["rows"] == 3
    assert profile["complete_rows"] == 2
    assert profile["incomplete_rows"] == 1
    assert profile["self_pairs"] == 1
    assert profile["unique_prompts"] == 3
    # The incomplete row contributes no numeric fields but is not dropped.
    assert profile["margin"]["count"] == 2


def test_profile_deterministic_key_order():
    rows = [
        _pair("p2", win="v/b.mp4", lose="v/x.mp4"),
        _pair("p1", win="v/a.mp4", lose="v/x.mp4"),
        _pair("p3", win="v/a.mp4", lose="v/y.mp4"),
    ]
    profile = ps.profile_pairs(rows)
    # count desc, then key asc
    assert list(profile["winners"]["top"].items()) == [("v/a.mp4", 2),
                                                       ("v/b.mp4", 1)]
    again = ps.profile_pairs(list(reversed(rows)))
    assert json.dumps(profile, sort_keys=True) == json.dumps(again,
                                                             sort_keys=True)


def test_cli_tolerates_bad_json_lines(tmp_path):
    pairs = _write_pairs(tmp_path, [
        json.dumps(_pair("p1")),
        "{not json",
        json.dumps(["a", "list", "is", "not", "a", "row"]),
        "",
        json.dumps(_pair("p2", win="v/p2_a.mp4", lose="v/p2_b.mp4")),
    ])
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pairs_stats.py"),
         "--pairs", str(pairs)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    profile = json.loads(proc.stdout)
    assert profile["rows"] == 2
    assert profile["bad_json_lines"] == 2
    assert profile["unique_prompts"] == 2
    assert profile["pairs"] == str(pairs)


def test_cli_missing_pairs_exits_1(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pairs_stats.py"),
         "--pairs", str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert proc.stderr.strip()


def test_cli_deterministic_stdout_and_out_file(tmp_path):
    lines = [
        json.dumps(_pair("p1", win="v/champ.mp4", lose="v/p1_b.mp4")),
        json.dumps(_pair("p2", win="v/champ.mp4", lose="v/p2_b.mp4",
                         win_score=4.0, lose_score=1.0)),
        json.dumps({"prompt_id": "p3"}),  # incomplete row
    ]
    pairs = _write_pairs(tmp_path, lines)
    cmd = [sys.executable, str(ROOT / "scripts" / "pairs_stats.py"),
           "--pairs", str(pairs)]
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

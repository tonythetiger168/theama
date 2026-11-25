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
    assert report["dropped_rows"][0"]["line"] == 1

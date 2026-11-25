# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P1a pair-latent preflight (data/pair_latent_check.py,
scripts/check_pair_latents.py, and the train_dpo.py fail-fast wiring).

Covers the pure validator (drop-reason precedence, injected exists,
deterministic report), the gate CLI contract (exit codes, --show-errors,
--no-files), and that train_dpo.py refuses an unusable manifest before
importing the GPU trainer.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.pair_latent_check import (  # noqa: E402
    DROP_REASONS,
    summarize,
    validate_pair_latents,
)

_spec = importlib.util.spec_from_file_location(
    "check_pair_latents", ROOT / "scripts" / "check_pair_latents.py"
)
cpl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpl)


def _ok_row(index: int = 0, win: str = "c/p0_w.pt", lose: str = "c/p0_l.pt") -> dict:
    return {
        "index": index,
        "prompt_id": f"p{index}",
        "prompt": f"prompt {index}",
        "model": "ti2v-5b",
        "win_latent": win,
        "lose_latent": lose,
        "reused": False,
    }


def _to_text(rows: list[dict]) -> str:
    return "\n".join(json.dumps(r) for r in rows) + "\n"


ALL = lambda p: True  # noqa: E731
NONE = lambda p: False  # noqa: E731


# ---- pure validator ---------------------------------------------------------


def test_all_usable_passthrough_in_order():
    rows = [_ok_row(i, f"c/p{i}_w.pt", f"c/p{i}_l.pt") for i in range(3)]
    report = validate_pair_latents(_to_text(rows), exists=ALL)
    assert report["rows"] == 3
    assert report["usable"] == 3
    assert report["usable_rows"] == rows  # file order preserved
    assert report["dropped"] == dict.fromkeys(DROP_REASONS, 0)
    assert report["dropped_rows"] == []
    assert report["bad_json_lines"] == 0


def test_error_row_reports_encode_failed_before_missing_keys():
    # Failure rows carry no win/lose keys; the root cause must still be
    # encode_failed, not missing_keys (same precedence as the joint loader).
    row = {"index": 0, "prompt_id": "p0", "prompt": "x", "model": "ti2v-5b",
           "error": "RuntimeError: decode blew up"}
    report = validate_pair_latents(_to_text([row]), exists=ALL)
    assert report["usable"] == 0
    assert report["dropped"]["encode_failed"] == 1
    assert report["dropped"]["missing_keys"] == 0
    assert report["dropped_rows"][0]["reason"] == "encode_failed"
    assert "decode blew up" in report["dropped_rows"][0]["detail"]


def test_skipped_row_reports_budget_exhausted():
    row = {"index": 1, "prompt_id": "p1", "prompt": "x", "model": "ti2v-5b",
           "skipped": True, "retries": 3, "last_error": "boom"}
    report = validate_pair_latents(_to_text([row]), exists=ALL)
    assert report["dropped"]["skipped_budget_exhausted"] == 1
    assert report["dropped_rows"][0]["index"] == 1


def test_missing_and_mistyped_keys():
    no_lose = {"index": 0, "prompt": "x", "win_latent": "c/w.pt"}
    bad_type = _ok_row(1)
    bad_type["win_latent"] = None
    report = validate_pair_latents(_to_text([no_lose, bad_type, [1, 2]]),
                                   exists=ALL)
    assert report["dropped"]["missing_keys"] == 3
    details = [r["detail"] for r in report["dropped_rows"]]
    assert "missing: lose_latent" in details[0]
    assert "win_latent is not a path string" in details[1]
    assert details[2] == "row is not a JSON object"
    assert report["dropped_rows"][2]["index"] is None


def test_missing_latent_file_uses_injected_exists():
    rows = [_ok_row(0), _ok_row(1, "c/p1_w.pt", "c/p1_l.pt")]
    exists = lambda p: p != "c/p1_l.pt"  # noqa: E731
    report = validate_pair_latents(_to_text(rows), exists=exists)
    assert report["usable"] == 1
    assert report["dropped"]["missing_latent_file"] == 1
    assert report["dropped_rows"][0]["detail"] == "lose_latent: c/p1_l.pt"


def test_bad_json_lines_counted_valid_rows_survive():
    text = json.dumps(_ok_row(0)) + "\n{bad json\n\n" + json.dumps(_ok_row(1))
    report = validate_pair_latents(text, exists=ALL)
    assert report["rows"] == 3
    assert report["bad_json_lines"] == 1
    assert report["usable"] == 2


def test_report_deterministic_across_runs():
    rows = [_ok_row(0), {"index": 1, "error": "x"}, _ok_row(2)]
    text = _to_text(rows)
    a = validate_pair_latents(text, exists=ALL)
    b = validate_pair_latents(text, exists=ALL)
    assert json.dumps(summarize(a), sort_keys=True) == json.dumps(
        summarize(b), sort_keys=True
    )
    assert list(a["dropped"].keys()) == list(DROP_REASONS)


# ---- gate CLI ----------------------------------------------------------------


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_pair_latents.py"), *cli_args],
        capture_output=True,
        text=True,
    )


def test_cli_clean_manifest_exit_0(tmp_path):
    win = tmp_path / "w.pt"
    lose = tmp_path / "l.pt"
    win.write_bytes(b"x")
    lose.write_bytes(b"y")
    manifest = tmp_path / "pairs_latents.jsonl"
    manifest.write_text(_to_text([_ok_row(0, str(win), str(lose))]))
    proc = _run_cli("--manifest", str(manifest))
    assert proc.returncode == 0
    summary = json.loads(proc.stdout)
    assert summary["usable"] == 1
    assert summary["dropped"] == dict.fromkeys(DROP_REASONS, 0)


def test_cli_rejects_unusable_rows_and_show_errors(tmp_path):
    rows = [
        {"index": 0, "prompt_id": "p0", "prompt": "x", "error": "boom"},
        {"index": 1, "prompt_id": "p1", "prompt": "y", "skipped": True,
         "retries": 2, "last_error": "z"},
    ]
    manifest = tmp_path / "pairs_latents.jsonl"
    manifest.write_text(_to_text(rows))
    proc = _run_cli("--manifest", str(manifest), "--show-errors", "--no-files")
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert summary["dropped"]["encode_failed"] == 1
    assert summary["dropped"]["skipped_budget_exhausted"] == 1
    err_lines = [json.loads(ln) for ln in proc.stderr.splitlines()
                 if ln.startswith("{")]
    assert [r["reason"] for r in err_lines] == [
        "encode_failed",
        "skipped_budget_exhausted",
    ]


def test_cli_missing_manifest_exit_1(tmp_path):
    proc = _run_cli("--manifest", str(tmp_path / "nope.jsonl"))
    assert proc.returncode == 1
    assert proc.stderr.strip()


def test_cli_no_files_skips_disk_check(tmp_path):
    manifest = tmp_path / "pairs_latents.jsonl"
    manifest.write_text(_to_text([_ok_row(0)]))  # .pt files do not exist
    assert _run_cli("--manifest", str(manifest)).returncode == 1
    assert _run_cli("--manifest", str(manifest), "--no-files").returncode == 0


# ---- train_dpo fail-fast wiring ----------------------------------------------


def test_train_dpo_refuses_unusable_manifest_before_trainer_import(tmp_path):
    # The preflight must fire before `from theama.train.dpo import ...` —
    # proven here by exiting 1 quickly on a box without diffusers.
    manifest = tmp_path / "pairs_latents.jsonl"
    manifest.write_text(
        json.dumps({"index": 0, "prompt_id": "p0", "prompt": "x",
                    "error": "boom"})
        + "\n"
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_dpo.py"),
            "--pairs",
            str(manifest),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "no usable pairs" in proc.stderr
    assert "encode_failed" in proc.stderr

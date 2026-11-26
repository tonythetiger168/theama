# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/train_dpo.py (P1a DPO training entry point).

train_dpo is the final stage of the P1a chain (runbook §2: gen_candidates
-> score_eval_set -> check_scores -> make_pref_pairs -> encode_pairs ->
check_pair_latents -> train_dpo). Its CLI contract — read the YAML config,
preflight the pairs manifest with the REAL pure-stdlib
data/pair_latent_check validator BEFORE the backbone trainer is even
imported (a bad manifest fails fast on CPU instead of raising
KeyError/FileNotFoundError inside the GPU loop), report dropped rows with
reasons on stderr, refuse to start when nothing usable remains, then hand
``(cfg, usable_rows, out)`` to DPOWanTrainer and print its metrics JSON —
had no main()-level coverage (only py_compile in test_p1a.py). A preflight
wiring bug here only surfaces after the Wan backbone is loaded on the GPU
pod, so it is pinned on CPU with a fake theama.train.dpo module (the
validator and YAML parsing run for real). No GPU or torch required.

Mirrors tests/test_p1a_score_eval_set.py. Deliberately does NOT duplicate
validate_pair_latents coverage (test_p1a_pair_latent_check.py owns it).
"""

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "train_dpo_cli", ROOT / "scripts" / "train_dpo.py"
)
td = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(td)  # inserts ROOT/src into sys.path itself

METRICS = {"final_loss": 0.125, "steps": 500, "grad_norm": 1.0}


def _success_row(tmp_path: Path, idx: int) -> dict:
    """A usable manifest row whose .pt files really exist (default exists)."""
    win = tmp_path / "latents" / f"pair{idx}_win.pt"
    lose = tmp_path / "latents" / f"pair{idx}_lose.pt"
    win.parent.mkdir(parents=True, exist_ok=True)
    win.write_bytes(b"\x00fake")
    lose.write_bytes(b"\x00fake")
    return {
        "index": idx,
        "prompt_id": f"p{idx:02d}",
        "prompt": f"prompt {idx}",
        "model": "ti2v-5b",
        "win_latent": str(win),
        "lose_latent": str(lose),
        "reused": False,
    }


def _write_pairs(path: Path, lines) -> None:
    """Write manifest lines (dicts are JSON-serialized, strs verbatim)."""
    text = "".join(
        (json.dumps(r) if isinstance(r, dict) else r) + "\n" for r in lines
    )
    path.write_text(text, encoding="utf-8")


def _write_config(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps({"dpo": cfg}), encoding="utf-8")  # JSON is YAML
    return path


def _dropped_counts(err: str) -> dict:
    """Extract the sort_keys JSON dict after "dropped " on stderr."""
    return json.loads(re.search(r"dropped (\{[^}]*\})", err).group(1))


def _fake_trainer_module(monkeypatch):
    """Install a fake theama.train.dpo into sys.modules; return call log."""
    calls = {"init": [], "train": []}

    class FakeDPOWanTrainer:
        def __init__(self, cfg):
            self.cfg = cfg
            calls["init"].append(cfg)

        def train(self, pairs, out):
            calls["train"].append((pairs, out))
            return dict(METRICS)

    mod = types.ModuleType("theama.train.dpo")
    mod.DPOWanTrainer = FakeDPOWanTrainer
    monkeypatch.setitem(sys.modules, "theama.train.dpo", mod)
    return calls


def _run(monkeypatch, tmp_path, capsys, pair_lines, extra_args=(),
         fake_trainer=True):
    """Run td.main(); return (exit_code, calls, out_dir)."""
    pairs_file = tmp_path / "pairs_latents.jsonl"
    if pair_lines is not None:
        _write_pairs(pairs_file, pair_lines)
    out_dir = tmp_path / "run1"
    argv = [
        "train_dpo.py", "--pairs", str(pairs_file), "--out", str(out_dir),
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    calls = _fake_trainer_module(monkeypatch) if fake_trainer else None
    code = td.main()
    return code, calls, out_dir


# ------------------------------------------------------------ fail fast


def test_missing_pairs_file_exits_1_before_trainer_import(
    monkeypatch, tmp_path, capsys
):
    # None tripwire: if main() imported theama.train.dpo on this path the
    # import itself would raise — the trainer must stay untouched.
    monkeypatch.setitem(sys.modules, "theama.train.dpo", None)
    code, _, _ = _run(monkeypatch, tmp_path, capsys, pair_lines=None,
                      fake_trainer=False)
    assert code == 1
    err = capsys.readouterr().err
    assert "pairs_latents.jsonl" in err


def test_all_rows_dropped_refuses_to_start(monkeypatch, tmp_path, capsys):
    rows = [
        {"index": 0, "prompt_id": "p00", "prompt": "x", "model": "ti2v-5b",
         "error": "encode boom"},
        {"index": 1, "prompt_id": "p01", "prompt": "y", "model": "ti2v-5b",
         "skipped": True, "retries": 3, "last_error": "oom"},
    ]
    monkeypatch.setitem(sys.modules, "theama.train.dpo", None)
    code, _, _ = _run(monkeypatch, tmp_path, capsys, rows, fake_trainer=False)
    assert code == 1
    err = capsys.readouterr().err
    assert "no usable pairs" in err
    assert "0/2 rows usable" in err
    dropped = _dropped_counts(err)
    assert dropped["encode_failed"] == 1
    assert dropped["skipped_budget_exhausted"] == 1
    assert dropped["missing_keys"] == 0
    assert dropped["missing_latent_file"] == 0


def test_bad_json_lines_reported_in_stderr(monkeypatch, tmp_path, capsys):
    rows = ["{not json",
            {"index": 0, "prompt_id": "p00", "prompt": "x",
             "model": "ti2v-5b", "error": "boom"}]
    monkeypatch.setitem(sys.modules, "theama.train.dpo", None)
    code, _, _ = _run(monkeypatch, tmp_path, capsys, rows, fake_trainer=False)
    assert code == 1
    err = capsys.readouterr().err
    assert "no usable pairs" in err
    assert "1 bad JSON line(s)" in err


# ------------------------------------------------------------- training


def test_happy_path_trains_on_all_usable_rows(monkeypatch, tmp_path, capsys):
    rows = [_success_row(tmp_path, 0), _success_row(tmp_path, 1)]
    code, calls, out_dir = _run(monkeypatch, tmp_path, capsys, rows)
    assert code == 0
    captured = capsys.readouterr()
    assert captured.err == ""  # nothing dropped -> silent stderr
    assert len(calls["init"]) == 1
    assert calls["train"] == [(rows, str(out_dir))]
    # metrics printed as indent-2 JSON, round-trips to the trainer's dict
    out = captured.out
    assert json.loads(out) == METRICS
    assert out.startswith("{\n  ")  # indent=2, not compact


def test_dropped_rows_warn_but_training_uses_usable_only(
    monkeypatch, tmp_path, capsys
):
    good = _success_row(tmp_path, 0)
    stale = _success_row(tmp_path, 1)
    Path(stale["win_latent"]).unlink()  # cache pruned after encoding
    bad_keys = {"index": 2, "prompt_id": "p02", "prompt": "z"}
    code, calls, _ = _run(
        monkeypatch, tmp_path, capsys, [good, stale, bad_keys]
    )
    assert code == 0
    assert calls["train"][0][0] == [good]  # usable rows only, file order
    err = capsys.readouterr().err
    assert "1/3 rows usable" in err
    dropped = _dropped_counts(err)
    assert dropped["missing_latent_file"] == 1
    assert dropped["missing_keys"] == 1
    assert dropped["encode_failed"] == 0


def test_config_dpo_section_passed_verbatim(monkeypatch, tmp_path, capsys):
    cfg = {"train": {"lr": 0.0005, "beta": 250}, "lora": {"rank": 8}}
    config = _write_config(tmp_path / "custom.yaml", cfg)
    code, calls, _ = _run(
        monkeypatch, tmp_path, capsys, [_success_row(tmp_path, 0)],
        extra_args=("--config", str(config)),
    )
    assert code == 0
    assert calls["init"] == [cfg]


def test_default_config_is_repo_dpo_yaml(monkeypatch, tmp_path, capsys):
    import yaml

    expected = yaml.safe_load(
        (ROOT / "configs" / "dpo_ti2v_5b.yaml").read_text()
    )["dpo"]
    code, calls, _ = _run(
        monkeypatch, tmp_path, capsys, [_success_row(tmp_path, 0)]
    )
    assert code == 0
    assert calls["init"] == [expected]


def test_rerun_byte_identical_stdout(monkeypatch, tmp_path, capsys):
    rows = [_success_row(tmp_path, 0)]
    outs = []
    for _ in range(2):
        code, _, _ = _run(monkeypatch, tmp_path, capsys, rows)
        assert code == 0
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1]

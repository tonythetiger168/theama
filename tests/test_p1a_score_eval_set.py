# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/score_eval_set.py (P1a DPO scoring stage).

score_eval_set walks gen_candidates' output, scores each clip with a
registered reward model (GPU), and writes scores.jsonl + a summary sidecar.
Its resume contract — successful rows for the SAME reward model are kept and
compacted, error rows are retried, a fully scored rerun never constructs the
scorer (no GPU model load), --no-resume re-scores from scratch WITHOUT
appending duplicate rows, and failed clips keep exit 0 so a rerun retries
them — decides whether a crashed GPU scoring run resumes cheaply or silently
ships duplicate/broken rows into check_scores / make_pref_pairs. A bug here
costs GPU hours, so it is covered on CPU with a fake theama.reward.score
module (the real load_successful_scores / summarize_scores are pure stdlib
and run for real). No GPU, torch, or transformers required.

Mirrors tests/test_p1a_gen_candidates.py. Deliberately does NOT duplicate
collect() coverage (test_p1a.py owns it).
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "score_eval_set_cli", ROOT / "scripts" / "score_eval_set.py"
)
se = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(se)  # inserts ROOT/src into sys.path itself

# Real pure-stdlib helpers (CPU-safe) — the fake module re-exports them so the
# resume/compaction/summary logic under test is the production code.
from theama.reward.score import load_successful_scores, summarize_scores

PROMPTS = [
    {"id": "p01", "prompt": "a calm lake at dawn"},
    {"id": "p02", "prompt": "busy street timelapse"},
    {"id": "p03", "prompt": "night market crowd"},
]
DEFAULT_SCORE = (0.75, {"flicker": 0.5})


def _write_prompts(path: Path, rows=PROMPTS) -> None:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def _touch_clip(videos: Path, pid: str, k: int = 1) -> Path:
    d = videos / pid
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"cand_{k}.mp4"
    p.write_bytes(b"\x00fake")
    return p


def _rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _write_scores(path: Path, rows: list[dict]) -> None:
    # Same serialization main() uses, so a resume rewrite is byte-identical.
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _success_row(videos: Path, pid: str, overall=0.111, reward="videoreward",
                 k: int = 1) -> dict:
    return {
        "prompt_id": pid,
        "prompt": next(r["prompt"] for r in PROMPTS if r["id"] == pid),
        "video": str(videos / pid / f"cand_{k}.mp4"),
        "reward": reward,
        "overall": overall,
        "dims": {"flicker": 0.25},
    }


class _FakeScoreResult:
    def __init__(self, overall, dims):
        self.overall = overall
        self.dims = dims


def _fake_reward_module(monkeypatch, score_map=None, fail=()) -> dict:
    """Install a fake theama.reward.score into sys.modules; return call log."""
    calls = {"constructed": [], "load": 0, "score": []}
    score_map = score_map or {}
    fail = set(fail)

    class FakeScorer:
        def __init__(self, spec_name, device="cuda"):
            self.spec_name = spec_name
            self.device = device
            calls["constructed"].append((spec_name, device))

        def load(self):
            calls["load"] += 1

        def score(self, video, prompt):
            calls["score"].append((video, prompt))
            if video in fail:
                raise RuntimeError("boom")
            overall, dims = score_map.get(video, DEFAULT_SCORE)
            return _FakeScoreResult(overall, dims)

    mod = types.ModuleType("theama.reward.score")
    mod.VideoRewardScorer = FakeScorer
    mod.load_successful_scores = load_successful_scores
    mod.summarize_scores = summarize_scores
    monkeypatch.setitem(sys.modules, "theama.reward.score", mod)
    return calls


def _run(monkeypatch, tmp_path, clips, extra_args=(), score_map=None, fail=()):
    """Run se.main() with a fake reward module; return (exit_code, calls, videos)."""
    calls = _fake_reward_module(monkeypatch, score_map=score_map, fail=fail)
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)  # exist_ok: the determinism test re-runs in-place
    for pid, k in clips:
        _touch_clip(videos, pid, k)
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    argv = [
        "score_eval_set.py", "--videos", str(videos),
        "--prompts", str(prompts), *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    code = se.main()
    return code, calls, videos


# ---------------------------------------------------------------- fresh run


def test_fresh_run_scores_in_collect_order_and_writes_everything(
    monkeypatch, tmp_path, capsys
):
    code, calls, videos = _run(monkeypatch, tmp_path, [("p01", 1), ("p02", 1)])
    assert code == 0
    assert calls["constructed"] == [("videoreward", "cuda")]
    assert calls["load"] == 1
    assert calls["score"] == [
        (str(videos / "p01" / "cand_1.mp4"), "a calm lake at dawn"),
        (str(videos / "p02" / "cand_1.mp4"), "busy street timelapse"),
    ]
    rows = _rows(videos / "scores.jsonl")
    assert len(rows) == 2
    for row, pid in zip(rows, ("p01", "p02")):
        assert row["prompt_id"] == pid
        assert row["prompt"] == next(r["prompt"] for r in PROMPTS if r["id"] == pid)
        assert row["video"] == str(videos / pid / "cand_1.mp4")
        assert row["reward"] == "videoreward"
        assert row["overall"] == 0.75
        assert row["dims"] == {"flicker": 0.5}
        assert "error" not in row
    summary = json.loads((videos / "scores.summary.json").read_text())
    assert summary == {
        "reward": "videoreward", "scored": 2, "failed": 0,
        "mean_overall": 0.75, "dims": {"flicker": 0.5},
    }
    assert "2 clips matched: 0 already scored, 2 to score" in capsys.readouterr().out


def test_summary_aggregates_only_numeric_dims(monkeypatch, tmp_path):
    va = str(tmp_path / "videos" / "p01" / "cand_1.mp4")
    vb = str(tmp_path / "videos" / "p02" / "cand_1.mp4")
    score_map = {
        va: (0.5, {"flicker": 0.25, "motion": 0.5}),
        vb: (1.0, {"flicker": 0.75}),
    }
    code, _, videos = _run(
        monkeypatch, tmp_path, [("p01", 1), ("p02", 1)], score_map=score_map
    )
    assert code == 0
    summary = json.loads((videos / "scores.summary.json").read_text())
    assert summary["scored"] == 2 and summary["failed"] == 0
    assert summary["mean_overall"] == 0.75
    assert summary["dims"] == {"flicker": 0.5, "motion": 0.5}


def test_out_override_writes_beside_custom_path(monkeypatch, tmp_path):
    out = tmp_path / "custom_scores.jsonl"
    code, _, videos = _run(
        monkeypatch, tmp_path, [("p01", 1)], extra_args=["--out", str(out)]
    )
    assert code == 0
    assert len(_rows(out)) == 1
    assert json.loads((tmp_path / "custom_scores.summary.json").read_text())["scored"] == 1
    assert not (videos / "scores.jsonl").exists()


# ------------------------------------------------------------------- resume


def test_resume_keeps_success_rows_and_retries_error_rows(
    monkeypatch, tmp_path, capsys
):
    videos = tmp_path / "videos"
    videos.mkdir()
    _touch_clip(videos, "p01")
    _touch_clip(videos, "p02")
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    out = videos / "scores.jsonl"
    _write_scores(out, [
        _success_row(videos, "p01", overall=0.125),
        {"prompt_id": "p02", "prompt": PROMPTS[1]["prompt"],
         "video": str(videos / "p02" / "cand_1.mp4"),
         "reward": "videoreward", "error": "RuntimeError: old"},
    ])
    calls = _fake_reward_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "score_eval_set.py", "--videos", str(videos), "--prompts", str(prompts),
    ])
    assert se.main() == 0
    # only the error row's clip is re-scored
    assert calls["score"] == [
        (str(videos / "p02" / "cand_1.mp4"), "busy street timelapse")
    ]
    rows = _rows(out)
    assert len(rows) == 2
    assert rows[0]["prompt_id"] == "p01" and rows[0]["overall"] == 0.125  # kept verbatim
    assert rows[1]["prompt_id"] == "p02" and rows[1]["overall"] == 0.75
    assert all("error" not in r for r in rows)  # error row compacted away
    assert "2 clips matched: 1 already scored, 1 to score" in capsys.readouterr().out


def test_resume_fully_cached_never_constructs_scorer(monkeypatch, tmp_path, capsys):
    videos = tmp_path / "videos"
    videos.mkdir()
    _touch_clip(videos, "p01")
    _touch_clip(videos, "p02")
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    out = videos / "scores.jsonl"
    _write_scores(out, [_success_row(videos, "p01"), _success_row(videos, "p02")])
    before = out.read_bytes()
    calls = _fake_reward_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "score_eval_set.py", "--videos", str(videos), "--prompts", str(prompts),
    ])
    assert se.main() == 0
    assert calls["constructed"] == [] and calls["load"] == 0 and calls["score"] == []
    assert out.read_bytes() == before  # compaction rewrite is byte-identical
    summary = json.loads((videos / "scores.summary.json").read_text())
    assert summary["scored"] == 2 and summary["failed"] == 0
    assert "2 clips matched: 2 already scored, 0 to score" in capsys.readouterr().out


def test_resume_drops_rows_from_other_reward_models(monkeypatch, tmp_path):
    videos = tmp_path / "videos"
    videos.mkdir()
    _touch_clip(videos, "p01")
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    out = videos / "scores.jsonl"
    _write_scores(out, [_success_row(videos, "p01", overall=0.125, reward="other-x")])
    calls = _fake_reward_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "score_eval_set.py", "--videos", str(videos), "--prompts", str(prompts),
    ])
    assert se.main() == 0
    assert len(calls["score"]) == 1  # other-reward row not reusable
    rows = _rows(out)
    assert len(rows) == 1 and rows[0]["reward"] == "videoreward"
    assert rows[0]["overall"] == 0.75  # old 0.125 row compacted away, not kept


def test_no_resume_truncates_instead_of_appending(monkeypatch, tmp_path):
    """--no-resume must not leave duplicate rows behind (resume-contract bug:
    the old file used to be appended to, growing duplicates on every re-run)."""
    videos = tmp_path / "videos"
    videos.mkdir()
    _touch_clip(videos, "p01")
    _touch_clip(videos, "p02")
    prompts = tmp_path / "prompts.jsonl"
    _write_prompts(prompts)
    out = videos / "scores.jsonl"
    _write_scores(out, [
        _success_row(videos, "p01", overall=0.125),
        {"prompt_id": "p02", "prompt": PROMPTS[1]["prompt"],
         "video": str(videos / "p02" / "cand_1.mp4"),
         "reward": "videoreward", "error": "RuntimeError: old"},
    ])
    calls = _fake_reward_module(monkeypatch)
    monkeypatch.setattr(sys, "argv", [
        "score_eval_set.py", "--videos", str(videos), "--prompts", str(prompts),
        "--no-resume",
    ])
    assert se.main() == 0
    assert len(calls["score"]) == 2  # everything re-scored
    rows = _rows(out)
    assert len(rows) == 2  # exactly one row per clip — no duplicates
    assert all(r["overall"] == 0.75 and "error" not in r for r in rows)


def test_no_resume_rerun_is_byte_identical(monkeypatch, tmp_path):
    code, _, videos = _run(
        monkeypatch, tmp_path, [("p01", 1), ("p02", 1)], extra_args=["--no-resume"]
    )
    assert code == 0
    first = (videos / "scores.jsonl").read_bytes()
    first_summary = (videos / "scores.summary.json").read_bytes()
    code, _, _ = _run(
        monkeypatch, tmp_path, [("p01", 1), ("p02", 1)], extra_args=["--no-resume"]
    )
    assert code == 0
    assert (videos / "scores.jsonl").read_bytes() == first
    assert (videos / "scores.summary.json").read_bytes() == first_summary


# ------------------------------------------------------- failure / strict


def test_scoring_failure_keeps_exit_zero_and_writes_error_row(
    monkeypatch, tmp_path, capsys
):
    vb = str(tmp_path / "videos" / "p02" / "cand_1.mp4")
    code, calls, videos = _run(
        monkeypatch, tmp_path, [("p01", 1), ("p02", 1)], fail={vb}
    )
    assert code == 0  # documented contract: rerun retries the failed row
    rows = _rows(videos / "scores.jsonl")
    assert len(rows) == 2
    assert rows[0]["prompt_id"] == "p01" and rows[0]["overall"] == 0.75
    assert rows[1]["prompt_id"] == "p02"
    assert rows[1]["error"].startswith("RuntimeError: boom")
    assert "overall" not in rows[1]
    summary = json.loads((videos / "scores.summary.json").read_text())
    assert summary["scored"] == 1 and summary["failed"] == 1
    assert "rerun to retry" in capsys.readouterr().err


def test_strict_reraises_before_writing_error_row(monkeypatch, tmp_path):
    va = str(tmp_path / "videos" / "p01" / "cand_1.mp4")
    with pytest.raises(RuntimeError, match="boom"):
        _run(monkeypatch, tmp_path, [("p01", 1), ("p02", 1)],
             extra_args=["--strict"], fail={va})
    out = tmp_path / "videos" / "scores.jsonl"
    assert _rows(out) == []  # first clip failed: nothing written, no error row
    assert not (tmp_path / "videos" / "scores.summary.json").exists()


# ---------------------------------------------------------------- truncation


def test_smoke_scores_only_two_clips(monkeypatch, tmp_path):
    code, calls, videos = _run(
        monkeypatch, tmp_path, [("p01", 1), ("p02", 1), ("p03", 1)],
        extra_args=["--smoke"],
    )
    assert code == 0
    assert len(calls["score"]) == 2
    assert len(_rows(videos / "scores.jsonl")) == 2


def test_limit_truncates(monkeypatch, tmp_path):
    code, calls, videos = _run(
        monkeypatch, tmp_path, [("p01", 1), ("p02", 1), ("p03", 1)],
        extra_args=["--limit", "1"],
    )
    assert code == 0
    assert len(calls["score"]) == 1
    assert len(_rows(videos / "scores.jsonl")) == 1


def test_limit_zero_is_clean_exit_1_before_scorer(monkeypatch, tmp_path, capsys):
    code, calls, videos = _run(
        monkeypatch, tmp_path, [("p01", 1)], extra_args=["--limit", "0"]
    )
    assert code == 1
    assert calls["constructed"] == []  # scorer never touched on an empty plan
    assert "no MP4s matched to eval prompts" in capsys.readouterr().err
    assert not (videos / "scores.jsonl").exists()

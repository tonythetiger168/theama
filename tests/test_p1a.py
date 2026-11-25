# SPDX-License-Identifier: Apache-2.0
"""CPU tests for P1a: reward registry, pair building, DPO math."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.latent_cache import (
    FAILURES_NAME,
    clear_failure,
    is_valid_latent,
    latent_pair_paths,
    load_failure_ledger,
    load_reusable_rows,
    pair_failure_key,
    plan_encoding,
    read_cache_meta,
    record_failure,
    reset_failure_ledger,
    run_encoding,
    save_latent_atomic,
    summarize_encoding,
    write_cache_meta,
)
from theama.reward.config import ALLOWED_LICENSES, REWARD_MODELS, get_reward_spec
from theama.reward.pairs import build_pairs, coverage_report
from theama.reward.score import load_successful_scores, summarize_scores
from theama.train.dpo import dpo_loss, flow_match_pair, velocity_mse


# ---- reward registry -------------------------------------------------------

def test_reward_registry_license_whitelist():
    assert REWARD_MODELS, "registry must not be empty"
    for name, spec in REWARD_MODELS.items():
        assert spec.license in ALLOWED_LICENSES, f"{name}: {spec.license}"
        assert get_reward_spec(name) is spec


def test_reward_registry_unknown_rejected():
    with pytest.raises(KeyError):
        get_reward_spec("minimax-h3-reward")


def test_videoreward_entry():
    spec = get_reward_spec("videoreward")
    assert spec.repo_id == "KlingTeam/VideoReward"
    assert spec.license == "Apache-2.0"
    assert spec.multi_dim


# ---- preference pair building ----------------------------------------------

def _rows(scores_by_prompt):
    rows = []
    for pid, entries in scores_by_prompt.items():
        for i, s in enumerate(entries):
            rows.append({
                "prompt_id": pid,
                "prompt": f"prompt {pid}",
                "video": f"/v/{pid}/cand_{i}.mp4",
                "overall": s,
            })
    return rows


def test_build_pairs_best_vs_worst():
    rows = _rows({"T2V-001": [0.9, 0.5, 0.7, 0.2]})
    pairs = build_pairs(rows, margin=0.5)
    assert len(pairs) == 1
    p = pairs[0]
    assert p.win_video.endswith("cand_0.mp4")   # 0.9
    assert p.lose_video.endswith("cand_3.mp4")  # 0.2
    assert p.margin == pytest.approx(0.7)


def test_build_pairs_margin_filter():
    rows = _rows({"T2V-001": [0.55, 0.50, 0.52]})
    assert build_pairs(rows, margin=0.5) == []


def test_build_pairs_single_candidate_skipped():
    rows = _rows({"T2V-001": [0.9], "T2V-002": [0.1, 0.9]})
    pairs = build_pairs(rows, margin=0.5)
    assert [p.prompt_id for p in pairs] == ["T2V-002"]


def test_build_pairs_deterministic_tie_break():
    rows = _rows({"T2V-001": [0.9, 0.9, 0.1]})
    p1 = build_pairs(rows, margin=0.5)
    p2 = build_pairs(rows, margin=0.5)
    assert p1 == p2
    assert p1[0].win_video.endswith("cand_0.mp4")  # path-sorted tie-break


def test_coverage_report():
    rows = _rows({"A": [0.9, 0.1], "B": [0.5, 0.4], "C": [0.8, 0.2]})
    pairs = build_pairs(rows, margin=0.5)
    rep = coverage_report(rows, pairs)
    assert rep["prompts_total"] == 3
    assert rep["prompts_paired"] == 2
    assert rep["pair_rate"] == pytest.approx(2 / 3)


# ---- DPO math (pure torch, CPU) --------------------------------------------

def test_flow_match_endpoints():
    x = torch.randn(2, 4, 3, 8, 8)
    noise = torch.randn_like(x)
    x_t0, v0 = flow_match_pair(x, noise, 0.0)
    x_t1, _ = flow_match_pair(x, noise, 1.0)
    assert torch.allclose(x_t0, x)
    assert torch.allclose(x_t1, noise)
    assert torch.allclose(v0, noise - x)


def test_dpo_loss_prefers_policy_beating_ref_on_wins():
    zero = torch.tensor(0.0)
    one = torch.tensor(1.0)
    # beta kept small so logits stay unsaturated and the loss is measurable
    good = dpo_loss(zero, one, one, zero, beta=2)   # policy wins the pair
    bad = dpo_loss(one, zero, zero, one, beta=2)    # policy loses the pair
    assert good < bad
    assert float(good) > 0.0  # logsigmoid loss is always positive


def test_dpo_loss_zero_when_policy_equals_ref():
    e = torch.tensor(0.3)
    loss = dpo_loss(e, e.clone(), e.clone(), e, beta=500)
    assert float(loss) == pytest.approx(0.693147, rel=1e-4)  # log 2


def test_dpo_loss_gradient_direction():
    model_err_w = torch.tensor(0.5, requires_grad=True)
    model_err_l = torch.tensor(0.5, requires_grad=True)
    ref = torch.tensor(0.5)
    loss = dpo_loss(model_err_w, ref, model_err_l, ref, beta=500)
    loss.backward()
    # gradient DESCENT moves errors opposite to the grad sign:
    assert model_err_w.grad > 0  # descent shrinks policy error on the win
    assert model_err_l.grad < 0  # descent grows policy error on the lose


def test_velocity_mse_shape():
    pred = torch.randn(3, 2, 5)
    target = torch.randn(3, 2, 5)
    err = velocity_mse(pred, target)
    assert err.shape == (3,)


# ---- scores.jsonl resume / summary helpers ----------------------------------

def _write_scores(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_load_successful_scores_missing_file(tmp_path):
    assert load_successful_scores(tmp_path / "nope.jsonl") == {}


def test_load_successful_scores_filters_errors_and_reward(tmp_path):
    f = tmp_path / "scores.jsonl"
    _write_scores(f, [
        {"prompt_id": "A", "video": "/v/a.mp4", "reward": "videoreward", "overall": 0.9},
        {"prompt_id": "B", "video": "/v/b.mp4", "reward": "videoreward",
         "error": "RuntimeError: boom"},
        {"prompt_id": "C", "video": "/v/c.mp4", "reward": "other-model", "overall": 0.5},
        {"prompt_id": "D", "video": "/v/d.mp4", "overall": 0.7},  # legacy: no reward key
    ])
    kept = load_successful_scores(f, reward="videoreward")
    assert sorted(kept) == ["/v/a.mp4", "/v/d.mp4"]  # errors + other-reward dropped
    assert load_successful_scores(f, reward="other-model") == {
        "/v/c.mp4": {"prompt_id": "C", "video": "/v/c.mp4",
                     "reward": "other-model", "overall": 0.5}
    }


def test_load_successful_scores_dedup_latest_wins(tmp_path):
    f = tmp_path / "scores.jsonl"
    _write_scores(f, [
        {"video": "/v/a.mp4", "overall": 0.1},
        {"video": "/v/a.mp4", "overall": 0.9},
    ])
    kept = load_successful_scores(f)
    assert kept["/v/a.mp4"]["overall"] == pytest.approx(0.9)


def test_load_successful_scores_skips_malformed_lines(tmp_path):
    f = tmp_path / "scores.jsonl"
    f.write_text(
        '{"video": "/v/a.mp4", "overall": 0.5}\n'
        "not json at all\n"
        "\n"
        '{"video": "/v/b.mp4"}\n'  # no overall -> treated as failed row
    )
    assert sorted(load_successful_scores(f)) == ["/v/a.mp4"]


def test_summarize_scores_mixed_rows():
    rows = [
        {"overall": 0.8, "dims": {"visual_quality": 0.9, "text_alignment": 0.7}},
        {"overall": 0.4, "dims": {"visual_quality": 0.5}},
        {"video": "/v/x.mp4", "error": "RuntimeError: boom"},
    ]
    s = summarize_scores(rows)
    assert s["scored"] == 2
    assert s["failed"] == 1
    assert s["mean_overall"] == pytest.approx(0.6)
    assert s["dims"]["visual_quality"] == pytest.approx(0.7)   # mean over rows having it
    assert s["dims"]["text_alignment"] == pytest.approx(0.7)


def test_summarize_scores_empty():
    assert summarize_scores([]) == {
        "scored": 0, "failed": 0, "mean_overall": None, "dims": {}
    }


# ---- encode_pairs latent-cache helpers --------------------------------------


def _latent(path: Path, fill: float = 0.0) -> None:
    save_latent_atomic(torch.full((1, 4, 2, 3, 3), fill), path)


def _pairs(n: int) -> list[dict]:
    return [
        {
            "prompt_id": f"T2V-{i:03d}",
            "prompt": f"prompt {i}",
            "win_video": f"/v/{i}/win.mp4",
            "lose_video": f"/v/{i}/lose.mp4",
        }
        for i in range(n)
    ]


def test_latent_pair_paths_naming(tmp_path):
    win, lose = latent_pair_paths(tmp_path, 7)
    assert win.name == "pair0007_win.pt"
    assert lose.name == "pair0007_lose.pt"


def test_is_valid_latent(tmp_path):
    good = tmp_path / "good.pt"
    _latent(good)
    assert is_valid_latent(good)
    assert not is_valid_latent(tmp_path / "missing.pt")

    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"half-written tensor")  # truncated-run artifact
    assert not is_valid_latent(corrupt)

    wrong_ndim = tmp_path / "2d.pt"
    torch.save(torch.randn(3, 3), wrong_ndim)
    assert not is_valid_latent(wrong_ndim)

    nonfinite = tmp_path / "nan.pt"
    torch.save(torch.full((1, 4, 2, 3, 3), float("nan")), nonfinite)
    assert not is_valid_latent(nonfinite)


def test_save_latent_atomic_roundtrip_and_overwrite(tmp_path):
    dest = tmp_path / "x.pt"
    t1 = torch.randn(1, 4, 2, 3, 3)
    save_latent_atomic(t1, dest)
    assert torch.equal(torch.load(dest, weights_only=True), t1)
    assert not (tmp_path / "x.pt.tmp").exists()  # temp file cleaned up

    t2 = torch.randn(1, 4, 2, 3, 3)
    save_latent_atomic(t2, dest)
    assert torch.equal(torch.load(dest, weights_only=True), t2)


def test_load_reusable_rows_filters(tmp_path):
    f = tmp_path / "pairs_latents.jsonl"
    rows = [
        {"index": 0, "prompt_id": "A", "model": "ti2v-5b",
         "win_latent": "w", "lose_latent": "l"},
        {"index": 1, "prompt_id": "B", "model": "ti2v-5b",
         "error": "RuntimeError: boom"},                          # retried
        {"index": 2, "prompt_id": "C", "model": "other-model",
         "win_latent": "w", "lose_latent": "l"},                  # wrong model
        {"index": "3", "prompt_id": "D", "model": "ti2v-5b"},     # non-int idx
        {"prompt_id": "E", "model": "ti2v-5b"},                   # legacy, no idx
        {"index": 0, "prompt_id": "A", "model": "ti2v-5b",
         "win_latent": "w2", "lose_latent": "l2"},                # dup: later wins
    ]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows) + "not json\n")
    kept = load_reusable_rows(f, model="ti2v-5b")
    assert sorted(kept) == [0]
    assert kept[0]["win_latent"] == "w2"
    assert load_reusable_rows(tmp_path / "nope.jsonl", model="ti2v-5b") == {}


def test_summarize_encoding():
    rows = [
        {"index": 0, "reused": False},
        {"index": 1, "reused": True},
        {"index": 2, "error": "boom"},
    ]
    assert summarize_encoding(rows) == {
        "total": 3, "encoded": 1, "reused": 1, "failed": 1
    }
    assert summarize_encoding([]) == {
        "total": 0, "encoded": 0, "reused": 0, "failed": 0
    }


def test_plan_encoding_manifest_resume_and_guards(tmp_path):
    pairs = _pairs(3)
    win0, lose0 = latent_pair_paths(tmp_path, 0)
    _latent(win0)
    _latent(lose0)
    manifest = tmp_path / "pairs_latents.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in [
        {"index": 0, "prompt_id": "T2V-000", "prompt": "prompt 0",
         "model": "ti2v-5b", "win_latent": str(win0), "lose_latent": str(lose0)},
        # prompt_id drifted at index 1 -> must re-encode even with a row
        {"index": 1, "prompt_id": "STALE", "prompt": "old", "model": "ti2v-5b",
         "win_latent": "w", "lose_latent": "l"},
        # error row at index 2, no valid files on disk -> retried
        {"index": 2, "prompt_id": "T2V-002", "model": "ti2v-5b",
         "error": "RuntimeError: boom"},
    ]))
    plan = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert sorted(plan.reusable) == [0]
    assert plan.reusable[0]["reused"] is True
    assert plan.todo == [1, 2]
    assert not plan.cache_invalidated


def test_plan_encoding_legacy_file_fallback(tmp_path):
    pairs = _pairs(3)
    win0, lose0 = latent_pair_paths(tmp_path, 0)
    _latent(win0)
    _latent(lose0)                      # pair 0: both sides -> reused
    win1, _ = latent_pair_paths(tmp_path, 1)
    _latent(win1)                       # pair 1: one side only -> re-encode
    corrupt = tmp_path / "pair0002_win.pt"
    corrupt.write_bytes(b"junk")        # pair 2: corrupt -> re-encode
    plan = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert sorted(plan.reusable) == [0]
    assert plan.reusable[0]["prompt_id"] == "T2V-000"  # row synthesized
    assert plan.todo == [1, 2]


def test_plan_encoding_drifted_row_blocks_legacy_fallback(tmp_path):
    """A manifest row rejected by the prompt_id drift guard marks the
    on-disk .pt files stale: they must NOT be re-adopted via the legacy
    file fallback (same hole the AV cache closed in the round-9 build)."""
    pairs = _pairs(2)
    win0, lose0 = latent_pair_paths(tmp_path, 0)
    _latent(win0)
    _latent(lose0)  # valid files, but the manifest row below has drifted
    manifest = tmp_path / "pairs_latents.jsonl"
    manifest.write_text(json.dumps({
        "index": 0, "prompt_id": "STALE", "prompt": "old",
        "model": "ti2v-5b", "win_latent": str(win0), "lose_latent": str(lose0),
    }) + "\n")
    plan = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert plan.reusable == {}          # fallback must not rescue index 0
    assert plan.todo == [0, 1]

    # After re-encoding under the current prompt_id, resume reuses normally.
    def encode(video_path, dest):
        _latent(Path(dest))

    summary = run_encoding(plan, encode)
    assert summary == {"model": "ti2v-5b", "total": 2,
                       "encoded": 2, "reused": 0, "failed": 0}
    plan2 = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert sorted(plan2.reusable) == [0, 1]
    assert plan2.todo == []


def test_plan_encoding_meta_model_mismatch_invalidates(tmp_path):
    pairs = _pairs(1)
    win, lose = latent_pair_paths(tmp_path, 0)
    _latent(win)
    _latent(lose)
    write_cache_meta(tmp_path, "other-model")
    plan = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert plan.cache_invalidated
    assert plan.todo == [0]
    assert read_cache_meta(tmp_path)["model"] == "other-model"


def test_run_encoding_three_round_resume(tmp_path):
    pairs = _pairs(3)
    calls: list[str] = []

    def make_encoder(fail_on: str | None):
        def encode(video_path, dest):
            calls.append(video_path)
            if video_path == fail_on:
                raise RuntimeError("simulated GPU failure")
            _latent(Path(dest))
        return encode

    # round 1: pair 1 fails -> error row, pairs 0/2 encoded
    plan1 = plan_encoding(pairs, tmp_path, "ti2v-5b")
    s1 = run_encoding(plan1, make_encoder("/v/1/win.mp4"))
    assert s1 == {"model": "ti2v-5b", "total": 3,
                  "encoded": 2, "reused": 0, "failed": 1}
    assert len(calls) == 5  # pair0 win+lose, pair1 win (fails), pair2 win+lose
    manifest_rows = [
        json.loads(l) for l in plan1.manifest_path.read_text().splitlines()
    ]
    assert sum("error" in r for r in manifest_rows) == 1

    # round 2: only the failed pair re-runs; cached pairs cost zero calls
    calls.clear()
    plan2 = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert plan2.todo == [1]
    s2 = run_encoding(plan2, make_encoder(None))
    assert s2 == {"model": "ti2v-5b", "total": 3,
                  "encoded": 1, "reused": 2, "failed": 0}
    assert calls == ["/v/1/win.mp4", "/v/1/lose.mp4"]

    # round 3: everything cached -> encoder never needed (encode_fn=None ok)
    plan3 = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert plan3.todo == []
    s3 = run_encoding(plan3, None)
    assert s3 == {"model": "ti2v-5b", "total": 3,
                  "encoded": 0, "reused": 3, "failed": 0}
    assert json.loads(plan3.summary_path.read_text()) == s3
    assert read_cache_meta(tmp_path)["model"] == "ti2v-5b"


def test_run_encoding_strict_raises(tmp_path):
    plan = plan_encoding(_pairs(2), tmp_path, "ti2v-5b")

    def bad_encoder(video_path, dest):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        run_encoding(plan, bad_encoder, strict=True)


# ---- latent-cache retry budget (failure ledger) ------------------------------
#
# Same opt-in design as the text cache (test_m2_text_cache): a pair_failures.json
# ledger counts per-pair failures, budget-exhausted pairs are planned as skipped,
# and the default (no max_retries) keeps byte-identical pre-ledger behaviour.


def _pair_encoder(calls: list, fail_on: tuple = ()):
    """Fake encode_fn(video_path, dest): records calls, optionally fails."""

    def encode(video_path, dest):
        calls.append(video_path)
        if video_path in fail_on:
            raise RuntimeError("boom")
        _latent(Path(dest))

    return encode


def _quiet(msg):  # keep pytest output clean
    pass


def test_pair_failure_ledger_roundtrip(tmp_path):
    assert load_failure_ledger(tmp_path) == {}
    pair = _pairs(1)[0]
    assert pair_failure_key(3) == "pair0003"
    assert record_failure(tmp_path, 3, pair, "RuntimeError: boom") == 1
    assert record_failure(tmp_path, 3, pair, "RuntimeError: boom again") == 2
    ledger = load_failure_ledger(tmp_path)
    entry = ledger["pair0003"]
    assert entry["count"] == 2
    assert entry["prompt_id"] == "T2V-000"
    assert "boom again" in entry["error"]
    # A different index lands on a different key.
    record_failure(tmp_path, 7, pair, "x")
    assert len(load_failure_ledger(tmp_path)) == 2
    # Success clears only that pair.
    clear_failure(tmp_path, 3)
    ledger = load_failure_ledger(tmp_path)
    assert "pair0003" not in ledger
    assert len(ledger) == 1
    # Reset deletes the file entirely.
    assert reset_failure_ledger(tmp_path)
    assert not (tmp_path / FAILURES_NAME).exists()
    assert not reset_failure_ledger(tmp_path)
    # Corrupt ledger degrades to {} instead of crashing.
    (tmp_path / FAILURES_NAME).write_text("{not json")
    assert load_failure_ledger(tmp_path) == {}


def test_plan_skips_pair_past_retry_budget(tmp_path):
    pairs = _pairs(2)
    fail = ("/v/1/win.mp4",)  # pair 1 dies on the win side every run
    calls1: list = []
    s1 = run_encoding(
        plan_encoding(pairs, tmp_path, "ti2v-5b", max_retries=2),
        _pair_encoder(calls1, fail), progress=_quiet, max_retries=2,
    )
    assert s1["failed"] == 1 and s1["encoded"] == 1  # pair 0 cached from here on
    # Second run retries only pair 1 and fills its budget (count reaches 2).
    calls2: list = []
    s2 = run_encoding(
        plan_encoding(pairs, tmp_path, "ti2v-5b", max_retries=2),
        _pair_encoder(calls2, fail), progress=_quiet, max_retries=2,
    )
    assert calls2 == ["/v/1/win.mp4"]
    assert s2["failed"] == 1 and s2["reused"] == 1
    # Third run: pair 1 is skipped by the planner, never encoded; pair 0 reused.
    plan3 = plan_encoding(pairs, tmp_path, "ti2v-5b", max_retries=2)
    assert plan3.todo == []
    assert set(plan3.reusable) == {0}
    assert set(plan3.skipped) == {1}
    skipped_row = plan3.skipped[1]
    assert skipped_row["skipped"] is True
    assert skipped_row["retries"] == 2
    assert skipped_row["prompt_id"] == "T2V-001"
    assert "boom" in skipped_row["last_error"]
    calls3: list = []
    s3 = run_encoding(plan3, _pair_encoder(calls3), progress=_quiet, max_retries=2)
    assert calls3 == []
    # Skipped rows are not counted as failed, reused, or encoded.
    assert s3["failed"] == 0 and s3["encoded"] == 0
    assert s3["reused"] == 1
    assert s3["skipped"] == 1
    rows = [json.loads(l) for l in plan3.manifest_path.read_text().splitlines()]
    assert any(r.get("skipped") for r in rows)
    assert not any("error" in r for r in rows)
    # The summary sidecar carries the skipped count too.
    assert json.loads(plan3.summary_path.read_text())["skipped"] == 1


def test_pair_success_clears_ledger_entry(tmp_path):
    pairs = _pairs(1)
    run_encoding(
        plan_encoding(pairs, tmp_path, "ti2v-5b", max_retries=3),
        _pair_encoder([], fail_on=("/v/0/win.mp4",)), progress=_quiet, max_retries=3,
    )
    assert load_failure_ledger(tmp_path) != {}
    # The pair now encodes — success wipes its failure history.
    summary = run_encoding(
        plan_encoding(pairs, tmp_path, "ti2v-5b", max_retries=3),
        _pair_encoder([]), progress=_quiet, max_retries=3,
    )
    assert summary["encoded"] == 1 and summary["failed"] == 0
    assert load_failure_ledger(tmp_path) == {}


def test_pair_no_ledger_io_when_max_retries_unset(tmp_path):
    pairs = _pairs(2)
    summary = run_encoding(
        plan_encoding(pairs, tmp_path, "ti2v-5b"),
        _pair_encoder([], fail_on=("/v/1/win.mp4",)),
        progress=_quiet,
    )
    assert summary["failed"] == 1
    assert "skipped" not in summary  # default behaviour is byte-identical
    assert not (tmp_path / FAILURES_NAME).exists()
    # Legacy retry loop still works: the failure row lands back in todo.
    plan2 = plan_encoding(pairs, tmp_path, "ti2v-5b")
    assert plan2.todo == [1] and plan2.skipped == {}


def test_pair_strict_records_failure_before_reraising(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        run_encoding(
            plan_encoding(_pairs(1), tmp_path, "ti2v-5b", max_retries=2),
            _pair_encoder([], fail_on=("/v/0/win.mp4",)),
            strict=True,
            progress=_quiet,
            max_retries=2,
        )
    # The abort did not lose the failure: the ledger saw the attempt.
    ledger = load_failure_ledger(tmp_path)
    assert ledger["pair0000"]["count"] == 1


def test_pair_invalid_max_retries_rejected(tmp_path):
    with pytest.raises(ValueError):
        plan_encoding(_pairs(1), tmp_path, "ti2v-5b", max_retries=0)
    with pytest.raises(ValueError):
        run_encoding(
            plan_encoding(_pairs(1), tmp_path, "ti2v-5b"),
            _pair_encoder([]),
            progress=_quiet,
            max_retries=True,
        )


# ---- config + scripts -------------------------------------------------------

def test_dpo_yaml():
    cfg = yaml.safe_load((ROOT / "configs" / "dpo_ti2v_5b.yaml").read_text())["dpo"]
    assert cfg["base_model"] == "ti2v-5b"
    assert cfg["reward_model"] in REWARD_MODELS
    assert cfg["train"]["beta"] > 0
    assert cfg["lora"]["rank"] >= 8
    gates = cfg["gates"]
    assert 0.5 < gates["blind_win_rate_vs_base"] <= 1.0
    assert gates["vbench_tolerance"] > 0


@pytest.mark.parametrize("script", [
    "gen_candidates.py", "score_eval_set.py", "make_pref_pairs.py",
    "encode_pairs.py", "train_dpo.py",
])
def test_p1a_scripts_compile(script):
    subprocess.run(
        [sys.executable, "-m", "py_compile", str(ROOT / "scripts" / script)],
        check=True,
    )

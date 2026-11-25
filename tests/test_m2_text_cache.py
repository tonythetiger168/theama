# SPDX-License-Identifier: Apache-2.0
"""CPU tests for theama.data.text_cache (P1b content-addressed text cache).

Encoders are deterministic stand-ins that save small tensors — the cache
logic under test is pure planning/manifest machinery, no tokenizer and no
GPU. Covers: content-addressed naming (caption edit => new path, so drift
cannot reuse stale embeddings), fresh/resume planning, seq_len recovery
from the manifest, error-row retry, meta.json model invalidation,
same-run caption dedup, corrupt/NaN/wrong-ndim re-encoding, strict mode,
the empty-caption guard, and the opt-in retry budget (failure ledger:
record/clear/skip semantics, exhausted-budget rows, strict-mode recording).
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.latent_cache import (  # noqa: E402
    META_NAME,
    save_latent_atomic,
)
from theama.data.text_cache import (  # noqa: E402
    FAILURES_NAME,
    MANIFEST_NAME,
    SUMMARY_NAME,
    clear_failure,
    is_valid_text_latent,
    load_failure_ledger,
    plan_text_encoding,
    record_failure,
    reset_failure_ledger,
    run_text_encoding,
    text_latent_path,
)

EMB_SHAPE = (8, 16)  # (L, D) small stand-in for (512, 4096)


def _samples(*captions: str) -> list[dict]:
    return [{"caption": c} for c in captions]


def _encoder(calls: list, fail_on: tuple = (), shape=EMB_SHAPE, seq_len: int = 7):
    """Fake encode_fn: records calls, optionally fails, saves real tensors."""

    def encode(sample, dest):
        calls.append(sample["caption"])
        if sample["caption"] in fail_on:
            raise RuntimeError("boom")
        save_latent_atomic(torch.zeros(*shape), dest)
        return seq_len

    return encode


def _quiet(msg):  # keep pytest output clean
    pass


def _read_manifest(out_dir: Path) -> list[dict]:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_text_latent_path_is_content_addressed(tmp_path):
    a = text_latent_path(tmp_path, "一隻貓在屋頂上")
    again = text_latent_path(tmp_path, "一隻貓在屋頂上")
    b = text_latent_path(tmp_path, "一隻貓在屋頂上。")  # one char different
    assert a == again
    assert a != b  # caption drift lands on a different file — structurally
    assert a.name.startswith("text_") and a.suffix == ".pt"
    with pytest.raises(ValueError):
        text_latent_path(tmp_path, "")
    with pytest.raises(ValueError):
        text_latent_path(tmp_path, "   ")
    with pytest.raises(ValueError):
        text_latent_path(tmp_path, None)


def test_is_valid_text_latent_ndim2(tmp_path):
    good = tmp_path / "good.pt"
    save_latent_atomic(torch.zeros(*EMB_SHAPE), good)
    assert is_valid_text_latent(good)
    bad5d = tmp_path / "bad5d.pt"
    save_latent_atomic(torch.zeros(1, 16, 2, 4, 4), bad5d)
    assert not is_valid_text_latent(bad5d)  # video latent must not pass
    nan = tmp_path / "nan.pt"
    save_latent_atomic(torch.full(EMB_SHAPE, float("nan")), nan)
    assert not is_valid_text_latent(nan)
    assert not is_valid_text_latent(tmp_path / "missing.pt")


def test_fresh_plan_all_todo(tmp_path):
    plan = plan_text_encoding(_samples("a", "b", "c"), tmp_path, "umt5-xxl")
    assert plan.todo == [0, 1, 2]
    assert plan.reusable == {}
    assert not plan.cache_invalidated


def test_first_run_writes_manifest_summary_meta(tmp_path):
    samples = _samples("hello world", "你好世界")
    plan = plan_text_encoding(samples, tmp_path, "umt5-xxl")
    calls: list = []
    summary = run_text_encoding(plan, _encoder(calls), progress=_quiet)
    assert summary == {"model": "umt5-xxl", "total": 2, "encoded": 2,
                       "reused": 0, "failed": 0}
    assert calls == ["hello world", "你好世界"]
    rows = _read_manifest(tmp_path)
    assert len(rows) == 2
    for row in rows:
        assert row["shape"] == list(EMB_SHAPE)
        assert row["seq_len"] == 7
        assert row["reused"] is False
        assert Path(row["text_latent"]).exists()
    meta = json.loads((tmp_path / META_NAME).read_text())
    assert meta["model"] == "umt5-xxl"
    sidecar = json.loads((tmp_path / SUMMARY_NAME).read_text())
    assert sidecar == summary


def test_resume_reuses_without_encoder(tmp_path):
    samples = _samples("a", "b")
    run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder([]),
        progress=_quiet,
    )
    plan2 = plan_text_encoding(samples, tmp_path, "umt5-xxl")
    assert plan2.todo == []
    assert set(plan2.reusable) == {0, 1}
    # seq_len survived the round-trip through the manifest.
    assert plan2.reusable[0]["seq_len"] == 7
    summary = run_text_encoding(plan2, None, progress=_quiet)  # no encoder needed
    assert summary["reused"] == 2 and summary["encoded"] == 0


def test_caption_edit_reencodes_only_that_sample(tmp_path):
    samples = _samples("a", "b")
    run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder([]),
        progress=_quiet,
    )
    old_path = text_latent_path(tmp_path, "a")
    assert old_path.exists()
    edited = _samples("a — edited", "b")
    plan = plan_text_encoding(edited, tmp_path, "umt5-xxl")
    assert plan.todo == [0]
    assert set(plan.reusable) == {1}
    calls: list = []
    run_text_encoding(plan, _encoder(calls), progress=_quiet)
    assert calls == ["a — edited"]
    assert text_latent_path(tmp_path, "a — edited").exists()
    # Old embedding is orphaned garbage, never served for the new caption.
    assert old_path.exists()
    assert plan.reusable[1]["text_latent"] != str(old_path)


def test_meta_model_mismatch_invalidates_cache(tmp_path):
    samples = _samples("a")
    run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder([]),
        progress=_quiet,
    )
    plan = plan_text_encoding(samples, tmp_path, "umt5-large")
    assert plan.cache_invalidated
    assert plan.todo == [0]
    # --no-resume also forces a full re-encode.
    plan_noresume = plan_text_encoding(samples, tmp_path, "umt5-xxl", resume=False)
    assert plan_noresume.todo == [0]


def test_error_row_retried_on_next_run(tmp_path):
    samples = _samples("a", "b")
    calls: list = []
    summary = run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder(calls, fail_on=("a",)),
        progress=_quiet,
    )
    assert summary["failed"] == 1 and summary["encoded"] == 1
    rows = _read_manifest(tmp_path)
    assert rows[0]["error"].startswith("RuntimeError")
    assert not text_latent_path(tmp_path, "a").exists()
    # Retry: only the failed caption is encoded.
    plan2 = plan_text_encoding(samples, tmp_path, "umt5-xxl")
    assert plan2.todo == [0]
    calls2: list = []
    summary2 = run_text_encoding(plan2, _encoder(calls2), progress=_quiet)
    assert calls2 == ["a"]
    assert summary2["failed"] == 0 and summary2["encoded"] == 1
    assert summary2["reused"] == 1


def test_duplicate_captions_encode_once_per_run(tmp_path):
    samples = _samples("same caption", "other", "same caption")
    calls: list = []
    summary = run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder(calls),
        progress=_quiet,
    )
    assert calls == ["same caption", "other"]  # third sample never encoded
    assert summary["encoded"] == 2 and summary["reused"] == 1
    rows = _read_manifest(tmp_path)
    assert rows[0]["text_latent"] == rows[2]["text_latent"]
    assert rows[2]["reused"] is True
    assert rows[2]["seq_len"] == 7  # shared rows keep seq_len


def test_nan_embedding_refused_not_manifested(tmp_path):
    def bad_encoder(sample, dest):
        save_latent_atomic(torch.full(EMB_SHAPE, float("nan")), dest)

    plan = plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl")
    summary = run_text_encoding(plan, bad_encoder, progress=_quiet)
    assert summary["failed"] == 1
    assert "refusing to cache" in _read_manifest(tmp_path)[0]["error"]
    # Poisoned file fails validation, so the next plan re-encodes it.
    plan2 = plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl")
    assert plan2.todo == [0]


def test_wrong_ndim_embedding_refused(tmp_path):
    def bad_encoder(sample, dest):
        save_latent_atomic(torch.zeros(1, 8, 16), dest)  # 3D, not (L, D)

    summary = run_text_encoding(
        plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl"),
        bad_encoder,
        progress=_quiet,
    )
    assert summary["failed"] == 1
    assert "refusing to cache" in _read_manifest(tmp_path)[0]["error"]


def test_corrupt_pt_reencoded(tmp_path):
    samples = _samples("a")
    run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder([]),
        progress=_quiet,
    )
    dest = text_latent_path(tmp_path, "a")
    dest.write_bytes(dest.read_bytes()[:8])  # truncated write simulation
    plan = plan_text_encoding(samples, tmp_path, "umt5-xxl")
    assert plan.todo == [0]


def test_strict_reraises(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        run_text_encoding(
            plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl"),
            _encoder([], fail_on=("a",)),
            strict=True,
            progress=_quiet,
        )


def test_bad_seq_len_type_is_a_failure_row(tmp_path):
    def weird_encoder(sample, dest):
        save_latent_atomic(torch.zeros(*EMB_SHAPE), dest)
        return "7"  # not an int

    summary = run_text_encoding(
        plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl"),
        weird_encoder,
        progress=_quiet,
    )
    assert summary["failed"] == 1
    assert "seq_len" in _read_manifest(tmp_path)[0]["error"]


# ---- retry budget (failure ledger) ---------------------------------------------


def test_failure_ledger_roundtrip(tmp_path):
    assert load_failure_ledger(tmp_path) == {}
    assert record_failure(tmp_path, "一隻貓", "RuntimeError: boom") == 1
    assert record_failure(tmp_path, "一隻貓", "RuntimeError: boom again") == 2
    ledger = load_failure_ledger(tmp_path)
    entry = ledger[text_latent_path(tmp_path, "一隻貓").stem]
    assert entry["count"] == 2
    assert entry["caption"] == "一隻貓"
    assert "boom again" in entry["error"]
    # A different caption lands on a different key.
    record_failure(tmp_path, "狗", "x")
    assert len(load_failure_ledger(tmp_path)) == 2
    # Success clears only that caption.
    clear_failure(tmp_path, "一隻貓")
    ledger = load_failure_ledger(tmp_path)
    assert text_latent_path(tmp_path, "一隻貓").stem not in ledger
    assert len(ledger) == 1
    # Reset deletes the file entirely.
    assert reset_failure_ledger(tmp_path)
    assert not (tmp_path / FAILURES_NAME).exists()
    assert not reset_failure_ledger(tmp_path)
    # Corrupt ledger degrades to {} instead of crashing.
    (tmp_path / FAILURES_NAME).write_text("{not json")
    assert load_failure_ledger(tmp_path) == {}


def test_plan_skips_caption_past_retry_budget(tmp_path):
    samples = _samples("good", "bad")
    calls1: list = []
    summary1 = run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl", max_retries=2),
        _encoder(calls1, fail_on=("bad",)), progress=_quiet, max_retries=2,
    )
    assert calls1 == ["good", "bad"]  # "good" is cached from here on
    assert summary1["failed"] == 1 and summary1["encoded"] == 1
    # Second run retries only "bad" and fills its budget (count reaches 2).
    calls2: list = []
    summary2 = run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl", max_retries=2),
        _encoder(calls2, fail_on=("bad",)), progress=_quiet, max_retries=2,
    )
    assert calls2 == ["bad"]
    assert summary2["failed"] == 1 and summary2["reused"] == 1
    # Third run: "bad" is skipped by the planner, never encoded; "good" reused.
    plan3 = plan_text_encoding(samples, tmp_path, "umt5-xxl", max_retries=2)
    assert plan3.todo == []
    assert set(plan3.reusable) == {0}
    assert set(plan3.skipped) == {1}
    skipped_row = plan3.skipped[1]
    assert skipped_row["skipped"] is True
    assert skipped_row["retries"] == 2
    assert "boom" in skipped_row["last_error"]
    calls3: list = []
    summary3 = run_text_encoding(
        plan3, _encoder(calls3), progress=_quiet, max_retries=2,
    )
    assert calls3 == []
    # Skipped rows are not counted as failed, reused, or encoded.
    assert summary3["failed"] == 0 and summary3["encoded"] == 0
    assert summary3["reused"] == 1
    assert summary3["skipped"] == 1
    rows = _read_manifest(tmp_path)
    assert any(r.get("skipped") for r in rows)
    assert not any("error" in r for r in rows)
    # The summary sidecar carries the skipped count too.
    sidecar = json.loads((tmp_path / SUMMARY_NAME).read_text())
    assert sidecar["skipped"] == 1


def test_success_clears_ledger_entry(tmp_path):
    samples = _samples("flaky")
    calls1: list = []
    run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl", max_retries=3),
        _encoder(calls1, fail_on=("flaky",)), progress=_quiet, max_retries=3,
    )
    assert load_failure_ledger(tmp_path) != {}
    # The caption now encodes — success wipes its failure history.
    summary = run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl", max_retries=3),
        _encoder([]), progress=_quiet, max_retries=3,
    )
    assert summary["encoded"] == 1 and summary["failed"] == 0
    assert load_failure_ledger(tmp_path) == {}


def test_no_ledger_io_when_max_retries_unset(tmp_path):
    samples = _samples("a", "b")
    summary = run_text_encoding(
        plan_text_encoding(samples, tmp_path, "umt5-xxl"),
        _encoder([], fail_on=("b",)),
        progress=_quiet,
    )
    assert summary["failed"] == 1
    assert "skipped" not in summary  # default behaviour is byte-identical
    assert not (tmp_path / FAILURES_NAME).exists()
    # Legacy retry loop still works: the failure row lands back in todo.
    plan2 = plan_text_encoding(samples, tmp_path, "umt5-xxl")
    assert plan2.todo == [1] and plan2.skipped == {}


def test_strict_records_failure_before_reraising(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        run_text_encoding(
            plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl", max_retries=2),
            _encoder([], fail_on=("a",)),
            strict=True,
            progress=_quiet,
            max_retries=2,
        )
    # The abort did not lose the failure: the ledger saw the attempt.
    ledger = load_failure_ledger(tmp_path)
    assert ledger[text_latent_path(tmp_path, "a").stem]["count"] == 1


def test_invalid_max_retries_rejected(tmp_path):
    with pytest.raises(ValueError):
        plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl", max_retries=0)
    with pytest.raises(ValueError):
        run_text_encoding(
            plan_text_encoding(_samples("a"), tmp_path, "umt5-xxl"),
            _encoder([]),
            progress=_quiet,
            max_retries=True,
        )

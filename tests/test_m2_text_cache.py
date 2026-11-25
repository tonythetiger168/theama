# SPDX-License-Identifier: Apache-2.0
"""CPU tests for theama.data.text_cache (P1b content-addressed text cache).

Encoders are deterministic stand-ins that save small tensors — the cache
logic under test is pure planning/manifest machinery, no tokenizer and no
GPU. Covers: content-addressed naming (caption edit => new path, so drift
cannot reuse stale embeddings), fresh/resume planning, seq_len recovery
from the manifest, error-row retry, meta.json model invalidation,
same-run caption dedup, corrupt/NaN/wrong-ndim re-encoding, strict mode,
and the empty-caption guard.
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
    MANIFEST_NAME,
    SUMMARY_NAME,
    is_valid_text_latent,
    plan_text_encoding,
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

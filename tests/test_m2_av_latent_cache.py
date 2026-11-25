# SPDX-License-Identifier: Apache-2.0
"""CPU tests for theama.data.av_latent_cache (P1b plan/resume AV cache).

Encoders are deterministic stand-ins that save small tensors — the cache
logic under test is pure planning/manifest machinery, no media decode and
no GPU. Covers: naming compatibility with the smoke layout, the ndim=3
relaxation of is_valid_latent, fresh/resume planning, error-row retry,
(video,audio) drift guard, meta.json model invalidation, legacy fallback
seeding, corrupt-.pt re-encoding, strict mode, and the post-encode
contract check (invalid latents are never manifested as success).
"""

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_cache import (  # noqa: E402
    FAILURES_NAME,
    MANIFEST_NAME,
    SUMMARY_NAME,
    av_failure_key,
    av_latent_paths,
    clear_failure,
    is_valid_audio_latent,
    is_valid_video_latent,
    load_failure_ledger,
    plan_av_encoding,
    record_failure,
    reset_failure_ledger,
    run_av_encoding,
)
from theama.data.latent_cache import (  # noqa: E402
    META_NAME,
    is_valid_latent,
    save_latent_atomic,
)

VIDEO_SHAPE = (1, 16, 2, 4, 4)
AUDIO_SHAPE = (1, 8, 2048)


def _samples(n: int) -> list[dict]:
    return [
        {
            "video": f"/media/clip{i}.mp4",
            "audio": f"/media/clip{i}.wav",
            "caption": f"caption {i}",
            "source": "pexels:1",
        }
        for i in range(n)
    ]


def _encoder(calls: list, fail_on: tuple = (), audio_shape=AUDIO_SHAPE):
    """Fake encode_fn: records calls, optionally fails, saves real tensors."""

    def encode(sample, video_dest, audio_dest):
        calls.append(sample["video"])
        if sample["video"] in fail_on:
            raise RuntimeError("boom")
        save_latent_atomic(torch.zeros(*VIDEO_SHAPE), video_dest)
        save_latent_atomic(torch.zeros(*audio_shape), audio_dest)

    return encode


def _quiet(msg):  # keep pytest output clean
    pass


def _read_manifest(out_dir: Path) -> list[dict]:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_av_latent_paths_match_smoke_layout(tmp_path):
    video, audio = av_latent_paths(tmp_path, 3)
    assert video.name == "sample0003_video.pt"
    assert audio.name == "sample0003_audio.pt"


def test_is_valid_latent_ndim_relaxation(tmp_path):
    v = tmp_path / "v.pt"
    a = tmp_path / "a.pt"
    save_latent_atomic(torch.zeros(*VIDEO_SHAPE), v)
    save_latent_atomic(torch.zeros(*AUDIO_SHAPE), a)
    assert is_valid_latent(v)  # default stays 5D (pair-cache behaviour)
    assert not is_valid_latent(a)  # 3D rejected under the default
    assert is_valid_latent(a, ndim=3)
    assert not is_valid_latent(v, ndim=3)
    assert is_valid_video_latent(v) and not is_valid_video_latent(a)
    assert is_valid_audio_latent(a) and not is_valid_audio_latent(v)
    bad = tmp_path / "nan.pt"
    save_latent_atomic(torch.full(AUDIO_SHAPE, float("nan")), bad)
    assert not is_valid_audio_latent(bad)


def test_plan_fresh_all_todo(tmp_path):
    plan = plan_av_encoding(_samples(3), tmp_path, model="ti2v-5b")
    assert plan.todo == [0, 1, 2]
    assert plan.reusable == {}
    assert not plan.cache_invalidated


def test_run_writes_manifest_summary_and_meta(tmp_path):
    calls: list = []
    plan = plan_av_encoding(_samples(3), tmp_path, model="ti2v-5b")
    summary = run_av_encoding(plan, _encoder(calls), progress=_quiet)
    assert calls == ["/media/clip0.mp4", "/media/clip1.mp4", "/media/clip2.mp4"]
    assert summary == {
        "model": "ti2v-5b",
        "total": 3,
        "encoded": 3,
        "reused": 0,
        "failed": 0,
    }
    rows = _read_manifest(tmp_path)
    assert [r["index"] for r in rows] == [0, 1, 2]
    row = rows[0]
    assert row["video_shape"] == list(VIDEO_SHAPE)
    assert row["audio_shape"] == list(AUDIO_SHAPE)
    assert row["reused"] is False
    assert row["video"] == "/media/clip0.mp4" and row["audio"] == "/media/clip0.wav"
    on_disk = json.loads((tmp_path / SUMMARY_NAME).read_text())
    assert on_disk == summary
    meta = json.loads((tmp_path / META_NAME).read_text())
    assert meta["model"] == "ti2v-5b"


def test_resume_reuses_cache_and_skips_encoder(tmp_path):
    samples = _samples(3)
    calls: list = []
    run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b"), _encoder(calls), progress=_quiet
    )
    plan = plan_av_encoding(samples, tmp_path, "ti2v-5b")
    assert plan.todo == []
    assert sorted(plan.reusable) == [0, 1, 2]
    assert all(r["reused"] for r in plan.reusable.values())

    def forbidden(*args):  # encoder must not load when fully cached
        raise AssertionError("encoder called on a fully cached run")

    summary = run_av_encoding(plan, forbidden, progress=_quiet)
    assert summary["reused"] == 3 and summary["encoded"] == 0
    assert all(r["reused"] for r in _read_manifest(tmp_path))


def test_error_row_retried_on_next_run(tmp_path):
    samples = _samples(3)
    calls: list = []
    plan = plan_av_encoding(samples, tmp_path, "ti2v-5b")
    summary = run_av_encoding(
        plan, _encoder(calls, fail_on=("/media/clip1.mp4",)), progress=_quiet
    )
    assert summary["failed"] == 1 and summary["encoded"] == 2
    rows = {r["index"]: r for r in _read_manifest(tmp_path)}
    assert "error" in rows[1] and "RuntimeError" in rows[1]["error"]
    assert rows[1]["video"] == "/media/clip1.mp4"

    plan2 = plan_av_encoding(samples, tmp_path, "ti2v-5b")
    assert plan2.todo == [1]
    assert sorted(plan2.reusable) == [0, 2]
    calls.clear()
    summary2 = run_av_encoding(plan2, _encoder(calls), progress=_quiet)
    assert calls == ["/media/clip1.mp4"]
    assert summary2 == {
        "model": "ti2v-5b", "total": 3, "encoded": 1, "reused": 2, "failed": 0
    }


def test_drift_guard_media_change_reencodes_caption_edit_reuses(tmp_path):
    samples = _samples(2)
    run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b"), _encoder([]), progress=_quiet
    )
    captioned = [dict(s) for s in samples]
    captioned[0]["caption"] = "rewritten caption"
    plan = plan_av_encoding(captioned, tmp_path, "ti2v-5b")
    assert plan.todo == []  # latents do not depend on the caption

    moved = [dict(s) for s in samples]
    moved[1]["video"] = "/media/other.mp4"
    plan2 = plan_av_encoding(moved, tmp_path, "ti2v-5b")
    assert plan2.todo == [1]
    assert sorted(plan2.reusable) == [0]


def test_meta_model_mismatch_invalidates_whole_cache(tmp_path):
    samples = _samples(2)
    run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b"), _encoder([]), progress=_quiet
    )
    plan = plan_av_encoding(samples, tmp_path, "other-vae")
    assert plan.cache_invalidated
    assert plan.todo == [0, 1]
    assert plan.reusable == {}


def test_legacy_fallback_seeds_from_smoke_layout(tmp_path):
    # A smoke-run directory has .pt files but no usable manifest rows.
    video, audio = av_latent_paths(tmp_path, 0)
    save_latent_atomic(torch.zeros(*VIDEO_SHAPE), video)
    save_latent_atomic(torch.zeros(*AUDIO_SHAPE), audio)
    plan = plan_av_encoding(_samples(1), tmp_path, "ti2v-5b")
    assert plan.todo == []
    row = plan.reusable[0]
    assert row["reused"] is True
    assert row["video_shape"] == list(VIDEO_SHAPE)
    assert row["audio_shape"] == list(AUDIO_SHAPE)


def test_corrupt_pt_is_reencoded(tmp_path):
    samples = _samples(2)
    run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b"), _encoder([]), progress=_quiet
    )
    _, audio = av_latent_paths(tmp_path, 1)
    audio.write_bytes(b"\x00truncated")  # killed mid-write
    plan = plan_av_encoding(samples, tmp_path, "ti2v-5b")
    assert plan.todo == [1]
    calls: list = []
    summary = run_av_encoding(plan, _encoder(calls), progress=_quiet)
    assert calls == ["/media/clip1.mp4"]
    assert summary["encoded"] == 1 and summary["reused"] == 1
    assert is_valid_audio_latent(audio)


def test_strict_reraises(tmp_path):
    plan = plan_av_encoding(_samples(1), tmp_path, "ti2v-5b")
    with pytest.raises(RuntimeError, match="boom"):
        run_av_encoding(
            plan,
            _encoder([], fail_on=("/media/clip0.mp4",)),
            strict=True,
            progress=_quiet,
        )


def test_contract_violation_is_recorded_not_cached(tmp_path):
    # Encoder honours the call but leaves a 2D "audio latent" — the runner
    # must record a failure row instead of manifesting corrupt output.
    calls: list = []
    plan = plan_av_encoding(_samples(1), tmp_path, "ti2v-5b")
    summary = run_av_encoding(
        plan, _encoder(calls, audio_shape=(8, 2048)), progress=_quiet
    )
    assert summary["failed"] == 1 and summary["encoded"] == 0
    (row,) = _read_manifest(tmp_path)
    assert "refusing to cache" in row["error"]
    # Next run treats the sample as pending again (no success row, and the
    # legacy fallback rejects the 2D file as well).
    plan2 = plan_av_encoding(_samples(1), tmp_path, "ti2v-5b")
    assert plan2.todo == [0]


# ---- retry budget (failure ledger) ---------------------------------------------
#
# Same opt-in design as the text cache (test_m2_text_cache): an av_failures.json
# ledger counts per-sample failures, budget-exhausted samples are planned as
# skipped, and the default (no max_retries) keeps byte-identical pre-ledger
# behaviour.


def test_av_failure_ledger_roundtrip(tmp_path):
    assert load_failure_ledger(tmp_path) == {}
    sample = _samples(1)[0]
    assert av_failure_key(3) == "sample0003"
    assert record_failure(tmp_path, 3, sample, "RuntimeError: boom") == 1
    assert record_failure(tmp_path, 3, sample, "RuntimeError: boom again") == 2
    ledger = load_failure_ledger(tmp_path)
    entry = ledger["sample0003"]
    assert entry["count"] == 2
    assert entry["caption"] == "caption 0"
    assert entry["video"] == "/media/clip0.mp4"
    assert "boom again" in entry["error"]
    # A different index lands on a different key.
    record_failure(tmp_path, 7, sample, "x")
    assert len(load_failure_ledger(tmp_path)) == 2
    # Success clears only that sample.
    clear_failure(tmp_path, 3)
    ledger = load_failure_ledger(tmp_path)
    assert "sample0003" not in ledger
    assert len(ledger) == 1
    # Reset deletes the file entirely.
    assert reset_failure_ledger(tmp_path)
    assert not (tmp_path / FAILURES_NAME).exists()
    assert not reset_failure_ledger(tmp_path)
    # Corrupt ledger degrades to {} instead of crashing.
    (tmp_path / FAILURES_NAME).write_text("{not json")
    assert load_failure_ledger(tmp_path) == {}


def test_plan_skips_sample_past_retry_budget(tmp_path):
    samples = _samples(2)
    fail = ("/media/clip1.mp4",)
    calls1: list = []
    s1 = run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b", max_retries=2),
        _encoder(calls1, fail_on=fail), progress=_quiet, max_retries=2,
    )
    assert s1["failed"] == 1 and s1["encoded"] == 1  # sample 0 cached from here on
    # Second run retries only sample 1 and fills its budget (count reaches 2).
    calls2: list = []
    s2 = run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b", max_retries=2),
        _encoder(calls2, fail_on=fail), progress=_quiet, max_retries=2,
    )
    assert calls2 == ["/media/clip1.mp4"]
    assert s2["failed"] == 1 and s2["reused"] == 1
    # Third run: sample 1 is skipped by the planner, never encoded.
    plan3 = plan_av_encoding(samples, tmp_path, "ti2v-5b", max_retries=2)
    assert plan3.todo == []
    assert set(plan3.reusable) == {0}
    assert set(plan3.skipped) == {1}
    skipped_row = plan3.skipped[1]
    assert skipped_row["skipped"] is True
    assert skipped_row["retries"] == 2
    assert skipped_row["video"] == "/media/clip1.mp4"
    assert "boom" in skipped_row["last_error"]
    calls3: list = []
    s3 = run_av_encoding(plan3, _encoder(calls3), progress=_quiet, max_retries=2)
    assert calls3 == []
    # Skipped rows are not counted as failed, reused, or encoded.
    assert s3["failed"] == 0 and s3["encoded"] == 0
    assert s3["reused"] == 1
    assert s3["skipped"] == 1
    rows = _read_manifest(tmp_path)
    assert any(r.get("skipped") for r in rows)
    assert not any("error" in r for r in rows)
    # The summary sidecar carries the skipped count too.
    assert json.loads((tmp_path / SUMMARY_NAME).read_text())["skipped"] == 1


def test_av_success_clears_ledger_entry(tmp_path):
    samples = _samples(1)
    run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b", max_retries=3),
        _encoder([], fail_on=("/media/clip0.mp4",)), progress=_quiet, max_retries=3,
    )
    assert load_failure_ledger(tmp_path) != {}
    # The sample now encodes — success wipes its failure history.
    summary = run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b", max_retries=3),
        _encoder([]), progress=_quiet, max_retries=3,
    )
    assert summary["encoded"] == 1 and summary["failed"] == 0
    assert load_failure_ledger(tmp_path) == {}


def test_av_no_ledger_io_when_max_retries_unset(tmp_path):
    samples = _samples(2)
    summary = run_av_encoding(
        plan_av_encoding(samples, tmp_path, "ti2v-5b"),
        _encoder([], fail_on=("/media/clip1.mp4",)),
        progress=_quiet,
    )
    assert summary["failed"] == 1
    assert "skipped" not in summary  # default behaviour is byte-identical
    assert not (tmp_path / FAILURES_NAME).exists()
    # Legacy retry loop still works: the failure row lands back in todo.
    plan2 = plan_av_encoding(samples, tmp_path, "ti2v-5b")
    assert plan2.todo == [1] and plan2.skipped == {}


def test_av_strict_records_failure_before_reraising(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        run_av_encoding(
            plan_av_encoding(_samples(1), tmp_path, "ti2v-5b", max_retries=2),
            _encoder([], fail_on=("/media/clip0.mp4",)),
            strict=True,
            progress=_quiet,
            max_retries=2,
        )
    # The abort did not lose the failure: the ledger saw the attempt.
    ledger = load_failure_ledger(tmp_path)
    assert ledger["sample0000"]["count"] == 1


def test_av_invalid_max_retries_rejected(tmp_path):
    with pytest.raises(ValueError):
        plan_av_encoding(_samples(1), tmp_path, "ti2v-5b", max_retries=0)
    with pytest.raises(ValueError):
        run_av_encoding(
            plan_av_encoding(_samples(1), tmp_path, "ti2v-5b"),
            _encoder([]),
            progress=_quiet,
            max_retries=True,
        )

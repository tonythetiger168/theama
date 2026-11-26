# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/encode_pairs.py (P1a DPO pair latent pre-encode CLI).

The module layers are covered elsewhere — theama.data.latent_cache by
tests/test_m2_av_latent_cache.py, theama.reward.pairs_check by
tests/test_p1a_pairs_check.py; this file pins the *CLI contract* of the
script itself — the part that decides whether a crashed GPU encode run
resumes cheaply or burns a VAE load for nothing:

- preflight gate: pairs.jsonl is validated with the REAL
  theama.reward.pairs_check (pure stdlib) before planning — unusable rows
  or bad JSON lines refuse to start (exit 1, per-row reasons, planner
  never reached), because pair indices key the latent cache and rows must
  never be dropped in place;
- --smoke / --limit truncation, empty-file exit 1;
- the VAE stack (diffusers + torch + theama.inference.config) is imported
  and loaded **only** when the plan has work to do — a fully cached rerun
  never pays the model-load cost (mirrors encode_texts / gen_candidates
  resume economics);
- --no-resume / --max-retries / --reset-failures / --strict pass-through
  (reset happens before planning);
- failed pairs keep exit 0 (documented contract: retried next run,
  --strict to abort) with a stderr hint that mentions --reset-failures
  once a retry budget is in play;
- encode_fn wiring: WanPipeline from the model's repo_id in bf16, VAE to
  cuda/float32, win half encoded before lose half, latents saved via
  latent_cache.save_latent_atomic;
- encode_video math: frames scaled (x/127.5 - 1) and moved to cuda fp32
  before encode, latent normalized per channel as (latent - mean)/std
  before .cpu().

No GPU, torch, diffusers, or network required — heavy deps are injected
fakes, mirroring tests/test_m2_encode_texts.py.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "encode_pairs", ROOT / "scripts" / "encode_pairs.py"
)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

MODEL_REPO = "Wan-AI/Fake-TI2V-5B"
Z_DIM = 2
LATENTS_MEAN = [0.5, -0.5]
LATENTS_STD = [2.0, 4.0]
# two 2x2 frames; 0/127.5/255 map to exactly -1/0/+1 after (x/127.5 - 1)
FRAMES = [
    [[[0.0, 127.5, 255.0], [255.0, 127.5, 0.0]],
     [[127.5, 0.0, 255.0], [0.0, 255.0, 127.5]]],
    [[[255.0, 0.0, 127.5], [127.5, 255.0, 0.0]],
     [[0.0, 127.5, 255.0], [255.0, 0.0, 127.5]]],
]
# fake VAE output (1, Z_DIM, 1, 1, 1); per-channel (raw - mean) / std is
# exactly (+1.0, -1.0) with the constants above
RAW_LATENT = np.array([[[[[2.5]]], [[[-4.5]]]]])


# ---------------------------------------------------------------- fakes


class _FT:
    """Numpy-backed stand-in for a torch tensor (only the ops encode_video
    uses). Records device/dtype so tests can assert the cuda/fp32 hoist."""

    def __init__(self, arr, device="cpu", dtype=None):
        self.arr = np.asarray(arr, dtype=np.float64)
        self.device = device
        self.dtype = dtype

    # shape ops -------------------------------------------------------
    def permute(self, *dims):
        return _FT(np.transpose(self.arr, dims), self.device, self.dtype)

    def unsqueeze(self, dim):
        return _FT(np.expand_dims(self.arr, dim), self.device, self.dtype)

    def view(self, *shape):
        return _FT(self.arr.reshape(shape), self.device, self.dtype)

    # device/dtype moves ------------------------------------------------
    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def cpu(self):
        self.device = "cpu"
        return self

    # arithmetic --------------------------------------------------------
    @staticmethod
    def _arr(other):
        return other.arr if isinstance(other, _FT) else other

    def __truediv__(self, other):
        return _FT(self.arr / self._arr(other), self.device, self.dtype)

    def __rtruediv__(self, other):
        return _FT(self._arr(other) / self.arr, self.device, self.dtype)

    def __sub__(self, other):
        return _FT(self.arr - self._arr(other), self.device, self.dtype)

    def __rsub__(self, other):
        return _FT(self._arr(other) - self.arr, self.device, self.dtype)

    def __mul__(self, other):
        return _FT(self.arr * self._arr(other), self.device, self.dtype)

    def __rmul__(self, other):
        return _FT(self._arr(other) * self.arr, self.device, self.dtype)


class _NoGrad:
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fakes(monkeypatch, tmp_path, todo, reusable=(), skipped=None,
                   cache_invalidated=False, ledger_cleared=False,
                   run_summary=None, with_vae=True):
    """Inject fake latent_cache (+ optionally diffusers/torch/config).

    Returns the recorder dict. `todo`/`reusable` are index lists the fake
    plan reports (`skipped` an index->row dict); the fake runner encodes
    every todo pair through the script's real encode_fn unless run_summary
    is given (then it is returned verbatim and encode_fn is unused).
    """
    rec = {
        "encode_calls": [],
        "saves": [],
        "load_video_calls": [],
        "vae_moves": [],
        "order": [],
    }

    latent_cache = types.ModuleType("theama.data.latent_cache")

    def plan_encoding(pairs, out, model, resume=True, max_retries=None):
        rec["order"].append("plan")
        rec["plan_args"] = {
            "pairs": list(pairs),
            "out": Path(out),
            "model": model,
            "resume": resume,
            "max_retries": max_retries,
        }
        plan = types.SimpleNamespace(
            pairs=list(pairs),
            out_dir=Path(out),
            model=model,
            reusable=list(reusable),
            todo=list(todo),
            skipped=dict(skipped or {}),
            cache_invalidated=cache_invalidated,
            manifest_path=Path(out) / "pairs_latents.jsonl",
            summary_path=Path(out) / "pairs_latents.summary.json",
        )
        rec["plan"] = plan
        return plan

    def reset_failure_ledger(out):
        rec["order"].append("reset")
        rec["reset_out"] = Path(out)
        return ledger_cleared

    def run_encoding(plan, encode_fn, strict=False, progress=None,
                     max_retries=None):
        rec["run_args"] = {
            "encode_fn": encode_fn,
            "strict": strict,
            "max_retries": max_retries,
        }
        if run_summary is not None:
            return run_summary
        rows = []
        for index in plan.todo:
            pair = plan.pairs[index]
            win = plan.out_dir / f"pair_{index:016x}_win.pt"
            lose = plan.out_dir / f"pair_{index:016x}_lose.pt"
            encode_fn(pair["win_video"], win)
            encode_fn(pair["lose_video"], lose)
            rows.append({
                "index": index,
                "prompt_id": pair["prompt_id"],
                "win_latent": str(win),
                "lose_latent": str(lose),
                "reused": False,
            })
        plan.manifest_path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        summary = {"encoded": len(rows), "reused": len(plan.reusable),
                   "failed": 0}
        plan.summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def save_latent_atomic(tensor, dest):
        rec["saves"].append({"dest": Path(dest), "tensor": tensor})

    latent_cache.plan_encoding = plan_encoding
    latent_cache.reset_failure_ledger = reset_failure_ledger
    latent_cache.run_encoding = run_encoding
    latent_cache.save_latent_atomic = save_latent_atomic
    monkeypatch.setitem(sys.modules, "theama.data.latent_cache", latent_cache)

    if with_vae:
        vae = types.SimpleNamespace()
        vae.config = types.SimpleNamespace(
            latents_mean=LATENTS_MEAN, latents_std=LATENTS_STD, z_dim=Z_DIM
        )

        def vae_to(device=None, dtype=None):
            rec["vae_moves"].append((device, dtype))
            return vae

        def vae_encode(x):
            rec["vae_input"] = x
            return types.SimpleNamespace(
                latent_dist=types.SimpleNamespace(
                    sample=lambda: _FT(RAW_LATENT, device=x.device)
                )
            )

        vae.to = vae_to
        vae.encode = vae_encode

        diffusers = types.ModuleType("diffusers")

        class WanPipeline:
            @staticmethod
            def from_pretrained(repo, torch_dtype=None):
                rec["pipe_load"] = {"repo": repo, "torch_dtype": torch_dtype}
                return types.SimpleNamespace(vae=vae)

        diffusers.WanPipeline = WanPipeline

        diffusers_utils = types.ModuleType("diffusers.utils")

        def load_video(path):
            rec["load_video_calls"].append(path)
            return FRAMES

        diffusers_utils.load_video = load_video
        diffusers.utils = diffusers_utils

        torch = types.ModuleType("torch")
        torch.bfloat16 = "bf16-sentinel"
        torch.float32 = "fp32-sentinel"
        torch.from_numpy = lambda arr: _FT(np.asarray(arr))
        torch.tensor = lambda data: _FT(np.asarray(data))
        torch.no_grad = _NoGrad()

        config = types.ModuleType("theama.inference.config")
        config.MODELS = {"ti2v-5b": types.SimpleNamespace(repo_id=MODEL_REPO)}

        monkeypatch.setitem(sys.modules, "diffusers", diffusers)
        monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "theama.inference.config", config)
    return rec


# ---------------------------------------------------------------- helpers


def _row(tmp_path, pid, win_score=0.9, lose_score=0.1):
    """A validate_pairs-clean row; both clip files exist on disk."""
    win = tmp_path / f"{pid}_win.mp4"
    lose = tmp_path / f"{pid}_lose.mp4"
    win.touch()
    lose.touch()
    return {
        "prompt_id": pid,
        "prompt": f"prompt {pid}",
        "win_video": str(win),
        "lose_video": str(lose),
        "win_score": win_score,
        "lose_score": lose_score,
    }


def _pairs_file(tmp_path, rows, extra_lines=()):
    path = tmp_path / "pairs.jsonl"
    lines = [json.dumps(r) for r in rows] + list(extra_lines)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def _argv(tmp_path, pairs_path, *extra):
    return [
        "--pairs", str(pairs_path),
        "--out", str(tmp_path / "latents"),
        *extra,
    ]


def _run(monkeypatch, tmp_path, rows, todo, extra=(), extra_lines=(), **kw):
    pairs_path = _pairs_file(tmp_path, rows, extra_lines)
    rec = _install_fakes(monkeypatch, tmp_path, todo, **kw)
    argv = sys.argv
    sys.argv = ["encode_pairs.py", *_argv(tmp_path, pairs_path, *extra)]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    return code, rec


GOOD_SUMMARY = {"encoded": 0, "reused": 2, "failed": 0}


# ------------------------------------------------------------------ tests
# preflight gate (real theama.reward.pairs_check)


def test_missing_pairs_file_exit_1(monkeypatch, tmp_path, capsys):
    rec = _install_fakes(monkeypatch, tmp_path, todo=[])
    argv = sys.argv
    sys.argv = ["encode_pairs.py",
                "--pairs", str(tmp_path / "nope.jsonl"),
                "--out", str(tmp_path / "latents")]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    assert code == 1
    assert "plan_args" not in rec  # planner never reached
    assert capsys.readouterr().err  # the OS error is reported


def test_unusable_row_refuses_to_start(monkeypatch, tmp_path, capsys):
    row = _row(tmp_path, "p1")
    row["lose_video"] = row["win_video"]  # self-pair carries no DPO signal
    code, rec = _run(monkeypatch, tmp_path, [row], todo=[0])
    assert code == 1
    err = capsys.readouterr().err
    assert "self_pair" in err  # per-row reason is printed
    assert "refusing to start" in err
    assert "make_pref_pairs" in err
    assert "plan_args" not in rec  # planner never reached


def test_bad_json_line_refuses_to_start(monkeypatch, tmp_path, capsys):
    row = _row(tmp_path, "p1")
    code, rec = _run(monkeypatch, tmp_path, [row], todo=[0],
                     extra_lines=("NOT JSON",))
    assert code == 1
    err = capsys.readouterr().err
    assert "bad JSON line" in err
    assert "refusing to start" in err
    assert "plan_args" not in rec


def test_missing_video_file_refuses_to_start(monkeypatch, tmp_path, capsys):
    row = _row(tmp_path, "p1")
    Path(row["lose_video"]).unlink()  # clip pruned after pairing
    code, rec = _run(monkeypatch, tmp_path, [row], todo=[0])
    assert code == 1
    assert "missing_video_file" in capsys.readouterr().err
    assert "plan_args" not in rec


# truncation / empty


def test_smoke_truncates_to_two(monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(3)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[0, 1],
                     extra=("--smoke",))
    assert code == 0
    assert [p["prompt_id"] for p in rec["plan_args"]["pairs"]] == ["p0", "p1"]


def test_limit_truncates(monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(3)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[0],
                     extra=("--limit", "1"))
    assert code == 0
    assert [p["prompt_id"] for p in rec["plan_args"]["pairs"]] == ["p0"]


def test_empty_pairs_file_exit_1(monkeypatch, tmp_path, capsys):
    code, rec = _run(monkeypatch, tmp_path, [], todo=[])
    assert code == 1
    assert "no pairs to encode" in capsys.readouterr().err
    assert "plan_args" not in rec


# resume economy


def test_fully_cached_never_imports_torch_diffusers_or_config(
        monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    _install_fakes(monkeypatch, tmp_path, todo=[], reusable=[0, 1],
                   run_summary=GOOD_SUMMARY)
    # tripwires: any import of the heavy stack raises
    for name in ("torch", "diffusers", "diffusers.utils",
                 "theama.inference.config"):
        monkeypatch.setitem(sys.modules, name, None)
    pairs_path = _pairs_file(tmp_path, rows)
    argv = sys.argv
    sys.argv = ["encode_pairs.py", *_argv(tmp_path, pairs_path)]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    assert code == 0


def test_fully_cached_passes_encode_fn_none(monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                     run_summary=GOOD_SUMMARY)
    assert code == 0
    assert rec["run_args"]["encode_fn"] is None


def test_no_resume_passes_resume_false(monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                     extra=("--no-resume",), run_summary=GOOD_SUMMARY)
    assert code == 0
    assert rec["plan_args"]["resume"] is False


# flag pass-through


def test_max_retries_forwarded_to_plan_and_run(monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                     extra=("--max-retries", "3"), run_summary=GOOD_SUMMARY)
    assert code == 0
    assert rec["plan_args"]["max_retries"] == 3
    assert rec["run_args"]["max_retries"] == 3


def test_reset_failures_clears_ledger_before_planning(monkeypatch, tmp_path,
                                                      capsys):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                     extra=("--reset-failures",), ledger_cleared=True,
                     run_summary=GOOD_SUMMARY)
    assert code == 0
    assert rec["reset_out"] == tmp_path / "latents"
    assert rec["order"] == ["reset", "plan"]  # ledger cleared before planning
    assert "cleared failure ledger" in capsys.readouterr().out


def test_strict_forwarded(monkeypatch, tmp_path):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                     extra=("--strict",), run_summary=GOOD_SUMMARY)
    assert code == 0
    assert rec["run_args"]["strict"] is True


def test_cache_invalidated_message(monkeypatch, tmp_path, capsys):
    rows = [_row(tmp_path, "p0")]
    code, _ = _run(monkeypatch, tmp_path, rows, todo=[0],
                   cache_invalidated=True)
    assert code == 0
    assert "re-encoding everything" in capsys.readouterr().out


# failure contract


def test_failed_pairs_keep_exit_0_with_retry_hint(monkeypatch, tmp_path,
                                                  capsys):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, _ = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                   run_summary={"encoded": 1, "reused": 1, "failed": 1})
    assert code == 0  # documented contract: failures retried on next run
    err = capsys.readouterr().err
    assert "1 pair(s) failed to encode" in err
    assert "rerun to retry" in err


def test_failed_hint_mentions_reset_when_budget_capped(monkeypatch, tmp_path,
                                                       capsys):
    rows = [_row(tmp_path, f"p{i}") for i in range(2)]
    code, _ = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                   extra=("--max-retries", "3"),
                   run_summary={"encoded": 1, "reused": 1, "failed": 1})
    assert code == 0
    assert "--reset-failures" in capsys.readouterr().err


def test_skipped_budget_exhausted_suffix_in_counts(monkeypatch, tmp_path,
                                                   capsys):
    rows = [_row(tmp_path, f"p{i}") for i in range(3)]
    skipped = {2: {"index": 2, "prompt_id": "p2", "skipped": True,
                   "retries": 3}}
    code, _ = _run(monkeypatch, tmp_path, rows, todo=[], reusable=[0, 1],
                   skipped=skipped, run_summary=GOOD_SUMMARY)
    assert code == 0
    assert "retry budget exhausted" in capsys.readouterr().out


# encode_fn / encode_video wiring


def test_fresh_run_loads_pipeline_and_encodes_win_then_lose(
        monkeypatch, tmp_path):
    rows = [_row(tmp_path, "p0"), _row(tmp_path, "p1")]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[0, 1])
    assert code == 0
    # pipeline from the default model's repo, bf16; VAE hoisted to cuda fp32
    assert rec["plan_args"]["model"] == "ti2v-5b"
    assert rec["pipe_load"] == {"repo": MODEL_REPO,
                                "torch_dtype": "bf16-sentinel"}
    assert rec["vae_moves"] == [("cuda", "fp32-sentinel")]
    # win half encoded before lose half, in pair order
    assert rec["load_video_calls"] == [
        rows[0]["win_video"], rows[0]["lose_video"],
        rows[1]["win_video"], rows[1]["lose_video"],
    ]
    # every half saved via save_latent_atomic, on-cpu tensors
    assert len(rec["saves"]) == 4
    assert all(s["tensor"].device == "cpu" for s in rec["saves"])
    # manifest + summary sidecar written
    manifest = [json.loads(line) for line in
                rec["plan"].manifest_path.read_text().splitlines()]
    assert [r["prompt_id"] for r in manifest] == ["p0", "p1"]
    assert all(not r["reused"] for r in manifest)
    summary = json.loads(rec["plan"].summary_path.read_text())
    assert summary == {"encoded": 2, "reused": 0, "failed": 0}


def test_encode_video_normalization_math(monkeypatch, tmp_path):
    rows = [_row(tmp_path, "p0")]
    code, rec = _run(monkeypatch, tmp_path, rows, todo=[0])
    assert code == 0
    # frames scaled to [-1, 1] and moved to cuda/fp32 before encode
    x = rec["vae_input"]
    expected_x = (
        np.stack([np.asarray(f, dtype=np.float32) for f in FRAMES])
        .transpose(3, 0, 1, 2)[None] / 127.5 - 1.0
    )
    np.testing.assert_allclose(x.arr, expected_x)
    assert (x.device, x.dtype) == ("cuda", "fp32-sentinel")
    # saved latent = (raw - mean) / std per channel, back on cpu
    expected = (
        (RAW_LATENT - np.array(LATENTS_MEAN).reshape(1, Z_DIM, 1, 1, 1))
        / np.array(LATENTS_STD).reshape(1, Z_DIM, 1, 1, 1)
    )
    assert np.array_equal(expected, np.array([[[[[1.0]]], [[[-1.0]]]]]))
    assert len(rec["saves"]) == 2  # win + lose
    for save in rec["saves"]:
        np.testing.assert_allclose(save["tensor"].arr, expected)
        assert save["tensor"].device == "cpu"

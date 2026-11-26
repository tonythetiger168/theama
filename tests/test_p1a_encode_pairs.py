# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/encode_pairs.py (P1a DPO pair-latent pre-encode CLI).

The cache core (theama.data.latent_cache: plan_encoding / run_encoding /
failure ledger) and the preflight gate (theama.reward.pairs_check +
scripts/check_pairs.py, incl. the refuse-before-cache-import wiring) are
covered by their own tests; this file pins the *CLI contract* of the
encode_pairs script itself — the part that decides whether a crashed GPU
encode run resumes cheaply or burns a VAE load for nothing:

- the Wan pipeline (diffusers + torch) is imported and the VAE loaded
  **only** when the plan has work to do — a fully cached rerun never pays
  the load cost (mirrors encode_texts / gen_candidates resume economics);
- encode_fn wiring: win before lose, frames scaled /127.5-1, latent
  normalized (x - latents_mean) * (1/latents_std) from vae.config, saved
  via latent_cache.save_latent_atomic to the dest the planner chose;
- --smoke / --limit truncation, zero pairs after truncation exit 1;
- --no-resume / --max-retries / --reset-failures / --strict pass-through;
- failed pairs keep exit 0 (documented contract: retried next run,
  --strict to abort) with a stderr hint;
- --model selects both the planner cache namespace and the pipeline repo.

No GPU, torch, diffusers, or network required — all heavy deps are
injected fakes, mirroring tests/test_m2_encode_texts.py. The pairs
preflight runs the *real* theama.reward.pairs_check (pure stdlib).
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest  # noqa: F401  (fixtures only)

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "encode_pairs", ROOT / "scripts" / "encode_pairs.py"
)
ep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep)

MODEL_REPO = "Wan-AI/Fake-TI2V-5B"
OTHER_REPO = "Wan-AI/Fake-A14B"
MEAN = [0.5, 0.25, -0.5, 1.0]
STD = [2.0, 4.0, 0.5, 1.0]
ZDIM = 4


# ---------------------------------------------------------------- fakes


class _NoGrad:
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeTensor:
    """Records every op; ops return self so chains keep one tag."""

    def __init__(self, rec, tag):
        self._rec = rec
        self.tag = tag

    def _op(self, name, *args):
        self._rec["ops"].append((self.tag, name, args))
        return self

    def permute(self, *dims):
        return self._op("permute", *dims)

    def unsqueeze(self, dim):
        return self._op("unsqueeze", dim)

    def view(self, *shape):
        return self._op("view", *shape)

    def to(self, *args):
        return self._op("to", *args)

    def cpu(self):
        return self._op("cpu")

    @property
    def device(self):
        return "fake-device"

    def __truediv__(self, other):
        return self._op("truediv", other)

    def __rtruediv__(self, other):
        return self._op("rtruediv", other)

    def __sub__(self, other):
        return self._op("sub", getattr(other, "tag", other))

    def __mul__(self, other):
        return self._op("mul", getattr(other, "tag", other))


class _FakeVAE:
    def __init__(self, rec):
        self._rec = rec
        self.config = types.SimpleNamespace(
            latents_mean=MEAN, latents_std=STD, z_dim=ZDIM
        )

    def to(self, *args):
        self._rec["vae_moves"].append(args)
        return self

    def encode(self, x):
        self._rec["vae_encode_inputs"].append(x.tag)
        rec = self._rec
        tag = f"latent{rec['counters']['latent']}"
        rec["counters"]["latent"] += 1
        latent = _FakeTensor(rec, tag)
        return types.SimpleNamespace(
            latent_dist=types.SimpleNamespace(sample=lambda: latent)
        )


def _install_fakes(monkeypatch, *, todo=(0,), reusable=(), skipped=(),
                   cache_invalidated=False, ledger_cleared=False,
                   run_summary=None):
    """Inject fake latent_cache/config/diffusers/torch/numpy modules.

    Returns the recorder dict. `todo`/`reusable`/`skipped` are pair indices
    into the (truncated) pairs list the fake plan reports; the fake runner
    encodes every todo pair through the script's real encode_fn (win then
    lose) unless run_summary is given (then it is returned verbatim and
    encode_fn is unused).
    """
    rec = {
        "ops": [], "calls": [], "saves": [], "load_video_calls": [],
        "asarray_calls": [], "pipe_loads": [], "vae_moves": [],
        "vae_encode_inputs": [], "from_numpy_calls": 0, "stack_calls": 0,
        "counters": {"x": 0, "latent": 0},
    }

    latent_cache = types.ModuleType("theama.data.latent_cache")

    def plan_encoding(pairs, out_dir, model, resume=True, max_retries=None):
        rec["calls"].append("plan")
        rec["plan_args"] = {
            "pairs": list(pairs),
            "out": Path(out_dir),
            "model": model,
            "resume": resume,
            "max_retries": max_retries,
        }
        out = Path(out_dir)
        plan = types.SimpleNamespace(
            pairs=list(pairs),
            model=model,
            reusable={
                i: {"index": i, "prompt_id": pairs[i]["prompt_id"],
                    "model": model, "reused": True,
                    "win_latent": str(out / f"pair{i:04d}_win.pt"),
                    "lose_latent": str(out / f"pair{i:04d}_lose.pt")}
                for i in reusable
            },
            todo=list(todo),
            skipped={
                i: {"index": i, "prompt_id": pairs[i]["prompt_id"],
                    "skipped": True}
                for i in skipped
            },
            cache_invalidated=cache_invalidated,
            manifest_path=out / "pairs_latents.jsonl",
            summary_path=out / "pairs_latents.summary.json",
        )
        rec["plan"] = plan
        return plan

    def run_encoding(plan, encode_fn, strict=False, progress=None,
                     max_retries=None):
        rec["calls"].append("run")
        rec["run_args"] = {
            "encode_fn": encode_fn,
            "strict": strict,
            "max_retries": max_retries,
        }
        if run_summary is not None:
            return run_summary
        out = plan.manifest_path.parent
        rows = dict(plan.reusable)
        for index in plan.todo:
            pair = plan.pairs[index]
            win = out / f"pair{index:04d}_win.pt"
            lose = out / f"pair{index:04d}_lose.pt"
            encode_fn(pair["win_video"], win)
            encode_fn(pair["lose_video"], lose)
            rows[index] = {
                "index": index,
                "prompt_id": pair["prompt_id"],
                "model": plan.model,
                "win_latent": str(win),
                "lose_latent": str(lose),
                "reused": False,
            }
            if progress:
                progress(f"[{index + 1}/{len(plan.pairs)}] "
                         f"{pair['prompt_id']}")
        plan.manifest_path.write_text(
            "".join(json.dumps(rows[i]) + "\n" for i in sorted(rows)),
            encoding="utf-8",
        )
        summary = {"model": plan.model, "encoded": len(plan.todo),
                   "reused": len(plan.reusable), "failed": 0}
        plan.summary_path.write_text(json.dumps(summary, indent=2) + "\n",
                                     encoding="utf-8")
        return summary

    def reset_failure_ledger(out):
        rec["calls"].append("reset")
        rec["reset_out"] = Path(out)
        return ledger_cleared

    def save_latent_atomic(tensor, dest):
        rec["saves"].append((tensor.tag, Path(dest)))

    latent_cache.plan_encoding = plan_encoding
    latent_cache.run_encoding = run_encoding
    latent_cache.reset_failure_ledger = reset_failure_ledger
    latent_cache.save_latent_atomic = save_latent_atomic

    config = types.ModuleType("theama.inference.config")
    config.MODELS = {
        "ti2v-5b": types.SimpleNamespace(repo_id=MODEL_REPO),
        "fake-big": types.SimpleNamespace(repo_id=OTHER_REPO),
    }

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16-sentinel"
    torch.float32 = "f32-sentinel"
    torch.no_grad = _NoGrad()

    def _from_numpy(arr):
        rec["from_numpy_calls"] += 1
        rec["ops"].append(("torch", "from_numpy", ()))
        tag = f"x{rec['counters']['x']}"
        rec["counters"]["x"] += 1
        return _FakeTensor(rec, tag)

    def _tensor(data):
        rec["ops"].append(("torch", "tensor", tuple(data)))
        if list(data) == MEAN:
            tag = "mean"
        elif list(data) == STD:
            tag = "std"
        else:
            tag = "const"
        return _FakeTensor(rec, tag)

    torch.from_numpy = _from_numpy
    torch.tensor = _tensor

    numpy = types.ModuleType("numpy")
    numpy.float32 = "np-f32"

    def _asarray(frame, dtype=None):
        rec["asarray_calls"].append((frame, dtype))
        return frame

    def _stack(seq):
        rec["stack_calls"] += 1
        return ("stack", tuple(seq))

    numpy.asarray = _asarray
    numpy.stack = _stack

    diffusers = types.ModuleType("diffusers")

    class WanPipeline:
        @staticmethod
        def from_pretrained(repo_id, torch_dtype=None):
            rec["pipe_loads"].append({"repo_id": repo_id,
                                      "torch_dtype": torch_dtype})
            vae = _FakeVAE(rec)
            rec["vae"] = vae
            return types.SimpleNamespace(vae=vae)

    diffusers.WanPipeline = WanPipeline

    diffusers_utils = types.ModuleType("diffusers.utils")

    def load_video(path):
        rec["load_video_calls"].append(str(path))
        return ["f0", "f1"]

    diffusers_utils.load_video = load_video
    diffusers.utils = diffusers_utils

    monkeypatch.setitem(sys.modules, "theama.data.latent_cache", latent_cache)
    monkeypatch.setitem(sys.modules, "theama.inference.config", config)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "diffusers.utils", diffusers_utils)
    return rec


def _make_pairs(tmp_path, n=3):
    clips = tmp_path / "clips"
    clips.mkdir(exist_ok=True)
    rows = []
    for i in range(n):
        win = clips / f"p{i}_w.mp4"
        lose = clips / f"p{i}_l.mp4"
        win.write_bytes(b"")
        lose.write_bytes(b"")
        rows.append({
            "prompt_id": f"p{i}",
            "prompt": f"prompt {i}",
            "win_video": str(win),
            "lose_video": str(lose),
            "win_score": 4.5,
            "lose_score": 2.0,
            "margin": 2.5,
        })
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("".join(json.dumps(r) + "\n" for r in rows),
                     encoding="utf-8")
    return pairs, rows


def _run(monkeypatch, tmp_path, extra=(), n_pairs=3, **fakes_kw):
    pairs_path, rows = _make_pairs(tmp_path, n_pairs)
    rec = _install_fakes(monkeypatch, **fakes_kw)
    argv = sys.argv
    sys.argv = ["encode_pairs.py", "--pairs", str(pairs_path),
                "--out", str(tmp_path / "latents"), *extra]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    return code, rec, rows


# ------------------------------------------------------------------ tests


def test_fresh_run_loads_pipeline_and_encodes_win_then_lose(
        monkeypatch, tmp_path, capsys):
    code, rec, rows = _run(monkeypatch, tmp_path, todo=[0], reusable=[1, 2])
    assert code == 0
    assert rec["calls"] == ["plan", "run"]  # no ledger I/O without the flag
    # pipeline from the model's repo in bf16; VAE hoisted to cuda/f32 once
    assert rec["pipe_loads"] == [{"repo_id": MODEL_REPO,
                                  "torch_dtype": "bf16-sentinel"}]
    assert rec["vae_moves"] == [("cuda", "f32-sentinel")]
    # win encoded before lose, each saved to the planner-chosen dest
    assert rec["load_video_calls"] == [rows[0]["win_video"],
                                       rows[0]["lose_video"]]
    assert [s[0] for s in rec["saves"]] == ["latent0", "latent1"]
    assert [s[1].name for s in rec["saves"]] == ["pair0000_win.pt",
                                                 "pair0000_lose.pt"]
    # manifest: reused rows echoed, fresh row records both latent paths
    manifest = [json.loads(line) for line in
                rec["plan"].manifest_path.read_text().splitlines()]
    assert [m["index"] for m in manifest] == [0, 1, 2]
    assert manifest[0]["reused"] is False
    assert manifest[0]["win_latent"].endswith("pair0000_win.pt")
    assert manifest[0]["lose_latent"].endswith("pair0000_lose.pt")
    assert [m["reused"] for m in manifest[1:]] == [True, True]
    summary = json.loads(rec["plan"].summary_path.read_text())
    assert summary == {"model": "ti2v-5b", "encoded": 1, "reused": 2,
                       "failed": 0}
    out = capsys.readouterr().out
    assert "3 pairs: 2 cached, 1 to encode" in out
    assert "wrote manifest ->" in out
    assert "summary: " in out


def test_encode_video_scales_frames_and_normalizes_latent(
        monkeypatch, tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, todo=[0])
    assert code == 0
    ops = rec["ops"]
    # frames: 2 per video, fp32, stacked, one tensor per video (win + lose)
    assert rec["asarray_calls"] == [("f0", "np-f32"), ("f1", "np-f32")] * 2
    assert rec["stack_calls"] == 2
    assert rec["from_numpy_calls"] == 2
    # pixels -> model input: NCHW permute, batch dim, /127.5-1, cuda fp32
    assert [op for op in ops if op[0] == "x0"] == [
        ("x0", "permute", (3, 0, 1, 2)),
        ("x0", "unsqueeze", (0,)),
        ("x0", "truediv", (127.5,)),
        ("x0", "sub", (1.0,)),
        ("x0", "to", ("cuda", "f32-sentinel")),
    ]
    assert rec["vae_encode_inputs"] == ["x0", "x1"]
    # Wan normalization: (latent - mean) * (1/std), stats from vae.config
    assert ("torch", "tensor", tuple(MEAN)) in ops
    assert ("torch", "tensor", tuple(STD)) in ops
    assert ("mean", "view", (1, ZDIM, 1, 1, 1)) in ops
    assert ("std", "view", (1, ZDIM, 1, 1, 1)) in ops
    assert ("mean", "to", ("fake-device",)) in ops
    assert ("std", "rtruediv", (1.0,)) in ops
    assert [op for op in ops if op[0] == "latent0"] == [
        ("latent0", "sub", ("mean",)),
        ("latent0", "mul", ("std",)),
        ("latent0", "cpu", ()),
    ]


def test_fully_cached_resume_never_imports_torch_or_diffusers(
        monkeypatch, tmp_path):
    pairs_path, _rows = _make_pairs(tmp_path)
    _install_fakes(monkeypatch, todo=[], reusable=[0, 1, 2])
    # tripwires: any `import torch` / `from diffusers import ...` raises
    for mod in ("torch", "diffusers", "diffusers.utils", "numpy"):
        monkeypatch.setitem(sys.modules, mod, None)
    argv = sys.argv
    sys.argv = ["encode_pairs.py", "--pairs", str(pairs_path),
                "--out", str(tmp_path / "latents")]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    assert code == 0


def test_fully_cached_resume_passes_encode_fn_none(monkeypatch, tmp_path):
    code, rec, _ = _run(
        monkeypatch, tmp_path, todo=[], reusable=[0, 1, 2],
        run_summary={"model": "ti2v-5b", "encoded": 0, "reused": 3,
                     "failed": 0})
    assert code == 0
    assert rec["run_args"]["encode_fn"] is None


def test_smoke_flag_truncates_to_two(monkeypatch, tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--smoke",), todo=[0])
    assert code == 0
    assert [p["prompt_id"] for p in rec["plan_args"]["pairs"]] == ["p0", "p1"]


def test_limit_truncates(monkeypatch, tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--limit", "1"),
                        todo=[0])
    assert code == 0
    assert [p["prompt_id"] for p in rec["plan_args"]["pairs"]] == ["p0"]


def test_limit_zero_exit_1_before_planner(monkeypatch, tmp_path, capsys):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--limit", "0"),
                        todo=[])
    assert code == 1
    assert "no pairs to encode" in capsys.readouterr().err
    assert "plan_args" not in rec  # planner never reached


def test_missing_pairs_file_exit_1(monkeypatch, tmp_path, capsys):
    rec = _install_fakes(monkeypatch)
    argv = sys.argv
    sys.argv = ["encode_pairs.py", "--pairs", str(tmp_path / "nope.jsonl"),
                "--out", str(tmp_path / "latents")]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    assert code == 1
    assert capsys.readouterr().err.strip()  # OSError message printed
    assert "plan_args" not in rec  # preflight never reached


def test_unusable_pairs_refused_before_planning(monkeypatch, tmp_path,
                                                capsys):
    # real pairs_check: a row missing keys rejects the whole file (the
    # cache is index-keyed, so rows are never dropped in place)
    code, rec, _ = _run(monkeypatch, tmp_path, todo=[0])
    assert code == 0  # sanity: the helper's own rows are usable
    pairs_path = tmp_path / "pairs.jsonl"
    pairs_path.write_text(
        pairs_path.read_text(encoding="utf-8")
        + json.dumps({"prompt_id": "p9", "prompt": "x",
                      "win_video": "w.mp4"})
        + "\n",
        encoding="utf-8",
    )
    rec = _install_fakes(monkeypatch)
    argv = sys.argv
    sys.argv = ["encode_pairs.py", "--pairs", str(pairs_path),
                "--out", str(tmp_path / "latents2")]
    try:
        code = ep.main()
    finally:
        sys.argv = argv
    assert code == 1
    err = capsys.readouterr().err
    assert "missing_keys" in err
    assert "refusing to start" in err
    assert "plan_args" not in rec  # planner never reached


def test_no_resume_passes_resume_false(monkeypatch, tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--no-resume",),
                        todo=[0])
    assert code == 0
    assert rec["plan_args"]["resume"] is False


def test_max_retries_forwarded_to_plan_and_run(monkeypatch, tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--max-retries", "3"),
                        todo=[], reusable=[0, 1, 2])
    assert code == 0
    assert rec["plan_args"]["max_retries"] == 3
    assert rec["run_args"]["max_retries"] == 3


def test_reset_failures_clears_ledger_before_planning(monkeypatch, tmp_path,
                                                      capsys):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--reset-failures",),
                        todo=[], reusable=[0, 1, 2], ledger_cleared=True)
    assert code == 0
    assert rec["calls"] == ["reset", "plan", "run"]  # cleared pre-plan
    assert rec["reset_out"] == tmp_path / "latents"
    assert "cleared failure ledger" in capsys.readouterr().out


def test_strict_forwarded(monkeypatch, tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--strict",),
                        todo=[], reusable=[0, 1, 2])
    assert code == 0
    assert rec["run_args"]["strict"] is True


def test_failed_pairs_keep_exit_0_with_stderr_hint(monkeypatch, tmp_path,
                                                   capsys):
    code, _rec, _rows = _run(
        monkeypatch, tmp_path, todo=[], reusable=[0, 1, 2],
        run_summary={"model": "ti2v-5b", "encoded": 0, "reused": 2,
                     "failed": 1})
    assert code == 0  # documented contract: failures retried on next run
    err = capsys.readouterr().err
    assert "1 pair(s) failed to encode" in err
    assert "rerun to retry" in err


def test_failed_hint_mentions_reset_when_budget_capped(monkeypatch, tmp_path,
                                                       capsys):
    code, _rec, _rows = _run(
        monkeypatch, tmp_path, extra=("--max-retries", "3"),
        todo=[], reusable=[0, 1, 2],
        run_summary={"model": "ti2v-5b", "encoded": 0, "reused": 2,
                     "failed": 1})
    assert code == 0
    assert "--reset-failures" in capsys.readouterr().err


def test_cache_invalidated_message(monkeypatch, tmp_path, capsys):
    code, _rec, _rows = _run(monkeypatch, tmp_path, todo=[0],
                             cache_invalidated=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "cache model mismatch" in out
    assert "re-encoding everything" in out


def test_skipped_budget_exhausted_suffix_in_counts(monkeypatch, tmp_path,
                                                   capsys):
    code, _rec, _rows = _run(
        monkeypatch, tmp_path, todo=[], reusable=[0, 1], skipped=[2],
        run_summary={"model": "ti2v-5b", "encoded": 0, "reused": 2,
                     "failed": 0})
    assert code == 0
    assert "1 skipped (retry budget exhausted)" in capsys.readouterr().out


def test_model_flag_selects_planner_namespace_and_repo(monkeypatch,
                                                       tmp_path):
    code, rec, _ = _run(monkeypatch, tmp_path, extra=("--model", "fake-big"),
                        todo=[0])
    assert code == 0
    # cache is tied to the model (normalization is VAE-specific)…
    assert rec["plan_args"]["model"] == "fake-big"
    # …and the pipeline comes from that model's repo
    assert rec["pipe_loads"] == [{"repo_id": OTHER_REPO,
                                  "torch_dtype": "bf16-sentinel"}]

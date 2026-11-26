# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/encode_texts.py (P1b text-cache pre-encode CLI).

The module layer (theama.data.text_cache) is covered by
tests/test_m2_text_cache.py; this file pins the *CLI contract* of the
script itself — the part that decides whether a crashed GPU encode run
resumes cheaply or burns a model load for nothing:

- the text encoder (torch + transformers) is imported and loaded **only**
  when the plan has work to do — a fully cached rerun never pays the
  load cost (mirrors gen_eval_set / gen_candidates resume economics);
- --smoke / --limit truncation, zero-usable-manifest exit 1;
- --encoder override beats --model repo resolution;
- --no-resume / --max-retries / --reset-failures / --strict pass-through;
- failed captions keep exit 0 (documented contract: retried next run,
  --strict to abort) with a stderr hint;
- encode_fn wiring: tokenizer kwargs, bf16 + cuda + eval, seq_len from
  the attention mask, save via latent_cache.save_latent_atomic.

No GPU, torch, transformers, or network required — all heavy deps are
injected fakes, mirroring tests/test_p1a_gen_candidates.py.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "encode_texts", ROOT / "scripts" / "encode_texts.py"
)
et = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(et)

MODEL_REPO = "Wan-AI/Fake-TI2V-5B"
CAPTIONS = ["a calm lake at dawn", "busy street timelapse", "night market"]


# ---------------------------------------------------------------- fakes


class _FakeMask:
    """Stands in for tokens.attention_mask: mask[0].sum().item() -> n."""

    def __init__(self, n):
        self._n = n

    def __getitem__(self, _i):
        return self

    def sum(self):
        return self

    def item(self):
        return self._n


class _FakeTokens(dict):
    """dict so text_encoder(**tokens) works; attribute access for mask."""

    def __init__(self, seq_len):
        super().__init__(input_ids=[1, 2, 3])
        self.attention_mask = _FakeMask(seq_len)

    def to(self, _device):
        return self


class _FakeEmb:
    def __init__(self, rec):
        self._rec = rec

    def cpu(self):
        self._rec["emb_cpu"] = self._rec.get("emb_cpu", 0) + 1
        return self


class _FakeEncoder:
    def __init__(self, rec):
        self._rec = rec
        self.moves = []
        self.eval_called = False

    def to(self, device):
        self.moves.append(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **kw):
        self._rec["encoder_calls"].append(kw)
        return types.SimpleNamespace(last_hidden_state=[_FakeEmb(self._rec)])


class _NoGrad:
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_fakes(monkeypatch, captions, todo, reusable=(), skipped=(),
                   cache_invalidated=False, ledger_cleared=False,
                   run_summary=None):
    """Inject fake dataset/text_cache/config/transformers/torch/latent_cache.

    Returns the recorder dict. `todo`/`reusable`/`skipped` are lists of
    sample dicts the fake plan reports; the fake runner encodes every
    todo sample through the script's real encode_fn unless run_summary
    is given (then it is returned verbatim and encode_fn is unused).
    """
    rec = {"tokenizer_calls": [], "encoder_calls": [], "saves": []}

    dataset = types.ModuleType("theama.data.dataset")

    def load_manifest(path):
        rec["manifest_arg"] = path
        report = types.SimpleNamespace(
            samples=[types.SimpleNamespace(caption=c) for c in captions],
        )
        report.summary = lambda: f"{len(captions)} usable"
        return report

    dataset.load_manifest = load_manifest

    text_cache = types.ModuleType("theama.data.text_cache")

    def plan_text_encoding(samples, out, encoder_id, resume=True,
                           max_retries=None):
        rec["plan_args"] = {
            "samples": list(samples),
            "out": Path(out),
            "encoder_id": encoder_id,
            "resume": resume,
            "max_retries": max_retries,
        }
        plan = types.SimpleNamespace(
            reusable=list(reusable),
            todo=list(todo),
            skipped=list(skipped),
            cache_invalidated=cache_invalidated,
            manifest_path=Path(out) / "text_latents.manifest.jsonl",
            summary_path=Path(out) / "text_latents.summary.json",
        )
        rec["plan"] = plan
        return plan

    def reset_failure_ledger(out):
        rec["reset_out"] = Path(out)
        return ledger_cleared

    def run_text_encoding(plan, encode_fn, strict=False, progress=None,
                          max_retries=None):
        rec["run_args"] = {
            "encode_fn": encode_fn,
            "strict": strict,
            "max_retries": max_retries,
        }
        if run_summary is not None:
            return run_summary
        rows = []
        for i, sample in enumerate(plan.todo):
            dest = plan.manifest_path.parent / f"text_{i:016x}.pt"
            seq_len = encode_fn(sample, dest)
            rows.append({"caption": sample["caption"], "seq_len": seq_len})
        plan.manifest_path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
        summary = {
            "encoded": len(rows),
            "reused": len(plan.reusable),
            "failed": 0,
        }
        plan.summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return summary

    text_cache.plan_text_encoding = plan_text_encoding
    text_cache.reset_failure_ledger = reset_failure_ledger
    text_cache.run_text_encoding = run_text_encoding

    config = types.ModuleType("theama.inference.config")
    config.MODELS = {"ti2v-5b": types.SimpleNamespace(repo_id=MODEL_REPO)}

    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(repo, subfolder=None):
            rec["tokenizer_load"] = {"repo": repo, "subfolder": subfolder}

            def tokenize(caption, max_length=None, padding=None,
                         truncation=None, return_tensors=None):
                rec["tokenizer_calls"].append({
                    "caption": caption,
                    "max_length": max_length,
                    "padding": padding,
                    "truncation": truncation,
                    "return_tensors": return_tensors,
                })
                return _FakeTokens(seq_len=3)

            return tokenize

    class UMT5EncoderModel:
        @staticmethod
        def from_pretrained(repo, subfolder=None, torch_dtype=None):
            rec["encoder_load"] = {
                "repo": repo,
                "subfolder": subfolder,
                "torch_dtype": torch_dtype,
            }
            encoder = _FakeEncoder(rec)
            rec["encoder"] = encoder
            return encoder

    transformers.AutoTokenizer = AutoTokenizer
    transformers.UMT5EncoderModel = UMT5EncoderModel

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16-sentinel"
    torch.no_grad = _NoGrad()

    latent_cache = types.ModuleType("theama.data.latent_cache")

    def save_latent_atomic(emb, dest):
        rec["saves"].append(Path(dest))

    latent_cache.save_latent_atomic = save_latent_atomic

    monkeypatch.setitem(sys.modules, "theama.data.dataset", dataset)
    monkeypatch.setitem(sys.modules, "theama.data.text_cache", text_cache)
    monkeypatch.setitem(sys.modules, "theama.inference.config", config)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "theama.data.latent_cache", latent_cache)
    return rec


def _argv(tmp_path, *extra):
    return [
        "--manifest", str(tmp_path / "av.jsonl"),
        "--out", str(tmp_path / "texts"),
        *extra,
    ]


def _todo(captions):
    return [{"caption": c} for c in captions]


# ------------------------------------------------------------------ tests


def test_fresh_run_loads_encoder_and_encodes_every_todo(monkeypatch, tmp_path):
    rec = _install_fakes(monkeypatch, CAPTIONS, todo=_todo(CAPTIONS))
    argv = sys.argv
    sys.argv = ["encode_texts.py", *_argv(tmp_path)]
    try:
        rc = et.main()
    finally:
        sys.argv = argv
    assert rc == 0
    # encoder loaded from the model's repo, right subfolders, bf16, cuda, eval
    assert rec["tokenizer_load"] == {"repo": MODEL_REPO,
                                     "subfolder": "tokenizer"}
    assert rec["encoder_load"] == {"repo": MODEL_REPO,
                                   "subfolder": "text_encoder",
                                   "torch_dtype": "bf16-sentinel"}
    assert rec["encoder"].moves == ["cuda"]
    assert rec["encoder"].eval_called
    # every todo sample encoded + saved, seq_len recorded in the manifest
    assert len(rec["saves"]) == 3
    assert rec["emb_cpu"] == 3
    rows = [json.loads(line) for line in
            rec["plan"].manifest_path.read_text().splitlines()]
    assert [r["caption"] for r in rows] == CAPTIONS
    assert [r["seq_len"] for r in rows] == [3, 3, 3]
    summary = json.loads(rec["plan"].summary_path.read_text())
    assert summary == {"encoded": 3, "reused": 0, "failed": 0}


def _run(monkeypatch, tmp_path, captions, todo, extra=(), **kw):
    rec = _install_fakes(monkeypatch, captions, todo=todo, **kw)
    argv = sys.argv
    sys.argv = ["encode_texts.py", *_argv(tmp_path, *extra)]
    try:
        code = et.main()
    finally:
        sys.argv = argv
    return code, rec


def test_fully_cached_resume_never_imports_torch_or_transformers(
        monkeypatch, tmp_path):
    _install_fakes(monkeypatch, CAPTIONS, todo=[], reusable=_todo(CAPTIONS))
    # tripwires: any `import torch` / `from transformers import ...` raises
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    argv = sys.argv
    sys.argv = ["encode_texts.py", *_argv(tmp_path)]
    try:
        code = et.main()
    finally:
        sys.argv = argv
    assert code == 0


def test_fully_cached_resume_passes_encode_fn_none(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["run_args"]["encode_fn"] is None


def test_smoke_flag_truncates_to_two(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS,
                     todo=_todo(CAPTIONS[:2]), extra=("--smoke",))
    assert code == 0
    assert [s["caption"] for s in rec["plan_args"]["samples"]] == CAPTIONS[:2]


def test_limit_truncates(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS,
                     todo=_todo(CAPTIONS[:1]), extra=("--limit", "1"))
    assert code == 0
    assert [s["caption"] for s in rec["plan_args"]["samples"]] == CAPTIONS[:1]


def test_zero_usable_samples_exit_1_before_planning(monkeypatch, tmp_path,
                                                    capsys):
    code, rec = _run(monkeypatch, tmp_path, [], todo=[])
    assert code == 1
    assert "plan_args" not in rec  # planner never reached
    assert "zero usable samples" in capsys.readouterr().err


def test_encoder_override_beats_model_repo(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS), extra=("--encoder", "org/custom-t5"),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["plan_args"]["encoder_id"] == "org/custom-t5"


def test_default_encoder_comes_from_model_repo(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["plan_args"]["encoder_id"] == MODEL_REPO


def test_no_resume_passes_resume_false(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS), extra=("--no-resume",),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["plan_args"]["resume"] is False


def test_max_retries_forwarded_to_plan_and_run(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS), extra=("--max-retries", "3"),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["plan_args"]["max_retries"] == 3
    assert rec["run_args"]["max_retries"] == 3


def test_reset_failures_clears_ledger_and_reports(monkeypatch, tmp_path,
                                                  capsys):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS), ledger_cleared=True,
                     extra=("--reset-failures",),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["reset_out"] == tmp_path / "texts"
    assert "cleared failure ledger" in capsys.readouterr().out


def test_failed_samples_keep_exit_0_with_stderr_hint(monkeypatch, tmp_path,
                                                     capsys):
    code, _ = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                   reusable=_todo(CAPTIONS),
                   run_summary={"encoded": 0, "reused": 2, "failed": 1})
    assert code == 0  # documented contract: failures retried on next run
    err = capsys.readouterr().err
    assert "1 sample(s) failed to encode" in err
    assert "rerun to retry" in err


def test_failed_hint_mentions_reset_when_budget_capped(monkeypatch, tmp_path,
                                                       capsys):
    code, _ = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                   reusable=_todo(CAPTIONS), extra=("--max-retries", "3"),
                   run_summary={"encoded": 0, "reused": 2, "failed": 1})
    assert code == 0
    assert "--reset-failures" in capsys.readouterr().err


def test_strict_forwarded(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                     reusable=_todo(CAPTIONS), extra=("--strict",),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["run_args"]["strict"] is True


def test_cache_invalidated_message(monkeypatch, tmp_path, capsys):
    code, _ = _run(monkeypatch, tmp_path, CAPTIONS, todo=_todo(CAPTIONS),
                   cache_invalidated=True)
    assert code == 0
    assert "re-encoding everything" in capsys.readouterr().out


def test_skipped_budget_exhausted_suffix_in_counts(monkeypatch, tmp_path,
                                                   capsys):
    code, _ = _run(monkeypatch, tmp_path, CAPTIONS, todo=[],
                   reusable=_todo(CAPTIONS[:2]),
                   skipped=_todo(CAPTIONS[2:]),
                   run_summary={"encoded": 0, "reused": 2, "failed": 0})
    assert code == 0
    assert "retry budget exhausted" in capsys.readouterr().out


def test_encode_fn_tokenizer_kwargs_and_max_length(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, CAPTIONS[:1],
                     todo=_todo(CAPTIONS[:1]), extra=("--max-length", "64"))
    assert code == 0
    call = rec["tokenizer_calls"][0]
    assert call == {
        "caption": CAPTIONS[0],
        "max_length": 64,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }

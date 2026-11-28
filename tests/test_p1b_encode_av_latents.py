# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/encode_av_latents.py (P1b AV-latent pre-encode CLI).

The module layer (theama.data.av_latent_cache) is covered by
tests/test_p1b_av_latent_cache.py; this file pins the *CLI contract* of
the script itself — the part that decides whether a crashed GPU encode
run resumes cheaply or pays a Wan VAE + DAC load for nothing:

- torch / diffusers (WanPipeline) / DAC codec / decode helpers are
  imported and loaded **only** when the plan has work to do — a fully
  cached rerun never pays the load cost (mirrors encode_texts.py);
- --smoke / --limit truncation, zero-usable-manifest exit 1;
- --no-resume / --max-retries / --reset-failures / --strict pass-through;
- failed samples keep exit 0 (documented contract: retried next run,
  --strict to abort) with a stderr hint that names --reset-failures when
  the retry budget is capped;
- encode_fn wiring: decode_video fps/seconds/height/width, Wan VAE
  per-channel latents_mean / latents_std normalization, DAC pad_to_hop +
  encode_stereo mid+side tokens, both latents saved via
  latent_cache.save_latent_atomic.

No GPU, torch, diffusers, or network required — all heavy deps are
injected fakes, mirroring tests/test_m2_encode_texts.py.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "encode_av_latents", ROOT / "scripts" / "encode_av_latents.py"
)
eav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eav)

MODEL_REPO = "Wan-AI/Fake-TI2V-5B"
SAMPLES = [
    {"video": "/data/v0.mp4", "audio": "/data/a0.wav",
     "caption": "a calm lake at dawn", "source": "test"},
    {"video": "/data/v1.mp4", "audio": "/data/a1.wav",
     "caption": "busy street timelapse", "source": "test"},
    {"video": "/data/v2.mp4", "audio": "/data/a2.wav",
     "caption": "night market", "source": "test"},
]
Z_DIM = 16


# ---------------------------------------------------------------- fakes


class _FakeTensor:
    """Permissive tensor stand-in that records the ops the script applies."""

    def __init__(self, rec, tag):
        self._rec = rec
        self.tag = tag
        self.device = "cpu"

    def _rec_op(self, name, *args):
        self._rec.setdefault("tensor_ops", []).append((self.tag, name, args))
        return self

    def permute(self, *dims):
        return self._rec_op("permute", *dims)

    def unsqueeze(self, dim):
        return self._rec_op("unsqueeze", dim)

    def view(self, *shape):
        return self._rec_op("view", *shape)

    def float(self):
        return self._rec_op("float")

    def to(self, *args, **kwargs):
        return self._rec_op("to", *args, *kwargs.values())

    def cpu(self):
        self._rec.setdefault("cpu_calls", []).append(self.tag)
        return self

    def __truediv__(self, other):
        return self._rec_op("truediv", other)

    def __rtruediv__(self, other):
        return self._rec_op("rtruediv", other)

    def __sub__(self, other):
        return self._rec_op("sub", getattr(other, "tag", other))

    def __rsub__(self, other):
        return self._rec_op("rsub", other)

    def __mul__(self, other):
        return self._rec_op("mul", getattr(other, "tag", other))


class _NoGrad:
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeVAE:
    class config:
        latents_mean = [0.15] * Z_DIM
        latents_std = [0.5] * Z_DIM
        z_dim = Z_DIM

    def __init__(self, rec):
        self._rec = rec

    def to(self, device, dtype):
        self._rec["vae_moves"].append((device, dtype))
        return self

    def encode(self, x):
        self._rec["vae_encode_input"] = x
        latent = _FakeTensor(self._rec, "video_latent")
        return types.SimpleNamespace(
            latent_dist=types.SimpleNamespace(sample=lambda: latent)
        )


def _install_fakes(monkeypatch, samples, todo, reusable=(), skipped=(),
                   cache_invalidated=False, ledger_cleared=False,
                   run_summary=None):
    """Inject fake dataset/av_latent_cache/torch/diffusers/codec modules.

    Returns the recorder dict. `todo`/`reusable`/`skipped` are lists of
    sample dicts the fake plan reports; the fake runner encodes every
    todo sample through the script's real encode_fn unless run_summary
    is given (then it is returned verbatim and encode_fn is unused).
    """
    rec = {"saves": [], "vae_moves": [], "tensor_ops": [], "cpu_calls": []}

    dataset = types.ModuleType("theama.data.dataset")

    def load_manifest(path):
        rec["manifest_arg"] = path
        report = types.SimpleNamespace(
            samples=[
                types.SimpleNamespace(
                    video_path=s["video"], audio_path=s["audio"],
                    caption=s["caption"], source=s["source"],
                )
                for s in samples
            ],
        )
        report.summary = lambda: f"{len(samples)} usable"
        return report

    dataset.load_manifest = load_manifest

    av_latent_cache = types.ModuleType("theama.data.av_latent_cache")

    def plan_av_encoding(samples_, out, model, resume=True, max_retries=None):
        rec["plan_args"] = {
            "samples": list(samples_),
            "out": Path(out),
            "model": model,
            "resume": resume,
            "max_retries": max_retries,
        }
        plan = types.SimpleNamespace(
            samples=list(samples_),
            reusable=list(reusable),
            todo=list(todo),
            skipped=list(skipped),
            cache_invalidated=cache_invalidated,
            out_dir=Path(out),
            model=model,
            manifest_path=Path(out) / "av_latents.manifest.jsonl",
            summary_path=Path(out) / "av_latents.summary.json",
        )
        rec["plan"] = plan
        return plan

    def reset_failure_ledger(out):
        rec["reset_out"] = Path(out)
        return ledger_cleared

    def run_av_encoding(plan, encode_fn, strict=False, progress=None,
                        max_retries=None):
        rec["run_args"] = {
            "encode_fn": encode_fn,
            "strict": strict,
            "max_retries": max_retries,
        }
        if run_summary is not None:
            return run_summary
        rows = []
        for index, _entry in enumerate(plan.todo):
            video_dest = plan.out_dir / f"sample{index:04d}_video.pt"
            audio_dest = plan.out_dir / f"sample{index:04d}_audio.pt"
            encode_fn(plan.samples[index], video_dest, audio_dest)
            rows.append({
                "index": index,
                "caption": plan.samples[index]["caption"],
                "video": plan.samples[index]["video"],
                "audio": plan.samples[index]["audio"],
                "video_latent": str(video_dest),
                "audio_latent": str(audio_dest),
            })
            if progress:
                progress(f"encoded sample {index}")
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

    av_latent_cache.plan_av_encoding = plan_av_encoding
    av_latent_cache.reset_failure_ledger = reset_failure_ledger
    av_latent_cache.run_av_encoding = run_av_encoding

    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16-sentinel"
    torch.float32 = "f32-sentinel"
    torch.no_grad = _NoGrad()

    def from_numpy(arr):
        return _FakeTensor(rec, "ndarray")

    torch.from_numpy = from_numpy

    def tensor(data):
        rec.setdefault("tensor_ctor", []).append(list(data))
        return _FakeTensor(rec, "meanstd")

    torch.tensor = tensor

    av_latent_smoke = types.ModuleType("theama.data.av_latent_smoke")

    def pad_to_hop(wave, hop_length):
        rec["pad_to_hop"] = {"hop_length": hop_length, "wave": wave}
        return wave

    av_latent_smoke.pad_to_hop = pad_to_hop

    decode = types.ModuleType("theama.data.decode")

    def decode_video(path, fps=None, seconds=None, height=None, width=None):
        rec.setdefault("decode_video_calls", []).append({
            "path": path, "fps": fps, "seconds": seconds,
            "height": height, "width": width,
        })
        return object()

    def decode_audio(path, sample_rate=None, seconds=None):
        rec.setdefault("decode_audio_calls", []).append({
            "path": path, "sample_rate": sample_rate, "seconds": seconds,
        })
        return object()

    decode.decode_video = decode_video
    decode.decode_audio = decode_audio

    latent_cache = types.ModuleType("theama.data.latent_cache")

    def save_latent_atomic(t, dest):
        rec["saves"].append((t.tag, Path(dest)))

    latent_cache.save_latent_atomic = save_latent_atomic

    config = types.ModuleType("theama.inference.config")
    config.MODELS = {
        "ti2v-5b": types.SimpleNamespace(repo_id=MODEL_REPO),
        "wan2.2": types.SimpleNamespace(repo_id="Wan-AI/Fake-Wan2.2"),
    }

    diffusers = types.ModuleType("diffusers")

    class WanPipeline:
        @staticmethod
        def from_pretrained(repo, torch_dtype=None):
            rec["pipeline_load"] = {"repo": repo, "torch_dtype": torch_dtype}
            vae = _FakeVAE(rec)
            rec["vae"] = vae
            return types.SimpleNamespace(vae=vae)

    diffusers.WanPipeline = WanPipeline

    audio_codec = types.ModuleType("theama.model.audio_codec")

    def get_codec_spec(name):
        rec["codec_name"] = name
        return types.SimpleNamespace(hop_length=512)

    def encode_stereo(codec, wave):
        rec["encode_stereo_wave"] = wave
        return _FakeTensor(rec, "audio_tokens")

    audio_codec.get_codec_spec = get_codec_spec
    audio_codec.encode_stereo = encode_stereo

    dac_codec = types.ModuleType("theama.model.dac_codec")

    class DACAudioCodec:
        @staticmethod
        def from_pretrained(device=None):
            rec["codec_load"] = {"device": device}
            codec = object()
            rec["codec"] = codec
            return codec

    dac_codec.DACAudioCodec = DACAudioCodec

    monkeypatch.setitem(sys.modules, "theama.data.dataset", dataset)
    monkeypatch.setitem(sys.modules, "theama.data.av_latent_cache",
                        av_latent_cache)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "theama.data.av_latent_smoke",
                        av_latent_smoke)
    monkeypatch.setitem(sys.modules, "theama.data.decode", decode)
    monkeypatch.setitem(sys.modules, "theama.data.latent_cache", latent_cache)
    monkeypatch.setitem(sys.modules, "theama.inference.config", config)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers)
    monkeypatch.setitem(sys.modules, "theama.model.audio_codec", audio_codec)
    monkeypatch.setitem(sys.modules, "theama.model.dac_codec", dac_codec)
    return rec



def _argv(tmp_path, *extra):
    return [
        "--manifest", str(tmp_path / "av.jsonl"),
        "--out", str(tmp_path / "latents"),
        *extra,
    ]


def _run(monkeypatch, tmp_path, samples, todo, extra=(), **kw):
    rec = _install_fakes(monkeypatch, samples, todo=todo, **kw)
    argv = sys.argv
    sys.argv = ["encode_av_latents.py", *_argv(tmp_path, *extra)]
    try:
        code = eav.main()
    finally:
        sys.argv = argv
    return code, rec


# ------------------------------------------------------------------ tests


def test_fresh_run_loads_vae_and_codec_and_encodes_every_todo(
        monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES))
    assert code == 0
    # pipeline + codec loaded once, from the model repo, right dtypes
    assert rec["pipeline_load"] == {"repo": MODEL_REPO,
                                    "torch_dtype": "bf16-sentinel"}
    assert rec["vae_moves"] == [("cuda", "f32-sentinel")]
    assert rec["codec_load"] == {"device": "cuda"}
    assert rec["codec_name"] == "dac_44k"
    # every todo sample encoded on both paths, atomically saved in order
    assert rec["saves"] == [
        ("video_latent", tmp_path / "latents" / "sample0000_video.pt"),
        ("audio_tokens", tmp_path / "latents" / "sample0000_audio.pt"),
        ("video_latent", tmp_path / "latents" / "sample0001_video.pt"),
        ("audio_tokens", tmp_path / "latents" / "sample0001_audio.pt"),
        ("video_latent", tmp_path / "latents" / "sample0002_video.pt"),
        ("audio_tokens", tmp_path / "latents" / "sample0002_audio.pt"),
    ]
    # decode kwargs carry the CLI's fps/seconds/size/sample-rate
    assert rec["decode_video_calls"][0] == {
        "path": "/data/v0.mp4", "fps": 24, "seconds": 5,
        "height": 240, "width": 432,
    }
    assert rec["decode_audio_calls"][0] == {
        "path": "/data/a0.wav", "sample_rate": 44100, "seconds": 5,
    }
    assert rec["pad_to_hop"]["hop_length"] == 512
    # both latents round-trip through .cpu() before the atomic save
    assert rec["cpu_calls"] == ["video_latent", "audio_tokens"] * 3
    rows = [json.loads(line) for line in
            rec["plan"].manifest_path.read_text().splitlines()]
    assert [r["caption"] for r in rows] == [s["caption"] for s in SAMPLES]
    summary = json.loads(rec["plan"].summary_path.read_text())
    assert summary == {"encoded": 3, "reused": 0, "failed": 0}


def test_fully_cached_resume_never_imports_torch_or_diffusers(
        monkeypatch, tmp_path):
    rec = _install_fakes(monkeypatch, SAMPLES, todo=[], reusable=list(SAMPLES),
                         run_summary={"encoded": 0, "reused": 3, "failed": 0})
    # tripwires: any `import torch` / `from diffusers import ...` raises
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "diffusers", None)
    argv = sys.argv
    sys.argv = ["encode_av_latents.py", *_argv(tmp_path)]
    try:
        code = eav.main()
    finally:
        sys.argv = argv
    assert code == 0
    assert rec["run_args"]["encode_fn"] is None
    assert rec["saves"] == []
    assert rec.get("pipeline_load") is None
    assert rec.get("codec_load") is None


def test_fully_cached_resume_passes_encode_fn_none(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=[],
                     reusable=list(SAMPLES),
                     run_summary={"encoded": 0, "reused": 3, "failed": 0})
    assert code == 0
    assert rec["run_args"]["encode_fn"] is None


def test_smoke_truncates_to_two_samples(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES[:2]),
                     extra=("--smoke",))
    assert code == 0
    assert [s["caption"] for s in rec["plan_args"]["samples"]] == [
        "a calm lake at dawn", "busy street timelapse",
    ]


def test_limit_truncates_samples(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES[:2]),
                     extra=("--limit", "2"))
    assert code == 0
    assert len(rec["plan_args"]["samples"]) == 2


def test_zero_usable_samples_exit_1_before_planning(monkeypatch, tmp_path,
                                                    capsys):
    code, rec = _run(monkeypatch, tmp_path, [], todo=[])
    assert code == 1
    assert "plan_args" not in rec  # planner never reached
    assert "zero usable samples" in capsys.readouterr().err


def test_no_resume_passes_resume_false(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                     extra=("--no-resume",))
    assert code == 0
    assert rec["plan_args"]["resume"] is False


def test_max_retries_forwarded_to_plan_and_run(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                     extra=("--max-retries", "3"))
    assert code == 0
    assert rec["plan_args"]["max_retries"] == 3
    assert rec["run_args"]["max_retries"] == 3
    assert "reset_out" not in rec  # no --reset-failures: ledger untouched


def test_reset_failures_clears_ledger_and_reports(monkeypatch, tmp_path,
                                                  capsys):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                     ledger_cleared=True, extra=("--reset-failures",))
    assert code == 0
    assert rec["reset_out"] == tmp_path / "latents"
    assert "cleared failure ledger" in capsys.readouterr().out


def test_strict_passed_to_runner(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                     extra=("--strict",))
    assert code == 0
    assert rec["run_args"]["strict"] is True


def test_failed_samples_keep_exit_0_with_stderr_hint(monkeypatch, tmp_path,
                                                     capsys):
    code, _ = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                   run_summary={"encoded": 2, "reused": 0, "failed": 1})
    assert code == 0  # documented contract: failures retried on next run
    err = capsys.readouterr().err
    assert "1 sample(s) failed to encode" in err
    assert "rerun to retry" in err
    assert "--reset-failures" not in err  # budget uncapped: no reset hint


def test_failed_hint_mentions_reset_when_budget_capped(monkeypatch, tmp_path,
                                                       capsys):
    code, _ = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                   extra=("--max-retries", "3"),
                   run_summary={"encoded": 2, "reused": 0, "failed": 1})
    assert code == 0
    assert "--reset-failures" in capsys.readouterr().err


def test_cache_invalidated_reports_model_mismatch(monkeypatch, tmp_path,
                                                  capsys):
    code, _ = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                   cache_invalidated=True)
    assert code == 0
    out = capsys.readouterr().out
    assert "cache model mismatch" in out
    assert "re-encoding everything" in out


def test_skipped_suffix_mentions_retry_budget(monkeypatch, tmp_path, capsys):
    code, _ = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES[:2]),
                   skipped=[SAMPLES[2]],
                   run_summary={"encoded": 2, "reused": 0, "failed": 0})
    assert code == 0
    out = capsys.readouterr().out
    assert "1 skipped (retry budget exhausted)" in out


def test_video_latent_normalized_with_vae_config(monkeypatch, tmp_path):
    """Video latent must be normalized by per-channel latents_mean/std."""
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES[:1]))
    assert code == 0
    # mean then std, both from vae.config, viewed as (1, z, 1, 1, 1)
    assert rec["tensor_ctor"] == [[0.15] * Z_DIM, [0.5] * Z_DIM]
    ops = [name for _, name, _ in rec["tensor_ops"]]
    assert "view" in ops and "sub" in ops and "rtruediv" in ops
    # saved latent went through .cpu() first
    assert rec["cpu_calls"] == ["video_latent", "audio_tokens"]


def test_model_and_codec_flags_forward_to_pipeline(monkeypatch, tmp_path):
    code, rec = _run(monkeypatch, tmp_path, SAMPLES, todo=list(SAMPLES),
                     extra=("--model", "wan2.2", "--codec", "dac_24k"))
    assert code == 0
    assert rec["plan_args"]["model"] == "wan2.2"
    assert rec["pipeline_load"]["repo"] == "Wan-AI/Fake-Wan2.2"
    assert rec["codec_name"] == "dac_24k"

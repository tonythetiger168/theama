# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/gen_cjk_images.py (M7a CJK T2I image generation).

The resume contract — existing readable PNGs are skipped (the pipeline is
never loaded when nothing is pending), corrupt or partial PNGs are deleted
and regenerated, a prompt whose generation raises is skipped so one bad
prompt does not abort a 100-prompt run, and the exit code is nonzero until
every eval row has a readable PNG — decides whether a crashed GPU run
continues where it stopped or silently ships gaps into score_cjk_text.py.
A bug here costs GPU hours, so it is covered on CPU with fake
diffusers/torch modules, mirroring tests/test_p1a_gen_candidates.py.
No GPU, torch, diffusers, or model weights required; PIL writes the
fixture PNGs (Pillow is a declared dependency in pyproject.toml).

This file deliberately does not touch tests/test_m7a.py (seed schema,
model-table sync, resolution snap) or test_m7a_script_split.py (scorer
side) — zero overlap by construction.
"""

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "gen_cjk_images", ROOT / "scripts" / "gen_cjk_images.py"
)
gci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gci)

ROWS = [
    {"id": "cjk01", "category": "signage", "prompt": "夜市招牌，寫著「牛肉麵」",
     "expected_text": "牛肉麵"},
    {"id": "cjk02", "category": "poster", "prompt": "電影海報，標題「山之巔」",
     "expected_text": "山之巔"},
    {"id": "cjk03", "category": "menu", "prompt": "菜單一角，列著「珍珠奶茶」",
     "expected_text": "珍珠奶茶"},
]


def _write_eval(path: Path, rows=ROWS) -> None:
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _valid_png(path: Path) -> None:
    Image.new("RGB", (8, 8), (7, 7, 7)).save(path)


def _jsonl_rows(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _fake_diffusers(calls, fail_prompts=(), accept_guidance=True):
    """diffusers module stub: AutoPipelineForText2Image -> recording FakePipe.

    __call__ signatures are real (no **kwargs) so gci.pipe_signature sees
    exactly the parameter set a production pipeline would expose.
    """
    mod = types.ModuleType("diffusers")

    class FakeImage:
        def save(self, path):
            Image.new("RGB", (8, 8), (2, 4, 6)).save(path)

    def record(prompt, kwargs):
        calls["gen"].append((prompt, kwargs))
        if prompt in fail_prompts:
            raise RuntimeError("diffusion boom")
        return types.SimpleNamespace(images=[FakeImage()])

    if accept_guidance:
        class FakePipe:
            def __call__(self, prompt, height, width, num_inference_steps,
                         generator, guidance_scale):
                return record(prompt, {
                    "height": height, "width": width,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator, "guidance_scale": guidance_scale,
                })
    else:
        class FakePipe:
            def __call__(self, prompt, height, width, num_inference_steps,
                         generator):
                return record(prompt, {
                    "height": height, "width": width,
                    "num_inference_steps": num_inference_steps,
                    "generator": generator,
                })

    FakePipe.to = lambda self, device: self
    FakePipe.enable_model_cpu_offload = lambda self: None

    class AutoPipelineForText2Image:
        @staticmethod
        def from_pretrained(repo_id, torch_dtype=None):
            calls["load"] += 1
            calls["repo_id"] = repo_id
            calls["torch_dtype"] = torch_dtype
            return FakePipe()

    mod.AutoPipelineForText2Image = AutoPipelineForText2Image
    return mod


def _run_main(monkeypatch, tmp_path, rows=ROWS, extra_args=(), fail_prompts=(),
              accept_guidance=True, model="qwen-image-2512", pre_valid=(),
              pre_corrupt=()):
    """Run gci.main() with fake diffusers/torch and a sentinel _generator."""
    calls = {"load": 0, "gen": []}
    tmp_path.mkdir(parents=True, exist_ok=True)  # allow tmp_path / "a" subdirs
    monkeypatch.setitem(sys.modules, "diffusers",
                        _fake_diffusers(calls, fail_prompts, accept_guidance))
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    monkeypatch.setitem(sys.modules, "torch", torch)
    # The real _generator needs a CUDA device; a sentinel records (seed, id).
    monkeypatch.setattr(gci, "_generator", lambda seed, pid: ("gen", seed, pid))

    eval_path = tmp_path / "eval.jsonl"
    _write_eval(eval_path, rows)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for pid in pre_valid:
        _valid_png(out_dir / f"{pid}.png")
    for pid in pre_corrupt:
        (out_dir / f"{pid}.png").write_text("not a png", encoding="utf-8")
    argv = [
        "gen_cjk_images.py", "--model", model,
        "--eval-set", str(eval_path), "--out", str(out_dir),
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    code = gci.main()
    return code, calls, out_dir


# ------------------------------------------------------------------- png_ok


def test_png_ok_accepts_valid_png(tmp_path):
    p = tmp_path / "ok.png"
    _valid_png(p)
    assert gci.png_ok(p) is True


def test_png_ok_rejects_missing_and_empty(tmp_path):
    assert gci.png_ok(tmp_path / "nope.png") is False
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    assert gci.png_ok(empty) is False


def test_png_ok_rejects_corrupt_and_non_png(tmp_path):
    good = tmp_path / "good.png"
    _valid_png(good)
    truncated = tmp_path / "trunc.png"
    truncated.write_bytes(good.read_bytes()[:20])  # header only, no IEND
    assert gci.png_ok(truncated) is False
    text = tmp_path / "text.png"
    text.write_text("definitely not a png", encoding="utf-8")
    assert gci.png_ok(text) is False


# ------------------------------------------------------------ pipe_signature


def test_pipe_signature_reflects_call_parameters():
    class WithGuidance:
        def __call__(self, prompt, height, width, num_inference_steps,
                     generator, guidance_scale):
            return None

    class NoGuidance:
        def __call__(self, prompt, height, width, num_inference_steps,
                     generator):
            return None

    assert "guidance_scale" in gci.pipe_signature(WithGuidance())
    assert "guidance_scale" not in gci.pipe_signature(NoGuidance())


def test_pipe_signature_uninspectable_returns_empty(monkeypatch):
    import inspect as _inspect

    def boom(_target):
        raise ValueError("no signature")

    monkeypatch.setattr(_inspect, "signature", boom)
    # A plain callable so __call__ exists; the patched inspect.signature
    # raises ValueError, which pipe_signature must degrade to an empty set.
    assert gci.pipe_signature(lambda: None) == set()


# ------------------------------------------------------------- resume / CLI


def test_fresh_run_generates_all_and_writes_manifest_and_summary(
        monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(monkeypatch, tmp_path)
    assert code == 0
    assert calls["load"] == 1
    assert calls["repo_id"] == "Qwen/Qwen-Image-2512"
    assert calls["torch_dtype"] == "bf16"
    assert [p for p, _ in calls["gen"]] == [r["prompt"] for r in ROWS]
    manifest = _jsonl_rows(out_dir / "manifest.jsonl")
    assert [m["id"] for m in manifest] == ["cjk01", "cjk02", "cjk03"]
    for m in manifest:
        assert m["model"] == "qwen-image-2512"
        assert m["repo_id"] == "Qwen/Qwen-Image-2512"
        assert m["seed"] == 42
        # Model-card defaults applied (qwen-image-2512: 30 steps, CFG 3.5).
        assert m["steps"] == 30
        assert m["guidance"] == 3.5
        assert (m["width"], m["height"]) == (1024, 1024)
        assert isinstance(m["seconds_wall"], float)
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 3
    assert summary["ok"] == 3
    assert summary["failed"] == []


def test_full_resume_never_loads_pipeline(monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path, pre_valid=("cjk01", "cjk02", "cjk03")
    )
    assert code == 0
    assert calls["load"] == 0          # lazy: nothing pending -> no model load
    assert calls["gen"] == []
    assert _jsonl_rows(out_dir / "manifest.jsonl") == []
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] == 3


def test_corrupt_png_is_deleted_and_regenerated(monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path,
        pre_valid=("cjk01", "cjk03"), pre_corrupt=("cjk02",),
    )
    assert code == 0
    assert [p for p, _ in calls["gen"]] == [ROWS[1]["prompt"]]  # only cjk02
    assert gci.png_ok(out_dir / "cjk02.png") is True            # real PNG now
    assert [m["id"] for m in _jsonl_rows(out_dir / "manifest.jsonl")] == ["cjk02"]


def test_generation_failure_keeps_mining_and_final_gate_fails(
        monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path, fail_prompts=(ROWS[1]["prompt"],),
    )
    assert code == 1                                   # final gate reports the gap
    assert [p for p, _ in calls["gen"]] == [r["prompt"] for r in ROWS]  # kept mining
    assert gci.png_ok(out_dir / "cjk01.png") is True
    assert not (out_dir / "cjk02.png").exists()        # no partial write left behind
    assert gci.png_ok(out_dir / "cjk03.png") is True
    assert [m["id"] for m in _jsonl_rows(out_dir / "manifest.jsonl")] == [
        "cjk01", "cjk03"]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["ok"] == 2
    assert summary["failed"] == ["cjk02"]


def test_guidance_passed_only_when_pipeline_accepts_it(monkeypatch, tmp_path):
    _, calls_with, _ = _run_main(monkeypatch, tmp_path / "a", accept_guidance=True)
    assert all(kw["guidance_scale"] == 3.5 for _, kw in calls_with["gen"])
    _, calls_without, _ = _run_main(
        monkeypatch, tmp_path / "b", accept_guidance=False
    )
    assert all("guidance_scale" not in kw for _, kw in calls_without["gen"])


def test_per_prompt_generator_carries_seed_and_id(monkeypatch, tmp_path):
    code, calls, _ = _run_main(monkeypatch, tmp_path, extra_args=["--seed", "7"])
    assert code == 0
    assert [kw["generator"] for _, kw in calls["gen"]] == [
        ("gen", 7, "cjk01"), ("gen", 7, "cjk02"), ("gen", 7, "cjk03"),
    ]


def test_limit_truncates_rows(monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path, extra_args=["--limit", "2"]
    )
    assert code == 0
    assert [p for p, _ in calls["gen"]] == [r["prompt"] for r in ROWS[:2]]
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["ok"] == 2


def test_steps_and_guidance_overrides_recorded(monkeypatch, tmp_path):
    code, calls, out_dir = _run_main(
        monkeypatch, tmp_path, extra_args=["--steps", "7", "--guidance", "1.25"]
    )
    assert code == 0
    assert all(kw["num_inference_steps"] == 7 for _, kw in calls["gen"])
    assert all(kw["guidance_scale"] == 1.25 for _, kw in calls["gen"])
    for m in _jsonl_rows(out_dir / "manifest.jsonl"):
        assert m["steps"] == 7
        assert m["guidance"] == 1.25


def test_model_card_defaults_differ_per_model(monkeypatch, tmp_path):
    # z-image-turbo is guidance-distilled: 8 steps, CFG 0.0 per its card.
    code, calls, out_dir = _run_main(monkeypatch, tmp_path, model="z-image-turbo")
    assert code == 0
    assert calls["repo_id"] == "Tongyi-MAI/Z-Image-Turbo"
    assert all(kw["num_inference_steps"] == 8 for _, kw in calls["gen"])
    assert all(kw["guidance_scale"] == 0.0 for _, kw in calls["gen"])
    for m in _jsonl_rows(out_dir / "manifest.jsonl"):
        assert m["steps"] == 8
        assert m["guidance"] == 0.0

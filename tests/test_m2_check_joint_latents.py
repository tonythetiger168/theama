# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the P1b joint-cache preflight gate
(data/joint_cache_check.py + scripts/check_joint_latents.py).

Covers the pure manifest profiler (category precedence, zero-filled keys,
determinism), the dropped-report zero-fill, and the gate CLI contract:
clean caches exit 0 with models resolved from meta.json; dropped rows /
encode-error rows / missing manifests / zero served exit 1 with the true
cause on stderr; --allow-dropped relaxes row-level failures; and
--validate-files catches a corrupt .pt that the schema-only pass cannot
see. Fake caches are manifest + meta.json only — no encoders, no GPU.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.joint_cache_check import (  # noqa: E402
    JOIN_DROP_REASONS,
    PROFILE_KEYS,
    profile_manifest,
    zero_filled_dropped,
)

_spec = importlib.util.spec_from_file_location(
    "check_joint_latents", ROOT / "scripts" / "check_joint_latents.py"
)
cjl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cjl)

AV_MODEL = "wan2.2-vae+dac"
TXT_MODEL = "umT5-xxl"


def _av_row(i: int, model: str = AV_MODEL, caption: str | None = None) -> dict:
    return {
        "index": i,
        "model": model,
        "caption": caption if caption is not None else f"第 {i} 段",
        "source": f"pexels:{i}",
        "video_latent": f"v{i}.pt",
        "audio_latent": f"a{i}.pt",
        "reused": True,
    }


def _text_row(i: int, model: str = TXT_MODEL, caption: str | None = None) -> dict:
    return {
        "index": i,
        "model": model,
        "caption": caption if caption is not None else f"第 {i} 段",
        "text_latent": f"t{i}.pt",
        "seq_len": 4,
        "reused": True,
    }


def _to_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _write_cache(base: Path, name: str, manifest: str, model: str, rows: list[dict]):
    d = base / name
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"model": model}))
    (d / manifest).write_text(_to_text(rows), encoding="utf-8")
    return d


def _write_caches(tmp_path: Path, av_rows: list[dict], text_rows: list[dict]):
    av = _write_cache(tmp_path, "av", "av_latents.jsonl", AV_MODEL, av_rows)
    text = _write_cache(tmp_path, "text", "text_latents.jsonl", TXT_MODEL, text_rows)
    return av, text


def _run_cli(*cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_joint_latents.py"), *cli_args],
        capture_output=True,
        text=True,
    )


# ---- pure profiler -----------------------------------------------------------


def test_profile_counts_every_category():
    text = _to_text(
        [
            _av_row(0),
            {**_av_row(1), "skipped": True},
            {**_av_row(2), "error": "simulated VAE failure"},
            _av_row(3, model="older-encoder"),
            {"index": "4", "model": AV_MODEL},  # non-int index
            {"index": True, "model": AV_MODEL},  # bool index is not an int
        ]
    ) + "{bad json\n"
    prof = profile_manifest(text, AV_MODEL)
    assert prof == {
        "total_rows": 7,
        "bad_json_lines": 1,
        "error_rows": 1,
        "invalid_index_rows": 2,
        "model_mismatch_rows": 1,
        "kept_rows": 2,
        "skipped_rows": 1,
    }


def test_profile_zero_filled_on_empty():
    assert profile_manifest("", AV_MODEL) == {k: 0 for k in PROFILE_KEYS}
    assert profile_manifest("\n  \n", AV_MODEL) == {k: 0 for k in PROFILE_KEYS}


def test_profile_deterministic_fixed_key_order():
    text = _to_text([_av_row(0), {**_av_row(1), "error": "x"}])
    a, b = profile_manifest(text, AV_MODEL), profile_manifest(text, AV_MODEL)
    assert list(a) == list(PROFILE_KEYS)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_zero_filled_dropped_covers_all_six_reasons():
    filled = zero_filled_dropped({"missing_text_row": 2})
    assert list(filled) == list(JOIN_DROP_REASONS)
    assert filled["missing_text_row"] == 2
    assert sum(filled.values()) == 2
    assert zero_filled_dropped({}) == {r: 0 for r in JOIN_DROP_REASONS}


# ---- CLI ---------------------------------------------------------------------


def test_cli_clean_caches_exit_0(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(0), _av_row(1)], [_text_row(0), _text_row(1)])
    proc = _run_cli("--av-cache", str(av), "--text-cache", str(text))
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(proc.stdout)
    assert summary["served"] == 2
    assert summary["dropped"] == {r: 0 for r in JOIN_DROP_REASONS}
    assert summary["av_model"] == AV_MODEL and summary["text_model"] == TXT_MODEL
    assert summary["av_manifest"]["kept_rows"] == 2


def test_cli_dropped_row_exit_1_with_reason(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(0), _av_row(1)], [_text_row(0)])
    proc = _run_cli("--av-cache", str(av), "--text-cache", str(text))
    assert proc.returncode == 1
    summary = json.loads(proc.stdout)
    assert summary["served"] == 1
    assert summary["dropped"]["missing_text_row"] == 1
    assert "missing_text_row=1" in proc.stderr


def test_cli_allow_dropped_relaxes_to_exit_0(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(0), _av_row(1)], [_text_row(0)])
    proc = _run_cli("--av-cache", str(av), "--text-cache", str(text), "--allow-dropped")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["dropped"]["missing_text_row"] == 1


def test_cli_missing_manifest_exit_1(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(0)], [_text_row(0)])
    (av / "av_latents.jsonl").unlink()
    proc = _run_cli("--av-cache", str(av), "--text-cache", str(text))
    assert proc.returncode == 1
    assert proc.stdout == ""


def test_cli_error_row_fails_strict_relaxed_with_allow(tmp_path):
    bad = {**_av_row(1), "error": "simulated VAE failure"}
    av, text = _write_caches(tmp_path, [_av_row(0), bad], [_text_row(0), _text_row(1)])
    strict = _run_cli("--av-cache", str(av), "--text-cache", str(text))
    assert strict.returncode == 1
    assert "encode-error" in strict.stderr
    relaxed = _run_cli(
        "--av-cache", str(av), "--text-cache", str(text), "--allow-dropped"
    )
    assert relaxed.returncode == 0, relaxed.stderr
    summary = json.loads(relaxed.stdout)
    assert summary["av_manifest"]["error_rows"] == 1
    assert summary["dropped"]["missing_av_row"] == 1  # error row is invisible to the join


def test_cli_zero_served_fails_even_with_allow(tmp_path):
    av, text = _write_caches(tmp_path, [], [])
    proc = _run_cli("--av-cache", str(av), "--text-cache", str(text), "--allow-dropped")
    assert proc.returncode == 1
    assert "zero served records" in proc.stderr


def test_cli_validate_files_catches_corrupt_pt(tmp_path):
    # Manifests carry absolute latent paths (the production planners record
    # the destination paths they wrote); the dataset loads them as-is.
    av_dir, text_dir = tmp_path / "av", tmp_path / "text"
    av_row = {
        **_av_row(0),
        "video_latent": str(av_dir / "v0.pt"),
        "audio_latent": str(av_dir / "a0.pt"),
    }
    text_row = {**_text_row(0), "text_latent": str(text_dir / "t0.pt")}
    av, text = _write_caches(tmp_path, [av_row], [text_row])
    torch.save(torch.zeros(1, 4, 2, 3, 3), av / "v0.pt")
    torch.save(torch.zeros(1, 6, 16), av / "a0.pt")
    torch.save(torch.zeros(8, 16), text / "t0.pt")
    ok = _run_cli(
        "--av-cache", str(av), "--text-cache", str(text), "--validate-files"
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["validated_latents"] == 3
    (av / "a0.pt").write_bytes(b"not a tensor")
    # Schema-only pass cannot see the corruption; --validate-files must.
    assert _run_cli("--av-cache", str(av), "--text-cache", str(text)).returncode == 0
    bad = _run_cli(
        "--av-cache", str(av), "--text-cache", str(text), "--validate-files"
    )
    assert bad.returncode == 1
    assert "0:audio" in bad.stderr


def test_cli_deterministic_stdout_and_json_out(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(0)], [_text_row(0)])
    first = _run_cli("--av-cache", str(av), "--text-cache", str(text))
    second = _run_cli("--av-cache", str(av), "--text-cache", str(text))
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    out = tmp_path / "summary.json"
    third = _run_cli(
        "--av-cache", str(av), "--text-cache", str(text), "--json-out", str(out)
    )
    assert third.returncode == 0
    assert out.read_text(encoding="utf-8") == first.stdout == third.stdout

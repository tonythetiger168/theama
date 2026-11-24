# SPDX-License-Identifier: Apache-2.0
"""M1 CPU-verifiable tests: eval-set schema, model configs, MP4 mux roundtrip.

The GPU acceptance (real Wan 2.2 generation + VBench) runs on the user's
local machine via scripts/gen_eval_set.py + scripts/run_vbench.py.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def test_eval_set_schema():
    rows = [
        json.loads(line)
        for line in (ROOT / "eval" / "t2v_v1.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 50
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == 50
    assert ids == sorted(ids)
    for row in rows:
        assert row["prompt"].strip()
        assert row["category"]
    # CJK coverage — the brand's zh-Hant focus must be in the eval set.
    cjk = [r for r in rows if any("一" <= ch <= "鿿" for ch in r["prompt"])]
    assert len(cjk) >= 8


def test_model_specs_valid():
    from theama.infer.config import MODELS, WAN_SPATIAL_MULT, WAN_TEMPORAL_STRIDE

    for spec in MODELS.values():
        assert (spec.num_frames - 1) % WAN_TEMPORAL_STRIDE == 0
        assert spec.width % WAN_SPATIAL_MULT == 0
        assert spec.height % WAN_SPATIAL_MULT == 0
        assert spec.num_frames == spec.fps * spec.native_seconds + 1
    assert MODELS["t2v-a14b"].guidance_scale_2 is not None  # MoE dual expert
    assert MODELS["t2v-a14b"].boundary_ratio == pytest.approx(0.875)
    assert MODELS["ti2v-5b"].fps == 24
    assert MODELS["ti2v-5b"].num_frames == 121


def test_frames_for_seconds_snapping():
    from theama.infer.config import MODELS, frames_for_seconds

    spec = MODELS["ti2v-5b"]
    assert frames_for_seconds(spec, 5) == 121
    assert frames_for_seconds(spec, 2) == 49  # 2*24+1 = 49, already 4k+1
    n = frames_for_seconds(spec, 3.0)
    assert (n - 1) % 4 == 0 and n >= 73


def test_reference_json():
    ref = json.loads((ROOT / "eval" / "reference.json").read_text(encoding="utf-8"))
    a14b = ref["t2v-a14b"]
    assert a14b["vbench_overall_bf16"] == pytest.approx(0.8681)
    assert len(a14b["six_dim_subset"]) == 6
    assert a14b["tolerance"] == pytest.approx(0.02)


def test_config_eval_section():
    cfg = yaml.safe_load((ROOT / "configs" / "theama_base.yaml").read_text(encoding="utf-8"))
    assert cfg["eval"]["prompt_set"] == "eval/t2v_v1.jsonl"
    assert len(cfg["eval"]["vbench_subset"]) == 6


def _write_frames(frames_dir: Path, n: int = 49, w: int = 160, h: int = 96) -> None:
    for i in range(1, n + 1):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[..., 0] = (i * 5) % 256
        arr[..., 1] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
        arr[..., 2] = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
        Image.fromarray(arr).save(frames_dir / f"{i:04d}.png")


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_mux_silent_aac_roundtrip(tmp_path):
    from theama.infer.mux import mux_mp4, probe_mp4

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frames(frames_dir)
    out = tmp_path / "out.mp4"
    mux_mp4(frames_dir, out, fps=24)

    info = probe_mp4(out)
    assert info["video_codec"] == "h264"
    assert info["fps"] == pytest.approx(24.0)
    assert info["width"] == 160
    assert info["height"] == 96
    assert info["duration"] == pytest.approx(49 / 24, abs=0.3)
    assert info["has_audio"] is True
    assert info["audio_codec"] == "aac"
    assert info["audio_sample_rate"] == 44100
    assert info["audio_channels"] == 2


def test_mux_empty_frames_dir_raises(tmp_path):
    from theama.infer.mux import MuxError, mux_mp4

    with pytest.raises(MuxError):
        mux_mp4(tmp_path, tmp_path / "out.mp4")


def test_gpu_scripts_parse():
    """Scripts must at least be valid Python (compile-checked on CPU)."""
    for script in ("gen_eval_set.py", "run_vbench.py"):
        src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        compile(src, script, "exec")

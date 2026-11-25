# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/build_av_manifest.py (P1b data-engine builder).

Covers staging-dir scanning, caption/source rules, the license red line at
build time, deterministic output, and the round trip into
``theama.data.dataset.load_manifest`` (paths resolve relative to the
manifest's parent directory).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import load_manifest  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "build_av_manifest", ROOT / "scripts" / "build_av_manifest.py"
)
bam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bam)


def _stage(root: Path, stem: str, *, video=True, audio=True, caption=True):
    if video:
        (root / f"{stem}.mp4").write_bytes(b"\x00" * 16)
    if audio:
        (root / f"{stem}.wav").write_bytes(b"\x00" * 16)
    if caption:
        (root / f"{stem}.txt").write_text(f"caption for {stem}", encoding="utf-8")


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_av_manifest.py", *argv])
    return bam.main()


def test_build_rows_happy_path_sorted(tmp_path):
    _stage(tmp_path, "b_clip")
    _stage(tmp_path, "a_clip")
    rows, skips = bam.build_rows(tmp_path, default_source="pexels", probe=False)
    assert skips == []
    assert [r["video"] for r in rows] == ["a_clip.mp4", "b_clip.mp4"]  # sorted
    row = rows[0]
    assert row["audio"] == "a_clip.wav"
    assert row["caption"] == "caption for a_clip"
    assert row["source"] == "pexels"
    assert "duration_s" not in row  # probe=False keeps optional keys out


def test_sidecar_overrides_caption_and_source(tmp_path):
    _stage(tmp_path, "clip", caption=True)
    (tmp_path / "clip.json").write_text(
        json.dumps({"caption": "sidecar 文案", "source": "mixkit"}),
        encoding="utf-8",
    )
    rows, _ = bam.build_rows(tmp_path, default_source="pexels", probe=False)
    assert rows[0]["caption"] == "sidecar 文案"
    assert rows[0]["source"] == "mixkit"


def test_unpaired_and_captionless_media_skipped(tmp_path):
    _stage(tmp_path, "no_audio", audio=False)
    _stage(tmp_path, "no_video", video=False)
    _stage(tmp_path, "no_caption", caption=False)
    _stage(tmp_path, "ok")
    rows, skips = bam.build_rows(tmp_path, default_source="pexels", probe=False)
    assert [r["video"] for r in rows] == ["ok.mp4"]
    reasons = {s["stem"]: s["reason"] for s in skips}
    assert "without audio" in reasons["no_audio"]
    assert "without video" in reasons["no_video"]
    assert "no caption" in reasons["no_caption"]


def test_missing_source_skipped(tmp_path):
    _stage(tmp_path, "clip")
    rows, skips = bam.build_rows(tmp_path, default_source=None, probe=False)
    assert rows == []
    assert "no source" in skips[0]["reason"]


@pytest.mark.parametrize("bad", ["minimax-h3", "Hailuo-clip", "LTX-source"])
def test_forbidden_source_aborts_build(tmp_path, bad):
    _stage(tmp_path, "clip")
    (tmp_path / "clip.json").write_text(json.dumps({"source": bad}))
    with pytest.raises(ValueError, match="forbidden training source"):
        bam.build_rows(tmp_path, default_source=None, probe=False)


def test_extension_priority_deterministic(tmp_path):
    for ext in (".webm", ".mp4"):  # .mp4 must win regardless of dir order
        (tmp_path / f"clip{ext}").write_bytes(b"\x00" * 8)
    (tmp_path / "clip.m4a").write_bytes(b"\x00" * 8)
    (tmp_path / "clip.wav").write_bytes(b"\x00" * 8)
    (tmp_path / "clip.txt").write_text("c", encoding="utf-8")
    rows, _ = bam.build_rows(tmp_path, default_source="pexels", probe=False)
    assert rows[0]["video"] == "clip.mp4"
    assert rows[0]["audio"] == "clip.wav"


def test_bad_sidecar_json_skipped(tmp_path):
    _stage(tmp_path, "clip")
    (tmp_path / "clip.json").write_text("{not json", encoding="utf-8")
    rows, skips = bam.build_rows(tmp_path, default_source="pexels", probe=False)
    assert rows == []
    assert "bad sidecar" in skips[0]["reason"]


def test_main_writes_manifest_and_load_manifest_roundtrip(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()
    _stage(staged, "clip1")
    _stage(staged, "clip2")
    out = tmp_path / "av_manifest.jsonl"

    rc = _run_main(
        ["--root", str(staged), "--source", "pexels",
         "--output", str(out), "--no-probe"],
        monkeypatch,
    )
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["video"].startswith("staged/")  # relative to manifest dir

    report = load_manifest(out, strict=True)  # raises on any bad row
    assert len(report.samples) == 2
    assert report.samples[0].video_path.is_file()
    assert report.samples[0].source == "pexels"


def test_main_refuses_overwrite_without_force(tmp_path, monkeypatch, capsys):
    staged = tmp_path / "staged"
    staged.mkdir()
    _stage(staged, "clip1")
    out = tmp_path / "av_manifest.jsonl"
    argv = ["--root", str(staged), "--source", "pexels",
            "--output", str(out), "--no-probe"]
    assert _run_main(argv, monkeypatch) == 0
    first = out.read_text(encoding="utf-8")
    assert _run_main(argv, monkeypatch) == 1  # exists, no --force
    assert "output exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == first  # untouched
    assert _run_main([*argv, "--force"], monkeypatch) == 0


def test_main_forbidden_source_exit_1(tmp_path, monkeypatch, capsys):
    staged = tmp_path / "staged"
    staged.mkdir()
    _stage(staged, "clip")
    rc = _run_main(
        ["--root", str(staged), "--source", "minimax-h3-demo",
         "--output", str(tmp_path / "m.jsonl"), "--no-probe"],
        monkeypatch,
    )
    assert rc == 1
    assert "forbidden training source" in capsys.readouterr().err
    assert not (tmp_path / "m.jsonl").exists()  # nothing written


def test_probe_media_graceful_without_ffprobe(monkeypatch, tmp_path):
    monkeypatch.setattr(bam.shutil, "which", lambda _: None)
    assert bam._probe_media(tmp_path / "x.mp4") == {}

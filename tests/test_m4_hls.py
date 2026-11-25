# SPDX-License-Identifier: Apache-2.0
"""M4 HLS packager tests: pure ladder/playlist logic + ffmpeg integration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from theama.inference.hls import (  # noqa: E402
    build_ladder,
    codecs_string,
    render_master_playlist,
    summary_json,
)
from theama.inference.mux import MuxError  # noqa: E402
import package_hls  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


# ---------- pure ladder logic ----------

def test_ladder_4k_sdr_source_all_rungs_no_hdr():
    rungs = build_ladder(3840, 2160)
    assert [r["name"] for r in rungs] == ["2160p", "1080p", "720p", "480p"]
    assert [r["vcodec"] for r in rungs] == ["av1", "hevc", "h264", "h264"]
    assert all(r["hdr"] is None for r in rungs)
    assert [r["height"] for r in rungs] == [2160, 1080, 720, 480]


def test_ladder_4k_hdr10_tags_top_rung_only():
    rungs = build_ladder(3840, 2160, hdr="hdr10")
    assert rungs[0]["hdr"] == "hdr10"
    assert all(r["hdr"] is None for r in rungs[1:])


def test_ladder_drops_rungs_above_source():
    assert [r["name"] for r in build_ladder(1920, 1080)] == ["1080p", "720p", "480p"]
    assert [r["name"] for r in build_ladder(1280, 720)] == ["720p", "480p"]
    assert [r["name"] for r in build_ladder(854, 480)] == ["480p"]


def test_ladder_sub_480_clamped_single_rung_no_upscale():
    rungs = build_ladder(640, 360)
    assert len(rungs) == 1
    assert rungs[0]["height"] == 360 and rungs[0]["vcodec"] == "h264"
    assert rungs[0]["width"] == 640
    # never upscale: a 240p source stays 240p
    assert build_ladder(426, 240)[0]["height"] == 240


def test_ladder_width_proportional_and_even():
    # 4:3 source at 480p rung: 640 wide, not the 16:9 854
    rungs = build_ladder(1440, 1080)
    assert rungs[0]["width"] == 1440
    assert rungs[1]["width"] == 960  # 720p rung on 4:3
    assert all(r["width"] % 2 == 0 for r in rungs)


def test_ladder_bandwidth_and_codecs():
    rungs = build_ladder(3840, 2160, hdr="hlg")
    assert rungs[0]["bandwidth"] == (12000 + 192) * 1000
    assert rungs[3]["bandwidth"] == (1200 + 128) * 1000
    assert codecs_string(rungs[0]) == "av01.0.13M.10,mp4a.40.2"
    assert codecs_string(rungs[1]).startswith("hvc1.")
    assert codecs_string(rungs[2]) == "avc1.640028,mp4a.40.2"


def test_ladder_invalid_inputs_rejected():
    with pytest.raises(MuxError):
        build_ladder(3840, 2160, hdr="dolby-vision")
    with pytest.raises(MuxError):
        build_ladder(0, 1080)


def test_master_playlist_rendering_deterministic():
    rungs = build_ladder(3840, 2160, hdr="hdr10")
    text = render_master_playlist(rungs)
    assert text == render_master_playlist(build_ladder(3840, 2160, hdr="hdr10"))
    lines = text.splitlines()
    assert lines[0] == "#EXTM3U" and lines[1] == "#EXT-X-VERSION:7"
    assert lines[2] == ('#EXT-X-STREAM-INF:BANDWIDTH=12192000,'
                        'AVERAGE-BANDWIDTH=12192000,RESOLUTION=3840x2160,'
                        'CODECS="av01.0.13M.10,mp4a.40.2"')
    assert lines[3] == "2160p/index.m3u8"
    assert lines[-1] == "480p/index.m3u8"
    assert len(lines) == 2 + 2 * 4


# ---------- ffmpeg integration ----------

def _make_clip(path: Path, width: int, height: int, seconds: float = 1.0) -> None:
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
         "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True, text=True,
    )


@needs_ffmpeg
def test_package_360p_single_rung_verified(tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 640, 360)
    summary = package_hls.package_hls(src, tmp_path / "out", hls_time=2)
    assert summary["verified"] is True
    assert [v["name"] for v in summary["variants"]] == ["360p"]
    v = summary["variants"][0]
    assert v["vcodec"] == "h264" and v["height"] == 360 and v["segments"] >= 1
    master = (tmp_path / "out" / "master.m3u8").read_text()
    assert "360p/index.m3u8" in master


@needs_ffmpeg
def test_package_720p_two_h264_rungs(tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 1280, 720)
    summary = package_hls.package_hls(src, tmp_path / "out", hls_time=2)
    assert [v["name"] for v in summary["variants"]] == ["720p", "480p"]
    assert all(v["vcodec"] == "h264" for v in summary["variants"])
    for v in summary["variants"]:
        d = tmp_path / "out" / v["name"]
        assert (d / "init.mp4").exists()
        assert list(d.glob("seg_*.m4s"))
    master = (tmp_path / "out" / "master.m3u8").read_text()
    assert "RESOLUTION=1280x720" in master and "854x480" in master


@needs_ffmpeg
def test_cli_json_out_roundtrip_deterministic(tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 640, 360)
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    rc1 = package_hls.main([str(src), "--out-dir", str(tmp_path / "p1"),
                            "--hls-time", "2", "--json-out", str(out1)])
    rc2 = package_hls.main([str(src), "--out-dir", str(tmp_path / "p2"),
                            "--hls-time", "2", "--json-out", str(out2)])
    assert rc1 == rc2 == 0
    d1, d2 = json.loads(out1.read_text()), json.loads(out2.read_text())
    # paths differ between runs; everything else must be identical
    for d in (d1, d2):
        d.pop("input"), d.pop("master")
    assert d1 == d2
    assert d1["verified"] is True


@needs_ffmpeg
def test_cli_missing_input_exit_1(tmp_path, capsys):
    rc = package_hls.main([str(tmp_path / "nope.mp4"),
                           "--out-dir", str(tmp_path / "out")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()  # fail before writing anything

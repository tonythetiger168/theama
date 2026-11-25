# SPDX-License-Identifier: Apache-2.0
"""M4 DASH manifest tests: pure m3u8/MPD logic + ffmpeg integration."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from theama.inference.dash import (  # noqa: E402
    _iso_duration,
    _segment_timeline,
    package_dash,
    parse_master_playlist,
    parse_variant_playlist,
    render_mpd,
)
from theama.inference.hls import build_ladder, render_master_playlist  # noqa: E402
from theama.inference.mux import MuxError  # noqa: E402
import package_dash as package_dash_cli  # noqa: E402
import package_hls  # noqa: E402

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(
    FFMPEG is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)

_VARIANT = """#EXTM3U
#EXT-X-VERSION:7
#EXT-X-TARGETDURATION:5
#EXT-X-PLAYLIST-TYPE:VOD
#EXT-X-MAP:URI="init.mp4"
#EXTINF:4.000000,
seg_00000.m4s
#EXTINF:4.000000,
seg_00001.m4s
#EXTINF:1.500000,
seg_00002.m4s
#EXT-X-ENDLIST
"""

# 720p + 480p, both h264, 3x4s + 1.5s tail, 24 fps
_REPS = [
    {"name": "720p", "vcodec": "h264", "width": 1280, "height": 720,
     "bandwidth": 3128000, "fps": 24.0, "durations": [4.0, 4.0, 4.0, 1.5]},
    {"name": "480p", "vcodec": "h264", "width": 854, "height": 480,
     "bandwidth": 1328000, "fps": 24.0, "durations": [4.0, 4.0, 4.0, 1.5]},
]

_GOLDEN_MPD = """<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" mediaPresentationDuration="PT13.5S" maxSegmentDuration="PT4S" minBufferTime="PT2S" profiles="urn:mpeg:dash:profile:isoff-live:2011">
  <Period id="0" duration="PT13.5S">
    <AdaptationSet id="0" contentType="video" mimeType="video/mp4" segmentAlignment="true" startWithSAP="1">
      <Representation id="720p" bandwidth="3128000" width="1280" height="720" frameRate="24" codecs="avc1.640028,mp4a.40.2">
        <SegmentTemplate timescale="1000000" startNumber="0" initialization="720p/init.mp4" media="720p/seg_$Number%05d$.m4s">
          <SegmentTimeline>
            <S t="0" d="4000000" r="2"/>
            <S d="1500000"/>
          </SegmentTimeline>
        </SegmentTemplate>
      </Representation>
      <Representation id="480p" bandwidth="1328000" width="854" height="480" frameRate="24" codecs="avc1.640028,mp4a.40.2">
        <SegmentTemplate timescale="1000000" startNumber="0" initialization="480p/init.mp4" media="480p/seg_$Number%05d$.m4s">
          <SegmentTimeline>
            <S t="0" d="4000000" r="2"/>
            <S d="1500000"/>
          </SegmentTimeline>
        </SegmentTemplate>
      </Representation>
    </AdaptationSet>
  </Period>
</MPD>
"""


# ---------- pure parsing / rendering ----------

def test_parse_variant_playlist_ok():
    v = parse_variant_playlist(_VARIANT, "720p")
    assert v["init_uri"] == "init.mp4"
    assert v["segment_uris"] == ["seg_00000.m4s", "seg_00001.m4s", "seg_00002.m4s"]
    assert v["durations"] == [4.0, 4.0, 1.5]
    assert v["target_duration"] == 5.0


def test_parse_variant_rejects_bad_playlists():
    with pytest.raises(MuxError, match="not an m3u8"):
        parse_variant_playlist("hello\n")
    with pytest.raises(MuxError, match="EXT-X-MAP"):
        parse_variant_playlist("#EXTM3U\n#EXTINF:4.0,\nseg_00000.m4s\n")
    with pytest.raises(MuxError, match="segment URIs"):
        parse_variant_playlist(
            '#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n'
            "#EXTINF:4.0,\nseg_00000.m4s\n#EXTINF:4.0,\n"
        )
    with pytest.raises(MuxError, match="no EXTINF"):
        parse_variant_playlist('#EXTM3U\n#EXT-X-MAP:URI="init.mp4"\n')


def test_parse_master_playlist_ok():
    text = render_master_playlist(build_ladder(1280, 720))
    declared = parse_master_playlist(text)
    assert declared == [
        {"name": "720p", "uri": "720p/index.m3u8", "bandwidth": 3128000,
         "width": 1280, "height": 720, "vcodec": "h264"},
        {"name": "480p", "uri": "480p/index.m3u8", "bandwidth": 1328000,
         "width": 854, "height": 480, "vcodec": "h264"},
    ]


def test_parse_master_rejects_bad_input():
    with pytest.raises(MuxError, match="not an m3u8"):
        parse_master_playlist("nope\n")
    with pytest.raises(MuxError, match="without a following URI"):
        parse_master_playlist("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\n")
    with pytest.raises(MuxError, match="not a package_hls master"):
        parse_master_playlist(
            "#EXTM3U\n"
            '#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION=640x360,'
            'CODECS="avc1.640028,mp4a.40.2"\n'
            "360p/stream.m3u8\n"
        )
    with pytest.raises(MuxError, match="codec tag"):
        parse_master_playlist(
            "#EXTM3U\n"
            "#EXT-X-STREAM-INF:BANDWIDTH=1,RESOLUTION=640x360,"
            'CODECS="vp09.00.10.08"\n'
            "360p/index.m3u8\n"
        )


def test_segment_timeline_collapses_and_rejects():
    assert _segment_timeline([4.0, 4.0, 4.0, 1.5]) == [[4000000, 2], [1500000, 0]]
    assert _segment_timeline([2.0, 2.0]) == [[2000000, 1]]
    with pytest.raises(MuxError, match="non-positive"):
        _segment_timeline([4.0, 0.0])


def test_iso_duration_formatting():
    assert _iso_duration(4.0) == "PT4S"
    assert _iso_duration(13.5) == "PT13.5S"
    assert _iso_duration(12.345) == "PT12.345S"


def test_render_mpd_golden():
    assert render_mpd(_REPS) == _GOLDEN_MPD


def test_render_mpd_codec_groups_in_first_appearance_order():
    rungs = build_ladder(3840, 2160)  # av1, hevc, h264, h264
    reps = [dict(r, durations=[4.0], fps=24.0) for r in rungs]
    text = render_mpd(reps)
    assert text.count("<AdaptationSet") == 3  # per-codec grouping
    pos = {tag: text.index(f'codecs="{tag}') for tag in ("av01.", "hvc1.", "avc1.")}
    assert pos["av01."] < pos["hvc1."] < pos["avc1."]
    assert text.count("<Representation") == 4


def test_render_mpd_deterministic_and_empty_rejected():
    assert render_mpd(_REPS) == render_mpd([dict(r) for r in _REPS])
    with pytest.raises(MuxError, match="no representations"):
        render_mpd([])


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
def test_dash_over_720p_bundle_verified(tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 1280, 720)
    out = tmp_path / "out"
    package_hls.package_hls(src, out, hls_time=2)
    master_before = (out / "master.m3u8").read_bytes()

    summary = package_dash(out)
    assert summary["verified"] is True
    assert [r["name"] for r in summary["representations"]] == ["720p", "480p"]
    assert all(r["vcodec"] == "h264" for r in summary["representations"])
    assert all(r["segments"] >= 1 for r in summary["representations"])
    assert 0.5 < summary["duration"] < 3.0

    mpd = (out / "manifest.mpd").read_text(encoding="utf-8")
    assert '<Representation id="720p"' in mpd
    assert 'media="480p/seg_$Number%05d$.m4s"' in mpd
    assert "SegmentTimeline" in mpd
    # the HLS bundle is only read, never modified
    assert (out / "master.m3u8").read_bytes() == master_before


@needs_ffmpeg
def test_cli_json_out_deterministic(tmp_path):
    src = tmp_path / "clip.mp4"
    _make_clip(src, 640, 360)
    summaries = []
    for name in ("p1", "p2"):
        out = tmp_path / name
        package_hls.package_hls(src, out, hls_time=2)
        json_out = tmp_path / f"{name}.json"
        rc = package_dash_cli.main([str(out), "--json-out", str(json_out)])
        assert rc == 0
        d = json.loads(json_out.read_text())
        d.pop("hls_dir"), d.pop("manifest")  # paths differ between runs
        summaries.append(d)
    assert summaries[0] == summaries[1]
    assert summaries[0]["verified"] is True


@needs_ffmpeg
def test_cli_missing_master_exit_1(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = package_dash_cli.main([str(empty)])
    assert rc == 1
    assert "master.m3u8" in capsys.readouterr().err
    assert not (empty / "manifest.mpd").exists()  # fail before writing anything

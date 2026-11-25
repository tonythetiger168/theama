# SPDX-License-Identifier: Apache-2.0
"""MPEG-DASH manifest packaging on top of an HLS fMP4/CMAF bundle (M4).

The 2026-11-26 delivery-format decision (training-plan M4) ships every
finished film as an HLS fMP4/CMAF bundle plus — from the SAME segments,
"同批 segment 兼出 DASH" — a DASH manifest.mpd. This module never
re-encodes: it reads a verified package_hls() output directory
(master.m3u8 + <rung>/{init.mp4, seg_*.m4s, index.m3u8}), cross-checks
every variant playlist against the master-playlist declarations and the
files on disk, renders a deterministic static MPD (per-codec
AdaptationSets, Representation-level SegmentTemplate with an exact
SegmentTimeline rebuilt from EXTINF), then ffprobe-verifies the manifest
itself. A manifest that does not verify raises MuxError — never
half-shipped.

Avatar realtime (M6b) deliberately does NOT use this either: LL-HLS /
WebRTC only. This module is for film VOD distribution only.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .hls import codecs_string
from .mux import MuxError, _require, probe_mp4

# SegmentTemplate tick rate; EXTINF seconds are rounded to whole ticks.
_TIMESCALE = 1_000_000

# ffprobe codec names for the ladder codecs (mirror of hls._verify_rung).
_PROBE_CODEC = {"h264": "h264", "hevc": "hevc", "av1": "av1"}
# RFC 6381 tag prefix -> ladder vcodec name (reverse of hls._CODECS_TAG).
_TAG_TO_VCODEC = {"avc1": "h264", "hvc1": "hevc", "av01": "av1"}


def _iso_duration(seconds: float) -> str:
    """PT4S / PT13.5S — microseconds precision, trailing zeros trimmed."""
    text = f"{seconds:.6f}".rstrip("0").rstrip(".")
    return f"PT{text}S"


def _format_fps(fps: float) -> str:
    return f"{fps:.3f}".rstrip("0").rstrip(".")


def parse_variant_playlist(text: str, rung: str = "?") -> dict:
    """Parse a package_hls variant playlist (fMP4 VOD).

    Returns {"init_uri", "segment_uris", "durations", "target_duration"}.
    Raises MuxError on anything that is not a well-formed fMP4 variant:
    the EXTINF count must match the segment-URI count exactly, or the
    SegmentTimeline built from it would point at the wrong files.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0] != "#EXTM3U":
        raise MuxError(f"{rung}: not an m3u8 playlist")
    init_uri = None
    durations: list[float] = []
    uris: list[str] = []
    target = None
    for line in lines[1:]:
        if line.startswith("#EXT-X-MAP:"):
            m = re.search(r'URI="([^"]+)"', line)
            if not m:
                raise MuxError(f"{rung}: EXT-X-MAP without URI")
            init_uri = m.group(1)
        elif line.startswith("#EXTINF:"):
            durations.append(float(line[len("#EXTINF:"):].split(",", 1)[0]))
        elif line.startswith("#EXT-X-TARGETDURATION:"):
            target = float(line.split(":", 1)[1])
        elif not line.startswith("#"):
            uris.append(line)
    if init_uri is None:
        raise MuxError(f"{rung}: missing EXT-X-MAP (not an fMP4 variant)")
    if not durations:
        raise MuxError(f"{rung}: no EXTINF entries")
    if len(durations) != len(uris):
        raise MuxError(
            f"{rung}: {len(durations)} EXTINF entries vs "
            f"{len(uris)} segment URIs"
        )
    return {
        "init_uri": init_uri,
        "segment_uris": uris,
        "durations": durations,
        "target_duration": target,
    }


def parse_master_playlist(text: str) -> list[dict]:
    """Parse a package_hls master playlist into declared rungs, in order.

    Each entry: name / uri / bandwidth / width / height / vcodec (ladder
    name derived from the RFC 6381 tag prefix). Anything that is not the
    exact shape package_hls writes raises MuxError — this is a packaging
    tool, not a general m3u8 parser.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines or lines[0] != "#EXTM3U":
        raise MuxError("not an m3u8 master playlist")
    declared: list[dict] = []
    pending = None
    for line in lines[1:]:
        if line.startswith("#EXT-X-STREAM-INF:"):
            if pending is not None:
                raise MuxError("EXT-X-STREAM-INF without a following URI")
            pending = dict(
                re.findall(r'([\w-]+)=("[^"]*"|[^,]*)', line)
            )
        elif not line.startswith("#"):
            if pending is None:
                raise MuxError(f"variant URI {line!r} without EXT-X-STREAM-INF")
            uri = line
            if not uri.endswith("/index.m3u8"):
                raise MuxError(
                    f"unexpected variant URI {uri!r} — not a package_hls master"
                )
            codecs = pending.get("CODECS", "").strip('"')
            vtag = codecs.split(",", 1)[0].split(".", 1)[0]
            if vtag not in _TAG_TO_VCODEC:
                raise MuxError(f"unrecognised video codec tag in {codecs!r}")
            res = pending.get("RESOLUTION", "x").split("x", 1)
            declared.append({
                "name": uri[: -len("/index.m3u8")],
                "uri": uri,
                "bandwidth": int(pending["BANDWIDTH"]),
                "width": int(res[0]),
                "height": int(res[1]),
                "vcodec": _TAG_TO_VCODEC[vtag],
            })
            pending = None
    if pending is not None:
        raise MuxError("EXT-X-STREAM-INF without a following URI")
    if not declared:
        raise MuxError("no variants declared in master playlist")
    return declared


def _segment_timeline(durations: list[float]) -> list[list[int]]:
    """Collapse EXTINF seconds to [d_ticks, repeat] S entries."""
    entries: list[list[int]] = []
    for d in durations:
        ticks = round(d * _TIMESCALE)
        if ticks <= 0:
            raise MuxError(f"non-positive segment duration {d}")
        if entries and entries[-1][0] == ticks:
            entries[-1][1] += 1
        else:
            entries.append([ticks, 0])
    return entries


def render_mpd(reps: list[dict]) -> str:
    """Deterministic static MPD for the given representations (high -> low).

    Each rep: name / vcodec / width / height / bandwidth / durations (EXTINF
    seconds per segment) and optional fps. Representations group into
    per-codec AdaptationSets in first-appearance order (mixed codecs in one
    AdaptationSet are legal but confuse strict players); every
    Representation carries its own SegmentTemplate with rung-prefixed URIs
    and an exact SegmentTimeline. Segments stay muxed (audio rides the same
    fMP4), so the codecs attribute lists video + AAC-LC — same as HLS.
    """
    if not reps:
        raise MuxError("no representations to render")
    timelines = [_segment_timeline(r["durations"]) for r in reps]
    totals = [sum(d * (1 + r) for d, r in tl) for tl in timelines]
    total = max(totals)
    max_seg = max(d for tl in timelines for d, _r in tl)

    groups: dict[str, list] = {}
    for rep, tl in zip(reps, timelines):
        groups.setdefault(rep["vcodec"], []).append((rep, tl))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" type="static" '
        f'mediaPresentationDuration="{_iso_duration(total / _TIMESCALE)}" '
        f'maxSegmentDuration="{_iso_duration(max_seg / _TIMESCALE)}" '
        'minBufferTime="PT2S" '
        'profiles="urn:mpeg:dash:profile:isoff-live:2011">',
        f'  <Period id="0" duration="{_iso_duration(total / _TIMESCALE)}">',
    ]
    for set_id, (_vcodec, members) in enumerate(groups.items()):
        lines.append(
            f'    <AdaptationSet id="{set_id}" contentType="video" '
            'mimeType="video/mp4" segmentAlignment="true" startWithSAP="1">'
        )
        for rep, tl in members:
            fps_attr = (
                f' frameRate="{_format_fps(rep["fps"])}"' if rep.get("fps") else ""
            )
            lines.append(
                f'      <Representation id="{rep["name"]}" '
                f'bandwidth="{rep["bandwidth"]}" '
                f'width="{rep["width"]}" height="{rep["height"]}"'
                f'{fps_attr} codecs="{codecs_string(rep)}">'
            )
            lines.append(
                f'        <SegmentTemplate timescale="{_TIMESCALE}" '
                'startNumber="0" '
                f'initialization="{rep["name"]}/init.mp4" '
                f'media="{rep["name"]}/seg_$Number%05d$.m4s">'
            )
            lines.append("          <SegmentTimeline>")
            for i, (d, r) in enumerate(tl):
                t_attr = ' t="0"' if i == 0 else ""
                r_attr = f' r="{r}"' if r else ""
                lines.append(f"            <S{t_attr} d=\"{d}\"{r_attr}/>")
            lines.append("          </SegmentTimeline>")
            lines.append("        </SegmentTemplate>")
            lines.append("      </Representation>")
        lines.append("    </AdaptationSet>")
    lines.append("  </Period>")
    lines.append("</MPD>")
    return "\n".join(lines) + "\n"


def _probe_mpd(ffprobe: str, mpd_path: Path) -> None:
    """Final gate: ffprobe must parse the manifest and see a video stream."""
    proc = subprocess.run(
        [ffprobe, "-v", "error",
         "-protocol_whitelist", "file,crypto,data",
         "-show_entries", "stream=codec_type",
         "-of", "json", mpd_path.name],
        capture_output=True, text=True, cwd=mpd_path.parent,
    )
    if proc.returncode != 0:
        raise MuxError(
            f"ffprobe could not parse {mpd_path.name}: {proc.stderr[-2000:]}"
        )
    streams = json.loads(proc.stdout).get("streams", [])
    if not any(s.get("codec_type") == "video" for s in streams):
        raise MuxError(f"{mpd_path.name}: no video stream visible to ffprobe")


def package_dash(hls_dir: str | Path) -> dict:
    """Render + verify manifest.mpd over an existing package_hls bundle.

    Cross-checks master declarations vs variant playlists vs files on disk
    vs ffprobe of every variant, writes manifest.mpd, then verifies the
    manifest itself with ffprobe. Raises MuxError on any mismatch — a
    manifest that does not verify is not a manifest.
    """
    ffprobe = _require("ffprobe")
    root = Path(hls_dir)
    master_path = root / "master.m3u8"
    if not master_path.is_file():
        raise MuxError(f"{root}: master.m3u8 not found — run package_hls first")
    declared = parse_master_playlist(master_path.read_text(encoding="utf-8"))

    reps = []
    for d in declared:
        variant_dir = root / d["name"]
        playlist_path = variant_dir / "index.m3u8"
        if not playlist_path.is_file():
            raise MuxError(f"{d['name']}: index.m3u8 missing")
        variant = parse_variant_playlist(
            playlist_path.read_text(encoding="utf-8"), d["name"]
        )
        if not (variant_dir / variant["init_uri"]).is_file():
            raise MuxError(
                f"{d['name']}: init segment {variant['init_uri']} missing on disk"
            )
        missing = [u for u in variant["segment_uris"]
                   if not (variant_dir / u).is_file()]
        if missing:
            raise MuxError(
                f"{d['name']}: {len(missing)} segments missing on disk "
                f"(first: {missing[0]})"
            )
        probe = probe_mp4(playlist_path)
        expected = _PROBE_CODEC[d["vcodec"]]
        if probe["video_codec"] != expected:
            raise MuxError(
                f"{d['name']}: probed codec {probe['video_codec']} "
                f"!= declared {expected}"
            )
        if probe["width"] != d["width"] or probe["height"] != d["height"]:
            raise MuxError(
                f"{d['name']}: probed {probe['width']}x{probe['height']} "
                f"!= declared {d['width']}x{d['height']}"
            )
        reps.append({
            "name": d["name"],
            "vcodec": d["vcodec"],
            "width": d["width"],
            "height": d["height"],
            "bandwidth": d["bandwidth"],
            "fps": probe["fps"],
            "durations": variant["durations"],
        })

    mpd_path = root / "manifest.mpd"
    mpd_path.write_text(render_mpd(reps), encoding="utf-8")
    _probe_mpd(ffprobe, mpd_path)
    return {
        "hls_dir": str(root),
        "manifest": str(mpd_path),
        "representations": [
            {"name": r["name"], "vcodec": r["vcodec"],
             "width": r["width"], "height": r["height"],
             "bandwidth": r["bandwidth"], "segments": len(r["durations"]),
             "duration": round(sum(r["durations"]), 3)}
            for r in reps
        ],
        "duration": round(max(sum(r["durations"]) for r in reps), 3),
        "verified": True,
    }

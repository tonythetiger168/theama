#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build an A/V data-card manifest from a staged media directory (CPU).

The P1b/M2 pipeline consumes a jsonl data-card manifest (see
``theama.data.dataset.load_manifest``) with ``video``, ``audio``,
``caption``, ``source`` per row. Until now that file had to be authored by
hand; this script derives it from a staged directory of licensed clips:

    <root>/<stem>.mp4            # video (.mp4/.mov/.webm/.mkv)
    <root>/<stem>.wav            # paired audio (.wav/.mp3/.flac/.m4a)
    <root>/<stem>.txt            # caption (or sidecar json "caption")
    <root>/<stem>.json           # optional sidecar: {"caption": ..., "source": ...}

Every row carries a data-card ``source`` id; pass a default with
``--source`` (e.g. ``--source pexels``) and override per clip via sidecar.
The license red line is enforced at build time: forbidden sources
(MiniMax H3 / LTX / ...) abort the build, so a bad batch never produces a
manifest that only fails later at training-launch time.

Media paths are written relative to the manifest's parent directory, which
is exactly how ``load_manifest`` resolves them. When ``ffprobe`` is on
PATH, duration/fps/resolution are probed into optional row keys (readers
ignore unknown keys); without ffprobe the keys are simply omitted.

Rows are emitted sorted by stem for deterministic diffs. Unpaired media
and captionless clips are skipped and reported — a row is only written
when all four required fields are present.

    python scripts/build_av_manifest.py --root data/staged/pexx \
        --source pexels --output data/av_manifest.jsonl

Exit code 0 on success (even with skips), 1 on fatal errors (forbidden
source, output exists without --force, missing root).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import check_source  # noqa: E402

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv")  # priority order on stem clash
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a")

_PROBE_TIMEOUT_S = 15


def _load_sidecar(path: Path) -> dict:
    """Parse a ``<stem>.json`` sidecar; returns {} when absent."""
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"bad sidecar {path.name}: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"sidecar {path.name} must be a JSON object")
    return obj


def _probe_media(path: Path) -> dict:
    """Best-effort ffprobe metadata; {} when ffprobe is unavailable/fails."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height,avg_frame_rate",
                "-show_entries", "format=duration",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S,
        )
        info = json.loads(out.stdout or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return {}
    meta: dict = {}
    try:
        stream = (info.get("streams") or [{}])[0]
        if stream.get("width") and stream.get("height"):
            meta["width"] = int(stream["width"])
            meta["height"] = int(stream["height"])
        rate = stream.get("avg_frame_rate", "")
        if rate and rate != "0/0":
            num, den = rate.split("/")
            meta["fps"] = round(int(num) / int(den), 3)
        duration = info.get("format", {}).get("duration")
        if duration:
            meta["duration_s"] = round(float(duration), 3)
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    return meta


def build_rows(
    root: Path, *, default_source: str | None, probe: bool = True
) -> tuple[list[dict], list[dict]]:
    """Scan ``root`` and return ``(rows, skips)``.

    ``skips`` holds ``{"stem", "reason"}`` dicts for unpaired or
    captionless media so operators can fix the staging directory.
    """
    stems: dict[str, dict[str, list[Path]]] = {}
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        ext = entry.suffix.lower()
        slot = None
        if ext in VIDEO_EXTS:
            slot = "video"
        elif ext in AUDIO_EXTS:
            slot = "audio"
        if slot is None:
            continue
        stems.setdefault(entry.stem, {}).setdefault(slot, []).append(entry)

    def _pick(candidates: list[Path], exts: tuple[str, ...]) -> Path:
        """Deterministic choice: extension priority, then filename."""
        return min(candidates, key=lambda p: (exts.index(p.suffix.lower()), p.name))

    rows: list[dict] = []
    skips: list[dict] = []
    for stem in sorted(stems):
        slots = stems[stem]
        video = _pick(slots["video"], VIDEO_EXTS) if slots.get("video") else None
        audio = _pick(slots["audio"], AUDIO_EXTS) if slots.get("audio") else None
        if video is None:
            skips.append({"stem": stem, "reason": "audio without video"})
            continue
        if audio is None:
            skips.append({"stem": stem, "reason": "video without audio"})
            continue

        try:
            sidecar = _load_sidecar(root / f"{stem}.json")
        except ValueError as exc:
            skips.append({"stem": stem, "reason": str(exc)})
            continue

        caption = sidecar.get("caption")
        txt = root / f"{stem}.txt"
        if caption is None and txt.is_file():
            caption = txt.read_text(encoding="utf-8").strip()
        if not isinstance(caption, str) or not caption.strip():
            skips.append({"stem": stem, "reason": "no caption (.txt or sidecar)"})
            continue

        source = sidecar.get("source", default_source)
        if not isinstance(source, str) or not source.strip():
            skips.append({"stem": stem, "reason": "no source (pass --source or sidecar)"})
            continue
        source = source.strip()
        check_source(source)  # license red line — fatal, aborts the build

        row = {
            "video": video.name,
            "audio": audio.name,
            "caption": caption.strip(),
            "source": source,
        }
        if probe:
            row.update(_probe_media(video))
        rows.append(row)
    return rows, skips


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="staged media directory")
    ap.add_argument("--source", default=None, help="default data-card source id")
    ap.add_argument("--output", required=True, help="manifest jsonl to write")
    ap.add_argument("--force", action="store_true", help="overwrite --output")
    ap.add_argument("--no-probe", action="store_true", help="skip ffprobe metadata")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"root not found: {root}", file=sys.stderr)
        return 1
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"output exists (pass --force to overwrite): {out}", file=sys.stderr)
        return 1

    try:
        rows, skips = build_rows(root, default_source=args.source,
                                 probe=not args.no_probe)
    except ValueError as exc:  # forbidden source — license red line
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    base = out.parent.resolve()
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            record = dict(row)
            # Paths relative to the manifest dir — load_manifest's convention.
            record["video"] = os.path.relpath((root / row["video"]).resolve(), base)
            record["audio"] = os.path.relpath((root / row["audio"]).resolve(), base)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "rows": len(rows),
        "skipped": len(skips),
        "output": str(out),
        "sources": sorted({r["source"] for r in rows}),
        "probed": not args.no_probe and shutil.which("ffprobe") is not None,
    }
    print(json.dumps(summary, indent=2))
    for skip in skips:
        print(json.dumps(skip, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

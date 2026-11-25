#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile an A/V data-card manifest (P1b data engine, CPU).

    python scripts/manifest_stats.py --manifest data/av_manifest.jsonl
    python scripts/manifest_stats.py --manifest data/av_manifest.jsonl --out stats.json

Where ``check_av_manifest.py`` is the pass/fail gate, this tool is the
dataset profile you read *before* burning GPU hours on encode runs: row
counts, caption coverage and length distribution, probed media metadata
(duration / fps / resolution, present when ``build_av_manifest.py`` ran
with ffprobe on PATH), source mix, and a red-line visibility counter.
It only reads the manifest text; media files are never opened.

Exit code is 0 for any readable manifest (this tool describes, it does
not judge) and 1 when the manifest itself cannot be read. Output is one
deterministic JSON object on stdout (sorted keys, stable rounding), so
diffing two profiles is meaningful.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import FORBIDDEN_SOURCES  # noqa: E402

# Duration histogram edges in seconds: <2, 2-5, 5-10, 10-30, >=30.
_DURATION_BUCKETS = (
    ("lt_2s", 0.0, 2.0),
    ("2_to_5s", 2.0, 5.0),
    ("5_to_10s", 5.0, 10.0),
    ("10_to_30s", 10.0, 30.0),
    ("ge_30s", 30.0, None),
)


def _load_rows(manifest_path: str | Path) -> tuple[list[dict], int]:
    """Parse jsonl tolerantly; returns (object rows, bad-line count)."""
    rows: list[dict] = []
    bad = 0
    with open(manifest_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                bad += 1
    return rows, bad


def _is_forbidden(source: object) -> bool:
    if not isinstance(source, str):
        return False
    low = source.lower()
    return any(bad in low for bad in FORBIDDEN_SOURCES)


def _distribution(values: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1
    # Deterministic: count desc, then key asc.
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _num_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "total": round(sum(values), 3),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def profile_rows(rows: list[dict]) -> dict:
    """Pure profile of parsed manifest rows (unknown keys ignored)."""
    captions = [
        r["caption"] for r in rows
        if isinstance(r.get("caption"), str) and r["caption"].strip()
    ]
    durations = [
        float(r["duration_s"]) for r in rows
        if isinstance(r.get("duration_s"), (int, float))
    ]
    fps_values = [
        r["fps"] for r in rows if isinstance(r.get("fps"), (int, float))
    ]
    resolutions = [
        f"{int(r['width'])}x{int(r['height'])}"
        for r in rows
        if isinstance(r.get("width"), (int, float))
        and isinstance(r.get("height"), (int, float))
    ]

    profile: dict = {
        "rows": len(rows),
        "with_caption": len(captions),
        "missing_caption": len(rows) - len(captions),
        "caption_chars": (
            _num_stats([float(len(c)) for c in captions]) if captions else None
        ),
        "sources": _distribution(
            [r.get("source") for r in rows if r.get("source") is not None]
        ),
        "forbidden_sources": sum(1 for r in rows if _is_forbidden(r.get("source"))),
        "probed": {
            "duration_s": len(durations),
            "fps": len(fps_values),
            "resolution": len(resolutions),
        },
        "duration_s": None,
        "fps": None,
        "resolutions": None,
    }

    if durations:
        stats = _num_stats(durations)
        buckets: dict[str, int] = {}
        for name, lo, hi in _DURATION_BUCKETS:
            buckets[name] = sum(
                1 for d in durations
                if d >= lo and (hi is None or d < hi)
            )
        stats["buckets"] = buckets
        profile["duration_s"] = stats
    if fps_values:
        profile["fps"] = _distribution(fps_values)
    if resolutions:
        profile["resolutions"] = _distribution(resolutions)
    return profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="jsonl data-card manifest")
    ap.add_argument("--out", default=None, help="also write the profile JSON here")
    args = ap.parse_args()

    try:
        rows, bad_lines = _load_rows(args.manifest)
    except OSError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    profile = profile_rows(rows)
    profile["manifest"] = str(args.manifest)
    profile["bad_json_lines"] = bad_lines
    text = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

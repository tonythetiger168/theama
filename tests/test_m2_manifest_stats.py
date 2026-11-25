# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/manifest_stats.py (P1b data-engine profiler).

Covers the pure profile function (caption coverage, duration buckets,
probed-media distributions, red-line source visibility) and the CLI
contract (tolerant jsonl parsing, deterministic stdout, --out).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "manifest_stats", ROOT / "scripts" / "manifest_stats.py"
)
ms = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ms)


def _write_manifest(tmp_path: Path, lines: list[str]) -> Path:
    path = tmp_path / "av_manifest.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(**kw) -> dict:
    base = {
        "video": "clips/a.mp4",
        "audio": "clips/a.wav",
        "caption": "a caption",
        "source": "pexels:1",
    }
    base.update(kw)
    return base


def test_profile_empty_manifest():
    profile = ms.profile_rows([])
    assert profile["rows"] == 0
    assert profile["with_caption"] == 0
    assert profile["missing_caption"] == 0
    assert profile["caption_chars"] is None
    assert profile["duration_s"] is None
    assert profile["fps"] is None
    assert profile["resolutions"] is None
    assert profile["forbidden_sources"] == 0
    assert profile["probed"] == {"duration_s": 0, "fps": 0, "resolution": 0}


def test_profile_caption_coverage_and_lengths():
    rows = [
        _row(caption="abcd"),
        _row(caption="abcdef"),
        _row(caption=""),          # empty -> missing
        _row(caption="   "),       # whitespace-only -> missing
        {k: v for k, v in _row().items() if k != "caption"},  # key absent
    ]
    profile = ms.profile_rows(rows)
    assert profile["rows"] == 5
    assert profile["with_caption"] == 2
    assert profile["missing_caption"] == 3
    chars = profile["caption_chars"]
    assert chars["count"] == 2
    assert chars["min"] == 4.0
    assert chars["max"] == 6.0
    assert chars["median"] == 5.0
    assert chars["mean"] == 5.0
    assert chars["total"] == 10.0


def test_profile_duration_buckets_and_totals():
    rows = [
        _row(duration_s=1.5),    # lt_2s
        _row(duration_s=2.0),    # 2_to_5s (edge inclusive at the low bound)
        _row(duration_s=4.999),  # 2_to_5s
        _row(duration_s=7.25),   # 5_to_10s
        _row(duration_s=15.0),   # 10_to_30s
        _row(duration_s=45.0),   # ge_30s
        _row(),                  # unprobed: excluded from duration stats
    ]
    stats = ms.profile_rows(rows)["duration_s"]
    assert stats["count"] == 6
    assert stats["total"] == 75.749
    assert stats["min"] == 1.5
    assert stats["max"] == 45.0
    assert stats["buckets"] == {
        "lt_2s": 1,
        "2_to_5s": 2,
        "5_to_10s": 1,
        "10_to_30s": 1,
        "ge_30s": 1,
    }
    assert ms.profile_rows(rows)["probed"]["duration_s"] == 6


def test_profile_fps_and_resolution_distributions():
    rows = [
        _row(fps=24.0, width=432, height=240),
        _row(fps=24.0, width=432, height=240),
        _row(fps=30.0, width=1280, height=704),
        _row(),  # unprobed
    ]
    profile = ms.profile_rows(rows)
    assert profile["fps"] == {"24.0": 2, "30.0": 1}
    assert profile["resolutions"] == {"432x240": 2, "1280x704": 1}
    assert profile["probed"] == {"duration_s": 0, "fps": 3, "resolution": 3}


def test_profile_source_mix_and_red_line_visibility():
    rows = [
        _row(source="pexels:1"),
        _row(source="pexels:2"),
        _row(source="mixkit:9"),
        _row(source="minimax-h3:77"),   # forbidden (license red line)
        _row(source="LTX-Showcase:3"),  # forbidden, case-insensitive
    ]
    profile = ms.profile_rows(rows)
    assert profile["sources"] == {"pexels:1": 1, "pexels:2": 1,
                                  "minimax-h3:77": 1, "LTX-Showcase:3": 1,
                                  "mixkit:9": 1}
    assert profile["forbidden_sources"] == 2


def test_profile_deterministic_key_order():
    rows = [_row(source="b:2"), _row(source="a:1"), _row(source="a:3")]
    profile = ms.profile_rows(rows)
    # count desc, then key asc
    assert list(profile["sources"].items()) == [("a:1", 1), ("a:3", 1), ("b:2", 1)]
    again = ms.profile_rows(list(reversed(rows)))
    assert json.dumps(profile, sort_keys=True) == json.dumps(again, sort_keys=True)


def test_cli_tolerates_bad_json_lines(tmp_path):
    manifest = _write_manifest(tmp_path, [
        json.dumps(_row(duration_s=3.0)),
        "{not json",
        json.dumps(["a", "list", "is", "not", "a", "row"]),
        "",
        json.dumps(_row(duration_s=5.0)),
    ])
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "manifest_stats.py"),
         "--manifest", str(manifest)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    profile = json.loads(proc.stdout)
    assert profile["rows"] == 2
    assert profile["bad_json_lines"] == 2
    assert profile["duration_s"]["count"] == 2
    assert profile["manifest"] == str(manifest)


def test_cli_missing_manifest_exits_1(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "manifest_stats.py"),
         "--manifest", str(tmp_path / "nope.jsonl")],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
    assert proc.stderr.strip()


def test_cli_deterministic_stdout_and_out_file(tmp_path):
    lines = [
        json.dumps(_row(source="pexels:1", duration_s=2.5, fps=24.0,
                        width=432, height=240)),
        json.dumps(_row(source="pexels:2", duration_s=8.0)),
        json.dumps(_row(caption="", source="mixkit:1")),
    ]
    manifest = _write_manifest(tmp_path, lines)
    cmd = [sys.executable, str(ROOT / "scripts" / "manifest_stats.py"),
           "--manifest", str(manifest)]
    first = subprocess.run(cmd, capture_output=True, text=True)
    second = subprocess.run(cmd, capture_output=True, text=True)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout  # byte-identical across runs

    out = tmp_path / "stats.json"
    proc = subprocess.run(cmd + ["--out", str(out)], capture_output=True, text=True)
    assert proc.returncode == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written == json.loads(proc.stdout)

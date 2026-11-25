# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the M2 A/V data pipeline: manifest ingestion + source gate."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.dataset import (  # noqa: E402
    AVDataset,
    FORBIDDEN_SOURCES,
    check_source,
    load_manifest,
    parse_manifest_row,
)


def _write_manifest(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _row(video="v/a.mp4", audio="a/a.wav", caption="a cat", source="pexels:1"):
    return json.dumps(
        {"video": video, "audio": audio, "caption": caption, "source": source}
    )


# ---- check_source ------------------------------------------------------------

def test_check_source_rejects_every_red_line_keyword():
    for bad in FORBIDDEN_SOURCES:
        with pytest.raises(ValueError, match="forbidden training source"):
            check_source(f"some-{bad}-dataset")
    with pytest.raises(ValueError):
        check_source("MiniMax-H3-clips")  # case-insensitive
    check_source("pexels:12345")  # clean source passes


# ---- parse_manifest_row ------------------------------------------------------

def test_parse_row_resolves_relative_paths_against_manifest_dir(tmp_path):
    sample = parse_manifest_row(
        json.loads(_row()), lineno=1, base_dir=tmp_path
    )
    assert sample.video_path == tmp_path / "v/a.mp4"
    assert sample.audio_path == tmp_path / "a/a.wav"
    assert sample.caption == "a cat" and sample.source == "pexels:1"


def test_parse_row_keeps_absolute_paths():
    sample = parse_manifest_row(
        json.loads(_row(video="/data/v.mp4", audio="/data/a.wav")),
        lineno=1,
        base_dir=Path("/elsewhere"),
    )
    assert sample.video_path == Path("/data/v.mp4")
    assert sample.audio_path == Path("/data/a.wav")


def test_parse_row_rejects_non_object_and_missing_or_empty_keys(tmp_path):
    with pytest.raises(ValueError, match="JSON object"):
        parse_manifest_row(["not", "a", "dict"], lineno=1, base_dir=tmp_path)
    with pytest.raises(ValueError, match="missing required keys"):
        parse_manifest_row({"video": "v.mp4"}, lineno=1, base_dir=tmp_path)
    with pytest.raises(ValueError, match="non-empty string"):
        parse_manifest_row(
            json.loads(_row(caption="  ")), lineno=1, base_dir=tmp_path
        )
    with pytest.raises(ValueError, match="non-empty string"):
        parse_manifest_row(
            {"video": "v", "audio": "a", "caption": 3, "source": "s"},
            lineno=1,
            base_dir=tmp_path,
        )


# ---- load_manifest: tolerant mode --------------------------------------------

def test_load_manifest_tolerant_skips_blanks_bad_json_and_bad_rows(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            _row(video="v/ok.mp4", audio="a/ok.wav"),
            "",
            "{not json",
            _row(caption=""),  # schema violation
            json.dumps(["array", "row"]),
            _row(video="v/ok2.mp4", audio="a/ok2.wav", source="LTX-2-gen"),
        ],
    )
    report = load_manifest(path)
    assert [s.video_path.name for s in report.samples] == ["ok.mp4"]
    kinds = [e.kind for e in report.errors]
    assert kinds.count("json") == 1
    assert kinds.count("schema") == 2
    assert kinds.count("forbidden_source") == 1
    assert all(e.lineno > 0 and e.message for e in report.errors)
    # forbidden row must never appear in the samples
    assert all("ltx" not in s.source.lower() for s in report.samples)


def test_load_manifest_dedupes_exact_video_audio_pairs(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            _row(),
            _row(),  # exact duplicate
            _row(video="v/other.mp4"),  # same audio, different video: kept
        ],
    )
    report = load_manifest(path)
    assert len(report.samples) == 2
    assert report.duplicates == 1


def test_load_manifest_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(tmp_path / "nope.jsonl")


def test_load_manifest_strict_raises_on_first_bad_row(tmp_path):
    path = _write_manifest(tmp_path, [_row(), "{bad json"])
    with pytest.raises(ValueError, match=r"manifest line 2: invalid JSON"):
        load_manifest(path, strict=True)
    path2 = _write_manifest(tmp_path, [_row(source="hailuo-gen")])
    with pytest.raises(ValueError, match=r"manifest line 1:.*forbidden"):
        load_manifest(path2, strict=True)
    # strict mode on a clean manifest yields zero errors
    clean = _write_manifest(tmp_path, [_row(video="v/x.mp4", audio="a/x.wav")])
    report = load_manifest(clean, strict=True)
    assert len(report.samples) == 1 and not report.errors


def test_manifest_report_summary(tmp_path):
    path = _write_manifest(
        tmp_path,
        [
            _row(source="pexels:1"),
            _row(video="v/b.mp4", audio="a/b.wav", source="pexels:2"),
            _row(video="v/c.mp4", audio="a/c.wav", source="pexels:1"),
            _row(video="v/d.mp4", audio="a/d.wav", source="h3-gen"),
        ],
    )
    summary = load_manifest(path).summary()
    assert summary["samples"] == 3
    assert summary["sources"] == {"pexels:1": 2, "pexels:2": 1}
    assert summary["errors"] == 1 and summary["forbidden_rejected"] == 1
    assert summary["duplicates"] == 0


# ---- AVDataset integration ----------------------------------------------------

def test_av_dataset_loads_manifest_and_keeps_rejections(tmp_path):
    path = _write_manifest(tmp_path, [_row(), "{nope"])
    ds = AVDataset(path, seconds=5, fps=24, sample_rate=44100)
    assert len(ds) == 1
    assert len(ds.errors) == 1 and ds.duplicates == 0
    assert ds.samples[0].caption == "a cat"
    # decode path is live: fixture media files do not exist on disk, so
    # __getitem__ surfaces the missing file loudly instead of returning zeros
    from theama.data.decode import MediaDecodeError

    with pytest.raises(MediaDecodeError, match="not found"):
        ds[0]


def test_av_dataset_strict_raises_on_bad_manifest(tmp_path):
    path = _write_manifest(tmp_path, [_row(source="lightricks-demo")])
    with pytest.raises(ValueError, match="forbidden"):
        AVDataset(path, strict=True)


def test_av_dataset_check_source_static_passthrough():
    with pytest.raises(ValueError):
        AVDataset.check_source("minimax-h3")
    AVDataset.check_source("pexels:9")


# ---- CLI gate script -----------------------------------------------------------

def test_check_av_manifest_cli_exit_codes(tmp_path):
    script = ROOT / "scripts" / "check_av_manifest.py"
    clean = _write_manifest(tmp_path, [_row()])
    ok = subprocess.run(
        [sys.executable, str(script), "--manifest", str(clean)],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0
    summary = json.loads(ok.stdout)
    assert summary["samples"] == 1 and summary["errors"] == 0

    dirty = _write_manifest(tmp_path, [_row(source="h3-gen"), "{bad"])
    bad = subprocess.run(
        [sys.executable, str(script), "--manifest", str(dirty), "--show-errors"],
        capture_output=True,
        text=True,
    )
    assert bad.returncode == 1
    assert "license-red-line" in bad.stderr
    assert "forbidden_source" in bad.stderr

    missing = subprocess.run(
        [sys.executable, str(script), "--manifest", str(tmp_path / "absent.jsonl")],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 1 and "manifest not found" in missing.stderr

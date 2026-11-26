# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/check_av_manifest.py (P1b data-engine gate).

Covers the CLI contract around the real tolerant loader: exit codes, JSON
summary on stdout, --show-errors row visibility on stderr, duplicate
accounting, and the license red-line rejection path. Media files are never
decoded — the manifest validator only checks schema/source, so path strings
are enough.
"""

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "check_av_manifest", ROOT / "scripts" / "check_av_manifest.py"
)
cam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cam)


def _row(video="clip.mp4", audio="clip.wav", caption="a caption", source="pexels"):
    return {"video": video, "audio": audio, "caption": caption, "source": source}


def _write_manifest(path: Path, lines) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_main(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["check_av_manifest.py", *argv])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cam.main()
    return rc, out.getvalue(), err.getvalue()


def test_happy_path_summary_exit_0(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    _write_manifest(
        manifest,
        [json.dumps(_row(video="b.mp4", audio="b.wav", source="mixkit")),
         json.dumps(_row(video="a.mp4", audio="a.wav", source="pexels"))],
    )
    rc, out, err = _run_main(["--manifest", str(manifest)], monkeypatch)
    assert rc == 0
    assert err == ""
    assert out.startswith('{\n  "samples": 2,')  # indent-2 JSON contract
    summary = json.loads(out)
    assert summary == {
        "samples": 2,
        "errors": 0,
        "duplicates": 0,
        "sources": {"mixkit": 1, "pexels": 1},
        "forbidden_rejected": 0,
    }


def test_missing_manifest_exit_1(tmp_path, monkeypatch):
    missing = tmp_path / "nope.jsonl"
    rc, out, err = _run_main(["--manifest", str(missing)], monkeypatch)
    assert rc == 1
    assert out == ""
    assert f"manifest not found: {missing}" in err


def test_bad_json_row_rejected_without_show_errors(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    _write_manifest(manifest, [json.dumps(_row()), "{not json"])
    rc, out, err = _run_main(["--manifest", str(manifest)], monkeypatch)
    assert rc == 1
    summary = json.loads(out)
    assert summary["samples"] == 1
    assert summary["errors"] == 1
    assert summary["forbidden_rejected"] == 0
    assert "manifest rejected 1 row(s)" in err
    assert "license-red-line" not in err
    assert '"kind"' not in err  # per-row details stay behind --show-errors


def test_schema_error_shown_with_show_errors(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    bad = _row()
    del bad["audio"]
    _write_manifest(manifest, [json.dumps(bad)])
    rc, out, err = _run_main(
        ["--manifest", str(manifest), "--show-errors"], monkeypatch
    )
    assert rc == 1
    assert json.loads(out)["errors"] == 1
    first = json.loads(err.splitlines()[0])
    assert first["lineno"] == 1
    assert first["kind"] == "schema"
    assert "missing required keys: ['audio']" in first["message"]
    assert '"video": "clip.mp4"' in first["raw"]
    assert "manifest rejected 1 row(s)" in err


def test_forbidden_source_red_line_exit_1(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    _write_manifest(
        manifest,
        [json.dumps(_row()), json.dumps(_row(video="h.mp4", audio="h.wav", source="minimax-h3-demo"))],
    )
    rc, out, err = _run_main(
        ["--manifest", str(manifest), "--show-errors"], monkeypatch
    )
    assert rc == 1
    summary = json.loads(out)
    assert summary["samples"] == 1
    assert summary["errors"] == 1
    assert summary["forbidden_rejected"] == 1
    first = json.loads(err.splitlines()[0])
    assert first["kind"] == "forbidden_source"
    assert "forbidden training source" in first["message"]
    assert "incl. 1 license-red-line source(s)" in err


def test_duplicates_counted_not_errors(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    dup = json.dumps(_row())
    _write_manifest(manifest, [dup, dup, json.dumps(_row(video="other.mp4", audio="other.wav"))])
    rc, out, err = _run_main(["--manifest", str(manifest)], monkeypatch)
    assert rc == 0
    assert err == ""
    summary = json.loads(out)
    assert summary["samples"] == 2
    assert summary["duplicates"] == 1
    assert summary["errors"] == 0


def test_multiple_errors_all_listed_with_show_errors(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    bad_schema = _row()
    del bad_schema["caption"]
    _write_manifest(manifest, ["{bad", json.dumps(bad_schema)])
    rc, out, err = _run_main(
        ["--manifest", str(manifest), "--show-errors"], monkeypatch
    )
    assert rc == 1
    assert json.loads(out)["errors"] == 2
    kinds = [json.loads(line)["kind"] for line in err.splitlines()[:2]]
    assert kinds == ["json", "schema"]
    assert "manifest rejected 2 row(s)" in err


def test_rerun_byte_identical(tmp_path, monkeypatch):
    manifest = tmp_path / "av.jsonl"
    _write_manifest(manifest, [json.dumps(_row()), json.dumps(_row(video="x.mp4", audio="x.wav"))])
    argv = ["--manifest", str(manifest)]
    rc1, out1, err1 = _run_main(argv, monkeypatch)
    rc2, out2, err2 = _run_main(argv, monkeypatch)
    assert (rc1, out1, err1) == (rc2, out2, err2) == (0, out1, "")

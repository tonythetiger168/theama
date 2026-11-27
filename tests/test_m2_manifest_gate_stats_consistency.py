# SPDX-License-Identifier: Apache-2.0
"""Cross-tool consistency tests for the P1b A/V manifest gate and profiler.

scripts/check_av_manifest.py (the pre-encode acceptance *gate*, built on
the real tolerant ``load_manifest``) and scripts/manifest_stats.py (the
read-only distribution *profiler*) both read the same data-card manifest.
Each tool has only been tested against its own fixtures; a refactor of
either side could silently diverge and the first witness would be a
confused operator about to burn GPU hours on encode runs.

These tests feed ONE manifest through BOTH tools and lock the exact
relationships:

- total accounting: every non-empty line is exactly one of
  gate.sample / gate.error / gate.duplicate, and exactly one of
  stats.row / stats.bad_json_line, so
  ``samples + errors + duplicates == rows + bad_json_lines``;
- classification divergence (documented, stable): a non-dict JSON line
  is a *schema* error to the gate but a *bad_json_line* to the profiler;
- forbidden-source divergence (documented, stable): the gate validates
  the schema before the source, so a forbidden row with a schema defect
  is a *schema* error, while the profiler counts any parsed row whose
  source matches the red-line substrings -- hence
  ``stats.forbidden_sources >= gate.forbidden_rejected`` with equality
  when every forbidden row is schema-valid;
- on a fully clean manifest the two source distributions agree exactly;
- role difference: the profiler exits 0 on manifests the gate rejects,
  both CLIs are byte-identical on rerun, and both exit 1 on a missing
  manifest.

Media files are never decoded -- the validator only checks schema/source,
so path strings are enough. Pure CPU, no encoders, no GPU.
"""

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate = _load_script("check_av_manifest")
stats = _load_script("manifest_stats")


def _row(video="clip.mp4", audio="clip.wav", caption="a caption", source="pexels"):
    return {"video": video, "audio": audio, "caption": caption, "source": source}


def _write_manifest(path: Path, lines) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(mod, argv, monkeypatch, prog):
    monkeypatch.setattr(sys, "argv", [prog, *argv])
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = mod.main()
    return rc, out.getvalue(), err.getvalue()


def _run_gate(manifest: Path, monkeypatch, *extra):
    return _run(gate, ["--manifest", str(manifest), *extra], monkeypatch,
                "check_av_manifest.py")


def _run_stats(manifest: Path, monkeypatch, *extra):
    return _run(stats, ["--manifest", str(manifest), *extra], monkeypatch,
                "manifest_stats.py")


def _mixed_manifest(tmp_path: Path) -> Path:
    """Seven non-empty lines covering every classification on both tools."""
    lines = [
        # r1: valid pexels row
        json.dumps(_row(video="a.mp4", audio="a.wav", source="pexels")),
        # r2: valid mixkit row
        json.dumps(_row(video="b.mp4", audio="b.wav", source="mixkit")),
        # r3: exact duplicate of r1 (gate: duplicate; stats: row)
        json.dumps(_row(video="a.mp4", audio="a.wav", source="pexels")),
        # r4: unparseable JSON (gate: json error; stats: bad line)
        "{not json",
        # r5: non-dict JSON (gate: schema error; stats: bad line)
        "[1, 2]",
        # r6: schema-invalid (missing caption; gate: schema error; stats: row)
        json.dumps({"video": "c.mp4", "audio": "c.wav", "source": "pexels"}),
        # r7: schema-valid forbidden source (both sides count it forbidden)
        json.dumps(_row(video="d.mp4", audio="d.wav",
                        source="minimax-h3-video")),
        # blank line: skipped by both
        "",
    ]
    return _write_manifest(tmp_path / "av.jsonl", lines)


def test_clean_manifest_counts_and_sources_agree(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path / "av.jsonl", [
        json.dumps(_row(video="a.mp4", audio="a.wav", source="pexels",
                        caption="first")),
        json.dumps(_row(video="b.mp4", audio="b.wav", source="mixkit",
                        caption="second")),
        json.dumps(_row(video="c.mp4", audio="c.wav", source="pexels",
                        caption="third")),
    ])
    g_rc, g_out, _ = _run_gate(manifest, monkeypatch)
    s_rc, s_out, _ = _run_stats(manifest, monkeypatch)
    assert g_rc == 0 and s_rc == 0
    g, s = json.loads(g_out), json.loads(s_out)
    assert g["samples"] == 3 and g["errors"] == 0 and g["duplicates"] == 0
    assert s["rows"] == 3 and s["bad_json_lines"] == 0
    # On a clean manifest the two source distributions agree exactly.
    assert g["sources"] == s["sources"] == {"mixkit": 1, "pexels": 2}
    # Every gated sample carries a non-empty caption.
    assert s["with_caption"] == g["samples"] and s["missing_caption"] == 0
    assert s["forbidden_sources"] == g["forbidden_rejected"] == 0


def test_total_accounting_on_mixed_manifest(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path)
    g_rc, g_out, _ = _run_gate(manifest, monkeypatch)
    s_rc, s_out, _ = _run_stats(manifest, monkeypatch)
    g, s = json.loads(g_out), json.loads(s_out)

    # Gate side: 2 samples, 1 duplicate, 4 errors (json r4, schema r5,
    # schema r6, forbidden r7) -> 7 non-empty lines.
    assert (g["samples"], g["duplicates"], g["errors"]) == (2, 1, 4)
    # Profiler side: rows r1/r2/r3/r6/r7 = 5, bad lines r4/r5 = 2.
    assert (s["rows"], s["bad_json_lines"]) == (5, 2)
    # The invariant: both tools account for every non-empty line exactly once.
    assert g["samples"] + g["errors"] + g["duplicates"] == 7
    assert s["rows"] + s["bad_json_lines"] == 7


def test_nondict_line_classification_divergence(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path)
    _, _, g_err = _run_gate(manifest, monkeypatch, "--show-errors")
    s_rc, s_out, _ = _run_stats(manifest, monkeypatch)
    s = json.loads(s_out)
    # Gate: the "[1, 2]" line is a *schema* rejection, visible with
    # --show-errors; profiler: it is one of the two bad_json_lines.
    schema_errors = [json.loads(line) for line in g_err.splitlines()
                     if line.strip().startswith("{")]
    kinds = sorted(e["kind"] for e in schema_errors)
    assert kinds == ["forbidden_source", "json", "schema", "schema"]
    assert any("JSON object" in e["message"] for e in schema_errors
               if e["kind"] == "schema")
    assert s["bad_json_lines"] == 2


def test_forbidden_divergence_schema_precedence(tmp_path, monkeypatch):
    # A forbidden-source row that is ALSO schema-invalid: the gate checks
    # the schema first (schema error), the profiler still counts the row
    # against the red-line substrings.
    manifest = _write_manifest(tmp_path / "av.jsonl", [
        json.dumps(_row(video="a.mp4", audio="a.wav", source="pexels")),
        json.dumps({"video": "d.mp4", "audio": "d.wav",
                    "source": "minimax-h3-video"}),  # missing caption
    ])
    g_rc, g_out, _ = _run_gate(manifest, monkeypatch)
    _, s_out, _ = _run_stats(manifest, monkeypatch)
    g, s = json.loads(g_out), json.loads(s_out)
    assert g["errors"] == 1 and g["forbidden_rejected"] == 0
    assert s["forbidden_sources"] == 1
    # Locked relationship: profiler count >= gate count.
    assert s["forbidden_sources"] >= g["forbidden_rejected"]


def test_forbidden_counts_agree_when_schema_valid(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path)
    _, g_out, _ = _run_gate(manifest, monkeypatch)
    _, s_out, _ = _run_stats(manifest, monkeypatch)
    g, s = json.loads(g_out), json.loads(s_out)
    # r7 is schema-valid, so both tools count exactly one forbidden row.
    assert g["forbidden_rejected"] == s["forbidden_sources"] == 1


def test_profiler_counts_rows_the_gate_drops(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path)
    _, g_out, _ = _run_gate(manifest, monkeypatch)
    _, s_out, _ = _run_stats(manifest, monkeypatch)
    g, s = json.loads(g_out), json.loads(s_out)
    # pexels appears in r1 (kept), r3 (duplicate) and r6 (schema error):
    # the profiler sees all three, the gate keeps one.
    assert s["sources"]["pexels"] == 3
    assert g["sources"]["pexels"] == 1
    assert g["sources"] == {"mixkit": 1, "pexels": 1}
    assert s["rows"] == 5


def test_role_difference_exit_codes(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path)
    g_rc, _, g_err = _run_gate(manifest, monkeypatch)
    s_rc, _, _ = _run_stats(manifest, monkeypatch)
    # The gate judges (exit 1, rejection note on stderr); the profiler
    # only describes (exit 0 on the same manifest).
    assert g_rc == 1 and "rejected 4 row(s)" in g_err
    assert "license-red-line" in g_err
    assert s_rc == 0


def test_both_clis_byte_identical_on_rerun(tmp_path, monkeypatch):
    manifest = _mixed_manifest(tmp_path)
    g_out1 = _run_gate(manifest, monkeypatch)[1]
    g_out2 = _run_gate(manifest, monkeypatch)[1]
    s_out1 = _run_stats(manifest, monkeypatch)[1]
    s_out2 = _run_stats(manifest, monkeypatch)[1]
    assert g_out1 == g_out2
    assert s_out1 == s_out2


def test_missing_manifest_both_exit_1(tmp_path, monkeypatch):
    missing = tmp_path / "nope.jsonl"
    g_rc, _, g_err = _run_gate(missing, monkeypatch)
    s_rc, _, s_err = _run_stats(missing, monkeypatch)
    assert g_rc == 1 and "manifest not found" in g_err
    assert s_rc == 1 and "nope.jsonl" in s_err

# SPDX-License-Identifier: Apache-2.0
"""Cross-tool consistency tests for the P1b joint-cache gate and profiler.

scripts/check_joint_latents.py (the pre-GPU acceptance *gate*) and
scripts/joint_latents_stats.py (the read-only distribution *profiler*) both
classify the same two cache manifests. The profiler docstring promises its
row classification mirrors the gate's precedence so "the counts line up
with what the gate reports" — but each tool has only been tested against
its own fixtures. A refactor of either side could silently diverge and the
first witness would be a confused operator on the GPU pod.

These tests feed ONE manifest pair through BOTH tools and lock the exact
relationships:

- bad-JSON / error / skipped / success counts agree on joinable rows
  (valid integer index, matching model);
- the two documented divergences stay stable: a skipped row under the
  wrong model or with a non-integer index is *model_mismatch* /
  *invalid_index* to the gate (its precedence matches
  load_reusable_rows) but still a *skipped* row to the profiler (which
  counts every non-error row carrying the flag);
- join arithmetic: gate.served == stats.join.served_indices whenever no
  caption drift, and gate.served == served_indices - caption_mismatch_rows
  otherwise; one-sided indices map missing_av_row <-> text_only_indices
  and missing_text_row <-> av_only_indices; a skipped AV row surfaces as
  skipped_av_row to the gate and as a text_only index to the profiler;
- role difference: the profiler still exits 0 on caches that fail the
  gate, and both CLIs are byte-identical on rerun over the same caches.

Fake caches are manifest + meta.json only — no encoders, no GPU.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_cache import MANIFEST_NAME as AV_MANIFEST  # noqa: E402
from theama.data.joint_cache_check import profile_manifest  # noqa: E402
from theama.data.text_cache import MANIFEST_NAME as TEXT_MANIFEST  # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gate_cli = _load_script("check_joint_latents")
stats = _load_script("joint_latents_stats")

AV_MODEL = "wan2.2-vae+dac"
TXT_MODEL = "umT5-xxl"


def _av_row(i: int, model: str = AV_MODEL, caption: str | None = None) -> dict:
    return {
        "index": i,
        "model": model,
        "caption": caption if caption is not None else f"第 {i} 段",
        "source": f"pexels:{i}",
        "video_latent": f"v{i}.pt",
        "audio_latent": f"a{i}.pt",
        "reused": True,
    }


def _text_row(i: int, model: str = TXT_MODEL, caption: str | None = None) -> dict:
    return {
        "index": i,
        "model": model,
        "caption": caption if caption is not None else f"第 {i} 段",
        "text_latent": f"t{i}.pt",
        "seq_len": 4,
        "reused": True,
    }


def _to_text(rows: list[dict]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


def _write_caches(tmp_path: Path, av_rows: list[dict], text_rows: list[dict]):
    av = tmp_path / "av"
    text = tmp_path / "text"
    av.mkdir(parents=True)
    text.mkdir(parents=True)
    (av / "meta.json").write_text(json.dumps({"model": AV_MODEL}))
    (text / "meta.json").write_text(json.dumps({"model": TXT_MODEL}))
    (av / AV_MANIFEST).write_text(_to_text(av_rows), encoding="utf-8")
    (text / TEXT_MANIFEST).write_text(_to_text(text_rows), encoding="utf-8")
    return av, text


def _run(script, *cli_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / f"{script}.py"), *cli_args],
        capture_output=True,
        text=True,
    )


def _gate(av: Path, text: Path, *extra: str) -> subprocess.CompletedProcess:
    return _run("check_joint_latents", "--av-cache", str(av),
                "--text-cache", str(text), *extra)


def _stats(av: Path, text: Path) -> subprocess.CompletedProcess:
    return _run("joint_latents_stats", "--av-cache", str(av),
                "--text-cache", str(text))


# ---- pure-function classification agreement ----------------------------------


def test_counts_agree_on_joinable_rows_mixed_manifest():
    text = _to_text(
        [
            _av_row(0),
            _av_row(1),
            {**_av_row(2), "skipped": True},
            {**_av_row(3), "error": "simulated VAE failure"},
        ]
    ) + "{bad json\n\n"  # one unparseable line + one blank line
    gate = profile_manifest(text, AV_MODEL)
    rows, bad = stats._load_rows(text)
    prof = stats._profile_av(rows, bad)
    assert gate["bad_json_lines"] == prof["bad_json_lines"] == 1
    assert gate["error_rows"] == prof["error_rows"] == 1
    assert gate["skipped_rows"] == prof["skipped_rows"] == 1
    # The gate's kept set is what the join can see — budget-skipped rows
    # included (load_reusable_rows keeps them; the join drops them with an
    # explicit reason). The profiler's success set excludes them. On
    # joinable rows: kept == success + skipped.
    assert gate["kept_rows"] == 3
    assert prof["success_rows"] == 2
    assert gate["kept_rows"] == prof["success_rows"] + prof["skipped_rows"]
    # Gate counts every non-blank line; the profiler's row list holds only
    # parseable objects — together they partition the file.
    assert gate["total_rows"] == prof["rows"] + prof["bad_json_lines"] == 5


def test_skipped_row_with_wrong_model_diverges_as_documented():
    row = {**_av_row(0, model="older-encoder"), "skipped": True}
    gate = profile_manifest(_to_text([row]), AV_MODEL)
    rows, bad = stats._load_rows(_to_text([row]))
    prof = stats._profile_av(rows, bad)
    # Gate precedence mirrors load_reusable_rows: model mismatch first.
    assert gate["model_mismatch_rows"] == 1
    assert gate["skipped_rows"] == 0
    assert gate["kept_rows"] == 0
    # Profiler counts every non-error row carrying the flag.
    assert prof["skipped_rows"] == 1
    assert prof["success_rows"] == 0


def test_skipped_row_with_invalid_index_diverges_as_documented():
    row = {**_av_row(0), "index": "0", "skipped": True}
    gate = profile_manifest(_to_text([row]), AV_MODEL)
    rows, bad = stats._load_rows(_to_text([row]))
    prof = stats._profile_av(rows, bad)
    assert gate["invalid_index_rows"] == 1
    assert gate["skipped_rows"] == 0
    assert prof["skipped_rows"] == 1
    assert prof["success_rows"] == 0


# ---- CLI-level agreement over the same cache pair -----------------------------


def test_cli_clean_caches_gate_and_stats_agree(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(i) for i in range(3)],
                             [_text_row(i) for i in range(3)])
    g, s = _gate(av, text), _stats(av, text)
    assert g.returncode == 0, g.stderr
    assert s.returncode == 0, s.stderr
    gate_out, stats_out = json.loads(g.stdout), json.loads(s.stdout)
    assert gate_out["served"] == stats_out["join"]["served_indices"] == 3
    assert stats_out["join"]["caption_mismatch_rows"] == 0
    assert (gate_out["av_manifest"]["kept_rows"]
            == stats_out["av"]["success_rows"] == 3)
    assert (gate_out["text_manifest"]["kept_rows"]
            == stats_out["text"]["success_rows"] == 3)


def test_cli_one_sided_rows_map_to_missing_drop_reasons(tmp_path):
    av, text = _write_caches(tmp_path, [_av_row(0), _av_row(1)],
                             [_text_row(1), _text_row(2)])
    strict = _gate(av, text)
    assert strict.returncode == 1
    assert "missing_text_row=1" in strict.stderr
    assert "missing_av_row=1" in strict.stderr
    g = _gate(av, text, "--allow-dropped")
    assert g.returncode == 0, g.stderr
    s = _stats(av, text)
    assert s.returncode == 0
    gate_out, stats_out = json.loads(g.stdout), json.loads(s.stdout)
    assert gate_out["served"] == stats_out["join"]["served_indices"] == 1
    assert (gate_out["dropped"]["missing_text_row"]
            == stats_out["join"]["av_only_indices"] == 1)
    assert (gate_out["dropped"]["missing_av_row"]
            == stats_out["join"]["text_only_indices"] == 1)


def test_cli_caption_mismatch_served_relationship(tmp_path):
    av, text = _write_caches(
        tmp_path,
        [_av_row(0, caption="貓"), _av_row(1)],
        [_text_row(0, caption="狗"), _text_row(1)],
    )
    g = _gate(av, text, "--allow-dropped")
    assert g.returncode == 0, g.stderr
    s = _stats(av, text)
    gate_out, stats_out = json.loads(g.stdout), json.loads(s.stdout)
    # The profiler's served_indices counts shared indices whether or not
    # captions agree; the gate drops drifted pairs.
    assert stats_out["join"]["served_indices"] == 2
    assert stats_out["join"]["caption_mismatch_rows"] == 1
    assert gate_out["dropped"]["caption_mismatch"] == 1
    assert (gate_out["served"]
            == stats_out["join"]["served_indices"]
            - stats_out["join"]["caption_mismatch_rows"] == 1)


def test_cli_skipped_av_row_surfaces_as_text_only_to_profiler(tmp_path):
    av, text = _write_caches(
        tmp_path,
        [{**_av_row(0), "skipped": True}, _av_row(1)],
        [_text_row(0), _text_row(1)],
    )
    g = _gate(av, text, "--allow-dropped")
    assert g.returncode == 0, g.stderr
    s = _stats(av, text)
    gate_out, stats_out = json.loads(g.stdout), json.loads(s.stdout)
    assert gate_out["dropped"]["skipped_av_row"] == 1
    # The skipped row is invisible to the profiler's serveable set, so the
    # text side looks one-sided there.
    assert stats_out["join"]["text_only_indices"] == 1
    assert gate_out["served"] == stats_out["join"]["served_indices"] == 1


def test_cli_profiler_describes_where_gate_judges_and_both_deterministic(
    tmp_path,
):
    av, text = _write_caches(tmp_path, [_av_row(0)], [_text_row(1)])
    # Zero served records: the gate always fails, the profiler stays a
    # read-only description.
    assert _gate(av, text, "--allow-dropped").returncode == 1
    s1 = _stats(av, text)
    assert s1.returncode == 0
    assert json.loads(s1.stdout)["join"]["served_indices"] == 0
    # Both CLIs are byte-identical on rerun over unchanged caches.
    av2, text2 = _write_caches(tmp_path / "again", [_av_row(0), _av_row(1)],
                               [_text_row(0), _text_row(1)])
    g1 = _gate(av2, text2)
    g2 = _gate(av2, text2)
    assert g1.returncode == g2.returncode == 0
    assert g1.stdout == g2.stdout and g1.stderr == g2.stderr
    s2 = _stats(av2, text2)
    s3 = _stats(av2, text2)
    assert s2.stdout == s3.stdout and s2.stderr == s3.stderr

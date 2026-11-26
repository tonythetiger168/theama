# SPDX-License-Identifier: Apache-2.0
"""CPU tests for scripts/score_cjk_text.py main()-level CLI contract.

test_m7a_script_split.py covers the zh-Hans/zh-Hant pack routing and its
three CLI paths; test_m7a.py only py_compiles the script. The rest of the
main() wiring had no coverage: the tesseract binary / language-pack
preflights, the default --eval-set / --out locations, --psm pass-through,
the missing-image failure contract and --strict, resume compaction of
error rows and old scorer versions, and gen-manifest provenance (mixed-
model warning, backfill onto rows scored before the manifest existed).
A wiring bug here only surfaces on the GPU pod after gen_cjk_images has
produced the seed images.

tesseract is never invoked: ``shutil.which`` / ``tesseract_languages`` /
``ocr_image`` are monkeypatched; the REAL seed loader, resume filter,
summary builder and file IO run, so the logic under test is production
code (mirrors test_p1a_score_eval_set).
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load_score_script():
    """Import scripts/score_cjk_text.py without executing main (CPU-safe:
    tesseract is only invoked inside main)."""
    import importlib.util

    path = ROOT / "scripts" / "score_cjk_text.py"
    spec = importlib.util.spec_from_file_location("score_cjk_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_seed(path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def _seed_row(id, **extra):
    row = {
        "id": id,
        "category": "poster",
        "prompt": "海報中央寫「夜雨偵探」",
        "expected_text": "夜雨偵探",
    }
    row.update(extra)
    return row


def _prime_cli(mod, monkeypatch, packs=("chi_tra", "chi_sim", "eng")):
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(mod, "tesseract_languages", lambda: set(packs))


def _fake_ocr(mod, monkeypatch, text="夜雨偵探", calls=None):
    def fake(path, langs, psm):
        if calls is not None:
            calls.append((Path(path).name, langs, psm))
        return text

    monkeypatch.setattr(mod, "ocr_image", fake)


def _setup(tmp_path, n=2, images=True):
    seed = tmp_path / "seed.jsonl"
    rows = [_seed_row(f"T2I-X-{k:03d}") for k in range(1, n + 1)]
    _write_seed(seed, rows)
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    if images is True:
        for r in rows:
            (img_dir / f"{r['id']}.png").write_bytes(b"fake-png")
    return seed, img_dir


def _argv(img_dir, seed, *extra):
    return ["score_cjk_text.py", "--images", str(img_dir),
            "--eval-set", str(seed), *extra]


def _read_rows(path):
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines()]


# -- preflight: binary + language packs ------------------------------------


def test_cli_tesseract_binary_missing(tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=1)
    monkeypatch.setattr(mod.shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 1
    assert "tesseract not found" in capsys.readouterr().err
    assert not (img_dir / "text_scores.jsonl").exists()


def test_cli_no_chinese_language_pack(tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=1)
    _prime_cli(mod, monkeypatch, packs=("eng", "osd"))
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 1
    err = capsys.readouterr().err
    assert "no Chinese tesseract language data" in err
    assert "eng" in err  # available packs echoed for the operator
    assert not (img_dir / "text_scores.jsonl").exists()


def test_cli_chi_sim_only_is_preferred_when_chi_tra_absent(
        tmp_path, monkeypatch):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=1)
    _prime_cli(mod, monkeypatch, packs=("chi_sim", "eng"))
    calls = []
    _fake_ocr(mod, monkeypatch, calls=calls)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    assert calls == [("T2I-X-001.png", "chi_sim+eng", 11)]


# -- happy path / output contract ------------------------------------------


def test_cli_happy_path_rows_stdout_and_summary(tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=2)
    _prime_cli(mod, monkeypatch)
    # Row 1 OCRs perfectly; row 2 drops half the expected characters.
    texts = {"T2I-X-001.png": "夜雨偵探", "T2I-X-002.png": "夜雨"}
    monkeypatch.setattr(
        mod, "ocr_image", lambda path, langs, psm: texts[Path(path).name])
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    out = capsys.readouterr().out
    assert "T2I-X-001: hit=True recall=1.00" in out
    assert "T2I-X-002: hit=False recall=0.50" in out
    assert "wrote 2 rows" in out
    rows = _read_rows(img_dir / "text_scores.jsonl")
    assert [r["id"] for r in rows] == ["T2I-X-001", "T2I-X-002"]
    rec = rows[0]
    assert rec["scorer"] == mod.SCORER_VERSION
    assert rec["ocr_langs"] == "chi_tra+eng"  # chi_tra wins when both exist
    assert rec["psm"] == 11
    assert rec["exact_hit"] is True
    assert rec["char_recall"] == 1.0
    assert rows[1]["char_recall"] == 0.5
    summary = json.loads(
        (img_dir / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["total"] == 2
    assert summary["scored"] == 2
    assert summary["failed"] == []
    assert summary["exact_hit_rate"] == 0.5
    assert summary["mean_char_recall"] == 0.75


def test_cli_rerun_is_byte_identical_and_skips_ocr(
        tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=2)
    _prime_cli(mod, monkeypatch)
    _fake_ocr(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    first = (img_dir / "text_scores.jsonl").read_bytes()
    first_summary = (img_dir / "text_scores.summary.json").read_bytes()
    # Second run: every row is already scored — OCR must never run and the
    # compaction rewrite must reproduce both files byte-identically.
    calls = []
    _fake_ocr(mod, monkeypatch, calls=calls)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    assert calls == []
    assert (img_dir / "text_scores.jsonl").read_bytes() == first
    assert (img_dir / "text_scores.summary.json").read_bytes() == first_summary


def test_cli_out_override_lands_rows_and_summary(tmp_path, monkeypatch):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=1)
    _prime_cli(mod, monkeypatch)
    _fake_ocr(mod, monkeypatch)
    custom = tmp_path / "custom" / "scores.jsonl"
    custom.parent.mkdir()
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed, "--out", str(custom)))
    assert mod.main() == 0
    assert len(_read_rows(custom)) == 1
    assert json.loads((tmp_path / "custom" / "scores.summary.json")
                      .read_text(encoding="utf-8"))["scored"] == 1
    assert not (img_dir / "text_scores.jsonl").exists()


def test_cli_psm_override_reaches_ocr_and_records(
        tmp_path, monkeypatch):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=1)
    _prime_cli(mod, monkeypatch)
    calls = []
    _fake_ocr(mod, monkeypatch, calls=calls)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed, "--psm", "6"))
    assert mod.main() == 0
    assert calls == [("T2I-X-001.png", "chi_tra+eng", 6)]
    rec = _read_rows(img_dir / "text_scores.jsonl")[0]
    assert rec["psm"] == 6
    summary = json.loads(
        (img_dir / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["psm"] == 6


# -- failure contract -------------------------------------------------------


def test_cli_missing_image_writes_error_row_and_exits_1(
        tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=2, images=False)
    (img_dir / "T2I-X-001.png").write_bytes(b"fake-png")  # row 2 missing
    _prime_cli(mod, monkeypatch)
    _fake_ocr(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 1
    err = capsys.readouterr().err
    assert "T2I-X-002: FAILED" in err
    assert "FileNotFoundError" in err
    rows = _read_rows(img_dir / "text_scores.jsonl")
    assert "error" not in rows[0]
    assert rows[1]["error"].startswith("FileNotFoundError")
    assert rows[1]["scorer"] == mod.SCORER_VERSION
    summary = json.loads(
        (img_dir / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["scored"] == 1
    assert summary["failed"] == ["T2I-X-002"]


def test_cli_strict_reraises_before_writing_error_row(tmp_path, monkeypatch):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=1, images=False)
    _prime_cli(mod, monkeypatch)
    _fake_ocr(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed, "--strict"))
    with pytest.raises(FileNotFoundError):
        mod.main()
    # The exception propagates before the error row is appended and before
    # the summary sidecar is written.
    assert not (img_dir / "text_scores.summary.json").exists()
    scores = img_dir / "text_scores.jsonl"
    assert not scores.exists() or scores.read_text(encoding="utf-8") == ""


# -- resume compaction ------------------------------------------------------


def test_cli_resume_retries_error_rows_and_drops_old_scorer_version(
        tmp_path, monkeypatch):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=3)
    _prime_cli(mod, monkeypatch)
    kept = {"id": "T2I-X-001", "category": "poster", "scorer": "cjk-text-1",
            "ocr_langs": "chi_tra+eng", "exact_hit": True, "char_recall": 1.0}
    prior_error = {"id": "T2I-X-002", "category": "poster",
                   "scorer": "cjk-text-1", "error": "RuntimeError: boom"}
    old_version = {"id": "T2I-X-003", "category": "poster",
                   "scorer": "cjk-text-0", "ocr_langs": "chi_tra+eng",
                   "exact_hit": False, "char_recall": 0.25}
    kept_line = json.dumps(kept, ensure_ascii=False)
    (img_dir / "text_scores.jsonl").write_text(
        "\n".join([kept_line, json.dumps(prior_error, ensure_ascii=False),
                   json.dumps(old_version, ensure_ascii=False)]) + "\n",
        encoding="utf-8")
    calls = []
    _fake_ocr(mod, monkeypatch, calls=calls)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    # Only the error row and the old-version row are re-scored.
    assert [c[0] for c in calls] == ["T2I-X-002.png", "T2I-X-003.png"]
    lines = (img_dir / "text_scores.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert lines[0] == kept_line  # kept row verbatim, first in file order
    rows = [json.loads(line) for line in lines]
    assert len(rows) == 3
    assert all("error" not in r for r in rows)
    assert rows[2]["scorer"] == "cjk-text-1"  # re-scored under current version
    assert rows[2]["char_recall"] == 1.0      # old 0.25 score replaced


# -- gen-manifest provenance ------------------------------------------------


def _write_manifest(img_dir, rows):
    (img_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")


def test_cli_manifest_mixed_models_warns_but_scores(
        tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=2)
    _write_manifest(img_dir, [
        {"id": "T2I-X-001", "model": "qwen-image-2512", "repo_id": "Qwen/a"},
        {"id": "T2I-X-002", "model": "z-image-turbo", "repo_id": "Z/b"},
    ])
    _prime_cli(mod, monkeypatch)
    _fake_ocr(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    err = capsys.readouterr().err
    assert "manifest mixes models" in err
    rows = _read_rows(img_dir / "text_scores.jsonl")
    assert [r["model"] for r in rows] == ["qwen-image-2512", "z-image-turbo"]
    summary = json.loads(
        (img_dir / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["model"] is None
    assert summary["models"] == ["qwen-image-2512", "z-image-turbo"]


def test_cli_manifest_backfills_model_onto_pre_manifest_kept_row(
        tmp_path, monkeypatch):
    mod = _load_score_script()
    seed, img_dir = _setup(tmp_path, n=2)
    _write_manifest(img_dir, [
        {"id": "T2I-X-001", "model": "qwen-image-2512", "repo_id": "Qwen/a"},
        {"id": "T2I-X-002", "model": "qwen-image-2512", "repo_id": "Qwen/a"},
    ])
    _prime_cli(mod, monkeypatch)
    # Row 1 was scored before gen_cjk_images started writing the manifest.
    prior = {"id": "T2I-X-001", "category": "poster", "scorer": "cjk-text-1",
             "ocr_langs": "chi_tra+eng", "exact_hit": True, "char_recall": 1.0}
    (img_dir / "text_scores.jsonl").write_text(
        json.dumps(prior, ensure_ascii=False) + "\n", encoding="utf-8")
    calls = []
    _fake_ocr(mod, monkeypatch, calls=calls)
    monkeypatch.setattr(sys, "argv", _argv(img_dir, seed))
    assert mod.main() == 0
    assert [c[0] for c in calls] == ["T2I-X-002.png"]  # row 1 kept
    rows = _read_rows(img_dir / "text_scores.jsonl")
    assert rows[0]["model"] == "qwen-image-2512"  # backfilled
    assert rows[1]["model"] == "qwen-image-2512"  # fresh provenance
    summary = json.loads(
        (img_dir / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["model"] == "qwen-image-2512"


# -- defaults ---------------------------------------------------------------


def test_cli_default_eval_set_and_out_location(tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()  # no images: every seed row fails to score
    _prime_cli(mod, monkeypatch)
    _fake_ocr(mod, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["score_cjk_text.py", "--images",
                                      str(img_dir)])
    assert mod.main() == 1
    repo_rows = (ROOT / "eval" / "cjk_t2i_v1.jsonl").read_text(
        encoding="utf-8").splitlines()
    rows = _read_rows(img_dir / "text_scores.jsonl")
    assert len(rows) == len(repo_rows)  # default eval set is the repo seed
    assert rows[0]["id"] == json.loads(repo_rows[0])["id"]
    assert all("error" in r for r in rows)
    summary = json.loads(
        (img_dir / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["scored"] == 0
    assert len(summary["failed"]) == len(repo_rows)

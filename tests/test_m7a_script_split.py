# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the M7a zh-Hans/zh-Hant script split (2026-11-26 decision).

The gate moved to zh-Hans-first with paired simplified/traditional seed
rows: the loader carries an optional ``script`` tag, the scorer OCRs each
tagged row with its matching tesseract pack, aborts before scoring when a
required pack is missing, re-scores tagged rows whose prior score used the
wrong pack, and splits the summary's headline rates per script. Untagged
(v1.2) seeds must behave byte-identically to before.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.image.eval import load_t2i_eval  # noqa: E402


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


def test_loader_reads_optional_script_tag(tmp_path):
    seed = tmp_path / "seed.jsonl"
    _write_seed(seed, [
        _seed_row("T2I-X-001", script="hans"),
        _seed_row("T2I-X-002", script="hant"),
        _seed_row("T2I-X-003"),
    ])
    rows = load_t2i_eval(seed)
    assert [r.script for r in rows] == ["hans", "hant", None]


def test_loader_rejects_invalid_script_tag(tmp_path):
    seed = tmp_path / "bad.jsonl"
    _write_seed(seed, [_seed_row("T2I-X-001", script="schinese")])
    with pytest.raises(ValueError, match="'script' must be one of"):
        load_t2i_eval(seed)
    _write_seed(seed, [_seed_row("T2I-X-001", script=1)])
    with pytest.raises(ValueError, match="'script' must be one of"):
        load_t2i_eval(seed)


def test_missing_script_packs_preflight():
    mod = _load_score_script()
    rows = [
        SimpleNamespace(id="a", script="hans"),
        SimpleNamespace(id="b", script="hant"),
        SimpleNamespace(id="c", script=None),
    ]
    assert mod.missing_script_packs(rows, {"chi_tra", "eng"}) == {"chi_sim"}
    assert mod.missing_script_packs(rows, {"chi_sim", "chi_tra", "eng"}) == set()
    # Untagged seeds never require extra packs.
    assert mod.missing_script_packs(rows[2:], set()) == set()


def test_row_ocr_langs_mapping():
    mod = _load_score_script()
    assert mod.row_ocr_langs("hans", "chi_tra+eng") == "chi_sim+eng"
    assert mod.row_ocr_langs("hant", "chi_sim+eng") == "chi_tra+eng"
    assert mod.row_ocr_langs(None, "chi_tra+eng") == "chi_tra+eng"


def test_keep_scored_row_resume_rule():
    mod = _load_score_script()
    # Untagged rows always keep, even without a recorded language.
    assert mod.keep_scored_row({}, None) is True
    # Tagged rows keep only when the prior score used the tag's pack.
    assert mod.keep_scored_row({"ocr_langs": "chi_sim+eng"}, "hans") is True
    assert mod.keep_scored_row({"ocr_langs": "chi_tra+eng"}, "hans") is False
    assert mod.keep_scored_row({"ocr_langs": "chi_sim+eng"}, "hant") is False


def _scored_hit(id, category="poster", recall=1.0, hit=True):
    return {"id": id, "category": category, "scorer": "cjk-text-1",
            "exact_hit": hit, "char_recall": recall}


def test_per_script_stats_split_zero_fill_and_untagged():
    mod = _load_score_script()
    rows = [
        SimpleNamespace(id="a", script="hans"),
        SimpleNamespace(id="b", script="hans"),
        SimpleNamespace(id="c", script="hant"),
        SimpleNamespace(id="d", script=None),
    ]
    hits = [
        _scored_hit("a", recall=1.0, hit=True),
        _scored_hit("b", recall=0.5, hit=False),
        _scored_hit("c", recall=1.0, hit=True),
        _scored_hit("d", recall=1.0, hit=True),
    ]
    stats = mod.per_script_stats(rows, hits)
    assert stats["hans"] == {"scored": 2, "exact_hit_rate": 0.5,
                             "mean_char_recall": 0.75}
    assert stats["hant"] == {"scored": 1, "exact_hit_rate": 1.0,
                             "mean_char_recall": 1.0}
    assert stats["untagged"] == 1
    # Zero-fill: a seed tagging only one script still reports both keys.
    only_hans = mod.per_script_stats(rows[:1], hits[:1])
    assert only_hans["hant"] == {"scored": 0, "exact_hit_rate": 0.0,
                                 "mean_char_recall": 0.0}
    assert "untagged" not in only_hans


def test_per_script_stats_none_without_tags():
    mod = _load_score_script()
    rows = [SimpleNamespace(id="a", script=None), SimpleNamespace(id="b")]
    hits = [_scored_hit("a"), _scored_hit("b")]
    assert mod.per_script_stats(rows, hits) is None


def test_summary_per_script_integration():
    mod = _load_score_script()
    tagged_rows = [SimpleNamespace(id="a", category="poster", script="hans")]
    scored = {"a": _scored_hit("a")}
    summary = mod.build_summary(tagged_rows, scored, "chi_sim+eng", 11, None)
    assert summary["per_script"]["hans"] == {
        "scored": 1, "exact_hit_rate": 1.0, "mean_char_recall": 1.0}
    # Untagged rows: no per_script key — v1.2 summaries stay byte-identical.
    plain_rows = [SimpleNamespace(id="a", category="poster", script=None)]
    plain = mod.build_summary(plain_rows, scored, "chi_tra+eng", 11, None)
    assert "per_script" not in plain


def _prime_cli(mod, monkeypatch, packs):
    monkeypatch.setattr(mod.shutil, "which", lambda name: "/usr/bin/tesseract")
    monkeypatch.setattr(mod, "tesseract_languages", lambda: packs)


def test_cli_aborts_when_script_pack_missing(tmp_path, monkeypatch, capsys):
    mod = _load_score_script()
    seed = tmp_path / "seed.jsonl"
    _write_seed(seed, [_seed_row("T2I-X-001", script="hans")])
    images = tmp_path / "imgs"
    images.mkdir()
    _prime_cli(mod, monkeypatch, {"chi_tra", "eng"})
    monkeypatch.setattr(sys, "argv", [
        "score_cjk_text.py", "--images", str(images), "--eval-set", str(seed)])
    assert mod.main() == 1
    assert "chi_sim" in capsys.readouterr().err
    # Fail-fast: aborts before writing any scores file.
    assert not (images / "text_scores.jsonl").exists()


def test_cli_rescores_tagged_row_scored_with_wrong_pack(tmp_path, monkeypatch):
    mod = _load_score_script()
    seed = tmp_path / "seed.jsonl"
    _write_seed(seed, [_seed_row("T2I-X-001", script="hans")])
    images = tmp_path / "imgs"
    images.mkdir()
    (images / "T2I-X-001.png").write_bytes(b"fake-png")
    # Prior score from the pre-split era used the traditional pack.
    prior = {"id": "T2I-X-001", "category": "poster", "scorer": "cjk-text-1",
             "ocr_langs": "chi_tra+eng", "exact_hit": True, "char_recall": 1.0}
    (images / "text_scores.jsonl").write_text(
        json.dumps(prior, ensure_ascii=False) + "\n", encoding="utf-8")
    _prime_cli(mod, monkeypatch, {"chi_sim", "chi_tra", "eng"})
    calls = []
    monkeypatch.setattr(mod, "ocr_image",
                        lambda path, langs, psm: calls.append(langs) or "夜雨偵探")
    monkeypatch.setattr(sys, "argv", [
        "score_cjk_text.py", "--images", str(images), "--eval-set", str(seed)])
    assert mod.main() == 0
    # Wrong-pack prior row was dropped and re-scored with chi_sim.
    assert calls == ["chi_sim+eng"]
    out = [(json.loads(line)) for line in
           (images / "text_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(out) == 1
    assert out[0]["ocr_langs"] == "chi_sim+eng"
    summary = json.loads(
        (images / "text_scores.summary.json").read_text(encoding="utf-8"))
    assert summary["per_script"]["hans"]["scored"] == 1


def test_cli_resume_keeps_tagged_row_with_matching_pack(tmp_path, monkeypatch):
    mod = _load_score_script()
    seed = tmp_path / "seed.jsonl"
    _write_seed(seed, [_seed_row("T2I-X-001", script="hans")])
    images = tmp_path / "imgs"
    images.mkdir()
    (images / "T2I-X-001.png").write_bytes(b"fake-png")
    prior = {"id": "T2I-X-001", "category": "poster", "scorer": "cjk-text-1",
             "ocr_langs": "chi_sim+eng", "exact_hit": True, "char_recall": 1.0}
    (images / "text_scores.jsonl").write_text(
        json.dumps(prior, ensure_ascii=False) + "\n", encoding="utf-8")
    _prime_cli(mod, monkeypatch, {"chi_sim", "chi_tra", "eng"})
    calls = []
    monkeypatch.setattr(mod, "ocr_image",
                        lambda path, langs, psm: calls.append(langs) or "夜雨偵探")
    monkeypatch.setattr(sys, "argv", [
        "score_cjk_text.py", "--images", str(images), "--eval-set", str(seed)])
    assert mod.main() == 0
    # Matching-pack prior row is kept — OCR never runs.
    assert calls == []
    out = [(json.loads(line)) for line in
           (images / "text_scores.jsonl").read_text(encoding="utf-8").splitlines()]
    assert out == [prior]

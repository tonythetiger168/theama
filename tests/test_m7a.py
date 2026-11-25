# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the M7a T2I eval-set loader and the CJK seed set."""

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.image.eval import (  # noqa: E402
    load_t2i_eval,
    summarize_t2i_eval,
)

CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def test_cjk_t2i_seed_set_loads_and_covers_categories():
    prompts = load_t2i_eval(ROOT / "eval" / "cjk_t2i_v1.jsonl")
    assert len(prompts) == 48  # v1.1 seed (24→48); full 100-prompt gate remains M7a acceptance
    assert len({p.id for p in prompts}) == len(prompts)
    assert all(CJK_RE.search(p.prompt) for p in prompts)
    assert all(p.expected_text for p in prompts)
    # Contract: the text the scorer looks for must be spelled out in the prompt.
    assert all(p.expected_text in p.prompt for p in prompts)
    summary = summarize_t2i_eval(prompts)
    assert summary["prompts"] == 48
    assert summary["with_expected_text"] == 48
    for category in ("poster", "sign", "product", "ui", "menu",
                     "newspaper", "ticket", "receipt", "banner"):
        assert category in summary["categories"]


def test_loader_rejects_duplicate_ids(tmp_path):
    row = json.dumps({
        "id": "T2I-X-001",
        "category": "poster",
        "prompt": "海報上寫「測試」",
        "expected_text": "測試",
    })
    path = tmp_path / "dup.jsonl"
    path.write_text(row + "\n" + row + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate prompt id"):
        load_t2i_eval(path)


def test_loader_rejects_missing_expected_text(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({
        "id": "T2I-X-001",
        "category": "poster",
        "prompt": "海報上寫字",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_t2i_eval(path)


def test_loader_rejects_bad_json_and_empty_file(tmp_path):
    bad = tmp_path / "bad_json.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_t2i_eval(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_t2i_eval(empty)
    with pytest.raises(FileNotFoundError, match="not found"):
        load_t2i_eval(tmp_path / "missing.jsonl")


def _load_gen_script():
    """Import scripts/gen_cjk_images.py without executing main (CPU-safe:
    torch/diffusers imports are lazy inside the script)."""
    import importlib.util

    path = ROOT / "scripts" / "gen_cjk_images.py"
    spec = importlib.util.spec_from_file_location("gen_cjk_images", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gen_script_model_table_matches_config_and_cards():
    import yaml

    mod = _load_gen_script()
    cfg = yaml.safe_load((ROOT / "configs" / "theama_base.yaml").read_text())
    backbone = cfg["model"] if "model" in cfg else cfg
    # The two official M7a paths stay pinned to the config backbones.
    assert mod.IMAGE_MODELS["qwen-image-2512"] == backbone["image_backbone"]
    assert mod.IMAGE_MODELS["z-image-turbo"] == backbone["image_backbone_fast"]
    # Experimental third path: MIT-licensed Ming-Image-0.1-Design.
    assert mod.IMAGE_MODELS["ming-image-design"] == "inclusionAI/Ming-Image-0.1-Design"
    assert "ming-image-design" in mod.EXPERIMENTAL_MODELS
    assert not ({"qwen-image-2512", "z-image-turbo"} & mod.EXPERIMENTAL_MODELS)
    # Every model has card defaults; values pin the official cards.
    assert set(mod.MODEL_DEFAULTS) == set(mod.IMAGE_MODELS)
    assert mod.MODEL_DEFAULTS["ming-image-design"] == {"steps": 12, "guidance": 1.0}
    assert mod.MODEL_DEFAULTS["z-image-turbo"] == {"steps": 8, "guidance": 0.0}
    assert mod.MODEL_DEFAULTS["qwen-image-2512"] == {"steps": 30, "guidance": 3.5}


def test_gen_script_resolution_snap():
    mod = _load_gen_script()
    # Models without buckets pass through unchanged.
    assert mod.snap_resolution("qwen-image-2512", 1024, 768) == (1024, 768)
    # Ming-Image renders only 1024/2048 squares; requests snap to nearest.
    assert mod.snap_resolution("ming-image-design", 1024, 1024) == (1024, 1024)
    assert mod.snap_resolution("ming-image-design", 2048, 2048) == (2048, 2048)
    assert mod.snap_resolution("ming-image-design", 512, 512) == (1024, 1024)
    assert mod.snap_resolution("ming-image-design", 1200, 1200) == (1024, 1024)
    assert mod.snap_resolution("ming-image-design", 1800, 1800) == (2048, 2048)
    assert mod.snap_resolution("ming-image-design", 4096, 4096) == (2048, 2048)


def _load_score_script():
    """Import scripts/score_cjk_text.py without executing main (CPU-safe:
    tesseract is only invoked inside main)."""
    import importlib.util

    path = ROOT / "scripts" / "score_cjk_text.py"
    spec = importlib.util.spec_from_file_location("score_cjk_text", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_script_manifest_loader(tmp_path):
    mod = _load_score_script()
    # Absent sidecar -> None (older generator output stays scorable).
    assert mod.load_gen_manifest(tmp_path) is None
    lines = [
        json.dumps({"id": "T2I-C-001", "model": "z-image-turbo",
                    "repo_id": "Tongyi-MAI/Z-Image-Turbo"}),
        "{not json",                       # malformed line: skipped, not fatal
        json.dumps({"model": "no-id"}),    # no id: skipped
        "",                                 # blank line: skipped
        json.dumps({"id": "T2I-C-002", "model": "z-image-turbo",
                    "repo_id": "Tongyi-MAI/Z-Image-Turbo"}),
    ]
    (tmp_path / "manifest.jsonl").write_text("\n".join(lines) + "\n",
                                             encoding="utf-8")
    manifest = mod.load_gen_manifest(tmp_path)
    assert set(manifest) == {"T2I-C-001", "T2I-C-002"}
    assert manifest["T2I-C-001"]["model"] == "z-image-turbo"


class _Row:
    """Minimal stand-in for the eval loader's row type."""

    def __init__(self, id, category="poster"):
        self.id = id
        self.category = category


def _scored_hit(id, category="poster", recall=1.0, hit=True):
    return {"id": id, "category": category, "scorer": "cjk-text-1",
            "exact_hit": hit, "char_recall": recall}


def test_score_script_summary_carries_provenance():
    mod = _load_score_script()
    rows = [_Row("T2I-C-001"), _Row("T2I-C-002")]
    scored = {
        "T2I-C-001": _scored_hit("T2I-C-001", recall=1.0),
        "T2I-C-002": _scored_hit("T2I-C-002", recall=0.5, hit=False),
    }
    manifest = {
        "T2I-C-001": {"id": "T2I-C-001", "model": "qwen-image-2512",
                      "repo_id": "Qwen/Qwen-Image-2512"},
        "T2I-C-002": {"id": "T2I-C-002", "model": "qwen-image-2512",
                      "repo_id": "Qwen/Qwen-Image-2512"},
    }
    summary = mod.build_summary(rows, scored, "chi_tra+eng", 11, manifest)
    # Rates unchanged from the pre-provenance behaviour.
    assert summary["scored"] == 2
    assert summary["exact_hit_rate"] == 0.5
    assert summary["mean_char_recall"] == 0.75
    # Single-model manifest -> headline model + repo attribution.
    assert summary["model"] == "qwen-image-2512"
    assert summary["models"] == ["qwen-image-2512"]
    assert summary["repo_ids"] == ["Qwen/Qwen-Image-2512"]
    assert "no_provenance" not in summary
    # No manifest -> no provenance keys at all (backwards compatible).
    plain = mod.build_summary(rows, scored, "chi_tra+eng", 11, None)
    assert "model" not in plain and "models" not in plain


def test_score_script_summary_flags_mixed_models_and_orphans():
    mod = _load_score_script()
    rows = [_Row("T2I-C-001"), _Row("T2I-C-002"), _Row("T2I-C-003")]
    scored = {r.id: _scored_hit(r.id) for r in rows}
    manifest = {
        "T2I-C-001": {"id": "T2I-C-001", "model": "qwen-image-2512",
                      "repo_id": "Qwen/Qwen-Image-2512"},
        "T2I-C-002": {"id": "T2I-C-002", "model": "z-image-turbo",
                      "repo_id": "Tongyi-MAI/Z-Image-Turbo"},
        # T2I-C-003 scored but absent from the manifest -> no provenance.
    }
    summary = mod.build_summary(rows, scored, "chi_tra+eng", 11, manifest)
    # Mixed models: no single headline model, both listed.
    assert summary["model"] is None
    assert summary["models"] == ["qwen-image-2512", "z-image-turbo"]
    assert summary["no_provenance"] == ["T2I-C-003"]

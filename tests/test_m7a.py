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
    assert len(prompts) == 24  # v1 seed; full 100-prompt gate remains M7a acceptance
    assert len({p.id for p in prompts}) == len(prompts)
    assert all(CJK_RE.search(p.prompt) for p in prompts)
    assert all(p.expected_text for p in prompts)
    summary = summarize_t2i_eval(prompts)
    assert summary["prompts"] == 24
    assert summary["with_expected_text"] == 24
    for category in ("poster", "sign", "product", "ui", "menu"):
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

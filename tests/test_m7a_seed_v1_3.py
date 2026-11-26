# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the M7a v1.3 paired-script CJK seed (eval/cjk_t2i_v1_3.jsonl).

v1.3 (2026-11-26 gate decision: zh-Hans-first acceptance, zh-Hant tracked in
parallel) pairs every v1.2 traditional prompt with a same-semantics simplified
counterpart in ONE jsonl so a single score_cjk_text run produces both
per_script tracks. Layout: lines 1-100 are the v1.2 rows byte-preserved in
their four required fields plus ``script: "hant"``; lines 101-200 are the
simplified pairs (``script: "hans"``, id +100). ``pair_of`` back-references
make the pairing explicit in both directions (the loader tolerates unknown
keys by design).

The conversion itself (OpenCC tw2sp + a small mainland-vocabulary patch list)
is a generation-time step — these tests lock the STRUCTURE so a later edit to
either side of a pair fails CI loudly instead of silently desynchronising the
zh-Hans gate from its zh-Hant shadow.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.image.eval import (  # noqa: E402
    load_t2i_eval,
    summarize_t2i_eval,
)

V1_PATH = ROOT / "eval" / "cjk_t2i_v1.jsonl"
V1_3_PATH = ROOT / "eval" / "cjk_t2i_v1_3.jsonl"

# Signboard texts identical in both scripts (digits / script-neutral glyphs).
# Locked so editing one side of a pair breaks the test.
SCRIPT_NEUTRAL_IDS = {
    "T2I-CJK-003", "T2I-CJK-009", "T2I-CJK-010", "T2I-CJK-014", "T2I-CJK-018",
    "T2I-CJK-031", "T2I-CJK-032", "T2I-CJK-035", "T2I-CJK-044", "T2I-CJK-059",
    "T2I-CJK-063", "T2I-CJK-075", "T2I-CJK-078", "T2I-CJK-086", "T2I-CJK-088",
    "T2I-CJK-089", "T2I-CJK-095", "T2I-CJK-098",
}


def _load_v1_3():
    return load_t2i_eval(V1_3_PATH)


def test_v1_3_loads_200_rows_unique_ids():
    prompts = _load_v1_3()
    assert len(prompts) == 200
    assert len({p.id for p in prompts}) == 200


def test_script_split_is_100_hans_100_hant_zero_untagged():
    prompts = _load_v1_3()
    assert sum(1 for p in prompts if p.script == "hant") == 100
    assert sum(1 for p in prompts if p.script == "hans") == 100
    assert all(p.script is not None for p in prompts)


def test_block_layout_hant_then_hans_in_id_order():
    prompts = _load_v1_3()
    assert [p.id for p in prompts[:100]] == [f"T2I-CJK-{k:03d}" for k in range(1, 101)]
    assert [p.id for p in prompts[100:]] == [f"T2I-CJK-{k:03d}" for k in range(101, 201)]
    assert all(p.script == "hant" for p in prompts[:100])
    assert all(p.script == "hans" for p in prompts[100:])


def test_hant_block_preserves_v1_2_required_fields_verbatim():
    """The v1.2 gate rows carry over untouched; only tags are added."""
    v1 = {p.id: p for p in load_t2i_eval(V1_PATH)}
    assert len(v1) == 100
    for p in _load_v1_3()[:100]:
        base = v1[p.id]
        assert (p.category, p.prompt, p.expected_text) == (
            base.category, base.prompt, base.expected_text,
        )


def test_pairing_is_symmetric_and_category_preserving():
    prompts = _load_t2i_eval_raw()
    by_id = {row["id"]: row for row in prompts}
    for k in range(1, 101):
        hant = by_id[f"T2I-CJK-{k:03d}"]
        hans = by_id[f"T2I-CJK-{100+k:03d}"]
        assert hant["pair_of"] == hans["id"]
        assert hans["pair_of"] == hant["id"]
        assert hans["category"] == hant["category"]


def _load_t2i_eval_raw():
    with V1_3_PATH.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_expected_text_appears_verbatim_in_prompt_all_rows():
    """The scorer contract (expected_text spelled out in prompt) holds on both tracks."""
    prompts = _load_v1_3()
    assert all(p.expected_text in p.prompt for p in prompts)


def test_instruction_flip_no_traditional_residue_in_hans_rows():
    """A simplified prompt must not ask for traditional rendering."""
    prompts = _load_v1_3()
    hant_by_pair = {p.id: p for p in prompts[:100]}
    for hans in prompts[100:]:
        assert "繁體" not in hans.prompt and "繁体" not in hans.prompt
        assert "簡體" not in hans.prompt
        hant = hant_by_pair[f"T2I-CJK-{int(hans.id[8:]) - 100:03d}"]
        if "繁體" in hant.prompt:
            assert "简体" in hans.prompt


def test_script_neutral_expected_text_set_is_locked():
    prompts = _load_v1_3()
    pairs = zip(prompts[:100], prompts[100:])
    neutral = {a.id for a, b in pairs if a.expected_text == b.expected_text}
    assert neutral == SCRIPT_NEUTRAL_IDS


def test_summary_doubles_v1_2_categories():
    prompts = _load_v1_3()
    summary = summarize_t2i_eval(prompts)
    assert summary["prompts"] == 200
    assert summary["with_expected_text"] == 200
    v1_summary = summarize_t2i_eval(load_t2i_eval(V1_PATH))
    doubled = {k: v * 2 for k, v in v1_summary["categories"].items()}
    assert summary["categories"] == doubled

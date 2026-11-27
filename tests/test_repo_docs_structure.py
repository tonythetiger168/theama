# SPDX-License-Identifier: Apache-2.0
"""Repo-health gate for the long-lived markdown docs.

docs/session-state.md is the shared two-session checkpoint, and
docs/status.md / docs/model-roadmap.md are the handover entry points.
These long CJK files are re-pushed in full almost every round, so the
cheap structural invariants that the RESTORE/SAVE protocols rely on
are pinned here on CPU instead of being re-checked by hand:

- known docs exist and docs/*.md does not silently shrink
- no unresolved merge-conflict markers anywhere
- every markdown table keeps one consistent pipe count per block
  (automates the manual "pipe-count distribution unchanged" check
  performed on each status.md follow-through edit)
- code fences are balanced per file
- session-state.md keeps its header block (line 1 ``---`` followed by
  the keys RESTORE reads: schema / updated_at / head / last_increment /
  next_hour), head stays a 40-hex commit id, updated_at stays ISO-8601
- session-state.md keeps the five sections of the SAVE-protocol
  structural scan (the "five-heading" check done after every SAVE)

Content assertions are deliberately avoided: suite counts, ledger
lines and schedules change every round -- only the skeleton is gated.
"""

import re
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "docs"

KNOWN_DOCS = [
    "README.md",
    "AGENTS.md",
    "CLAUDE.md",
    "docs/session-state.md",
    "docs/status.md",
    "docs/model-roadmap.md",
]

# Git conflict markers are exactly seven characters wide. A bare run of
# seven equals signs could in principle be a setext heading underline,
# but every doc in this repo uses ATX headings, so it is treated as a
# marker here.
CONFLICT_MARKER = re.compile(r"^(<{7}|={7}|>{7})(\s|$)")

HEADER_KEY = re.compile(r"^([a-z_0-9]+):")
HEAD_SHA = re.compile(r"^head: [0-9a-f]{40}\s*$")
REQUIRED_HEADER_KEYS = {"schema", "updated_at", "head", "last_increment", "next_hour"}

# The five sections of the SAVE-protocol structural scan, written as
# escapes to keep this file pure ASCII:
#   "xian-kuang one-liner" / "this-round ledger" / RESTORE protocol /
#   SAVE protocol / red lines
REQUIRED_SECTIONS = [
    "## \u73fe\u6cc1\u4e00\u53e5\u8a71",
    "## \u672c\u8f2a\u8a18\u9304",
    "## RESTORE \u5354\u8b70",
    "## SAVE \u5354\u8b70",
    "## \u7d05\u7dda",
]


def _md_files():
    return sorted(DOCS_DIR.glob("*.md")) + [
        REPO_ROOT / name for name in ("README.md", "AGENTS.md", "CLAUDE.md")
    ]


def _table_blocks(lines):
    """Yield (end_lineno, block) for each run of pipe-led lines outside
    code fences."""
    in_code = False
    block = []
    blocks = []
    for i, ln in enumerate(lines, 1):
        if ln.strip().startswith("```"):
            in_code = not in_code
            if block:
                blocks.append((i - 1, block))
                block = []
            continue
        if in_code:
            continue
        if ln.lstrip().startswith("|"):
            block.append(ln)
        elif block:
            blocks.append((i - 1, block))
            block = []
    if block:
        blocks.append((len(lines), block))
    return blocks


def test_known_docs_exist():
    missing = [name for name in KNOWN_DOCS if not (REPO_ROOT / name).is_file()]
    assert not missing, f"known docs missing: {missing}"
    found = list(DOCS_DIR.glob("*.md"))
    assert len(found) >= len([n for n in KNOWN_DOCS if n.startswith("docs/")]), (
        f"docs/*.md shrank below the known set: {sorted(p.name for p in found)}"
    )


def test_no_conflict_markers():
    hits = []
    for path in _md_files():
        for i, ln in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if CONFLICT_MARKER.match(ln):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{i}: {ln[:20]!r}")
    assert not hits, "unresolved conflict markers:\n" + "\n".join(hits)


def test_tables_keep_consistent_pipe_counts():
    bad = []
    for path in _md_files():
        lines = path.read_text(encoding="utf-8").splitlines()
        for end_i, block in _table_blocks(lines):
            if len(block) < 2:
                continue
            counts = {ln.count("|") for ln in block}
            if len(counts) > 1:
                bad.append(
                    f"{path.relative_to(REPO_ROOT)}: table ending line {end_i} "
                    f"has pipe counts {sorted(counts)}"
                )
    assert not bad, "inconsistent tables:\n" + "\n".join(bad)


def test_code_fences_balanced():
    bad = []
    for path in _md_files():
        fences = sum(
            1
            for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip().startswith("```")
        )
        if fences % 2:
            bad.append(f"{path.relative_to(REPO_ROOT)}: {fences} fence lines")
    assert not bad, "unbalanced code fences:\n" + "\n".join(bad)


def _session_state_lines():
    return (DOCS_DIR / "session-state.md").read_text(encoding="utf-8").splitlines()


def test_session_state_header_block():
    lines = _session_state_lines()
    assert lines[0] == "---", "session-state.md must open with a --- line"
    block = []
    for ln in lines[1:]:
        if not ln.strip():
            break
        block.append(ln)
    keys = {m.group(1) for ln in block if (m := HEADER_KEY.match(ln))}
    missing = REQUIRED_HEADER_KEYS - keys
    assert not missing, f"header block lost keys RESTORE reads: {sorted(missing)}"
    head_lines = [ln for ln in block if ln.startswith("head:")]
    assert len(head_lines) == 1 and HEAD_SHA.match(head_lines[0]), (
        f"head must be a single 40-hex commit id, got: {head_lines!r}"
    )
    updated = next(ln.split(":", 1)[1] for ln in block if ln.startswith("updated_at:"))
    stamp = updated.strip().strip('"').replace("Z", "+00:00")
    datetime.fromisoformat(stamp)  # raises if the ISO-8601 shape drifts


def test_session_state_required_sections():
    lines = _session_state_lines()
    for prefix in REQUIRED_SECTIONS:
        assert any(ln.startswith(prefix) for ln in lines), (
            f"session-state.md lost SAVE-protocol section starting {prefix!r}"
        )

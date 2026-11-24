# SPDX-License-Identifier: Apache-2.0
"""Versioned T2I eval-set loader for the Theama Image line (M7a).

The CJK text-rendering eval set is data, not code: keep the loader strict and
side-effect free so CI can validate the jsonl without importing diffusers or
touching model weights. Rows use the same jsonl convention as eval/t2v_v1.jsonl
plus ``expected_text`` — the exact string a renderer must get right for the
sample to count as a text-accuracy pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_KEYS = ("id", "category", "prompt", "expected_text")


@dataclass(frozen=True)
class T2IEvalPrompt:
    """One text-to-image eval prompt."""

    id: str
    category: str
    prompt: str
    expected_text: str


def _parse_row(obj: object, *, lineno: int) -> T2IEvalPrompt:
    if not isinstance(obj, dict):
        raise ValueError(f"line {lineno}: row must be a JSON object")
    missing = [key for key in REQUIRED_KEYS if key not in obj]
    if missing:
        raise ValueError(f"line {lineno}: missing required keys: {missing}")
    values: dict[str, str] = {}
    for key in REQUIRED_KEYS:
        value = obj[key]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"line {lineno}: key {key!r} must be a non-empty string")
        values[key] = value.strip()
    return T2IEvalPrompt(
        id=values["id"],
        category=values["category"],
        prompt=values["prompt"],
        expected_text=values["expected_text"],
    )


def load_t2i_eval(path: str | Path) -> list[T2IEvalPrompt]:
    """Load and validate a T2I eval jsonl file.

    Strict by design: eval sets are small and versioned, so a malformed row or
    duplicate id should fail CI loudly instead of silently shrinking the gate.
    Blank lines are ignored; unknown keys are allowed for future growth.
    """
    eval_path = Path(path)
    if not eval_path.is_file():
        raise FileNotFoundError(f"T2I eval set not found: {eval_path}")
    prompts: list[T2IEvalPrompt] = []
    seen: set[str] = set()
    with eval_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {lineno}: invalid JSON: {exc}") from exc
            row = _parse_row(obj, lineno=lineno)
            if row.id in seen:
                raise ValueError(f"line {lineno}: duplicate prompt id: {row.id}")
            seen.add(row.id)
            prompts.append(row)
    if not prompts:
        raise ValueError(f"T2I eval set is empty: {eval_path}")
    return prompts


def summarize_t2i_eval(prompts: list[T2IEvalPrompt]) -> dict:
    """Small summary for logs / CI output."""
    categories: dict[str, int] = {}
    for row in prompts:
        categories[row.category] = categories.get(row.category, 0) + 1
    return {
        "prompts": len(prompts),
        "categories": dict(sorted(categories.items())),
        "with_expected_text": sum(1 for row in prompts if row.expected_text),
    }

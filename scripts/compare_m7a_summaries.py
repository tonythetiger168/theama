#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M7a CPU comparator: side-by-side view of per-model score summaries.

After ``gen_cjk_images.py`` + ``score_cjk_text.py`` have run for each M7a
candidate model (qwen-image-2512 / z-image-turbo, plus the diagnostic-only
ming-image-design path), every run leaves a ``<stem>.summary.json`` next to
the scores. This tool lines those summaries up so the gate decision reads
from one deterministic table instead of three files:

    python scripts/compare_m7a_summaries.py \
        eval_out/m7a/qwen/qwen.summary.json \
        eval_out/m7a/zimage/zimage.summary.json \
        eval_out/m7a/ming/ming.summary.json --out compare.json

Each model is labelled by the summary's provenance ``model`` field (written
from the generator manifest; falls back to the file stem when absent).
Models listed in gen_cjk_images.EXPERIMENTAL_MODELS are flagged
``diagnostic_only`` — they are shown for reference but are not part of the
M7a acceptance gate (see docs/model-roadmap.md).

Output: a text table on stdout (overall exact_hit / char_recall plus the
union of per-category char_recall rows, ``-`` where a model never scored
that category) and, with --out, a deterministic JSON object (sorted keys,
stable rounding) so diffs between runs are meaningful.

Like manifest_stats.py this tool describes, it does not judge: exit code is
0 whenever every input summary is readable and structurally valid, 1 on a
missing/unreadable/invalid file or an ambiguous (duplicate) model label.
Comparing requires at least two summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gen_cjk_images import EXPERIMENTAL_MODELS  # noqa: E402

REQUIRED_KEYS = (
    "total",
    "scored",
    "exact_hit_rate",
    "mean_char_recall",
    "per_category_char_recall",
)


def load_summary(path: Path) -> dict:
    """Read one summary sidecar; raises ValueError on structural problems."""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: unreadable summary ({exc})") from exc
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: summary is not a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in obj]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if not isinstance(obj["per_category_char_recall"], dict):
        raise ValueError(f"{path}: per_category_char_recall is not an object")
    return obj


def label_for(path: Path, summary: dict) -> str:
    """Provenance model name when present, else the file stem."""
    model = summary.get("model")
    if isinstance(model, str) and model:
        return model
    name = path.name
    return name[: -len(".summary.json")] if name.endswith(".summary.json") else path.stem


def build_comparison(entries: list[tuple[Path, dict]]) -> dict:
    """Assemble the comparison; pure so tests need no files.

    ``entries`` is a list of (path, summary) pairs in CLI order — the first
    entry is the reference column shown leftmost. Raises ValueError when two
    summaries resolve to the same label (the table would be ambiguous).
    """
    models = []
    seen: set[str] = set()
    for path, summary in entries:
        label = label_for(path, summary)
        if label in seen:
            raise ValueError(f"duplicate model label {label!r} ({path})")
        seen.add(label)
        models.append(
            {
                "label": label,
                "path": str(path),
                "diagnostic_only": label in EXPERIMENTAL_MODELS,
                "total": summary["total"],
                "scored": summary["scored"],
                "exact_hit_rate": summary["exact_hit_rate"],
                "mean_char_recall": summary["mean_char_recall"],
            }
        )
    categories: dict[str, dict[str, float | None]] = {}
    for (path, summary), model in zip(entries, models):
        per_cat = summary["per_category_char_recall"]
        for cat in sorted(per_cat):
            categories.setdefault(cat, {m["label"]: None for m in models})
            categories[cat][model["label"]] = per_cat[cat]
    spread = {
        cat: round(
            max(v for v in cols.values() if v is not None)
            - min(v for v in cols.values() if v is not None),
            4,
        )
        for cat, cols in sorted(categories.items())
    }
    return {
        "categories": {cat: categories[cat] for cat in sorted(categories)},
        "gate_models": [m["label"] for m in models if not m["diagnostic_only"]],
        "models": models,
        "per_category_spread": spread,
    }


def render_table(comparison: dict) -> str:
    """Human-readable text table: one row per category, one column per model."""
    models = comparison["models"]
    labels = [m["label"] for m in models]
    header = ["category", *labels, "spread"]
    rows = []
    for cat, cols in comparison["categories"].items():
        rows.append(
            [cat]
            + [f"{cols[l]:.4f}" if cols[l] is not None else "-" for l in labels]
            + [f"{comparison['per_category_spread'][cat]:.4f}"]
        )
    overall = ["OVERALL(char_recall)"] + [
        f"{m['mean_char_recall']:.4f}" for m in models
    ] + ["-"]
    exact = ["OVERALL(exact_hit)"] + [
        f"{m['exact_hit_rate']:.4f}" for m in models
    ] + ["-"]
    table = [header, overall, exact, *rows]
    widths = [max(len(r[i]) for r in table) for i in range(len(header))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r)).rstrip() for r in table]
    flagged = [m["label"] for m in models if m["diagnostic_only"]]
    if flagged:
        lines.append(f"diagnostic-only (not in the M7a gate): {', '.join(flagged)}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("summaries", nargs="+", help="score_cjk_text <stem>.summary.json files")
    ap.add_argument("--out", default=None, help="also write the comparison JSON here")
    args = ap.parse_args()

    if len(args.summaries) < 2:
        print("[error] need at least two summaries to compare", file=sys.stderr)
        return 1
    entries = []
    for raw in args.summaries:
        path = Path(raw)
        try:
            entries.append((path, load_summary(path)))
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
    try:
        comparison = build_comparison(entries)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    print(render_table(comparison))
    if args.out:
        Path(args.out).write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[ok] comparison written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

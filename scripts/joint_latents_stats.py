#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Profile the P1b joint latent caches (CPU, stdlib only).

    python scripts/joint_latents_stats.py \
        --av-cache data/av/latents_dev --text-cache data/av/texts
    python scripts/joint_latents_stats.py \
        --av-cache data/av/latents_dev --text-cache data/av/texts \
        --out joint_stats.json

Where ``check_joint_latents.py`` is the pass/fail gate, this tool is the
cache profile you read *before* burning GPU hours on joint training — the
P1b analog of ``scores_stats.py`` / ``pairs_stats.py`` (P1a). It answers
the questions the gate's verdict counts cannot: latent shape mix (a cache
that silently mixes resolutions or clip lengths trains badly even when it
passes), caption duplication (identical captions share one content-
addressed text embedding — worth seeing), index coverage and gaps,
cache-hit (``reused``) ratio, per-model row mix, and the join view:
how many indices each cache actually serves, which side drops which
indices, and caption drift between the two manifests.

It only reads the two manifest texts (``av_latents.jsonl`` /
``text_latents.jsonl``); ``.pt`` latent files are never opened. Row
classification mirrors ``theama.data.joint_cache_check.profile_manifest``
precedence: bad JSON lines, then ``error`` rows, then ``skipped`` rows,
then success rows — so the counts line up with what the gate reports.

Exit code is 0 for any readable pair of manifests (this tool describes,
it does not judge) and 1 when either manifest cannot be read. Output is
one deterministic JSON object on stdout (sorted keys, stable rounding),
so diffing two profiles is meaningful.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.data.av_latent_cache import MANIFEST_NAME as AV_MANIFEST_NAME  # noqa: E402
from theama.data.text_cache import MANIFEST_NAME as TEXT_MANIFEST_NAME  # noqa: E402


def _load_rows(text: str) -> tuple[list[dict], int]:
    """Parse jsonl tolerantly; returns (object rows, bad-line count)."""
    rows: list[dict] = []
    bad = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            rows.append(obj)
        else:
            bad += 1
    return rows, bad


def _success_rows(rows: list[dict]) -> list[dict]:
    """Rows the join can serve: not error rows, not budget-skipped rows."""
    return [r for r in rows if "error" not in r and not r.get("skipped")]


def _num_stats(values: list[float]) -> dict:
    return {
        "count": len(values),
        "min": round(min(values), 3),
        "median": round(statistics.median(values), 3),
        "mean": round(statistics.fmean(values), 3),
        "max": round(max(values), 3),
    }


def _distribution(values: list[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in values:
        key = str(v)
        counts[key] = counts.get(key, 0) + 1
    # Deterministic: count desc, then key asc.
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def _shape_distribution(rows: list[dict], key: str) -> dict[str, int]:
    shapes = []
    for r in rows:
        shape = r.get(key)
        if isinstance(shape, list) and all(isinstance(d, int) for d in shape):
            shapes.append(json.dumps(shape))
    return _distribution(shapes)


def _index_coverage(rows: list[dict]) -> dict:
    indices = [r["index"] for r in rows
               if isinstance(r.get("index"), int)
               and not isinstance(r.get("index"), bool)]
    unique = set(indices)
    return {
        "rows": len(indices),
        "unique": len(unique),
        "duplicate_rows": len(indices) - len(unique),
        "min": min(unique) if unique else None,
        "max": max(unique) if unique else None,
        "gaps": (max(unique) - min(unique) + 1 - len(unique))
        if unique else 0,
    }


def _caption_stats(rows: list[dict]) -> dict:
    captions = [r["caption"] for r in rows
                if isinstance(r.get("caption"), str)]
    per_caption = _distribution(captions)
    shared = sum(1 for n in per_caption.values() if n > 1)
    lengths = [float(len(c)) for c in captions]
    return {
        "rows": len(captions),
        "unique": len(per_caption),
        "shared_by_multiple_rows": shared,
        "max_rows_per_caption": max(per_caption.values()) if per_caption else 0,
        "length_chars": _num_stats(lengths) if lengths else None,
    }


def _profile_av(rows: list[dict], bad_lines: int) -> dict:
    success = _success_rows(rows)
    return {
        "rows": len(rows),
        "bad_json_lines": bad_lines,
        "success_rows": len(success),
        "error_rows": sum(1 for r in rows if "error" in r),
        "skipped_rows": sum(1 for r in rows
                            if "error" not in r and r.get("skipped")),
        "reused_rows": sum(1 for r in success if r.get("reused")),
        "index": _index_coverage(success),
        "captions": _caption_stats(success),
        "video_shapes": _shape_distribution(success, "video_shape"),
        "audio_shapes": _shape_distribution(success, "audio_shape"),
        "models": _distribution([r.get("model") for r in success
                                 if isinstance(r.get("model"), str)]),
    }


def _profile_text(rows: list[dict], bad_lines: int) -> dict:
    success = _success_rows(rows)
    seq_lens = [float(r["seq_len"]) for r in success
                if isinstance(r.get("seq_len"), (int, float))
                and not isinstance(r.get("seq_len"), bool)]
    return {
        "rows": len(rows),
        "bad_json_lines": bad_lines,
        "success_rows": len(success),
        "error_rows": sum(1 for r in rows if "error" in r),
        "skipped_rows": sum(1 for r in rows
                            if "error" not in r and r.get("skipped")),
        "reused_rows": sum(1 for r in success if r.get("reused")),
        "captions": _caption_stats(success),
        "shapes": _shape_distribution(success, "shape"),
        "seq_len": _num_stats(seq_lens) if seq_lens else None,
        "models": _distribution([r.get("model") for r in success
                                 if isinstance(r.get("model"), str)]),
    }


def _profile_join(av_rows: list[dict], text_rows: list[dict]) -> dict:
    """Cross-manifest view over serveable (success) rows only."""
    av_by_index = {
        r["index"]: r for r in _success_rows(av_rows)
        if isinstance(r.get("index"), int)
        and not isinstance(r.get("index"), bool)
    }
    text_by_index = {
        r["index"]: r for r in _success_rows(text_rows)
        if isinstance(r.get("index"), int)
        and not isinstance(r.get("index"), bool)
    }
    shared = set(av_by_index) & set(text_by_index)
    caption_mismatch = sum(
        1 for i in shared
        if av_by_index[i].get("caption") != text_by_index[i].get("caption")
    )
    return {
        "av_indices": len(av_by_index),
        "text_indices": len(text_by_index),
        "served_indices": len(shared),
        "av_only_indices": len(set(av_by_index) - set(text_by_index)),
        "text_only_indices": len(set(text_by_index) - set(av_by_index)),
        "caption_mismatch_rows": caption_mismatch,
    }


def profile_caches(av_text: str, text_text: str) -> dict:
    """Pure profile of the two cache manifests (unknown keys ignored)."""
    av_rows, av_bad = _load_rows(av_text)
    text_rows, text_bad = _load_rows(text_text)
    return {
        "av": _profile_av(av_rows, av_bad),
        "text": _profile_text(text_rows, text_bad),
        "join": _profile_join(av_rows, text_rows),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-cache", required=True, help="AV latent cache dir")
    ap.add_argument("--text-cache", required=True, help="text latent cache dir")
    ap.add_argument("--out", default=None, help="also write the JSON profile here")
    args = ap.parse_args()

    av_path = Path(args.av_cache) / AV_MANIFEST_NAME
    text_path = Path(args.text_cache) / TEXT_MANIFEST_NAME
    try:
        av_text = av_path.read_text(encoding="utf-8")
        text_text = text_path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"joint_latents_stats: cannot read manifest: {exc}",
              file=sys.stderr)
        return 1

    profile = profile_caches(av_text, text_text)
    payload = json.dumps(profile, indent=2, sort_keys=True,
                         ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

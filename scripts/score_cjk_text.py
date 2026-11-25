#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M7a CPU scorer: CJK text accuracy of generated eval images via OCR.

Scores the PNGs produced by gen_cjk_images.py against the eval seed's
``expected_text`` — the M7a gate is "繁中文字準確率達官方報告同量級", so the
metric must be a reproducible number, not eyeballing. Uses the tesseract CLI
(system package; ``apt install tesseract-ocr tesseract-ocr-chi-tra``) to
avoid adding a Python OCR dependency.

    python scripts/score_cjk_text.py --images eval_out/m7a/qwen

Metrics per row (after NFKC + whitespace/punctuation normalization):
  - exact_hit:    expected_text appears verbatim in the OCR text
  - char_recall:  fraction of expected characters present in the OCR text

Resume is on by default: rows already scored with the same SCORER_VERSION
are kept, missing/failed ones are (re)scored, and the file is compacted.
A <out_stem>.summary.json sidecar (headline rates + per-category means) is
written on every run. Exit code is nonzero unless every prompt has a score.

OCR of stylized posters/signs is noisy — treat exact_hit as the headline
gate and char_recall as the diagnostic, and re-run with --psm 6 if sparse
text mode under-reads a layout-heavy batch.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.image import load_t2i_eval  # noqa: E402

SCORER_VERSION = "cjk-text-1"


def normalize(text: str) -> str:
    """NFKC, drop whitespace and punctuation — compare characters only."""
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith(("Z", "P"))
        and not ch.isspace()
    )


def char_recall(expected: str, ocr_text: str) -> float:
    """Fraction of expected characters present anywhere in the OCR text."""
    if not expected:
        return 0.0
    pool = list(normalize(ocr_text))
    hits = 0
    for ch in normalize(expected):
        if ch in pool:
            pool.remove(ch)
            hits += 1
    return hits / len(normalize(expected))


def tesseract_languages() -> set[str]:
    out = subprocess.run(
        ["tesseract", "--list-langs"], capture_output=True, text=True
    )
    # Output format: "List of available languages in ...:" then one per line.
    return {line.strip() for line in out.stdout.splitlines()[1:] if line.strip()}


def ocr_image(path: Path, langs: str, psm: int) -> str:
    proc = subprocess.run(
        ["tesseract", str(path), "stdout", "-l", langs, "--psm", str(psm)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"tesseract exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of <prompt_id>.png files")
    ap.add_argument("--eval-set", default=str(ROOT / "eval" / "cjk_t2i_v1.jsonl"))
    ap.add_argument("--out", default=None,
                    help="default: <images>/text_scores.jsonl")
    ap.add_argument("--psm", type=int, default=11,
                    help="tesseract page segmentation (11=sparse text)")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first image that fails to score")
    args = ap.parse_args()

    if shutil.which("tesseract") is None:
        print("tesseract not found; install tesseract-ocr (+ chi-tra data)",
              file=sys.stderr)
        return 1
    available = tesseract_languages()
    for wanted in ("chi_tra", "chi_sim"):
        if wanted in available:
            langs = wanted + "+eng"
            break
    else:
        print(f"no Chinese tesseract language data found (have: {sorted(available)})",
              file=sys.stderr)
        return 1

    rows = load_t2i_eval(args.eval_set)
    images_dir = Path(args.images)
    out_path = Path(args.out) if args.out else images_dir / "text_scores.jsonl"

    # Resume: keep prior rows from this scorer version keyed by prompt id.
    kept: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scorer") == SCORER_VERSION and "error" not in row:
                kept[row["id"]] = row
        with out_path.open("w", encoding="utf-8") as fh:
            for row in kept.values():
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    scored = dict(kept)
    failures = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            if row.id in scored:
                continue
            image = images_dir / f"{row.id}.png"
            try:
                if not image.is_file():
                    raise FileNotFoundError(f"missing {image.name}")
                ocr_text = ocr_image(image, langs, args.psm)
                norm_expected = normalize(row.expected_text)
                rec = {
                    "id": row.id,
                    "category": row.category,
                    "image": str(image),
                    "scorer": SCORER_VERSION,
                    "ocr_langs": langs,
                    "psm": args.psm,
                    "expected_text": row.expected_text,
                    "ocr_text": ocr_text.strip(),
                    "exact_hit": norm_expected in normalize(ocr_text),
                    "char_recall": round(char_recall(row.expected_text, ocr_text), 4),
                }
                print(f"{row.id}: hit={rec['exact_hit']} "
                      f"recall={rec['char_recall']:.2f}", flush=True)
            except Exception as exc:
                if args.strict:
                    raise
                failures += 1
                rec = {
                    "id": row.id,
                    "category": row.category,
                    "image": str(image),
                    "scorer": SCORER_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"{row.id}: FAILED ({rec['error']})", file=sys.stderr, flush=True)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            scored[row.id] = rec

    hits = [r for r in scored.values() if "error" not in r]
    by_category: dict[str, list[float]] = {}
    for rec in hits:
        by_category.setdefault(rec["category"], []).append(rec["char_recall"])
    summary = {
        "scorer": SCORER_VERSION,
        "ocr_langs": langs,
        "psm": args.psm,
        "total": len(rows),
        "scored": len(hits),
        "failed": [r["id"] for r in scored.values() if "error" in r],
        "exact_hit_rate": round(sum(1 for r in hits if r["exact_hit"]) / len(hits), 4)
        if hits else 0.0,
        "mean_char_recall": round(sum(r["char_recall"] for r in hits) / len(hits), 4)
        if hits else 0.0,
        "per_category_char_recall": {
            cat: round(sum(v) / len(v), 4) for cat, v in sorted(by_category.items())
        },
    }
    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(scored)} rows -> {out_path}")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if (failures or len(hits) < len(rows)) else 0


if __name__ == "__main__":
    raise SystemExit(main())

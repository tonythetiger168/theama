#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M7a CPU scorer: CJK text accuracy of generated eval images via OCR.

Scores the PNGs produced by gen_cjk_images.py against the eval seed's
``expected_text`` — the M7a gate is "zh-Hans 文字準確率達 Qwen-Image 官方
報告同量級"（2026-11-26 拍板：簡中優先、zh-Hant 並行統計不擋放行）, so the
metric must be a reproducible number, not eyeballing. Uses the tesseract CLI
(system package; ``apt install tesseract-ocr tesseract-ocr-chi-tra
tesseract-ocr-chi-sim``) to avoid adding a Python OCR dependency.

    python scripts/score_cjk_text.py --images eval_out/m7a/qwen

Metrics per row (after NFKC + whitespace/punctuation normalization):
  - exact_hit:    expected_text appears verbatim in the OCR text
  - char_recall:  fraction of expected characters present in the OCR text

Script split (簡繁對照 seeds): seed rows tagged ``script: "hans"|"hant"``
are OCR'd with the matching tesseract pack (chi_sim / chi_tra) instead of
the run-level default, and the summary gains a ``per_script`` block with
per-script scored counts, exact-hit rates and mean char recall. Missing
language packs abort the run before any scoring (fail-fast); tagged rows
whose prior scores used the wrong pack are re-scored on resume. Untagged
(v1.2) seeds behave exactly as before — one run-level language, no
``per_script`` key in the summary.

Resume is on by default: rows already scored with the same SCORER_VERSION
are kept, missing/failed ones are (re)scored, and the file is compacted.
A <out_stem>.summary.json sidecar (headline rates + per-category means) is
written on every run. Exit code is nonzero unless every prompt has a score.

Provenance: when the images dir contains a manifest.jsonl written by
gen_cjk_images.py, the summary and per-row records carry the generator's
model/repo_id so parallel runs (qwen/zimage/ming) stay attributable, scored
images with no manifest row are listed as no_provenance, and a manifest
mixing multiple models triggers a stderr warning. Provenance is metadata —
a missing or malformed manifest never fails scoring.

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

# Seed-row script tag -> required tesseract pack (zh-Hans-first gate, 2026-11-26).
LANG_FOR_SCRIPT = {"hans": "chi_sim", "hant": "chi_tra"}


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


def missing_script_packs(rows, available: set[str]) -> set[str]:
    """Tesseract packs required by the seed's script tags but not installed."""
    needed = {LANG_FOR_SCRIPT[row.script] for row in rows
              if getattr(row, "script", None) in LANG_FOR_SCRIPT}
    return {pack for pack in sorted(needed) if pack not in available}


def row_ocr_langs(script: str | None, default_langs: str) -> str:
    """OCR language spec for one seed row (tagged rows get their own pack)."""
    if script in LANG_FOR_SCRIPT:
        return LANG_FOR_SCRIPT[script] + "+eng"
    return default_langs


def keep_scored_row(prior: dict, script: str | None) -> bool:
    """Resume filter: a tagged row is re-scored when its prior score used a
    different OCR language than the tag requires; untagged rows always keep."""
    if script not in LANG_FOR_SCRIPT:
        return True
    return prior.get("ocr_langs") == row_ocr_langs(script, "")


def per_script_stats(rows, hits: list[dict]) -> dict | None:
    """zh-Hans / zh-Hant split of the headline rates (2026-11-26 gate decision).

    Returns None when no scored row carries a script tag, so summaries of
    untagged (v1.2) seeds stay byte-identical to before. Both script keys are
    always present (zero-filled) for determinism; ``untagged`` appears only
    when a partially tagged seed left some hits without a tag.
    """
    script_by_id = {row.id: getattr(row, "script", None) for row in rows}
    tagged = [r for r in hits if script_by_id.get(r["id"]) in LANG_FOR_SCRIPT]
    if not tagged:
        return None
    stats: dict[str, object] = {}
    for tag in ("hans", "hant"):
        group = [r for r in tagged if script_by_id[r["id"]] == tag]
        stats[tag] = {
            "scored": len(group),
            "exact_hit_rate": round(
                sum(1 for r in group if r["exact_hit"]) / len(group), 4
            ) if group else 0.0,
            "mean_char_recall": round(
                sum(r["char_recall"] for r in group) / len(group), 4
            ) if group else 0.0,
        }
    untagged = len(hits) - len(tagged)
    if untagged:
        stats["untagged"] = untagged
    return stats


def load_gen_manifest(images_dir: Path) -> dict[str, dict] | None:
    """Provenance sidecar written by gen_cjk_images.py (<out>/manifest.jsonl).

    Returns {prompt_id: row}, or None when the sidecar is absent (images
    produced by an older generator or by hand). Malformed lines are skipped
    with a warning — provenance is metadata, scoring must not fail on it.
    """
    path = images_dir / "manifest.jsonl"
    if not path.exists():
        return None
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            print(f"[warn] skipping malformed manifest line in {path.name}",
                  file=sys.stderr)
            continue
        if isinstance(rec, dict) and rec.get("id"):
            rows[rec["id"]] = rec
    return rows


def build_summary(rows, scored: dict[str, dict], langs: str, psm: int,
                  manifest: dict[str, dict] | None) -> dict:
    """Assemble the summary sidecar; pure so tests need no tesseract."""
    hits = [r for r in scored.values() if "error" not in r]
    by_category: dict[str, list[float]] = {}
    for rec in hits:
        by_category.setdefault(rec["category"], []).append(rec["char_recall"])
    summary = {
        "scorer": SCORER_VERSION,
        "ocr_langs": langs,
        "psm": psm,
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
    per_script = per_script_stats(rows, hits)
    if per_script is not None:
        summary["per_script"] = per_script
    if manifest is not None:
        models = sorted({m["model"] for m in manifest.values() if m.get("model")})
        summary["models"] = models
        summary["model"] = models[0] if len(models) == 1 else None
        summary["repo_ids"] = sorted(
            {m["repo_id"] for m in manifest.values() if m.get("repo_id")}
        )
        no_provenance = sorted(r["id"] for r in hits if r["id"] not in manifest)
        if no_provenance:
            summary["no_provenance"] = no_provenance
    return summary


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
    missing = missing_script_packs(rows, available)
    if missing:
        print(f"seed rows carry script tags needing tesseract packs "
              f"{sorted(missing)} (have: {sorted(available)}) — install "
              f"tesseract-ocr-chi-sim / tesseract-ocr-chi-tra", file=sys.stderr)
        return 1
    images_dir = Path(args.images)
    out_path = Path(args.out) if args.out else images_dir / "text_scores.jsonl"
    manifest = load_gen_manifest(images_dir)
    if manifest is not None:
        models = {m.get("model") for m in manifest.values()}
        if len(models) > 1:
            print(f"[warn] manifest mixes models {sorted(models)} — this "
                  f"summary aggregates them; score each model in its own dir",
                  file=sys.stderr)

    # Resume: keep prior rows from this scorer version keyed by prompt id.
    # Tagged rows whose prior score used the wrong OCR pack are re-scored.
    script_by_id = {row.id: row.script for row in rows}
    kept: dict[str, dict] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if (row.get("scorer") == SCORER_VERSION and "error" not in row
                    and keep_scored_row(row, script_by_id.get(row["id"]))):
                if manifest is not None and row["id"] in manifest:
                    # Backfill provenance on rows scored before it existed.
                    row.setdefault("model", manifest[row["id"]].get("model"))
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
                row_langs = row_ocr_langs(row.script, langs)
                ocr_text = ocr_image(image, row_langs, args.psm)
                norm_expected = normalize(row.expected_text)
                rec = {
                    "id": row.id,
                    "category": row.category,
                    "image": str(image),
                    "scorer": SCORER_VERSION,
                    "ocr_langs": row_langs,
                    "psm": args.psm,
                    "expected_text": row.expected_text,
                    "ocr_text": ocr_text.strip(),
                    "exact_hit": norm_expected in normalize(ocr_text),
                    "char_recall": round(char_recall(row.expected_text, ocr_text), 4),
                }
                if manifest is not None and row.id in manifest:
                    rec["model"] = manifest[row.id].get("model")
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

    summary = build_summary(rows, scored, langs, args.psm, manifest)
    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(scored)} rows -> {out_path}")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if (failures or summary["scored"] < len(rows)) else 0


if __name__ == "__main__":
    raise SystemExit(main())

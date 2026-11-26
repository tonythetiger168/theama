#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score generated clips with a registered reward model (GPU).

Walks a candidates dir (<root>/<prompt_id>/cand_*.mp4, or a flat dir of
MP4s whose stem is the prompt id), scores each clip against its prompt from
the eval set, and writes scores.jsonl rows:

    {prompt_id, prompt, video, reward, overall, dims}

Resume is on by default: clips already present with a valid score for the
SAME reward model in the output file are skipped, and rows that previously
failed (recorded with an "error" field) are retried. Use --no-resume to
re-score from scratch and --strict to abort on the first scoring failure.
A scores.summary.json sidecar (counts, mean overall, per-dim means) is
written next to the output on every run.

    python scripts/score_eval_set.py --videos dpo_out/candidates --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def collect(videos: Path, prompt_by_id: dict[str, str]) -> list[tuple[str, str, str]]:
    """Return (prompt_id, video_path, prompt) triples."""
    items = []
    for mp4 in sorted(videos.rglob("*.mp4")):
        prompt_id = mp4.parent.name if mp4.parent.name in prompt_by_id else mp4.stem
        if prompt_id in prompt_by_id:
            items.append((prompt_id, str(mp4), prompt_by_id[prompt_id]))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--prompts", default=str(ROOT / "eval" / "t2v_v1.jsonl"))
    ap.add_argument("--reward", default="videoreward")
    ap.add_argument("--out", default=None, help="default: <videos>/scores.jsonl")
    ap.add_argument("--smoke", action="store_true", help="score only 2 clips")
    ap.add_argument("--limit", type=int, default=None, help="score at most N clips")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing output and re-score everything")
    ap.add_argument("--strict", action="store_true",
                    help="abort on the first clip that fails to score")
    args = ap.parse_args()

    prompt_by_id = {}
    for line in Path(args.prompts).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            prompt_by_id[row["id"]] = row["prompt"]

    items = collect(Path(args.videos), prompt_by_id)
    if args.smoke:
        items = items[:2]
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print("no MP4s matched to eval prompts", file=sys.stderr)
        return 1

    from theama.reward.score import (
        VideoRewardScorer,
        load_successful_scores,
        summarize_scores,
    )

    out_path = Path(args.out) if args.out else Path(args.videos) / "scores.jsonl"

    # Resume: keep prior successful rows for this reward model, compact the
    # file (dropping error rows so they are retried), score only the rest.
    # --no-resume re-scores from scratch: the existing file is truncated
    # (not appended to), so re-runs never accumulate duplicate rows that
    # downstream profilers (scores_stats) would flag.
    kept: dict[str, dict] = {}
    if out_path.exists():
        if not args.no_resume:
            kept = load_successful_scores(out_path, reward=args.reward)
        with out_path.open("w") as fh:
            for row in kept.values():
                fh.write(json.dumps(row) + "\n")
    todo = [item for item in items if item[1] not in kept]
    skipped = len(items) - len(todo)
    print(f"{len(items)} clips matched: {skipped} already scored, {len(todo)} to score")

    rows = list(kept.values())
    if todo:
        scorer = VideoRewardScorer(args.reward)
        scorer.load()
        failures = 0
        with out_path.open("a") as fh:
            for prompt_id, video, prompt in todo:
                try:
                    res = scorer.score(video, prompt)
                    row = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "video": video,
                        "reward": args.reward,
                        "overall": res.overall,
                        "dims": res.dims,
                    }
                    print(f"{prompt_id} {Path(video).name}: {res.overall:.4f}",
                          flush=True)
                except Exception as exc:
                    if args.strict:
                        raise
                    failures += 1
                    row = {
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "video": video,
                        "reward": args.reward,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    print(f"{prompt_id} {Path(video).name}: FAILED ({row['error']})",
                          file=sys.stderr, flush=True)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                rows.append(row)
        if failures:
            print(f"{failures} clip(s) failed to score (rerun to retry)",
                  file=sys.stderr)

    summary = {"reward": args.reward, **summarize_scores(rows)}
    summary_path = out_path.with_name(out_path.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {len(rows)} rows -> {out_path}")
    print(f"summary: {json.dumps(summary)} -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

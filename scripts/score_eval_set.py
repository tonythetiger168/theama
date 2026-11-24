#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Score generated clips with a registered reward model (GPU).

Walks a candidates dir (<root>/<prompt_id>/cand_*.mp4, or a flat dir of
MP4s whose stem is the prompt id), scores each clip against its prompt from
the eval set, and writes scores.jsonl rows:

    {prompt_id, prompt, video, overall, dims}

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
    args = ap.parse_args()

    prompt_by_id = {}
    for line in Path(args.prompts).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            prompt_by_id[row["id"]] = row["prompt"]

    items = collect(Path(args.videos), prompt_by_id)
    if args.smoke:
        items = items[:2]
    if not items:
        print("no MP4s matched to eval prompts", file=sys.stderr)
        return 1

    from theama.reward.score import VideoRewardScorer

    scorer = VideoRewardScorer(args.reward)
    scorer.load()

    out_path = Path(args.out) if args.out else Path(args.videos) / "scores.jsonl"
    with out_path.open("w") as fh:
        for prompt_id, video, prompt in items:
            res = scorer.score(video, prompt)
            row = {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "video": video,
                "overall": res.overall,
                "dims": res.dims,
            }
            fh.write(json.dumps(row) + "\n")
            print(f"{prompt_id} {Path(video).name}: {res.overall:.4f}", flush=True)
    print(f"wrote {len(items)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

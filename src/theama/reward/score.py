# SPDX-License-Identifier: Apache-2.0
"""Lazy VideoReward scorer wrapper.

Module import is CPU-safe (no torch/transformers at import time); heavy deps
load inside functions so the sandbox test-suite can import this package.

NOTE: the invocation in `_invoke` follows the KlingTeam/VideoReward model card
(AutoModel + AutoProcessor, trust_remote_code). If the card's call signature
has drifted, adjust ONLY `_invoke` — everything downstream (scores dict shape)
stays stable. First local GPU run of scripts/score_eval_set.py --smoke is the
verification step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .config import RewardSpec, get_reward_spec


@dataclass
class ScoreResult:
    """Scores for one (video, prompt) pair. dims may be empty if the backend
    only exposes a scalar overall reward."""

    overall: float
    dims: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"overall": self.overall, "dims": self.dims}


class VideoRewardScorer:
    """Scores videos with a registered reward model. GPU required."""

    def __init__(self, spec_name: str = "videoreward", device: str = "cuda"):
        self.spec: RewardSpec = get_reward_spec(spec_name)
        self.device = device
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch  # noqa: F401  (lazy: GPU-only)
        from transformers import AutoModel, AutoProcessor

        self._model = (
            AutoModel.from_pretrained(
                self.spec.repo_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            .to(self.device)
            .eval()
        )
        self._processor = AutoProcessor.from_pretrained(
            self.spec.repo_id, trust_remote_code=True
        )

    def score(self, video_path: str, prompt: str) -> ScoreResult:
        if self._model is None:
            self.load()
        return self._invoke(video_path, prompt)

    def _invoke(self, video_path: str, prompt: str) -> ScoreResult:
        """Single backend-specific call site (see module docstring)."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Model card pattern: processor builds inputs, model returns reward
        # logits whose last position is the scalar score (multi-dim heads
        # return a vector we map onto named dims).
        inputs = self._processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(self.device)
        import torch

        with torch.inference_mode():
            out = self._model(**inputs)
        scores = out.logits[0, -1].float().cpu().tolist()
        if isinstance(scores, float):
            return ScoreResult(overall=scores)
        dims = {
            name: float(v)
            for name, v in zip(
                ("visual_quality", "temporal_consistency", "dynamic_degree",
                 "text_alignment"),
                scores,
            )
        }
        return ScoreResult(overall=sum(scores) / len(scores), dims=dims)


def score_videos(
    items: list[tuple[str, str]], spec_name: str = "videoreward", device: str = "cuda"
) -> list[ScoreResult]:
    """Score a list of (video_path, prompt) pairs with one loaded model."""
    scorer = VideoRewardScorer(spec_name, device)
    scorer.load()
    return [scorer.score(v, p) for v, p in items]


# ---- scores.jsonl resume / summary helpers (pure logic, CPU-safe) ----------


def load_successful_scores(path: str | Path, reward: str | None = None) -> dict[str, dict]:
    """Read an existing scores.jsonl into {video_path: row} for resume.

    Only rows carrying a numeric ``overall`` are kept; error rows are dropped
    so a re-run retries them. When ``reward`` is given, rows recorded with a
    different reward model are dropped too (makes resume safe across scorer
    changes); rows with no ``reward`` key predate per-row tagging and are
    treated as the historical default "videoreward". On duplicate video
    paths the later row wins. Missing file / malformed lines are skipped.
    """
    kept: dict[str, dict] = {}
    p = Path(path)
    if not p.exists():
        return kept
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row.get("overall"), (int, float)):
            continue
        row_reward = row.get("reward", "videoreward")
        if reward is not None and row_reward != reward:
            continue
        video = row.get("video")
        if isinstance(video, str) and video:
            kept[video] = row
    return kept


def summarize_scores(rows: list[dict]) -> dict:
    """Aggregate score rows: scored/failed counts, mean overall, per-dim means.

    Rows with an ``error`` key (or no numeric ``overall``) count as failed and
    are excluded from the means. Returns ``mean_overall: None`` when nothing
    was scored.
    """
    scored = [r for r in rows if isinstance(r.get("overall"), (int, float))]
    failed = len(rows) - len(scored)
    dim_sums: dict[str, float] = {}
    dim_counts: dict[str, int] = {}
    for r in scored:
        dims = r.get("dims") or {}
        for name, value in dims.items():
            if isinstance(value, (int, float)):
                dim_sums[name] = dim_sums.get(name, 0.0) + float(value)
                dim_counts[name] = dim_counts.get(name, 0) + 1
    return {
        "scored": len(scored),
        "failed": failed,
        "mean_overall": (
            sum(float(r["overall"]) for r in scored) / len(scored) if scored else None
        ),
        "dims": {name: dim_sums[name] / dim_counts[name] for name in sorted(dim_sums)},
    }

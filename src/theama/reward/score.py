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

from dataclasses import dataclass, field

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

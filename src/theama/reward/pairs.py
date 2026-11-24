# SPDX-License-Identifier: Apache-2.0
"""Preference-pair construction from scored candidate sets (pure logic).

Input: scores.jsonl rows {prompt_id, prompt, video, overall} with several
candidate generations per prompt_id (different seeds). Output: one
(best, worst) pair per prompt when the score margin clears the threshold —
margin filtering is the main defence against noise-driven DPO drift.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrefPair:
    prompt_id: str
    prompt: str
    win_video: str
    lose_video: str
    win_score: float
    lose_score: float

    @property
    def margin(self) -> float:
        return self.win_score - self.lose_score

    def as_dict(self) -> dict:
        return {
            "prompt_id": self.prompt_id,
            "prompt": self.prompt,
            "win_video": self.win_video,
            "lose_video": self.lose_video,
            "win_score": self.win_score,
            "lose_score": self.lose_score,
            "margin": self.margin,
        }


def build_pairs(
    rows: list[dict], margin: float = 0.5, max_pairs_per_prompt: int = 1
) -> list[PrefPair]:
    """Group scored candidates by prompt and emit (best, worst) pairs.

    rows: dicts with keys prompt_id, prompt, video, overall.
    Deterministic: ties broken by video path so runs reproduce.
    Prompts with <2 usable candidates or margin < threshold yield nothing.
    """
    by_prompt: dict[str, list[dict]] = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_id"], []).append(r)

    pairs: list[PrefPair] = []
    for prompt_id in sorted(by_prompt):
        cands = sorted(
            by_prompt[prompt_id], key=lambda r: (-r["overall"], r["video"])
        )
        if len(cands) < 2:
            continue
        best = cands[0]
        # Pair the best candidate against the worst `max_pairs_per_prompt`
        # candidates (default 1 = classic best-vs-worst).
        for worst in cands[-max_pairs_per_prompt:]:
            if worst is best:
                continue
            if best["overall"] - worst["overall"] < margin:
                continue
            pairs.append(
                PrefPair(
                    prompt_id=prompt_id,
                    prompt=best["prompt"],
                    win_video=best["video"],
                    lose_video=worst["video"],
                    win_score=best["overall"],
                    lose_score=worst["overall"],
                )
            )
    return pairs


def coverage_report(rows: list[dict], pairs: list[PrefPair]) -> dict:
    """Small summary for the CLI: how many prompts produced usable pairs."""
    prompt_ids = {r["prompt_id"] for r in rows}
    paired = {p.prompt_id for p in pairs}
    return {
        "prompts_total": len(prompt_ids),
        "prompts_paired": len(paired),
        "pairs": len(pairs),
        "pair_rate": (len(paired) / len(prompt_ids)) if prompt_ids else 0.0,
    }

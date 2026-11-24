# SPDX-License-Identifier: Apache-2.0
"""Reward-model registry for P1a preference post-training.

License red line applies here too: only Apache-2.0 / MIT reward models may be
registered. Reward scores are used for ranking and DPO pair selection, never
as training targets distilled from closed/community-license models.

Picked 2026-11-24 (see docs/quality-path-wan3-h3.md):
- VideoReward (KlingTeam, Apache-2.0): Qwen2-VL-2B-based multi-dimensional
  video reward model. Dimensions cover visual quality, temporal consistency,
  dynamic degree and text alignment. ~5GB bf16, fits beside TI2V-5B on 24GB.
"""

from __future__ import annotations

from dataclasses import dataclass

ALLOWED_LICENSES = ("Apache-2.0", "MIT")


@dataclass(frozen=True)
class RewardSpec:
    name: str
    repo_id: str
    license: str
    base: str            # underlying VLM the reward head sits on
    vram_gb: int         # bf16 inference footprint (approx)
    multi_dim: bool      # returns per-dimension scores, not just a scalar


REWARD_MODELS: dict[str, RewardSpec] = {
    "videoreward": RewardSpec(
        "videoreward", "KlingTeam/VideoReward", "Apache-2.0",
        base="Qwen2-VL-2B", vram_gb=5, multi_dim=True,
    ),
}


def get_reward_spec(name: str) -> RewardSpec:
    """Return a registered reward spec, enforcing the license whitelist."""
    try:
        spec = REWARD_MODELS[name]
    except KeyError:
        raise KeyError(
            f"unknown reward model {name!r}; registered: {sorted(REWARD_MODELS)}"
        ) from None
    if spec.license not in ALLOWED_LICENSES:
        raise ValueError(
            f"reward model {name!r} is licensed {spec.license}; "
            f"only {ALLOWED_LICENSES} are allowed in this repo"
        )
    return spec

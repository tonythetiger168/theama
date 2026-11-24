# SPDX-License-Identifier: Apache-2.0
"""Reward models for P1a preference post-training (lazy, CPU-safe imports)."""

from .config import ALLOWED_LICENSES, REWARD_MODELS, RewardSpec, get_reward_spec

__all__ = ["ALLOWED_LICENSES", "REWARD_MODELS", "RewardSpec", "get_reward_spec"]

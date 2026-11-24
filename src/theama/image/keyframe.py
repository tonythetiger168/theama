# SPDX-License-Identifier: Apache-2.0
"""KeyframeBridge: Theama Image outputs -> Film I2V keyframes / Avatar references.

The image line is the hub of the three product lines:
  - first/last-frame pairs for the Film line's I2V pass (M7c)
  - identity reference images for the Avatar line's IdentityEncoder (M7c)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class KeyframeSpec:
    prompt: str
    aspect: tuple[int, int] = (1280, 720)  # match Film line native stage
    style_lock: str | None = None          # shared style suffix across keyframes
    seed: int | None = None


class KeyframeBridge:
    """Generates consistent keyframe sets for downstream video lines. M7c."""

    def __init__(self, image_backbone):
        self.image = image_backbone

    def first_last_frames(self, spec: KeyframeSpec) -> tuple:
        """Return (first_frame, last_frame) with a shared style/seed policy."""
        # TODO(M7c): fixed seed + style_lock suffix for cross-frame consistency;
        # feed into theama.model.VideoBackbone I2V path.
        raise NotImplementedError("M7c: first/last keyframe generation")

    def identity_reference(self, spec: KeyframeSpec):
        """Single portrait reference for avatar IdentityEncoder."""
        # TODO(M7c): portrait-safe crop/quality checks before handing to
        # theama.avatar.IdentityEncoder.
        raise NotImplementedError("M7c: identity reference generation")

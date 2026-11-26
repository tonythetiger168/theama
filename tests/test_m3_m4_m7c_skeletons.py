# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the M3 / M4 / M7c stage skeletons.

The three skeleton modules carry the last un-pinned declarative contracts in
src/ (everything else with working logic is covered; smoke_* GPU shell
scripts belong to the Claude lane). These tests pin the parts that are
already decided so the M3/M4/M7c implementations cannot silently drift:

- extend_30s: ExtendConfig defaults + SVIContinuation.num_segments() ceil
  arithmetic (segment count drives the M3 acceptance budget) + the
  documented NotImplementedError contract of extend().
- upscale_4k: DEFAULT_STAGES order and parameters (720p -> 2K regen ->
  tiled 4K, monotonically increasing resolution, decreasing denoise) +
  the TwoStageUpscaler default-stages fallback + NotImplementedError.
- keyframe: KeyframeSpec defaults (Film-native 1280x720 aspect, optional
  style_lock/seed) + KeyframeBridge wiring + both M7c entry points pinned
  to NotImplementedError.

Pure-python modules (dataclasses only) -- no fakes needed.
"""

from __future__ import annotations

import pytest

from theama.image.keyframe import KeyframeBridge, KeyframeSpec
from theama.inference.extend_30s import ExtendConfig, SVIContinuation
from theama.inference.upscale_4k import (
    DEFAULT_STAGES,
    TwoStageUpscaler,
    UpscaleStage,
)


# ---------------------------------------------------------------------------
# M3 -- extend_30s (SVI continuation)
# ---------------------------------------------------------------------------


class TestExtendConfig:
    def test_defaults(self):
        c = ExtendConfig()
        assert c.native_seconds == 5
        assert c.target_seconds == 30
        assert c.anchor_frames == 8
        assert c.audio_crossfade_ms == 250

    def test_custom_values_stored(self):
        c = ExtendConfig(native_seconds=4, target_seconds=24,
                         anchor_frames=16, audio_crossfade_ms=500)
        assert (c.native_seconds, c.target_seconds) == (4, 24)
        assert (c.anchor_frames, c.audio_crossfade_ms) == (16, 500)


class TestSVIContinuation:
    def test_default_config_when_none(self):
        backbone = object()
        svi = SVIContinuation(backbone)
        assert svi.backbone is backbone
        assert isinstance(svi.config, ExtendConfig)
        assert svi.config.target_seconds == 30

    def test_num_segments_default_is_six(self):
        # 30s target / 5s native = exactly 6 segments.
        assert SVIContinuation(None).num_segments() == 6

    def test_num_segments_ceil_for_non_multiples(self):
        # 31s needs a 7th partial segment; 29s still fits in 6.
        assert SVIContinuation(
            None, ExtendConfig(target_seconds=31)).num_segments() == 7
        assert SVIContinuation(
            None, ExtendConfig(target_seconds=29)).num_segments() == 6
        # custom native: 30 / 4 = 7.5 -> 8.
        assert SVIContinuation(None, ExtendConfig(
            native_seconds=4, target_seconds=30)).num_segments() == 8

    def test_num_segments_floors_at_one(self):
        # target shorter than (or equal to) one native segment still means
        # a single generation, never zero.
        assert SVIContinuation(
            None, ExtendConfig(target_seconds=3)).num_segments() == 1
        assert SVIContinuation(
            None, ExtendConfig(target_seconds=5)).num_segments() == 1
        assert SVIContinuation(
            None, ExtendConfig(target_seconds=0)).num_segments() == 1

    def test_extend_is_m3_notimplemented(self):
        svi = SVIContinuation(None)
        with pytest.raises(NotImplementedError, match="M3"):
            svi.extend(object(), object(), "a prompt")


# ---------------------------------------------------------------------------
# M4 -- upscale_4k (two-stage 4K regenerate)
# ---------------------------------------------------------------------------


class TestDefaultStages:
    def test_two_stages_in_order(self):
        assert [s.name for s in DEFAULT_STAGES] == ["regen2k", "regen4k"]

    def test_stage_parameters(self):
        regen2k, regen4k = DEFAULT_STAGES
        assert (regen2k.width, regen2k.height) == (2560, 1440)
        assert regen2k.denoise_strength == pytest.approx(0.35)
        assert regen2k.tiled is False
        assert (regen4k.width, regen4k.height) == (3840, 2160)
        assert regen4k.denoise_strength == pytest.approx(0.25)
        assert regen4k.tiled is True

    def test_resolution_ascends_denoise_descends(self):
        # Order contract: each stage upscales the previous one with a
        # gentler redraw; only the final 4K stage tiles.
        widths = [s.width for s in DEFAULT_STAGES]
        denoise = [s.denoise_strength for s in DEFAULT_STAGES]
        assert widths == sorted(widths) and len(set(widths)) == len(widths)
        assert denoise == sorted(denoise, reverse=True)
        assert [s.tiled for s in DEFAULT_STAGES] == [False, True]

    def test_upscale_stage_default_not_tiled(self):
        s = UpscaleStage("x", 100, 100, denoise_strength=0.5)
        assert s.tiled is False


class TestTwoStageUpscaler:
    def test_default_stages_fallback(self):
        backbone = object()
        up = TwoStageUpscaler(backbone)
        assert up.backbone is backbone
        assert up.stages is DEFAULT_STAGES

    def test_custom_stages_respected(self):
        stages = [UpscaleStage("only4k", 3840, 2160, 0.2, tiled=True)]
        up = TwoStageUpscaler(None, stages=stages)
        assert up.stages is stages

    def test_upscale_is_m4_notimplemented(self):
        with pytest.raises(NotImplementedError, match="M4"):
            TwoStageUpscaler(None).upscale(object(), "a prompt")


# ---------------------------------------------------------------------------
# M7c -- keyframe bridge (Image -> Film I2V / Avatar identity)
# ---------------------------------------------------------------------------


class TestKeyframeSpec:
    def test_defaults(self):
        spec = KeyframeSpec(prompt="a poster frame")
        # Film line native stage resolution; style/seed unlocked by default.
        assert spec.aspect == (1280, 720)
        assert spec.style_lock is None
        assert spec.seed is None

    def test_custom_values_stored(self):
        spec = KeyframeSpec(prompt="p", aspect=(1920, 1080),
                            style_lock="flat anime", seed=42)
        assert spec.aspect == (1920, 1080)
        assert spec.style_lock == "flat anime"
        assert spec.seed == 42


class TestKeyframeBridge:
    def test_stores_image_backbone(self):
        backbone = object()
        bridge = KeyframeBridge(backbone)
        assert bridge.image is backbone

    def test_first_last_frames_is_m7c_notimplemented(self):
        bridge = KeyframeBridge(None)
        with pytest.raises(NotImplementedError, match="M7c"):
            bridge.first_last_frames(KeyframeSpec(prompt="p"))

    def test_identity_reference_is_m7c_notimplemented(self):
        bridge = KeyframeBridge(None)
        with pytest.raises(NotImplementedError, match="M7c"):
            bridge.identity_reference(KeyframeSpec(prompt="p"))

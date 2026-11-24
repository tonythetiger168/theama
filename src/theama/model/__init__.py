# SPDX-License-Identifier: Apache-2.0
"""Model components: Wan 2.2 video backbone wrapper, parallel audio DiT, 3D RoPE."""

from theama.model.dit import VideoBackbone
from theama.model.audio_dit import AudioDiT, CrossModalBridge
from theama.model.audio_codec import (
    DAC_44K,
    AudioCodecSpec,
    MockAudioCodec,
)
from theama.model.dac_codec import DACAudioCodec
from theama.model.lora import LoRALinear, inject_lora_into_audio_dit, merge_lora

__all__ = [
    "VideoBackbone",
    "AudioDiT",
    "CrossModalBridge",
    "DAC_44K",
    "AudioCodecSpec",
    "MockAudioCodec",
    "DACAudioCodec",
    "LoRALinear",
    "inject_lora_into_audio_dit",
    "merge_lora",
]

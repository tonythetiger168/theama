# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the M2 audio codec latent interface (DAC 44.1 kHz)."""

import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.model.audio_codec import (
    CODEC_REGISTRY,
    DAC_44K,
    AudioCodecSpec,
    MockAudioCodec,
    decode_stereo,
    encode_stereo,
    get_codec_spec,
    mid_side_to_stereo,
    stereo_to_mid_side,
)
from theama.model.dac_codec import DACAudioCodec


# ---- spec values (verified against upstream conf/base.yml, MIT) ---------------

def test_dac_44k_spec_matches_upstream_config():
    assert DAC_44K.sample_rate == 44100
    assert DAC_44K.hop_length == 512  # encoder_rates [2, 4, 8, 8]
    assert DAC_44K.latent_dim == 1024  # encoder_dim 64 doubled over 4 stages
    assert DAC_44K.n_codebooks == 9
    assert DAC_44K.codebook_size == 1024
    assert DAC_44K.license == "MIT"
    assert DAC_44K.latent_frame_rate == pytest.approx(86.1328125)


def test_dit_token_dim_is_mid_plus_side():
    assert DAC_44K.stereo_layout == "mid_side"
    assert DAC_44K.dit_token_dim == 2 * DAC_44K.latent_dim


def test_registry_lookup_and_unknown():
    assert get_codec_spec("dac_44k") is DAC_44K
    assert CODEC_REGISTRY["dac_44k"].repo_id == "descript/dac_44khz"
    with pytest.raises(ValueError, match="unknown audio codec"):
        get_codec_spec("encodec")


# ---- frame/second math ---------------------------------------------------------

def test_frame_second_roundtrip():
    frames = DAC_44K.frames_for_seconds(5.0)  # 5s clip per M2 acceptance config
    assert frames * DAC_44K.hop_length >= 5 * DAC_44K.sample_rate  # pads up
    assert frames == 431  # ceil(220500 / 512)
    assert DAC_44K.seconds_for_frames(frames) == pytest.approx(5.0, abs=1e-2)
    # one video frame at 24fps spans ~3.59 latent frames — sync budget is fine
    assert DAC_44K.latent_frame_rate / 24 > 3


def test_frames_for_seconds_rejects_bad_input():
    with pytest.raises(ValueError, match="integer number of samples"):
        DAC_44K.frames_for_seconds(1 / 44100 / 2)  # half a sample
    with pytest.raises(ValueError, match="frames must be >= 0"):
        DAC_44K.seconds_for_frames(-1)


# ---- mid/side stereo layout ------------------------------------------------------

def test_mid_side_roundtrip_is_lossless():
    torch.manual_seed(0)
    wav = torch.randn(2, 2, 44100)
    mid, side = stereo_to_mid_side(wav)
    assert mid.shape == side.shape == (2, 1, 44100)
    assert torch.allclose(mid_side_to_stereo(mid, side), wav, atol=1e-6)


def test_mid_side_validates_stereo_shape():
    with pytest.raises(ValueError, match="stereo waveform"):
        stereo_to_mid_side(torch.randn(1, 44100))


# ---- mock codec -------------------------------------------------------------------

def test_mock_codec_encode_decode_shapes():
    codec = MockAudioCodec()
    frames = DAC_44K.frames_for_seconds(1.0)
    wav = torch.randn(2, 1, frames * DAC_44K.hop_length)
    lat = codec.encode(wav)
    assert lat.shape == (2, frames, DAC_44K.latent_dim)
    back = codec.decode(lat)
    assert back.shape == wav.shape


def test_mock_codec_is_deterministic():
    wav = torch.randn(1, 1, 512 * 4)
    assert torch.equal(MockAudioCodec(seed=3).encode(wav), MockAudioCodec(seed=3).encode(wav))


def test_mock_codec_validates_inputs():
    codec = MockAudioCodec()
    with pytest.raises(ValueError, match="mono waveform"):
        codec.encode(torch.randn(1, 2, 1024))
    with pytest.raises(ValueError, match="multiple of hop"):
        codec.encode(torch.randn(1, 1, 1000))
    with pytest.raises(ValueError, match="expected latents"):
        codec.decode(torch.randn(1, 4, 64))


def test_mock_latents_feed_audio_dit_contract():
    """Mock latents reshape to the (B, T_lat, 2*latent_dim) DiT token contract."""
    sys.path.insert(0, str(ROOT / "src"))
    from theama.model.audio_dit import AudioDiT

    codec = MockAudioCodec()
    wav = torch.randn(1, 2, 512 * 8)
    mid, side = stereo_to_mid_side(wav)
    tokens = torch.cat((codec.encode(mid), codec.encode(side)), dim=-1)
    assert tokens.shape[-1] == DAC_44K.dit_token_dim

    dit = AudioDiT(dim=64, layers=2, heads=4, video_dim=32,
                   bridge_every=1, latent_dim=DAC_44K.dit_token_dim)
    eps, _ = dit(tokens, torch.randn(1, 3, 32), torch.randn(1, 64))
    assert eps.shape == tokens.shape


# ---- real DAC wrapper contract (fake upstream model, no weights) -------------------

class _FakeDAC(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.seen_sample_rate = None

    def preprocess(self, waveform, sample_rate):
        self.seen_sample_rate = sample_rate
        return waveform

    def encode(self, waveform):
        batch, _, samples = waveform.shape
        frames = samples // self.spec.hop_length
        pooled = waveform.reshape(batch, frames, self.spec.hop_length).mean(dim=-1)
        continuous = pooled[:, None, :].repeat(1, self.spec.latent_dim, 1) * self.weight
        return continuous, None, None, None, None

    def decode(self, continuous):
        pooled = continuous.mean(dim=1)
        return pooled[:, None, :, None].expand(
            -1, -1, -1, self.spec.hop_length
        ).reshape(continuous.shape[0], 1, -1)


def _tiny_codec_spec():
    return AudioCodecSpec(
        name="tiny-dac",
        repo_id="example/tiny-dac",
        license="MIT",
        sample_rate=8,
        hop_length=2,
        latent_dim=4,
        n_codebooks=1,
        codebook_size=8,
        stereo_layout="mid_side",
    )


def test_dac_wrapper_transposes_upstream_latents_and_freezes_model():
    spec = _tiny_codec_spec()
    upstream = _FakeDAC(spec)
    codec = DACAudioCodec(upstream, spec)
    waveform = torch.arange(6.0).reshape(1, 1, 6).requires_grad_(True)
    latents = codec.encode(waveform)
    assert latents.shape == (1, 3, 4)
    assert not latents.requires_grad
    expected = waveform.detach().reshape(1, 3, 2).mean(dim=-1)
    assert torch.equal(latents[:, :, 0], expected)
    assert upstream.seen_sample_rate == 8
    assert not upstream.weight.requires_grad
    restored = codec.decode(latents)
    assert restored.shape == waveform.shape


def test_stereo_codec_helpers_roundtrip_layout():
    codec = MockAudioCodec()
    waveform = torch.randn(2, 2, DAC_44K.hop_length * 4)
    tokens = encode_stereo(codec, waveform)
    assert tokens.shape == (2, 4, DAC_44K.dit_token_dim)
    restored = decode_stereo(codec, tokens)
    assert restored.shape == waveform.shape
    with pytest.raises(ValueError, match="stereo latents"):
        decode_stereo(codec, tokens[..., :16])


# ---- config wiring -----------------------------------------------------------------

def test_m2_yaml_names_registered_codec():
    cfg = yaml.safe_load((ROOT / "configs" / "m2_joint_av.yaml").read_text())
    assert get_codec_spec(cfg["model"]["audio"]["codec"]) is DAC_44K

# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the Avatar line M6 skeletons (theama.avatar).

Locks the behaviour that already works in the skeletons so later M6a/M6b/M6c
implementation work refactors against a net, and pins the documented
not-yet-implemented surface (StreamingAvatarSession channels) so a silent
behaviour change cannot ship as a "skeleton".

Modules covered:
- identity.IdentityBank / IdentityEncoder (M6a skeleton, working logic)
- audio_drive.AudioDrive (M6a skeleton, working attention-pool forward)
- streaming.StreamConfig / StreamingAvatarSession (M6b/M6c skeleton)
"""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from theama.avatar.audio_drive import AudioDrive
from theama.avatar.identity import IdentityBank, IdentityEncoder
from theama.avatar.streaming import StreamConfig, StreamingAvatarSession

AUDIO_DIM, MOTION_DIM = 32, 16  # motion_dim must be divisible by heads (8)


# ---------------------------------------------------------------------------
# IdentityBank
# ---------------------------------------------------------------------------

def test_identity_bank_default_empty():
    bank = IdentityBank()
    assert bank.entries == {}


def test_identity_bank_none_entries_normalised():
    bank = IdentityBank(entries=None)
    assert bank.entries == {}


def test_identity_bank_register_swap_roundtrip():
    bank = IdentityBank()
    emb = torch.randn(8)
    bank.register("alice", emb)
    out = bank.swap("alice")
    assert out is emb  # hot-swap returns the registered tensor itself


def test_identity_bank_swap_unknown_raises_keyerror():
    bank = IdentityBank()
    with pytest.raises(KeyError, match="unknown identity: bob"):
        bank.swap("bob")


def test_identity_bank_register_overwrites():
    bank = IdentityBank()
    old = torch.zeros(4)
    new = torch.ones(4)
    bank.register("alice", old)
    bank.register("alice", new)
    assert bank.swap("alice") is new


def test_identity_bank_entries_independent():
    bank = IdentityBank()
    a, b = torch.randn(4), torch.randn(4)
    bank.register("a", a)
    bank.register("b", b)
    assert bank.swap("a") is a
    assert bank.swap("b") is b


# ---------------------------------------------------------------------------
# IdentityEncoder
# ---------------------------------------------------------------------------

def test_identity_encoder_default_dim():
    enc = IdentityEncoder()
    out = enc(torch.randn(2, 512))
    assert out.shape == (2, 1024)


def test_identity_encoder_custom_dim_and_dtype():
    enc = IdentityEncoder(dim=64).to(torch.float64)
    out = enc(torch.randn(3, 512, dtype=torch.float64))
    assert out.shape == (3, 64)
    assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# AudioDrive
# ---------------------------------------------------------------------------

def test_audio_drive_forward_shape():
    drive = AudioDrive(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM, window=4)
    audio = torch.randn(2, 50, AUDIO_DIM)
    out = drive(audio, n_video_frames=10)
    assert out.shape == (2, 10, MOTION_DIM)


def test_audio_drive_token_count_follows_video_grid():
    """Motion tokens track n_video_frames, independent of audio length Ta."""
    drive = AudioDrive(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM)
    short = drive(torch.randn(1, 20, AUDIO_DIM), n_video_frames=7)
    long = drive(torch.randn(1, 200, AUDIO_DIM), n_video_frames=7)
    assert short.shape == long.shape == (1, 7, MOTION_DIM)


def test_audio_drive_window_attribute_stored():
    drive = AudioDrive(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM, window=24)
    assert drive.window == 24


def test_audio_drive_frame_queries_expand_and_follow_dtype():
    drive = AudioDrive(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM)
    q = drive.frame_queries(batch=3, n=5, device=torch.device("cpu"),
                            dtype=torch.float64)
    assert q.shape == (3, 5, MOTION_DIM)
    assert q.dtype == torch.float64
    # every expanded query is the same learned base vector
    base = drive.frame_query.to(torch.float64)
    assert torch.allclose(q, base.expand(3, 5, MOTION_DIM))


def test_audio_drive_deterministic_in_eval():
    torch.manual_seed(0)
    drive = AudioDrive(audio_dim=AUDIO_DIM, motion_dim=MOTION_DIM).eval()
    audio = torch.randn(1, 30, AUDIO_DIM)
    with torch.no_grad():
        a = drive(audio, n_video_frames=4)
        b = drive(audio, n_video_frames=4)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# StreamConfig / StreamingAvatarSession
# ---------------------------------------------------------------------------

def test_stream_config_defaults():
    cfg = StreamConfig()
    assert (cfg.width, cfg.height) == (1280, 720)
    assert cfg.fps == 25
    assert cfg.chunk_frames == 5
    assert cfg.kv_window_frames == 60
    assert cfg.replay_anchor_every == 50


def test_session_init_defaults_and_kv_window():
    session = StreamingAvatarSession(backbone=object(), identity_bank=IdentityBank())
    assert session._frame_count == 0
    assert session._kv_cache.maxlen == session.config.kv_window_frames
    assert isinstance(session.config, StreamConfig)


def test_session_custom_config_respected():
    cfg = StreamConfig(kv_window_frames=12, chunk_frames=2)
    session = StreamingAvatarSession(None, IdentityBank(), config=cfg)
    assert session._kv_cache.maxlen == 12
    assert session.config.chunk_frames == 2


def test_session_channels_raise_not_implemented():
    """M6b/M6c channels are documented TODOs; pin the contract."""
    session = StreamingAvatarSession(None, IdentityBank())
    with pytest.raises(NotImplementedError, match="M6b"):
        session.push_audio(torch.randn(100))
    with pytest.raises(NotImplementedError, match="M6c"):
        session.push_instruction("swap reference")
    with pytest.raises(NotImplementedError, match="M6b"):
        session.next_frames()

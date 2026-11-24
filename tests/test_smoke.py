# SPDX-License-Identifier: Apache-2.0
"""CPU-only smoke tests: imports, config load, license red line, RoPE shapes."""

import torch
import yaml


def test_config_loads():
    cfg = yaml.safe_load(open("configs/theama_base.yaml"))
    assert cfg["generation"]["target_seconds"] == 30
    assert cfg["model"]["audio"]["sample_rate"] == 44100
    assert cfg["training"]["forbid_third_party_outputs"] is True


def test_license_red_line_enforced():
    from theama.data.dataset import AVDataset

    AVDataset.check_source("pexels:12345")  # clean source passes
    for bad in ("minimax-h3:gen-001", "ltx-2.5:out-9", "hailuo-clip"):
        try:
            AVDataset.check_source(bad)
            raise AssertionError(f"forbidden source accepted: {bad}")
        except ValueError:
            pass


def test_rope3d_shapes():
    from theama.model.rope3d import apply_rope, build_3d_rope

    t, h, w, dim = 4, 2, 2, 64
    freqs = build_3d_rope(t, h, w, dim)
    assert freqs.shape == (t * h * w, dim // 2)
    x = torch.randn(2, 8, t * h * w, dim)
    assert apply_rope(x, freqs).shape == x.shape


def test_audio_dit_forward_shapes():
    from theama.model.audio_dit import AudioDiT

    m = AudioDiT(dim=128, layers=4, heads=4, video_dim=256, bridge_every=2, latent_dim=32)
    a = torch.randn(2, 50, 32)
    v = torch.randn(2, 120, 256)
    cond = torch.randn(2, 128)
    a_out, v_out = m(a, v, cond)
    assert a_out.shape == a.shape
    assert v_out.shape == v.shape


def test_avatar_modules():
    import torch

    from theama.avatar import AudioDrive, IdentityEncoder, StreamingAvatarSession
    from theama.avatar.identity import IdentityBank as IB

    enc = IdentityEncoder(dim=64)
    emb = enc(torch.randn(1, 512))
    assert emb.shape == (1, 64)

    bank = IB()
    bank.register("alice", emb)
    assert bank.swap("alice") is emb
    try:
        bank.swap("bob")
        raise AssertionError("unknown identity accepted")
    except KeyError:
        pass

    drive = AudioDrive(audio_dim=32, motion_dim=16)
    out = drive(torch.randn(2, 40, 32), n_video_frames=10)
    assert out.shape == (2, 10, 16)

    session = StreamingAvatarSession(backbone=None, identity_bank=bank)
    assert session.config.height == 720


def test_image_modules():
    from theama.image import ImageBackbone, KeyframeBridge, KeyframeSpec
    from theama.image.t2i import EDIT_REPO, FAST_REPO, QUALITY_REPO

    assert "Qwen-Image-2512" in QUALITY_REPO
    assert "Z-Image-Turbo" in FAST_REPO
    assert "Edit-2511" in EDIT_REPO

    bb = ImageBackbone(mode="fast")
    assert bb.config.fast_steps == 8
    try:
        ImageBackbone(mode="ultra")
        raise AssertionError("unknown mode accepted")
    except ValueError:
        pass

    spec = KeyframeSpec(prompt="霓虹雨中的貓", style_lock="cinematic")
    assert spec.aspect == (1280, 720)
    bridge = KeyframeBridge(bb)
    assert bridge.image is bb


def test_package_version():
    import theama

    assert theama.__version__ == "0.1.0"

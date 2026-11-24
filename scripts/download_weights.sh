#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Download Wan 2.2 weights (Apache-2.0) — user-side, never committed to the repo.
set -euo pipefail

TARGET_DIR="${1:-checkpoints/wan2.2}"
mkdir -p "$TARGET_DIR"

# Diffusers-layout repos — src/theama/inference/ loads these directly.
echo "==> Wan 2.2 TI2V-5B Diffusers (dev/CI path, ~16GB VRAM w/ offload)"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$TARGET_DIR/ti2v-5b"

echo "==> Wan 2.2 T2V-A14B Diffusers (full backbone, 40GB+ w/ offload, 80GB-class native)"
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir "$TARGET_DIR/t2v-a14b"

echo "==> Wan 2.2 I2V-A14B Diffusers (keyframe-conditioned path for M7c)"
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir "$TARGET_DIR/i2v-a14b"

echo "==> VideoReward (P1a preference scoring, Apache-2.0, ~5GB)"
hf download KlingTeam/VideoReward --local-dir "$TARGET_DIR/videoreward"

echo "Done. Weights are under $TARGET_DIR (gitignored)."

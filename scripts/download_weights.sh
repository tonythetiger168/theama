#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Download Wan 2.2 weights (Apache-2.0) — user-side, never committed to the repo.
set -euo pipefail

TARGET_DIR="${1:-checkpoints/wan2.2}"
mkdir -p "$TARGET_DIR"

echo "==> Wan 2.2 TI2V-5B (dev/CI path, ~24GB VRAM)"
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir "$TARGET_DIR/ti2v-5b"

echo "==> Wan 2.2 T2V-A14B (full backbone, 80GB-class)"
hf download Wan-AI/Wan2.2-T2V-A14B --local-dir "$TARGET_DIR/t2v-a14b"

echo "Done. Weights are under $TARGET_DIR (gitignored)."

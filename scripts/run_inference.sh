#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# M1 inference entry: base 5s 720p pass + MP4 export (silent AAC track).
set -euo pipefail

PROMPT="${1:?usage: run_inference.sh <prompt> <out.mp4> [seconds]}"
OUT="${2:-outputs/out.mp4}"
SECONDS="${3:-5}"

mkdir -p "$(dirname "$OUT")"
python -m theama.inference.pipeline --prompt "$PROMPT" --out "$OUT" --seconds "$SECONDS" --dev

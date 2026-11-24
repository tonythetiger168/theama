#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# M1 inference entry: base 5s 720p pass + MP4 export (silent AAC track).
set -euo pipefail

PROMPT="${1:?usage: run_inference.sh <prompt> <out.mp4> [seconds]}"
OUT="${2:-outputs/out.mp4}"
SECONDS="${3:-5}"
MODEL="${THEAMA_MODEL:-ti2v-5b}"   # 24GB dev path; set THEAMA_MODEL=t2v-a14b for full backbone

mkdir -p "$(dirname "$OUT")"
PYTHONPATH=src python -m theama.infer.pipeline \
    --prompt "$PROMPT" --out "$OUT" --seconds "$SECONDS" --model "$MODEL"

#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# RunPod bootstrap: run once inside a fresh pod (pytorch template).
# Idempotent — safe to re-run. Weights and HF cache go on the network
# volume so they survive pod stop/start.
#
#   bash scripts/runpod_bootstrap.sh [workspace_dir]
#
# After bootstrap:
#   python scripts/gen_eval_set.py --model ti2v-5b --out "$WORKSPACE/eval_out/ti2v-5b"
#   python scripts/gen_candidates.py --model ti2v-5b --out "$WORKSPACE/dpo_out/candidates" --candidates 4
set -euo pipefail

WORKSPACE="${1:-/workspace}"
echo "==> workspace: $WORKSPACE (mount a network volume here)"
mkdir -p "$WORKSPACE"

cd "$WORKSPACE"
if [ -d theama/.git ]; then
  echo "==> repo exists, pulling latest"
  git -C theama pull --ff-only
else
  echo "==> cloning theama"
  git clone https://github.com/tonythetiger168/theama.git
fi
cd theama

echo "==> python deps (torch comes from the RunPod pytorch template)"
pip install -q -e .[dev]
pip install -q diffusers transformers accelerate peft soundfile av descript-audio-codec

export HF_HOME="$WORKSPACE/hf_cache"
echo "==> HF_HOME=$HF_HOME (persists on the network volume)"

echo "==> weights -> $WORKSPACE/checkpoints (Wan 2.2 x3 + VideoReward + DAC 44.1kHz)"
bash scripts/download_weights.sh "$WORKSPACE/checkpoints/wan2.2"

echo "==> CPU test suite"
PYTHONPATH=src python -m pytest tests/ -q -p no:cacheprovider

cat <<EOF

Bootstrap OK. GPU pipeline (in order):
  # 1. M2 real-weight contract smokes (JSON + WAV/VRAM reports)
  PYTHONPATH=src python scripts/smoke_m2_real_weights.py dac --weights "$WORKSPACE/checkpoints/wan2.2/dac_44khz/weights.pth" --out "$WORKSPACE/smoke_out/dac"
  PYTHONPATH=src python scripts/smoke_m2_real_weights.py wan --model-path "$WORKSPACE/checkpoints/wan2.2/ti2v-5b" --out "$WORKSPACE/smoke_out/wan_ti2v_5b"
  # 2. M1 acceptance — 50-clip eval set (add --limit 5 for a smoke first)
  PYTHONPATH=src python scripts/gen_eval_set.py --model ti2v-5b --out "$WORKSPACE/eval_out/ti2v-5b"
  # 3. P1a — candidates -> scores -> pairs -> latents -> DPO
  PYTHONPATH=src python scripts/gen_candidates.py --model ti2v-5b --out "$WORKSPACE/dpo_out/candidates" --candidates 4
  PYTHONPATH=src python scripts/score_eval_set.py --videos "$WORKSPACE/dpo_out/candidates" --smoke
  PYTHONPATH=src python scripts/score_eval_set.py --videos "$WORKSPACE/dpo_out/candidates"
  PYTHONPATH=src python scripts/make_pref_pairs.py --scores "$WORKSPACE/dpo_out/candidates/scores.jsonl" --out "$WORKSPACE/dpo_out/pairs.jsonl"
  PYTHONPATH=src python scripts/encode_pairs.py --pairs "$WORKSPACE/dpo_out/pairs.jsonl" --out "$WORKSPACE/dpo_out/latents"
  PYTHONPATH=src python scripts/train_dpo.py --pairs "$WORKSPACE/dpo_out/latents/pairs_latents.jsonl" --out "$WORKSPACE/dpo_out/run1"
EOF

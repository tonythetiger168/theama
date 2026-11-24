#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Theama M2 real-weight smoke on a fresh RunPod PyTorch instance.
#
# Recommended pod: RTX 4090 24 GB VRAM, >=32 GB CPU RAM, 40 GB network volume.
# Only the TI2V-5B transformer and DAC 44.1 kHz are downloaded; the A14B
# backbones and VideoReward are intentionally excluded from this M2-only path.
#
# From a cloned repository:
#   bash scripts/runpod_m2_smoke.sh [/workspace]
#
# The script is idempotent. Weights, HF cache, logs and smoke artifacts stay
# under the workspace. Real GPU results are not claimed until both JSON reports
# are written and M2_RUNPOD_SMOKE_OK is printed.
set -euo pipefail

WORKSPACE="${1:-/workspace}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trap 'status=$?; echo "M2_RUNPOD_FAILED status=$status line=$LINENO" >&2; exit "$status"' ERR

mkdir -p "$WORKSPACE"
WORKSPACE="$(cd "$WORKSPACE" && pwd)"
export WORKSPACE
echo "==> workspace: $WORKSPACE (use a mounted network volume)"

# Support both `bash scripts/...` inside a checkout and a standalone upload.
if [[ -f "$SCRIPT_DIR/../pyproject.toml" && -d "$SCRIPT_DIR/../src/theama" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
  if [[ -d "$REPO_ROOT/.git" ]]; then
    echo "==> updating existing checkout"
    git -C "$REPO_ROOT" pull --ff-only
  fi
else
  REPO_ROOT="$WORKSPACE/theama"
  if [[ -d "$REPO_ROOT/.git" ]]; then
    echo "==> updating $REPO_ROOT"
    git -C "$REPO_ROOT" pull --ff-only
  elif [[ -e "$REPO_ROOT" ]]; then
    echo "ERROR: $REPO_ROOT exists but is not a git checkout" >&2
    exit 2
  else
    echo "==> cloning Theama"
    git clone https://github.com/tonythetiger168/theama.git "$REPO_ROOT"
  fi
fi

cd "$REPO_ROOT"
export HF_HOME="$WORKSPACE/hf_cache"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CHECKPOINTS="$WORKSPACE/checkpoints/wan2.2"
TI2V_DIR="$CHECKPOINTS/ti2v-5b"
DAC_DIR="$CHECKPOINTS/dac_44khz"
OUT_DIR="$WORKSPACE/smoke_out"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$HF_HOME" "$TI2V_DIR" "$DAC_DIR" "$LOG_DIR"

for tool in git python3; do
  command -v "$tool" >/dev/null || { echo "ERROR: missing $tool" >&2; exit 2; }
done

echo "==> CUDA preflight (before any install or download)"
python3 - <<'PY'
import json
from pathlib import Path
import os
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; select a GPU RunPod PyTorch template")
name = torch.cuda.get_device_name(0)
total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU={name} VRAM={total_gib:.2f} GiB torch={torch.__version__} cuda={torch.version.cuda}")
if total_gib < 20:
    print("WARNING: this real-weight path is sized for 24 GB VRAM; smaller GPUs may OOM")
report = {
    "gpu_name": name,
    "gpu_total_gib": round(total_gib, 4),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
}
out = Path(os.environ["WORKSPACE"]) / "smoke_out" / "runpod_environment.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

free_gib="$(df -Pk "$WORKSPACE" | awk 'NR==2 {print int($4/1048576)}')"
echo "==> free space on $WORKSPACE: ${free_gib} GiB"
if (( free_gib < 30 )); then
  echo "WARNING: <30 GiB free on $WORKSPACE; TI2V-5B transformer + caches may not fit" >&2
fi

echo "==> Python dependencies (PyTorch comes from the RunPod template)"
# Pin the template's CUDA-matched torch stack. descript-audio-codec pulls in
# torchaudio; without this pin pip may replace torch with a PyPI build whose
# CUDA runtime does not match the pod driver.
TORCH_CONSTRAINTS="$OUT_DIR/torch_constraints.txt"
python3 - "$TORCH_CONSTRAINTS" <<'PY'
import sys
from importlib import metadata

pins = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        pins.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        if name == "torchaudio" and pins and pins[0].startswith("torch=="):
            # torchaudio is ABI-locked to torch; pip would otherwise pick the
            # newest torchaudio even when it cannot load against this torch.
            major, minor = pins[0].split("==")[1].split("+")[0].split(".")[:2]
            pins.append(f"torchaudio=={major}.{minor}.*")
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n".join(pins) + "\n")
print("pinned:", ", ".join(pins))
PY
python3 -m pip install -q -c "$TORCH_CONSTRAINTS" -e '.[dev]'
python3 -m pip install -q -c "$TORCH_CONSTRAINTS" \
  diffusers transformers accelerate soundfile descript-audio-codec "huggingface_hub>=0.34"
command -v hf >/dev/null || { echo "ERROR: hf CLI not found; upgrade huggingface_hub" >&2; exit 2; }
# Fail before the multi-GB download if the install broke torch or DAC imports.
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA lost after pip install"'
python3 -c 'import dac, diffusers; from diffusers import WanTransformer3DModel; print("imports OK: diffusers", diffusers.__version__)'

echo "==> TI2V-5B transformer only (Wan 2.2, Apache-2.0)"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --include 'transformer/*' \
  --local-dir "$TI2V_DIR"

if [[ ! -s "$DAC_DIR/weights.pth" ]]; then
  echo "==> DAC 44.1 kHz (Descript, MIT)"
  DAC_SOURCE="$(python3 -c 'import dac; print(dac.utils.download(model_type="44khz"))')"
  cp "$DAC_SOURCE" "$DAC_DIR/weights.pth"
else
  echo "==> DAC checkpoint already present; reusing"
fi

echo "==> CPU contract suite"
python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 | tee "$LOG_DIR/cpu_tests.log"

echo "==> official DAC stereo encode/decode smoke"
python3 scripts/smoke_m2_real_weights.py dac \
  --weights "$DAC_DIR/weights.pth" \
  --out "$OUT_DIR/dac" \
  --device cuda 2>&1 | tee "$LOG_DIR/dac_smoke.log"

echo "==> frozen TI2V-5B tiny A/V forward/backward/optimizer smoke"
python3 scripts/smoke_m2_real_weights.py wan \
  --model-path "$TI2V_DIR" \
  --out "$OUT_DIR/wan_ti2v_5b" \
  --device cuda \
  --dtype bfloat16 2>&1 | tee "$LOG_DIR/wan_smoke.log"

for report in \
  "$OUT_DIR/dac/m2_dac_smoke.json" \
  "$OUT_DIR/wan_ti2v_5b/m2_wan_smoke.json"; do
  [[ -s "$report" ]] || { echo "ERROR: missing report $report" >&2; exit 1; }
done

cat <<EOF

M2_RUNPOD_SMOKE_OK
DAC report: $OUT_DIR/dac/m2_dac_smoke.json
Wan report: $OUT_DIR/wan_ti2v_5b/m2_wan_smoke.json
DAC audio: $OUT_DIR/dac/input.wav and reconstructed.wav
Environment: $OUT_DIR/runpod_environment.json
Logs: $LOG_DIR

Paste both JSON reports back for acceptance review. These are real-weight
contract smokes, not prompt-to-video, perceptual-quality or A/V-sync gates.
EOF

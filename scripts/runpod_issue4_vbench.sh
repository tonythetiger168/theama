#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Theama follow-up GPU queue on one RunPod pod (network volume at /workspace):
#   A. issue #4 re-run: internal `wan_blocks` + LoRA real-weight smoke with the
#      norm_video_kv fix (auto-enabled for wan_blocks). Red flag clears when
#      audio loss and grad norm are back in single digits.
#   B. VBench 6-dim subset over the 50 M1 TI2V-5B clips (own baseline; there is
#      no official TI2V-5B reference, so this records numbers, it is not a gate).
#
#   THEAMA_SELF_TERMINATE=1 bash scripts/runpod_issue4_vbench.sh [/workspace] [max_hours=3]
#
# Expects the volume layout left by scripts/runpod_m1_eval.sh: weights in
# $WORKSPACE/checkpoints/wan2.2/ti2v-5b and clips in $WORKSPACE/m1_out/ti2v-5b
# (falls back to $WORKSPACE/pub/m1_out_50.tgz). Writes $WORKSPACE/queue_out/.
# VBench runs in its own venv because it pins transformers==4.33.2/numpy<2.
#
# With THEAMA_SELF_TERMINATE=1 the pod archives queue_out to
# $WORKSPACE/pub/queue_out.tgz and deletes itself on exit, and a watchdog
# does the same after max_hours even if a step hangs. A DONE marker stops a
# container restart from re-running the queue.
set -euo pipefail

WORKSPACE="${1:-/workspace}"
MAX_HOURS="${2:-3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORKSPACE"
WORKSPACE="$(cd "$WORKSPACE" && pwd)"
export WORKSPACE
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT_DIR="$WORKSPACE/queue_out"
LOG_DIR="$OUT_DIR/logs"
TI2V_DIR="$WORKSPACE/checkpoints/wan2.2/ti2v-5b"
CLIPS_DIR="$WORKSPACE/m1_out/ti2v-5b"
mkdir -p "$LOG_DIR" "$WORKSPACE/pub"

archive_and_delete() {
  [[ "${THEAMA_SELF_TERMINATE:-0}" == "1" ]] || return 0
  echo "==> THEAMA_SELF_TERMINATE: archiving queue_out ($1)"
  tar -czf "$WORKSPACE/pub/queue_out.tgz" -C "$WORKSPACE" queue_out || true
  sync
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    echo "==> deleting pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID" || echo "WARNING: self-terminate failed; delete the pod manually" >&2
  else
    echo "WARNING: RUNPOD_POD_ID or runpodctl missing; pod NOT deleted" >&2
  fi
}
trap 'archive_and_delete "exit status $?"' EXIT

if [[ -f "$OUT_DIR/DONE" ]]; then
  echo "==> $OUT_DIR/DONE exists; queue already ran (container restart?), not re-running"
  exit 0
fi

if [[ "${THEAMA_SELF_TERMINATE:-0}" == "1" ]]; then
  ( sleep "$(( MAX_HOURS * 3600 ))"
    echo "QUEUE_WATCHDOG_TIMEOUT after ${MAX_HOURS} h" | tee -a "$LOG_DIR/watchdog.log"
    archive_and_delete "watchdog ${MAX_HOURS} h" ) &
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$WORKSPACE/hf_cache"
echo "==> workspace=$WORKSPACE max_hours=$MAX_HOURS commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "==> CUDA preflight"
python3 - "$OUT_DIR/runpod_environment.json" <<'PY'
import json, os, sys
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; select a GPU RunPod PyTorch template")
props = torch.cuda.get_device_properties(0)
report = {
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_total_gib": round(props.total_memory / 2**30, 4),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "pod_id": os.environ.get("RUNPOD_POD_ID"),
}
print(report)
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(report, indent=2) + "\n")
PY

for tool in wget unzip; do
  command -v "$tool" >/dev/null || NEED_APT=1
done
if [[ "${NEED_APT:-0}" == "1" ]]; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq wget unzip >/dev/null
fi

echo "==> Python dependencies (pinned to the template torch stack)"
TORCH_CONSTRAINTS="$OUT_DIR/torch_constraints.txt"
python3 - "$TORCH_CONSTRAINTS" <<'PY'
import sys
from importlib import metadata

pins = []
for name in ("torch", "torchvision", "torchaudio"):
    try:
        pins.append(f"{name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        pass
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(pins) + "\n")
print("pinned:", ", ".join(pins))
PY
python3 -m pip install -q -c "$TORCH_CONSTRAINTS" -e '.[dev]'
python3 -m pip install -q -c "$TORCH_CONSTRAINTS" diffusers transformers accelerate "huggingface_hub>=0.34"

# ---------------------------------------------------------------- A: issue #4
echo "==> A. issue #4 internal wan_blocks + LoRA smoke (norm_video_kv)"
trap - ERR
set +e
python3 scripts/smoke_m2_real_weights.py wan \
  --model-path "$TI2V_DIR" \
  --out "$OUT_DIR/m2_internal" \
  --device cuda \
  --dtype bfloat16 \
  --bridge-location wan_blocks \
  --bridge-lora-rank 8 \
  --bridge-lora-alpha 16 2>&1 | tee "$LOG_DIR/m2_internal.log"
A_STATUS=${PIPESTATUS[0]}
set -e
echo "A_EXIT=$A_STATUS"

# ---------------------------------------------------------------- B: VBench
echo "==> B. VBench 6-dim subset over the M1 clips"
if (( $(find "$CLIPS_DIR" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l) < 50 )) \
   && [[ -s "$WORKSPACE/pub/m1_out_50.tgz" ]]; then
  echo "==> restoring clips from pub/m1_out_50.tgz"
  tar -xzf "$WORKSPACE/pub/m1_out_50.tgz" -C "$WORKSPACE"
fi
echo "clips: $(find "$CLIPS_DIR" -maxdepth 1 -name '*.mp4' 2>/dev/null | wc -l)"

set +e
VENV="$WORKSPACE/venv_vbench"
[[ -x "$VENV/bin/python" ]] || python3 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install -q -c "$TORCH_CONSTRAINTS" vbench 2>&1 | tee "$LOG_DIR/vbench_install.log"
VBENCH_CACHE_DIR="$WORKSPACE/vbench_cache" "$VENV/bin/python" scripts/run_vbench.py \
  --videos "$CLIPS_DIR" \
  --model ti2v-5b \
  --out-dir "$OUT_DIR/vbench" \
  --cache-dir "$WORKSPACE/vbench_cache" 2>&1 | tee "$LOG_DIR/vbench.log"
B_STATUS=${PIPESTATUS[0]}
set -e
echo "B_EXIT=$B_STATUS"

python3 - "$OUT_DIR" "$A_STATUS" "$B_STATUS" <<'PY'
import json, sys
from pathlib import Path

out, a_status, b_status = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

def load(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

m2 = load(out / "m2_internal" / "m2_wan_smoke.json")
metrics = (m2 or {}).get("metrics", {})
audio_loss, grad_norm = metrics.get("audio_loss"), metrics.get("grad_norm")
red_flag_cleared = (
    a_status == 0 and audio_loss is not None and grad_norm is not None
    and audio_loss < 10 and grad_norm < 10
)
vb = load(out / "vbench" / "ti2v-5b-m1_summary.json")
report = {
    "issue4": {"exit": a_status, "metrics": metrics, "red_flag_cleared": red_flag_cleared,
               "peak_allocated_gib": (m2 or {}).get("cuda_max_allocated_gib")},
    "vbench": {"exit": b_status, "summary": vb},
    "environment": load(out / "runpod_environment.json"),
}
(out / "queue_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
print("M2_INTERNAL_NORM_OK" if red_flag_cleared else "M2_INTERNAL_NORM_NOT_CLEARED")
print("VBENCH_OK" if b_status == 0 and vb and vb.get("dimensions") else "VBENCH_FAILED")
PY

touch "$OUT_DIR/DONE"
echo "QUEUE_DONE report=$OUT_DIR/queue_report.json"

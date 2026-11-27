#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Theama M1 eval-set generation smoke on a RunPod PyTorch pod.
#
#   bash scripts/runpod_m1_eval.sh [/workspace] [limit]
#
# Downloads the full Wan 2.2 TI2V-5B Diffusers repo (Apache-2.0) into the
# network volume (resuming the transformer already fetched by the M2 smoke),
# runs the CPU suite, then generates the first <limit> prompts (default 5) of
# eval/t2v_v1.jsonl at native 1280x704 / 121f / 24fps with CPU offload.
# Writes $WORKSPACE/m1_out/{ti2v-5b/*.mp4, m1_eval_report.json, logs/}.
# Prints M1_EVAL_SMOKE_OK only if every requested clip is a probe-valid MP4.
# This is a generation/plumbing gate, not the VBench quality gate.
#
# Unattended runs: set THEAMA_SELF_TERMINATE=1 so the pod archives m1_out to
# $WORKSPACE/pub/m1_out.tgz and deletes itself (runpodctl, $RUNPOD_POD_ID)
# when the script exits, pass or fail. Results stay on the network volume.
# Without this, a pod whose controller goes offline keeps billing (2026-11-26:
# a 10.6 h run idled 16 h more and cost $15.73 against an $8 cap).
set -euo pipefail

WORKSPACE="${1:-/workspace}"
LIMIT="${2:-5}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
trap 'status=$?; echo "M1_EVAL_FAILED status=$status line=$LINENO" >&2; exit "$status"' ERR

mkdir -p "$WORKSPACE"
WORKSPACE="$(cd "$WORKSPACE" && pwd)"
export WORKSPACE

self_terminate() {
  local status=$?
  [[ "${THEAMA_SELF_TERMINATE:-0}" == "1" ]] || return 0
  echo "==> THEAMA_SELF_TERMINATE: archiving m1_out (exit status $status)"
  mkdir -p "$WORKSPACE/pub"
  tar -czf "$WORKSPACE/pub/m1_out.tgz" -C "$WORKSPACE" m1_out || true
  sync
  if [[ -n "${RUNPOD_POD_ID:-}" ]] && command -v runpodctl >/dev/null; then
    echo "==> deleting pod $RUNPOD_POD_ID"
    runpodctl remove pod "$RUNPOD_POD_ID" || echo "WARNING: self-terminate failed; delete the pod manually" >&2
  else
    echo "WARNING: RUNPOD_POD_ID or runpodctl missing; pod NOT deleted" >&2
  fi
}
trap self_terminate EXIT
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export HF_HOME="$WORKSPACE/hf_cache"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TI2V_DIR="$WORKSPACE/checkpoints/wan2.2/ti2v-5b"
OUT_DIR="$WORKSPACE/m1_out"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$HF_HOME" "$TI2V_DIR" "$LOG_DIR"
echo "==> workspace=$WORKSPACE limit=$LIMIT commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "==> CUDA preflight"
python3 - <<'PY'
import json, os
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable; select a GPU RunPod PyTorch template")
props = torch.cuda.get_device_properties(0)
report = {
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_total_gib": round(props.total_memory / 2**30, 4),
    "torch_version": torch.__version__,
    "torch_is_nightly": "dev" in torch.__version__,
    "cuda_version": torch.version.cuda,
    "cpu_ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 2),
}
print(report)
out = Path(os.environ["WORKSPACE"]) / "m1_out" / "runpod_environment.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
PY

if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  echo "==> installing ffmpeg (mux_mp4/probe_mp4 need system ffmpeg)"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null
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
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    fh.write("\n".join(pins) + "\n")
print("pinned:", ", ".join(pins))
PY
python3 -m pip install -q -c "$TORCH_CONSTRAINTS" -e '.[dev]'
# ftfy: diffusers' Wan prompt cleaning calls ftfy.fix_text.
python3 -m pip install -q -c "$TORCH_CONSTRAINTS" \
  diffusers transformers accelerate ftfy sentencepiece "huggingface_hub>=0.34"
command -v hf >/dev/null || { echo "ERROR: hf CLI not found" >&2; exit 2; }
python3 -c 'import torch, ftfy; from diffusers import WanPipeline, AutoencoderKLWan; assert torch.cuda.is_available(); print("imports OK")'

echo "==> disk check for the full TI2V-5B repo"
python3 - "$TI2V_DIR" "$WORKSPACE" <<'PY'
import shutil, sys
from pathlib import Path
from huggingface_hub import HfApi

local, ws = Path(sys.argv[1]), sys.argv[2]
info = HfApi().model_info("Wan-AI/Wan2.2-TI2V-5B-Diffusers", files_metadata=True)
total = sum(s.size or 0 for s in info.siblings)
missing = sum(
    (s.size or 0) for s in info.siblings
    if not ((local / s.rfilename).exists() and (local / s.rfilename).stat().st_size == (s.size or 0))
)
free = shutil.disk_usage(ws).free
gib = 2**30
print(f"repo={total/gib:.1f} GiB, still to download={missing/gib:.1f} GiB, free={free/gib:.1f} GiB")
if missing + 3 * gib > free:
    raise SystemExit(
        f"ERROR: need {missing/gib + 3:.1f} GiB free on {ws}, have {free/gib:.1f} GiB; "
        "enlarge the network volume and re-run"
    )
PY

echo "==> Wan 2.2 TI2V-5B Diffusers, full repo (Apache-2.0)"
hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir "$TI2V_DIR"

echo "==> CPU contract suite"
python3 -m pytest tests/ -q -p no:cacheprovider 2>&1 | tee "$LOG_DIR/cpu_tests.log"

echo "==> eval-set generation: first $LIMIT prompts"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 15 \
  > "$LOG_DIR/nvidia_smi.csv" 2>/dev/null &
SMI_PID=$!
# Let a generation failure fall through to the report step (the ERR trap
# would otherwise fire on the failed pipeline even under `set +e`).
trap - ERR
set +e
python3 scripts/gen_eval_set.py \
  --model ti2v-5b \
  --model-path "$TI2V_DIR" \
  --out "$OUT_DIR/ti2v-5b" \
  --limit "$LIMIT" 2>&1 | tee "$LOG_DIR/gen_eval_set.log"
GEN_STATUS=${PIPESTATUS[0]}
set -e
kill "$SMI_PID" 2>/dev/null || true
trap 'status=$?; echo "M1_EVAL_FAILED status=$status line=$LINENO" >&2; exit "$status"' ERR

python3 - "$OUT_DIR" "$GEN_STATUS" "$LIMIT" <<'PY'
import csv, json, sys
from pathlib import Path

out, status, limit = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
gen = out / "ti2v-5b"
summary = json.loads((gen / "summary.json").read_text()) if (gen / "summary.json").exists() else None
manifest = []
if (gen / "manifest.jsonl").exists():
    manifest = [json.loads(l) for l in (gen / "manifest.jsonl").read_text().splitlines() if l.strip()]
peak_mib = None
smi = out / "logs" / "nvidia_smi.csv"
if smi.exists():
    vals = []
    for row in csv.reader(smi.read_text().splitlines()[1:]):
        try:
            vals.append(int(row[1].strip().split()[0]))
        except (IndexError, ValueError):
            pass
    peak_mib = max(vals) if vals else None
walls = [m["seconds_wall"] for m in manifest]
report = {
    "status": "ok" if status == 0 and summary and not summary["failed"] else "failed",
    "gate": "m1_eval_generation_smoke",
    "limit": limit,
    "gen_exit_status": status,
    "summary": summary,
    "clips": manifest,
    "seconds_per_clip_mean": round(sum(walls) / len(walls), 1) if walls else None,
    "nvidia_smi_peak_used_mib": peak_mib,
    "environment": json.loads((out / "runpod_environment.json").read_text()),
    "note": "plumbing/generation gate only; VBench quality gate not run",
}
(out / "m1_eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({k: report[k] for k in ("status", "summary", "seconds_per_clip_mean", "nvidia_smi_peak_used_mib")}))
sys.exit(0 if report["status"] == "ok" else 1)
PY

cat <<EOF

M1_EVAL_SMOKE_OK
Report: $OUT_DIR/m1_eval_report.json
Clips:  $OUT_DIR/ti2v-5b/*.mp4
Logs:   $LOG_DIR
EOF

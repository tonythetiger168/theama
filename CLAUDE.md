# Theama — Claude RunPod M2 Smoke Runbook

## Task

Run the complete M2 real-weight contract smoke workflow on RunPod. Do not use
the user's RTX 4060 8GB for the TI2V-5B transformer. All dependency installs,
weight downloads, tests and GPU smokes run inside the pod.

The canonical entry point is:

```bash
bash scripts/runpod_m2_smoke.sh /workspace
```

The script is idempotent and lives at `scripts/runpod_m2_smoke.sh` in this
repository. It installs dependencies, runs the CPU suite, downloads only the
required clean-license weights, runs both GPU smokes and writes reports/logs.

## Pod requirements

- GPU: RTX 4090 24GB is sufficient for the tiny contract smoke; do not upgrade
  to A100 before inspecting an actual failure or peak-memory report.
- CPU RAM: 32GB recommended.
- Storage: 40GB network volume mounted at `/workspace` so weights and reports
  survive pod stop/start.
- Template: a currently supported RunPod PyTorch template with CUDA available.
  Do not select a template marked Deprecated.
- Region: choose an available, cost-appropriate region.

If connected RunPod MCP/API tools are available, the agent may use them to
create or configure the pod after the user authorizes deployment and its cost.
Reading this file alone does not provide RunPod credentials or account access.
Never ask the user to paste API keys, PATs or SSH passwords into chat or commit
them to this repository. If no RunPod control tool is connected, give the user
these settings for manual deployment instead of claiming a pod was created.

## Execution

From a fresh pod terminal:

```bash
cd /workspace
git clone https://github.com/tonythetiger168/theama.git
cd theama
bash scripts/runpod_m2_smoke.sh /workspace
```

If `/workspace/theama` already exists:

```bash
cd /workspace/theama
git pull --ff-only
bash scripts/runpod_m2_smoke.sh /workspace
```

Do not substitute `scripts/runpod_bootstrap.sh` or
`scripts/download_weights.sh` for this M2-only run: those broader paths also
download A14B backbones and VideoReward, which are unnecessary here.

## Required acceptance evidence

Success requires all of the following, not merely a started pod or completed
weight download:

1. The CPU pytest suite passes.
2. `M2_DAC_SMOKE_OK` is printed.
3. `M2_WAN_SMOKE_OK` is printed.
4. The final marker is `M2_RUNPOD_SMOKE_OK`.
5. These files exist and are returned for review:

```text
/workspace/smoke_out/dac/m2_dac_smoke.json
/workspace/smoke_out/wan_ti2v_5b/m2_wan_smoke.json
/workspace/smoke_out/runpod_environment.json
```

The DAC smoke also writes `input.wav` and `reconstructed.wav` under
`/workspace/smoke_out/dac/`. Logs are under `/workspace/smoke_out/logs/`.

On failure, return the complete error plus the relevant log and
`runpod_environment.json`; do not claim GPU acceptance. The Wan smoke uses
random latent/text conditioning. It validates the real-weight tensor,
backward/optimizer and memory contracts, not prompt-to-video quality,
perceptual audio quality or A/V-sync acceptance.

## Save results and delete the pod

This cleanup is mandatory after both success and failure. Do not leave a pod
running or stopped while waiting for review: a stopped pod may still incur
storage charges, and the requested end state is a deleted pod with its network
volume preserved.

1. Create `results/runpod/m2-smoke/` in the local checkout on the machine
   running Claude. Copy the complete contents of `/workspace/smoke_out/` into
   that directory with the available RunPod file-transfer or SSH route. Keep
   the relative paths, including all JSON reports, DAC `input.wav` and
   `reconstructed.wav`, and every file under `logs/`. Do not commit generated
   results, credentials or weight files to the repository.
2. Save the complete terminal output and a `run-summary.md` in the same local
   directory. Record the command, pod ID, GPU, start/end time, exit status,
   observed success markers and result path. On failure, include the complete
   error/traceback and copy every partial artifact that exists; never discard
   failure evidence or claim GPU acceptance.
3. Verify the local copies before pod deletion. On success, all three required
   JSON reports and both DAC WAV files must exist and be non-empty, all copied
   JSON must parse, and the logs directory must be non-empty. On failure,
   verify that every copied file is non-empty and every copied JSON parses;
   list any expected artifacts that could not be produced in `run-summary.md`.
4. Only after the local evidence is verified, use the connected RunPod MCP/API
   to **delete** the pod, not merely stop it. Confirm deletion by pod ID with a
   subsequent get/list operation. Preserve the network volume and all saved
   results and weights on it; do not delete the volume.
5. Return the absolute local result path, pod ID, deleted/not-deleted
   confirmation, overall success/failure status and any error. Report the pod
   deletion separately so billing status is explicit. Never claim that results
   were saved or the pod was deleted without performing and verifying those
   actions.

If local transfer fails, make one bounded retry through another available
route. If it still fails, preserve a `smoke_out` archive on the network volume,
then delete the pod anyway to prevent further pod billing. In that exceptional
case, report clearly that local verification is incomplete and give the volume
archive path; do not describe the handoff as complete.

## License and data boundaries

Use only the TI2V-5B transformer from
`Wan-AI/Wan2.2-TI2V-5B-Diffusers` (Apache-2.0) and official DAC 44.1kHz (MIT).
The script intentionally downloads the transformer subfolder only.

MiniMax H3, LTX-2.x, Qwen-Image 2.1, FLUX 3 Klein and Krea 2 weights, code and
generated outputs must not enter this repository or its training data. API
blind benchmarking is allowed only under the project rules; their outputs must
not be used for training. Do not weaken these boundaries while troubleshooting.

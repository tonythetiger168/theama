# Theama — Current status and next steps (handoff)

Last updated: 2026-11-24. This file is the single entry point for any agent
(Claude, Kimi or others) continuing the work. Read it together with
`CLAUDE.md` (RunPod M2 runbook and license boundaries),
`docs/model-roadmap.md` and `docs/training-plan.md`.

## Milestone status

| Milestone | Code | GPU validation |
|---|---|---|
| M1 inference + eval harness | landed | **not run** |
| P1a VideoReward + LoRA-DPO | landed (encode_pairs resume/fault tolerance hardened) | **not run** |
| M2 audio DiT + joint A/V trainer | CPU trainer, DAC codec, external Wan adapter, bridge LoRA warmup, AV manifest data pipeline | **real-weight contract smoke PASSED 2026-11-24** (see below) |
| M3+ | planned | — |

## M2 real-weight RunPod smoke — 2026-11-24 result

Command: `bash scripts/runpod_m2_smoke.sh /workspace` (repo at commit
`c13d0f2`, i.e. after PR #2).

| Item | Value |
|---|---|
| Pod | `8glb2h730bo2mb` "theama-m2-smoke", 1x RTX 4090 24 GB (23.52 GiB visible), secure cloud, US, $0.89/hr, no network volume, 60 GB container disk |
| Image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` → torch `2.8.0.dev20250319+cu128` (**nightly**), CUDA 12.8 |
| Runtime / cost | ~6 min pod uptime (~4 min smoke), ≈ $0.09 |
| CPU suite on pod | 101 passed, 4 skipped |
| Markers | `M2_DAC_SMOKE_OK`, `M2_WAN_SMOKE_OK`, `M2_RUNPOD_SMOKE_OK`, exit 0 |
| Pod state after run | stopped (not terminated) |

Key report values (full JSON kept outside the repo, see below):

- **DAC 44.1 kHz** (`dac_stereo_encode_decode`, cuda): input `[1,2,44544]`,
  latent `[1,87,2048]` (mid/side stacked), restored `[1,2,44544]`,
  latent rate 86.13 Hz, reconstruction relative L2 **0.0935** (random-weight
  CPU emulation gives ≈1.0, so real weights were loaded), peak CUDA 0.67 GiB
  allocated / 0.90 GiB reserved.
- **Wan 2.2 TI2V-5B frozen** (`wan_5d_av_forward_backward`, bf16): video
  `[1,48,1,2,2]`, audio `[1,8,2048]`, text cond `[1,16,4096]`, bridge location
  `latent_pre_backbone`, 51,520 trainable params, 32 bridge grad tensors, 28
  updated, max bridge delta 1.22e-4, Wan frozen, loss 4.0 (audio 2.14 /
  video 1.875), grad norm 0.31, peak CUDA **9.37 GiB** allocated / 9.38 GiB
  reserved.

What this proves: real-weight load paths, tensor layouts, backward through the
frozen 5B transformer, optimizer updates of the bridges and memory headroom on
a 24 GB card. What it does **not** prove: prompt-to-video quality, perceptual
audio quality, A/V sync.

### Where the raw artifacts are

Per `CLAUDE.md`, generated results are not committed. They are on Tony's PC
(`trinity`) at `C:\Users\chien\aios\theama_m2_smoke\` (run.log, both JSON
reports, `dac/input.wav`, `logs/cpu_tests.log`, `logs/dac_smoke.log`,
`logs/wan_smoke.log`).

### Known gaps from this run

1. `runpod_environment.json` and `dac/reconstructed.wav` returned HTTP 404 when
   downloaded through the pod's `python -m http.server` proxy, although run.log
   shows both were written. The pod had no network volume, so they were lost at
   stop. The same environment facts are in run.log (above). A re-run to recover
   them was requested on 2026-11-24 and is waiting for Tony's approval on his
   PC. When re-running, list the served directory first and verify every file
   locally before stopping/deleting the pod.
2. `CLAUDE.md` now requires deleting (not stopping) the pod after verified
   copy and using a 40 GB network volume at `/workspace`. This run predates
   that rule: it used no volume and the pod `8glb2h730bo2mb` is stopped.
3. The image above ships a **nightly** torch. Fine for the smoke; pick an image
   with a stable torch release before any real training.
4. pip reported a harmless conflict (`httpcore` wants `h11<0.15`, `h11 0.16.0`
   installed).

## How pods are launched (no secrets here)

- Via the RunPod GraphQL API (`https://api.runpod.io/graphql`) with the
  `RUNPOD_API_KEY` stored in Tony's Windows user environment on `trinity`.
  `rest.runpod.io/v1` no longer resolves; old aevum scripts using it fail.
- `runpodctl`, the `runpod` Python package and a RunPod MCP server are not
  installed there.
- Starting, stopping or deleting a billed pod needs Tony's explicit approval.
- Existing network volume `aevum-data` (850 GB) is in EU-RO-1, which had no
  RTX 4090 stock on 2026-11-24; a 4090 run needs its own volume in a
  datacenter that has 4090s.
- Never commit or print API keys.

## Next steps (in order)

1. **(Optional) M2 re-run** to capture `runpod_environment.json` and
   `reconstructed.wav`, following the updated `CLAUDE.md` (40 GB volume,
   verify, then delete the pod).
2. **M1 GPU acceptance** on TI2V-5B (4090 is enough with CPU offload):
   `python scripts/gen_eval_set.py --model ti2v-5b --out $WORKSPACE/eval_out/ti2v-5b --limit 5`
   first, then the full 50-prompt `eval/t2v_v1.jsonl`, then
   `scripts/run_vbench.py`. Needs the full TI2V-5B Diffusers repo (VAE + text
   encoder), not only the transformer.
3. **P1a GPU run**: download VideoReward (`KlingTeam/VideoReward`, Apache-2.0),
   then the pipeline printed at the end of `scripts/runpod_bootstrap.sh`
   (candidates → `score_eval_set.py --smoke` → scores → pairs →
   `encode_pairs.py` → `train_dpo.py`). Verify `reward/score.py::_invoke`
   against the real model first.
4. **M2 integration**: move bridges inside Wan transformer blocks (current
   bridge is pre-backbone on the latent grid), connect the AV manifest
   pipeline to real licensed data, then a real bridge-warmup training run.
   Acceptance target: AV-sync < 40 ms.
5. Pick a stable-torch RunPod image and record it in `CLAUDE.md`.

License red lines in `CLAUDE.md` and `docs/license-risks.md` apply to every
step.

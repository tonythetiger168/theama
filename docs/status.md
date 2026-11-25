# Theama — Current status and next steps (handoff)

Last updated: 2026-11-26. This file is the single entry point for any agent
(Claude, Kimi or others) continuing the work. Read it together with
`CLAUDE.md` (RunPod M2 runbook and license boundaries),
`docs/model-roadmap.md` and `docs/training-plan.md`.

## Milestone status

| Milestone | Code | GPU validation |
|---|---|---|
| M1 inference + eval harness | landed | **not run** |
| P1a VideoReward + LoRA-DPO | landed; full preflight-gate chain: `check_scores.py` → `make_pref_pairs.py` fail-fast, `check_pairs.py` → `encode_pairs.py` fail-fast, `check_pair_latents.py` → `train_dpo.py` fail-fast; retry budgets + `pairs_stats.py` / `scores_stats.py` profilers | **not run** |
| P1b data engine | landed: `build_av_manifest.py` → `encode_texts.py` / `encode_av_latents.py` (retry budgets on all three caches) → joint latent loader (explicit `skipped_*_row` drop reasons) → joint training runner + runbook §5; joint-cache preflight gate `check_joint_latents.py` wired into runbook §5.1/§5.2 (fails before any GPU burn on dropped rows or corrupt/non-finite `.pt`) | **not run** (issue #4 open) |
| M2 audio DiT + joint A/V trainer | CPU trainer, DAC codec, external Wan adapter, internal Wan block bridges, bridge LoRA warmup, joint training runner, AV manifest data pipeline, **CFG text-conditioning dropout** (audio-stream null-condition training, default 0.0 off) | **external latent-bridge real-weight contract smoke PASSED** (RTX 4090 2026-11-24; full runbook re-run on A40 2026-11-25, see below); **internal `wan_blocks` real-weight GPU acceptance not run** |
| M4 distribution layer | **HLS fMP4/CMAF packager** (`package_hls.py`: 2160p AV1 → 1080p HEVC Safari fallback → 720p/480p H.264 ladder, per-rung ffprobe gates) + **DASH manifest.mpd over the same segments** (`package_dash.py`, zero re-encode); HDR10/HLG 10-bit HEVC/AV1 mux landed | n/a (CPU-verified packaging) |
| M7a CJK T2I eval | seed v1.2 (100 prompts) at acceptance-gate size; scorer script-split (zh-Hans gate / zh-Hant parallel stats); Ming-Image-0.1-Design third path plumbed as diagnostic-only | **not run** (Qwen-Image-2512 / Z-Image Turbo repro) |
| M3+ | planned | — |

Since 2026-11-25 the CPU suite has grown to **396 passed**; the repo is now
**public** (`github.com/tonythetiger168/theama`). The M5 north-star (per
Tony 2026-11-25): top the Artificial Analysis Video Arena with-audio boards —
T2V/editing above Wan 3.0, I2V above Seedance 2.5 (see
`docs/model-roadmap.md`).

## M2 external latent-bridge RunPod smoke — 2026-11-25 full runbook re-run (A40)

This is the run that satisfies every `CLAUDE.md` acceptance item (all three
JSON reports, both DAC WAVs, logs, terminal output, verified locally, pod
deleted, network volume kept).

| Item | Value |
|---|---|
| Pod | `pyf51osir0amhh` "theama-m2-smoke-24g", secure, CA-MTL-1, $0.59/hr — **deleted** 2026-11-25 00:37:57 UTC (absent from pod list afterwards) |
| GPU | NVIDIA A40 48 GB (44.42 GiB visible). No secure RTX 4090/3090/L4/A5000 was in stock; Tony approved "any secure ≥24 GB ≤ $1/hr". EU-SE-1 had a cheaper A40 but does not support network volumes. |
| Network volume | `theama-m2` (id `xb1dc3goq4`), 40 GB, CA-MTL-1 — **kept** (holds TI2V-5B transformer, DAC weights, `smoke_out.tgz`; incurs storage cost while it exists) |
| Image / torch | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, torch `2.8.0.dev20250319+cu128` (nightly), CUDA 12.8 |
| Commit | `4c2d828`, `bash scripts/runpod_m2_smoke.sh /workspace` |
| Time / cost | 00:34:26 → 00:37:06 UTC (~2 min 40 s), pod ~3.5 min, < $0.05 |
| CPU suite on pod | 147 passed, 23 skipped |
| Markers | `M2_DAC_SMOKE_OK`, `M2_WAN_SMOKE_OK`, `M2_RUNPOD_SMOKE_OK`, `EXIT=0` |
| Local results | `C:\Users\chien\aios\theama\results\runpod\m2-smoke\` on Tony's PC (full `smoke_out` tree + `run-summary.md`; not committed) |
| Verification | 3 JSONs parse; `input.wav` / `reconstructed.wav` 178,220 B each; logs + `terminal_output.log` non-empty |

Key values: DAC relative L2 **0.0942**, latent `[1,87,2048]`, peak 0.68 GiB.
Wan TI2V-5B bf16: 32 bridge grad tensors / 28 updated, Wan frozen, loss 4.19,
grad norm 0.52, peak **9.37 GiB** — identical memory to the 4090 run.

Same scope limits as below: contract smoke only, no prompt-to-video, perceptual
audio or A/V-sync gate.

## M2 external latent-bridge RunPod smoke — 2026-11-24 result (first run, RTX 4090)

Command: `bash scripts/runpod_m2_smoke.sh /workspace` (repo at commit
`c13d0f2`, i.e. after PR #2).

| Item | Value |
|---|---|
| Pod | `8glb2h730bo2mb` "theama-m2-smoke", 1x RTX 4090 24 GB (23.52 GiB visible), secure cloud, US, $0.89/hr, no network volume, 60 GB container disk |
| Image | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` → torch `2.8.0.dev20250319+cu128` (**nightly**), CUDA 12.8 |
| Runtime / cost | ~6 min pod uptime (~4 min smoke), ≈ $0.09 |
| CPU suite on pod | 101 passed, 4 skipped |
| Markers | `M2_DAC_SMOKE_OK`, `M2_WAN_SMOKE_OK`, `M2_RUNPOD_SMOKE_OK`, exit 0 |
| Pod state after run | stopped; deleted later on 2026-11-24 |

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

1. `runpod_environment.json` and `dac/reconstructed.wav` returned HTTP 404
   when downloaded through the pod's `python -m http.server` proxy and were lost
   at stop. **Resolved** by the 2026-11-25 re-run (tar the tree, then download).
2. This run used no network volume and only stopped its pod. **Resolved**:
   pod `8glb2h730bo2mb` was deleted 2026-11-24, and the re-run followed the
   delete-pod / keep-volume rule.
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
  On trinity, Claude Code auto mode only lets these through because Tony
  added `Bash(curl -s https://api.runpod.io/graphql:*)` to
  `permissions.allow` himself; agents must not edit those settings.
- Transfer results by tarring `/workspace/smoke_out` on the pod and downloading
  the archive; per-file fetches through `python -m http.server` lost files once.
- Secure-cloud stock changes within minutes; some datacenters (e.g. EU-SE-1)
  don't support network volumes.
- Existing network volume `aevum-data` (850 GB) is in EU-RO-1, which had no
  RTX 4090 stock on 2026-11-24; a 4090 run needs its own volume in a
  datacenter that has 4090s.
- Never commit or print API keys.

## Next steps (in order)

1. **External M2 smoke acceptance: done** (2026-11-25 A40 run). Reuse volume
   `theama-m2` in CA-MTL-1 for later runs to skip re-downloading weights, or
   delete it when no longer needed. A 4090 re-check is optional. Internal
   `wan_blocks` GPU acceptance remains open in step 4.
2. **M1 GPU acceptance** on TI2V-5B. One-command RunPod runner:
   `bash scripts/runpod_m1_eval.sh /workspace 5` (first 5 prompts of
   `eval/t2v_v1.jsonl`, native 1280x704 / 121f / 24fps, CPU offload). It
   downloads the full TI2V-5B Diffusers repo into the volume (fails fast if
   the volume is too small), installs ffmpeg/ftfy, runs the CPU suite, then
   writes `$WORKSPACE/m1_out/{ti2v-5b/*.mp4, m1_eval_report.json, logs/}` and
   prints `M1_EVAL_SMOKE_OK`. Then run all 50 (`... /workspace 50`), then
   `scripts/run_vbench.py` for the ±0.02 quality gate. Same save / verify /
   delete-pod / keep-volume procedure as `CLAUDE.md`.
3. **P1a GPU run**: download VideoReward (`KlingTeam/VideoReward`, Apache-2.0),
   then the pipeline printed at the end of `scripts/runpod_bootstrap.sh`
   (candidates → `score_eval_set.py --smoke` → scores → pairs →
   `encode_pairs.py` → `train_dpo.py`). Verify `reward/score.py::_invoke`
   against the real model first. Each stage is now fail-fast gated
   (`check_scores.py` / `check_pairs.py` / `check_pair_latents.py`); a bad
   manifest aborts before any GPU burn instead of crashing mid-loop.
4. **M2 internal integration**: internal Wan block bridges and the joint
   training runner have landed with CPU contracts. Run the real-weight
   `wan_blocks` + LoRA smoke (`--bridge-location wan_blocks
   --bridge-lora-rank 8 --bridge-lora-alpha 16`) — tracked as GitHub
   issue #4 — then connect the AV manifest pipeline to real licensed data
   and run a real bridge-warmup training run.
   Acceptance target: AV-sync < 40 ms.
5. Pick a stable-torch RunPod image and record it in `CLAUDE.md`.

License red lines in `CLAUDE.md` and `docs/license-risks.md` apply to every
step.

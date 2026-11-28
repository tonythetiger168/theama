# Theama — Current status and next steps (handoff)

Last updated: 2026-11-27. This file is the single entry point for any agent
(Claude, Kimi or others) continuing the work. Read it together with
`CLAUDE.md` (RunPod M2 runbook and license boundaries),
`docs/model-roadmap.md` and `docs/training-plan.md`.

## Milestone status

| Milestone | Code | GPU validation |
|---|---|---|
| M1 inference + eval harness | landed; VBench tolerance gate (`run_vbench.py`) pure-function + CLI CPU tests (fake-vbench injection); `gen_eval_set.py` probe_ok + resume skip/regenerate CPU tests (fake mux/pipeline injection); `inference/pipeline.py` one-shot CLI contract CPU tests (T2V/I2V branch wiring / `--seconds` frame snapping / offload+seed pass-through / argparse rejection / generation failure keeps exit non-zero without muxing, fake t2v/i2v module injection); `model/dit.py` Wan-backbone thin wrapper contract CPU tests (repo-id constants / BackboneConfig defaults incl. bf16 + MoE expert boundary 0.875 / lazy pipeline / Wan generate call conventions incl. 8n+1 frames and no_grad, fake diffusers injection) | **full 50-prompt generation run PASSED** (A40 2026-11-26, 50/50 probe-valid MP4s, mean 842 s/clip with CPU offload; see below); VBench ±0.02 gate **not run** |
| P1a VideoReward + LoRA-DPO | landed; full preflight-gate chain: `check_scores.py` → `make_pref_pairs.py` fail-fast, `check_pairs.py` → `encode_pairs.py` fail-fast, `check_pair_latents.py` → `train_dpo.py` fail-fast; retry budgets + `pairs_stats.py` / `scores_stats.py` profilers; runbook §2 command chain wired (three gates + profilers in-chain); `gen_candidates.py` + `encode_pairs.py` + `score_eval_set.py` + `make_pref_pairs.py` + `train_dpo.py` CLI contract CPU tests (resume lazy-load tripwires / preflight refuse / flag pass-through / failure-keeps-exit-0, fake torch+diffusers injection; `score_eval_set.py --no-resume` duplicate-row regression fixed; `make_pref_pairs.py` margin / max-per-prompt / reward-filter pass-through on binary-exact scores; `train_dpo.py` preflight fails fast before the backbone import) | **not run** |
| P1b data engine | landed: `build_av_manifest.py` → `encode_texts.py` / `encode_av_latents.py` (retry budgets on all three caches) → joint latent loader (explicit `skipped_*_row` drop reasons) → joint training runner + runbook §5; joint-cache preflight gate `check_joint_latents.py` wired into runbook §5.1/§5.2 (fails before any GPU burn on dropped rows or corrupt/non-finite `.pt`); cache profiler `joint_latents_stats.py` (shape mix / caption sharing / index coverage / join view — the read-before-GPU counterpart of the gate); `encode_texts.py` CLI contract CPU tests (resume lazy-load tripwires / exit-code contract, fake transformers/torch injection); `train_joint_av.py` `_real` CLI wiring CPU tests (arg gating / YAML config / frozen-Wan construction / joint-cache loader / RunnerConfig mapping, fake GPU/IO edges only); `check_av_manifest.py` manifest-gate CLI contract CPU tests (exit codes / indent-2 summary / `--show-errors` visibility / forbidden-source red line); gate-vs-profiler cross-consistency CPU tests (one manifest pair through both `check_joint_latents.py` and `joint_latents_stats.py`, count equalities + documented divergences locked); raw-manifest gate-vs-profiler cross-consistency CPU tests (`check_av_manifest.py` gate x `manifest_stats.py` profiler over one shared manifest: total accounting invariant, documented divergences pinned -- non-dict JSON line = schema error to the gate but bad_json_line to the profiler, forbidden+schema-defect row = schema to the gate but still red-line-counted by the profiler; clean-manifest source distributions exactly equal; profiler still exits 0 on gate-rejected manifests) | **not run** (issue #4 open) |
| M2 audio DiT + joint A/V trainer | CPU trainer, DAC codec, external Wan adapter, internal Wan block bridges, bridge LoRA warmup, joint training runner, AV manifest data pipeline, **CFG text-conditioning dropout** (audio-stream null-condition training, default 0.0 off), **`norm_video_kv` bridge-input norm** (parameter-free per-token LayerNorm on the v2a K/V for `wan_blocks` bridges; fail-closed adapter guard; default off keeps the latent-grid path bit-identical) | **external latent-bridge real-weight contract smoke PASSED** (RTX 4090 2026-11-24; full runbook re-run on A40 2026-11-25, see below); **internal `wan_blocks` + LoRA real-weight smoke ran 2026-11-26** (commit `2e16595`, issue #4): contract PASSED (100,352 trainable, 8 grad/4 updated, Wan frozen, peak 9.54 GiB) with a loss-scale red flag — audio loss 79.0 / grad norm 94.0 (~40x external) from un-normalised Wan block hidden states feeding the v2a K/V; CPU-side fix landed (`b5321003`..`d455e3dc`); **GPU re-run pending, bridge-warmup blocked until the red flag clears** |
| M4 distribution layer | **HLS fMP4/CMAF packager** (`package_hls.py`: 2160p AV1 → 1080p HEVC Safari fallback → 720p/480p H.264 ladder, per-rung ffprobe gates) + **DASH manifest.mpd over the same segments** (`package_dash.py`, zero re-encode); HDR10/HLG 10-bit HEVC/AV1 mux landed | n/a (CPU-verified packaging) |
| M6 avatar real-time | IdentityBank / IdentityEncoder / AudioDrive / StreamConfig skeleton contract CPU tests (register-swap identity roundtrip / output shape+dtype / token count follows the video frame grid / stream session deque wiring from config; M6b/M6c channels pinned to their documented NotImplementedError) | — |
| M7a CJK T2I eval | seed v1.2 (100 prompts) at acceptance-gate size; scorer script-split (zh-Hans gate / zh-Hant parallel stats); Ming-Image-0.1-Design third path plumbed as diagnostic-only; one-click generator `gen_cjk_images.py` resume-gate + per-prompt keep-mining CPU tests (a bad prompt no longer aborts a 100-prompt run); `score_cjk_text.py` main()-level CLI contract CPU tests (tesseract preflights / resume compaction / manifest provenance backfill, zero tesseract invocation); v1.3 paired-script seed `eval/cjk_t2i_v1_3.jsonl` (200 rows — the v1.2 hant block byte-preserved plus same-semantics hans pairs with symmetric `pair_of`, so one `score_cjk_text` run yields both script tracks) + structural CPU tests | **not run** (Qwen-Image-2512 / Z-Image Turbo repro) |
| M3+ | stage-skeleton contract CPU tests landed (`extend_30s` num_segments ceil arithmetic / `upscale_4k` DEFAULT_STAGES order + exact params / `KeyframeSpec` defaults, three entry points pinned to NotImplementedError); implementation planned | — |

As of 2026-11-27 the CPU suite stands at **662 passed**; the repo is now
**public** (`github.com/tonythetiger168/theama`). The M5 north-star (per
Tony 2026-11-25): top the Artificial Analysis Video Arena with-audio boards —
T2V/editing above Wan 3.0, I2V above MAGI-2 (Seedance 2.5 now #2; see
`docs/model-roadmap.md`).
Repo-health gates since: shell entry-script contract tests (`scripts/*.sh`
syntax / header contract / wiring coverage) and docs-structure tests
(`docs/*.md` existence / no conflict markers / per-table pipe-count
consistency / balanced code fences / session-state header schema /
SAVE-protocol headings).

## RunPod run 2026-11-26/27 (A40): full 50-prompt M1 generation

| Item | Value |
|---|---|
| Pod | `u4one3groo1q6z`, secure, CA-MTL-1, A40 48 GB, $0.59/hr — **deleted** 2026-11-27 ~10:30 UTC (absent from pod list afterwards) |
| Command | `bash scripts/runpod_m1_eval.sh /workspace 50` (commit `867071d`; resume skipped the 5 clips from the earlier rerun) |
| Generation | 07:48:34Z → 18:28Z on 2026-11-26 (~10.6 h, ≈ $6.27) |
| Total cost | **≈ $15.73** (26.7 h uptime) — **over Tony's $8 cap**. trinity went offline ~3 h in, so the 30-min checks and the 13 h hard stop never ran; the finished pod idled ~16 h |
| Markers | `M1_EVAL_SMOKE_OK`, exit 0, summary `{"total": 50, "ok": 50, "failed": []}` |
| Clips | 50/50 probe-valid, H.264 1280x704, 24 fps, 121 frames (5.06 s); mean **841.9 s/clip** (min 828.0, max 875.5) |
| Peak VRAM | **not recorded**: the container restarted ~18:26Z and re-ran the script, which skipped all 50 clips and overwrote `nvidia_smi.csv` and the logs (report shows 0). The 5-clip run on the same setup peaked at 33,679 MiB |
| Local results | `C:\Users\chien\aios\theama\results\runpod\m1-eval-50\` on trinity (`m1_out\`, `m1_out_50.tgz` 213.7 MB, `run-summary.md`; not committed). Tar also on the volume at `/workspace/pub/m1_out_50.tgz` |
| Volume | `theama-m2` expanded 40 → **80 GB** on 2026-11-26 (Tony's choice), kept |

**Lesson / fix:** a local controller must not be the only thing that stops a
billed pod. `scripts/runpod_m1_eval.sh` now supports
`THEAMA_SELF_TERMINATE=1`: on exit (pass or fail) it tars `m1_out` to
`$WORKSPACE/pub/m1_out.tgz` and deletes its own pod via
`runpodctl remove pod $RUNPOD_POD_ID`. Use it for every long unattended run,
and launch with a start command that does not re-run the job on container
restart.

## RunPod run 2026-11-26 (A40): M1 rerun + issue #4 internal smoke

One pod ran both jobs back to back.

| Item | Value |
|---|---|
| Pod | `bsmi95ocgchqot`, secure, CA-MTL-1, A40 48 GB, $0.59/hr — **deleted** 2026-11-26 01:13:00 UTC (absent from pod list afterwards) |
| Uptime / cost | 2026-11-25 23:54:12 → 2026-11-26 01:13:00 UTC (78.8 min), ≈ **$0.78** of the $2 cap |
| Network volume | `theama-m2` (`xb1dc3goq4`) **kept**; now ~5.1 GB free. Older outputs moved to `m1_out_run1_20261125/` and `smoke_out_m2_20261125/` on the volume |

### M1 eval generation smoke (rerun after the `save_frames` fix)

- Commit `0bf42cd`, `bash scripts/runpod_m1_eval.sh /workspace 5`.
- CPU suite on pod 395 passed / 1 skipped; `M1_EVAL_SMOKE_OK`, exit 0.
- 5/5 clips probe-valid: H.264 1280x704, 24 fps, 121 frames (5.06 s), silent
  AAC 44.1 kHz track. Wall time per clip 865.3 / 835.7 / 837.4 / 844.2 /
  829.3 s (mean **842 s**, CPU offload), nvidia-smi peak **33,679 MiB**.
- Run 1 (pod `31tuzdf6pcdj9z`) had failed on `numpy.ndarray.save` in
  `save_frames`; fixed in `ecf50d6` (`_to_pil`).
- Local results: `C:\Users\chien\aios\theama\results\runpod\m1-eval-rerun\`
  on trinity (MP4s, `m1_eval_report.json`, logs; not committed).
- Scope: plumbing/generation gate only. No VBench or quality judgement yet.

### Issue #4: internal `wan_blocks` + LoRA real-weight smoke

- Commit `2e16595`. External sanity first: CPU 409 passed / 1 skipped,
  `M2_DAC_SMOKE_OK`, `M2_WAN_SMOKE_OK` (9.37 GiB), `M2_RUNPOD_SMOKE_OK`.
- Internal run (`--bridge-location wan_blocks --bridge-lora-rank 8
  --bridge-lora-alpha 16`, bf16): `bridge_video_dim` 3072, bridge blocks
  `[14, 29]`, 100,352 trainable params, 8 grad tensors / 4 updated (consistent
  with LoRA B zero-init on step 1), max delta 1.01e-4, Wan frozen, CFG dropped
  0.0, peak **9.54 GiB** allocated / 9.56 GiB reserved.
- Loss 81.5 (audio **79.0**, video 2.375), grad norm **94.0**.
- Local results: `C:\Users\chien\aios\theama\results\runpod\m2-internal-lora\`
  on trinity; archive also on the volume at
  `/workspace/smoke_out_wan_internal_lora.tar.gz`.

**Red flag:** audio loss 79 is ~37x and grad norm 94 is ~180–300x the
external-bridge run (audio 2.14, grad norm 0.31–0.52). Likely cause: raw Wan block hidden states
(dim 3072, un-normalised) feed the audio bridge at a much larger scale than
latents do. Check hidden-state RMS at blocks 14/29 and add a norm / scale on
the bridge input (or a gate init) before any real bridge-warmup training.
The contract (shapes, backward, frozen Wan, memory) passes; the scale does not
look trainable as is.

**更新（2026-11-26）**：CPU 修復 `norm_video_kv` 已落地（commits
`b5321003`..`d455e3dc`，`run_wan_smoke` 對 `--bridge-location wan_blocks`
自動啟用）；GPU 重跑 pending，見 next step 4。

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
| Image / torch | `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` → torch `2.8.0.dev20250319+cu128` (**nightly**), CUDA 12.8 |
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
2. **M1 GPU acceptance** on TI2V-5B — full 50-prompt generation **PASSED**
   2026-11-26 (50/50 probe-valid, 842 s/clip; see the run section above).
   Remaining: `scripts/run_vbench.py` on those 50 clips for the ±0.02 quality
   gate (clips are on trinity and in `/workspace/pub/m1_out_50.tgz`). Run any
   long pod job with `THEAMA_SELF_TERMINATE=1`. Same save / verify /
   delete-pod / keep-volume procedure as `CLAUDE.md`.
3. **P1a GPU run**: download VideoReward (`KlingTeam/VideoReward`, Apache-2.0),
   then the pipeline printed at the end of `scripts/runpod_bootstrap.sh`
   (candidates → `score_eval_set.py --smoke` → scores → pairs →
   `encode_pairs.py` → `train_dpo.py`). Verify `reward/score.py::_invoke`
   against the real model first. Each stage is now fail-fast gated
   (`check_scores.py` / `check_pairs.py` / `check_pair_latents.py`); a bad
   manifest aborts before any GPU burn instead of crashing mid-loop.
4. **M2 internal integration**: the real-weight `wan_blocks` + LoRA smoke ran
   2026-11-26 (issue #4): contract PASSED but flagged a loss-scale red flag
   (audio loss 79.0 / grad norm 94.0, ~40x the external run) traced to raw
   Wan block hidden states feeding the v2a cross-attention K/V. The CPU-side
   fix (`norm_video_kv`, commits `b5321003`..`d455e3dc`) is landed and
   `run_wan_smoke` auto-enables it for `--bridge-location wan_blocks`. **Next
   GPU step: re-run the same smoke command at HEAD >= `d455e3dc` (no CLI
   change); the red flag clears if audio loss / grad norm return to
   single-digit, external-path scale. Bridge-warmup training stays blocked
   until that re-run passes.** Then connect the AV manifest pipeline to real
   licensed data and run the bridge-warmup training run.
   Acceptance target: AV-sync < 40 ms (not measurable with random
   conditioning; stays open for the trained run).
5. Pick a stable-torch RunPod image and record it in `CLAUDE.md`.
6. **MAGI-2 evaluation (2026-11-28, docs only)**: not adopted as backbone or
   fine-tune base (114B Preview needs 8xHopper, no training code; 480B flagship
   API-only). Kept as an Apache-2.0 code reference for M2 A/V alignment (it
   shares the Wan2.2 TI2V-5B VAE). CPU-only follow-ups in
   `docs/magi-2-evaluation.md`.

License red lines in `CLAUDE.md` and `docs/license-risks.md` apply to every
step.

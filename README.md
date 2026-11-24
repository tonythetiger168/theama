# Theama

**Theama** (Greek θέαμα — "spectacle") is an open-source audio-video generation
project on a fully permissive license stack, with three product lines:

1. **Theama Film** — **4K, 30-second MP4 movie clips with synchronized native
   audio**, at Wan 3.0 / MiniMax H3-class quality.
2. **Theama Avatar** — **real-time interactive digital humans** (voice-driven,
   mid-stream reference swaps, large body motion), at Vidu S2-Avatar class.
3. **Theama Image** — **text-to-image with best-in-class CJK/typography
   rendering** (Qwen-Image-2512 quality base + Z-Image Turbo fast path, both
   Apache-2.0), feeding keyframes to Film and identity references to Avatar.

## Design goals

| Goal | Target | Approach |
| --- | --- | --- |
| Video quality | Wan 3.0 / H3 class (M5 gate); Wan 2.2 A14B parity at M1 | Wan 2.2 MoE DiT backbone (27B total / 14B active, Apache-2.0) + preference post-training (P1a DPO) |
| Native audio | 44.1 kHz stereo, dialogue + SFX + ambience | Parallel audio DiT with bidirectional A/V cross-attention (clean-room, LTX-2.5 pattern; Kandinsky 6.0 MIT / MAGI-2 Apache-2.0 as permissive references) |
| Duration | 30 s per prompt | THEAMA-Extend: SVI-style autoregressive continuation with consistency anchors |
| Resolution | 4K (3840×2160) | THEAMA-4K: two-stage in-context regenerate 720p → 2K → 4K |
| Output | MP4 (H.264 / HEVC / AV1 + AAC, SDR / HDR10 / HLG) | `theama.inference` export / mux pipeline |
| Avatar | 720p real-time interactive | Wan-Animate base (Apache-2.0) + LatentSync (Apache-2.0) lip-sync fallback + causal streaming + replay anchors (clean-room SRF-class) |
| Image | CJK-text-accurate T2I + editing | Qwen-Image-2512 (Apache-2.0) quality base; Z-Image Turbo (Apache-2.0) 8-NFE fast path |
| License | End-to-end permissive | Apache-2.0 / MIT components only — no community-license weights anywhere in the stack |

## Why not MiniMax H3 or LTX-2.5 weights?

H3 (33B, native 32 kHz stereo, 15 s) and LTX-2.5 (22B, native 4K A/V) are the
closest open-weight models to our target quality — but neither can anchor a
truly open project:

- **MiniMax H3 Community License** (effective 2026-08-02) excludes local
  deployment in the US, EU, UK and South Korea, caps commercial revenue at
  $20M/yr, requires "MiniMax H3" attribution, and **forbids using model outputs
  to improve other models**. Only the H3-Base stage is open; Context-IR and
  Regenerate-2K stay API-only.
- **LTX-2.x Community License** caps free commercial use at $10M annual revenue.

Theama therefore builds on **Wan 2.2 (Apache-2.0)** and borrows *architecture
ideas only* from H3 / LTX-2.5. Full analysis: see Docs below.

## Repository layout

```
docs/            landscape survey, architecture, model stack & roadmap, training plan, license risks
src/theama/      model / data / train / inference / reward / avatar / image packages
configs/         base model, P1a DPO and M2 joint A/V configs
eval/            50-prompt T2V eval set + VBench reference scores
scripts/         weight download, inference, eval/VBench, P1a DPO pipeline, RunPod bootstrap
tests/           CPU-only tests (M1 / P1a / M2 / smoke)
```

## Quick start

```bash
pip install -e .[dev]                         # add [train] for DPO / M2 training deps
bash scripts/download_weights.sh              # Wan 2.2 + VideoReward (Apache-2.0) + DAC (MIT), user-side
pytest tests/                                 # CPU tests: M1 configs/eval set/mux, P1a DPO math, M2 audio DiT trainer
```

On a fresh RunPod pod, `bash scripts/runpod_bootstrap.sh [/workspace]` clones the
repo, installs deps, downloads weights to the network volume and runs the CPU tests.

M1 — inference + eval harness (GPU):

```bash
bash scripts/run_inference.sh "a cat walking through neon rain" out.mp4
python scripts/gen_eval_set.py --model ti2v-5b --out eval_out/ti2v-5b     # 50-prompt eval set
python scripts/run_vbench.py --videos eval_out/ti2v-5b --model ti2v-5b    # VBench gate vs reference
```

P1a — preference post-training (GPU, see `docs/training-plan.md`):

```bash
python scripts/gen_candidates.py --model ti2v-5b --out dpo_out/candidates --candidates 4
python scripts/score_eval_set.py --videos dpo_out/candidates               # -> scores.jsonl
python scripts/make_pref_pairs.py --scores dpo_out/candidates/scores.jsonl --out dpo_out/pairs.jsonl
python scripts/encode_pairs.py --pairs dpo_out/pairs.jsonl --out dpo_out/latents
python scripts/train_dpo.py --pairs dpo_out/latents/pairs_latents.jsonl --out dpo_out/run1
```

M2 — the joint A/V trainer (`theama.train.train_joint_av`, config
`configs/m2_joint_av.yaml`) has CPU-tested DAC latent and external Wan adapter
contracts, including a 5-D tiny A/V forward/backward smoke, plus LoRA injection
for bridge warmup (`theama.model.lora`, rank-8 default, merged back before the
full phase). Real-weight GPU smoke runners are ready; internal Wan block
bridges are still pending.

For a fresh RunPod PyTorch pod (recommended: RTX 4090 24GB, 32GB CPU RAM,
40GB network volume), the M2-only idempotent entry point is:

```bash
bash scripts/runpod_m2_smoke.sh /workspace
```

It installs dependencies, runs the CPU suite, downloads only the TI2V-5B
transformer and DAC (not A14B or VideoReward), then runs both real-weight
smokes. Reports and logs stay under `/workspace/smoke_out`; success is marked
by `M2_RUNPOD_SMOKE_OK` after both JSON reports have been written.

The individual commands are:

```bash
python scripts/smoke_m2_real_weights.py dac --weights checkpoints/wan2.2/dac_44khz/weights.pth --out smoke_out/dac
python scripts/smoke_m2_real_weights.py wan --model-path checkpoints/wan2.2/ti2v-5b --out smoke_out/wan_ti2v_5b
```

DAC writes a JSON report plus input/reconstructed WAVs. The Wan command checks
one frozen-TI2V-5B A/V optimizer step, bridge updates and peak CUDA memory using
random latent/text conditioning; it is a contract smoke, not a quality gate.

## Roadmap

Film line:

- **M1** — Wan 2.2 inference reproduction + eval harness (VBench + AV-sync metrics) — code landed 2026-11-24, GPU acceptance run pending
- **P1a** — Preference post-training: VideoReward (Apache-2.0) scoring + LoRA-DPO on Wan 2.2 — code landed 2026-11-24, GPU run pending
- **M2** — Audio DiT branch + joint A/V fine-tune (bridge warmup → full) — CPU trainer + DAC/Wan adapter landed 2026-11-24, real-weight GPU smoke pending
- **M3** — THEAMA-Extend: 5 s → 30 s continuation without drift
- **M4** — THEAMA-4K: two-stage 720p → 2K → 4K regenerate
- **M4b** — THEAMA-8K: 4K → 8K tiled regenerate + HEVC/AV1 + HDR10 delivery (halo tier, after M4 is stable) — HEVC/AV1 + HDR10/HLG mux landed 2026-11-24
- **M5** — Quality closure vs Wan 3.0 / H3 API on a fixed eval set (see docs/quality-path-wan3-h3.md)

Avatar line:

- **M6a** — Offline single-image + audio talking avatar (Wan-Animate base, Apache-2.0; LatentSync lip-sync fallback)
- **M6b** — Streaming real-time generation (causal DiT + rolling KV cache + replay anchors)
- **M6c** — Interactive layer: mid-stream reference swap, voice-directed control (Vidu S2 class)

Image line:

- **M7a** — Qwen-Image-2512 + Z-Image Turbo inference reproduction + CJK text-rendering eval
- **M7b** — Editing pipeline (Qwen-Image-Edit-2511, multi-reference)
- **M7c** — Integration: keyframes → Film I2V, identity references → Avatar

## License

Code in this repository: **Apache-2.0** (see `LICENSE`). Commercial use —
including in products and hosted services — needs no royalty or feedback.
Third-party model weights are downloaded by the user and remain under their own
licenses; see `NOTICE` and `docs/license-risks.md`. Avatars involve real
likenesses and voices — only use imagery and audio you have rights to.

## Docs

- [`docs/landscape-2026-11.md`](docs/landscape-2026-11.md) — global landscape & gap analysis
- [`docs/quality-path-wan3-h3.md`](docs/quality-path-wan3-h3.md) — quality bar & path to Wan 3.0 / H3 class
- [`docs/architecture.md`](docs/architecture.md) — hybrid backbone & three-line architecture
- [`docs/models.md`](docs/models.md) — model stack README (roles, licenses, sizes, status)
- [`docs/model-roadmap.md`](docs/model-roadmap.md) — model release roadmap & watchlist
- [`docs/training-plan.md`](docs/training-plan.md) — milestones & acceptance criteria
- [`docs/license-risks.md`](docs/license-risks.md) — license analysis (H3 / LTX / Qwen / FLUX)

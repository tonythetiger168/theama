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
| Video quality | Wan 2.2 A14B class | Wan 2.2 MoE DiT backbone (27B total / 14B active, Apache-2.0) |
| Native audio | 44.1 kHz stereo, dialogue + SFX + ambience | Parallel audio DiT with bidirectional A/V cross-attention (clean-room, LTX-2.5 pattern; Kandinsky 6.0 MIT / MAGI-2 Apache-2.0 as permissive references) |
| Duration | 30 s per prompt | THEAMA-Extend: SVI-style autoregressive continuation with consistency anchors |
| Resolution | 4K (3840×2160) | THEAMA-4K: two-stage in-context regenerate 720p → 2K → 4K |
| Output | MP4 (H.264 / HEVC / AV1 + AAC, SDR / HDR10 / HLG) | `theama.inference` export / mux pipeline |
| Avatar | 720p real-time interactive | Wan-Animate base (Apache-2.0) + causal streaming + replay anchors (clean-room SRF-class) |
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
docs/            landscape survey, architecture, training plan, license risks
src/theama/      model / data / train / inference / avatar / image packages
configs/         model & pipeline configs
scripts/         weight download + inference entry points
tests/           CPU-only smoke tests
```

## Quick start (M1 scope)

```bash
pip install -e .[dev]
bash scripts/download_weights.sh              # Wan 2.2 weights (Apache-2.0), user-side download
pytest tests/                                 # CPU tests: configs, eval set, MP4 mux
bash scripts/run_inference.sh "a cat walking through neon rain" out.mp4   # GPU
python scripts/gen_eval_set.py --model ti2v-5b --out eval_out/ti2v-5b     # GPU: 50-prompt eval set
python scripts/run_vbench.py --videos eval_out/t2v-a14b --model t2v-a14b  # GPU: VBench gate vs reference
```

## Roadmap

Film line:

- **M1** — Wan 2.2 inference reproduction + eval harness (VBench + AV-sync metrics) — code landed 2026-11-24, GPU acceptance run pending
- **P1a** — Preference post-training: VideoReward (Apache-2.0) scoring + LoRA-DPO on Wan 2.2 — code landed 2026-11-24, GPU run pending
- **M2** — Audio DiT branch + joint A/V fine-tune (bridge warmup → full) — CPU trainer landed 2026-11-24, DAC/Wan GPU wiring pending
- **M3** — THEAMA-Extend: 5 s → 30 s continuation without drift
- **M4** — THEAMA-4K: two-stage 720p → 2K → 4K regenerate
- **M4b** — THEAMA-8K: 4K → 8K tiled regenerate + HEVC/AV1 + HDR10 delivery (halo tier, after M4 is stable)
- **M5** — Quality closure vs Wan 3.0 / H3 API on a fixed eval set (see docs/quality-path-wan3-h3.md)

Avatar line:

- **M6a** — Offline single-image + audio talking avatar (Wan-Animate base, Apache-2.0)
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

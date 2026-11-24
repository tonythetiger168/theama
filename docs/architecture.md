# Theama 架構設計書

目標：**4K、30 秒、原生音訊、MP4 輸出**；全棧 Apache-2.0 / MIT。

## 管線總覽

```
prompt ──► [1] Base 生成（Wan 2.2 A14B, 720p, 5s）
        ──► [2] Audio DiT 同生（44.1kHz stereo）
        ──► [3] THEAMA-Extend（SVI 式接續，→ 30s）
        ──► [4] THEAMA-4K（two-stage regenerate 720p→2K→4K）
        ──► [5] Export（ffmpeg mux → MP4: H.264/H.265 + AAC）
```

## 模組設計

### 1. Video backbone — Wan 2.2 A14B（Apache-2.0）

- 27B MoE（14B active/step）：high-noise / low-noise 雙專家按 timestep 切換。
- 3D causal VAE；umt5-xxl text encoder。
- 開發與 CI 走 TI2V-5B（24GB 可跑）。
- 權重由使用者自行下載（`scripts/download_weights.sh`），不入庫。

### 2. Audio DiT（theama 新增，clean-room）

- **Codec**：Descript Audio Codec（DAC）44.1kHz stereo latent。
- **架構範式**：LTX-2.5 式非對稱雙流 —— video/audio 各自 self-attention，
  每 N 層插一個雙向 cross-attention bridge（`CrossModalBridge`），
  加 modality-aware CFG。**只借架構概念，不碰 LTX 權重/程式碼。**
- **乾淨授權對照實作**：Kandinsky 6.0（MIT，44.1kHz + lip-sync）、
  MAGI-2（Apache-2.0）— 可直接讀碼對齊實作細節。
- **時間對齊**：video latent 幀率與 audio latent 幀率不同，
  以 3D RoPE 的時間軸縮放映射（`model/rope3d.py`）。

### 3. THEAMA-Extend（5s → 30s）

- SVI（Stable Video Infinity）式 autoregressive continuation：
  以前一段尾部 `anchor_frames` 幀 + audio tail 為條件接續生成下一段。
- 一致性：anchor 幀低噪重繪 + audio latent crossfade（預設 250ms）。
- 5s × 6 段 = 30s；段間 drift 以 eval harness 的 identity/temporal 指標把關。

### 4. THEAMA-4K（720p → 2K → 4K）

- Stage 1（720p→2K）：in-context regenerate，video2video 低重繪幅度（denoise 0.35）。
- Stage 2（2K→4K）：tile-based diffusion upscale（denoise 0.25）+ temporal consistency。
- 概念對齊 H3 的 Base→Regenerate-2K 兩段式，但**自建**——H3 的 2K stage 本來就沒開放，
  任何路線都繞不開自建 HR stage。

### 5. Export

- ffmpeg mux：H.264（預設）/ H.265 + AAC 192k → `.mp4`。

## 訓練策略

- Objective：flow matching（logit-normal timestep 採樣）。
- 路線：先 LoRA 只訓 A/V bridge 層驗證音訊對齊 → 再 full fine-tune。
- 資料：授權乾淨的 AV pairs，需自建 data card（來源、授權、篩選流程）。
- **硬性禁令**：任何 MiniMax H3 / LTX 系列權重、程式碼、生成輸出不得進入
  訓練集或蒸餾流程（見 `license-risks.md`）。

## 硬體量級（誠實估算）

| 階段 | 規格 |
|---|---|
| M1 推理重現 | 單卡 24GB（TI2V-5B）/ 80GB（A14B） |
| M2 音訊 LoRA | 單卡 80GB |
| M3–M4 | 多卡節點（8×H100 級）|
| 4K 推理 | 分層 offload + tiled decode，48GB+ 目標 |

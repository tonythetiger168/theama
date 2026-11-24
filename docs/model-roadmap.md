# Theama 模型路線圖（Model Roadmap）

模型交付視角的路線圖；里程碑驗收細節見
[`training-plan.md`](training-plan.md)，品質對標邏輯見
[`quality-path-wan3-h3.md`](quality-path-wan3-h3.md)。

## 現況（2026-11-24）

| 階段 | 模型交付物 | 狀態 |
|---|---|---|
| M1 | Wan 2.2 推理重現（TI2V-5B / T2V-A14B / I2V-A14B）+ eval harness + MP4 mux | 程式落地，GPU 驗收待跑 |
| P1a | **theama-ti2v-5b-dpo**（VideoReward 偏好對 + LoRA-DPO） | 程式落地，GPU 驗收待跑 |
| M2 | **theama-audio-dit v1**（並聯音訊 DiT + 雙向 CrossModalBridge） | CPU trainer 落地（bridge warmup → full），DAC / Wan GPU 串接待做 |
| M4b（前置） | HEVC / AV1 + HDR10 / HLG 10-bit 封裝 | mux 落地；8K regenerate 本體待 M4 穩定 |

## 近期（下一季）

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| P1a→A14B | **theama-t2v-a14b-dpo** | 5B 全綠後同配方推 A14B（80GB 卡） |
| M2 | **theama-audio-dit v1** | 接 DAC codec 與 Wan adapter、LoRA 注入、AV 資料管線後上 GPU 訓練；驗收：AV-sync < 40ms |
| M3 | **theama-extend-30s** | SVI 式接續，anchor_frames=8；驗收：identity drift ≤ 單段 1.5× |
| M4 | **theama-4k-regen** | 720p→2K→4K 兩段 in-context 重建；驗收：LPIPS 不劣化 >10% |

## 中期

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| M4b | **theama-8k-regen** | 4K→8K tiled regenerate + HEVC/AV1 + HDR10（光環級，M4 穩定後才動） |
| M5 | 品質收斂版 Film 模型 | 盲測 vs H3 win/tie ≥50%、vs Wan 3.0 ≥40%；VBench 開源前三 |
| M6a | **theama-avatar-offline** | Wan-Animate（或 Wan-Animate-2-14B）底座 + LatentSync 墊底唇形 |
| M7a | **theama-image 雙路徑** | Qwen-Image-2512（品質）+ Z-Image Turbo（快速）推理重現 + CJK 評測 |

## 遠期

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| M6b | **theama-avatar-stream** | causal DiT + rolling KV + replay anchor；24GB 卡 720p 實時係數 ≥1.0 |
| M6c | **theama-avatar-interactive** | VLM agent 互動層：串流中換參考圖、語音指令（Vidu S2 class） |
| M7b/c | 編輯管線 + **KeyframeBridge** | Qwen-Image-Edit-2511；T2I→Film I2V、identity→Avatar 三線整合 |
| P2 | **omni-reference encoder** | 多圖/影片/音訊參考（對標 Wan 3.0 omni-reference；H3 Contextual Omni Rep clean-room） |

## Watchlist（上游釋出追蹤）

最近核查：**2026-11-24**。

- **Wan-Animate-2-14B**（Apache-2.0）— M6a 升級候選。repo id 已核對：
  `Wan-AI/Wan2.2-Animate-2-14B`（2026-08-07 發布，arXiv 2608.06009），另有
  `Wan2.2-Animate-2-14B-Diffusers` 與 `Wan2.2-Animate-2-14B-Distilled-Diffusers`
  （2026-08-06，base + distilled 權重）。animation / replacement 雙模式同一 checkpoint。
- **Wan 3.x 開權重** — 仍 API-only（2026-11-24 再次確認；2026-11-05 多方核查：Wan-AI HF 27 個 repo、
  Wan-Video GitHub 6 個 repo 均無 2.5/2.6/2.7/3.0 權重；官方
  `AlibabaCloud-Official/Wan3.0` 僅 README + license 檔）。骨幹續用 Wan 2.2。
  旁註：新開放控制模型 `Wan-Dancer-14B`（Apache-2.0，2026-07）可留意。
- **Qwen-Image 新版** — **Qwen-Image-2.1（7B，2026-09-20）統一 T2I+編輯+原生 RGBA，
  但為 Qwen Research License（非商用）→ 紅線確認，不碰**；2.1-Turbo（2026-10-09，8-step）
  同授權。Qwen-Image-3.0 為 hosted-only。Apache-2.0 線最新仍是
  Qwen-Image-2512 與 Qwen-Image-Edit-2511 —— M7a 選型不變。
- **開源 video reward 新模型** — awesome-rm-for-video-generation（2026-10 更新）新增
  AesRM（專家標註美學 RM，base + CoT 變體）、UnifiedReward-Flex、GT-SVJ（self-supervised
  judge）等；授權與權重可得性未逐一核實，VideoReward 仍為首選，暫不加第二把尺。

## 發布策略

- 自研微調權重（DPO LoRA、audio DiT、extend/4K 模組）以 **Apache-2.0** 發布於
  Hugging Face `chienhsinlin` namespace，模型卡遵循 HF model card 規範（訓練資料來源、
  評測分數、限制與合規聲明）。
- 第三方權重（Wan 2.2、Qwen-Image 等）不轉售不重新打包，使用者自行下載
  （`scripts/download_weights.sh`），權重授權各自保留。

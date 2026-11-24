# Theama 模型總覽（Model README）

全棧 **Apache-2.0 / MIT only**。每個模型列角色、底座、授權、規模、現況。
授權紅線見 [`license-risks.md`](license-risks.md)；落地時程見
[`model-roadmap.md`](model-roadmap.md)。

## Film 線（影片生成）

| 模型 | 角色 | 授權 | 規模 / 硬體 | 現況 |
|---|---|---|---|---|
| **Wan 2.2 TI2V-5B** | dev/CI 骨幹（T2V+I2V 雙任務） | Apache-2.0 | dense 5B；1280×704、121f@24fps、50 steps、CFG 5.0；16GB w/ offload | ✅ 使用中（M1） |
| **Wan 2.2 T2V-A14B** | 生產骨幹 | Apache-2.0 | MoE 27B total / 14B active；81f@16fps、CFG 3.0/4.0 雙專家、boundary 0.875；40GB+ w/ offload | ✅ 使用中（M1） |
| **Wan 2.2 I2V-A14B** | 圖生影片（M7c 關鍵幀路徑） | Apache-2.0 | 同上 | ✅ 使用中（M1） |
| **VideoReward**（KlingTeam） | P1a 偏好打分（Qwen2-VL-2B 多維：視覺品質/時序一致/動態程度/文字對齊） | Apache-2.0 | ~5GB bf16 | ✅ 已選定，GPU 驗證待跑 |
| **THEAMA Audio DiT** | 原生音訊（對白+音效+環境音 44.1kHz stereo） | 自研 clean-room（LTX-2.5 雙流範式；Kandinsky 6.0 MIT / MAGI-2 Apache-2.0 對照） | DAC latent，並聯 DiT | 🚧 M2 CPU trainer + Wan adapter 落地 |
| **Descript DAC 44.1kHz** | M2 音訊 continuous-latent codec | MIT | mono 1024 維、hop 512、86.13Hz；Theama 以 mid/side 組成立體聲 2048 維 tokens | 🚧 contract/mock 落地，真實權重 smoke 待跑 |
| **THEAMA-Extend** | 5s → 30s 自回歸接續 + consistency anchors | 自研（SVI 式） | — | 📋 M3 |
| **THEAMA-4K / 8K Regen** | 720p→2K→4K（→8K）in-context 重建 | 自研（H3 In-Context Regeneration 範式） | tiled VAE decode；80GB 卡 | 📋 M4 / M4b |

## Avatar 線（互動數字人）

| 模型 | 角色 | 授權 | 現況 |
|---|---|---|---|
| **Wan-Animate** | 離線 talking avatar 底座備案（replacement 需求回退此第一代） | Apache-2.0 | 📋 M6a 備案 |
| **Wan-Animate-2-14B** | M6a 底座（`Wan-AI/Wan2.2-Animate-2-14B`，animation-only） | Apache-2.0 | ✅ 已選定（base config 已更新） |
| **LatentSync** | 唇形管線墊底方案（Wan-Animate-2 非音訊驅動） | Apache-2.0 | 📋 M6a 備案 |
| **LongCat-Video-Avatar** | 底座備選 | MIT | 👀 watchlist |
| 實時化改造 | causal/block-causal attention + rolling KV cache + replay anchor | 自研 | 📋 M6b/M6c |

## Image 線（文生圖 / 編輯）

| 模型 | 角色 | 授權 | 現況 |
|---|---|---|---|
| **Qwen-Image-2512** | 品質底座（CJK 文字渲染最強開放權重） | Apache-2.0 | 📋 M7a |
| **Z-Image Turbo** | 快速路徑（8-NFE） | Apache-2.0 | 📋 M7a |
| **Qwen-Image-Edit-2511** | 多參考編輯 | Apache-2.0 | 📋 M7b |
| **KeyframeBridge** | T2I 首末幀 → Film I2V；identity 參考 → Avatar | 自研 | 📋 M7c |

## 永不使用（授權紅線）

MiniMax H3（地區排除 + $20M cap + 禁輸出訓練）、LTX-2.x（$10M cap）、
Qwen-Image 2.1（研究非商用）、FLUX 3 Klein（$5M cap）、Krea 2（$1M cap）。
架構概念可 clean-room 借鑑；權重、程式碼、生成輸出一律不進本專案。
防護：`AVDataset.check_source` 黑名單 + reward 註冊表白名單
（`src/theama/reward/config.py`）。

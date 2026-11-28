# MAGI-2 自研適用性評估（2026-11-28）

請求：Tony「eval magic-2 for 自研」（2026-11-28）。「magic-2」判讀為 Sand.ai 的
**MAGI-2**（repo 已在 watchlist 追蹤）。本文件為**文件層研究**：未下載權重、
未開 pod、未花 GPU 費用。

## 一句話結論

**不換底座，也不拿 MAGI-2 當自研起點。** Wan 2.2（TI2V-5B → A14B）骨幹不變。
MAGI-2 Preview 的價值在「讀碼對照」：單流 AV 聯合建模、A/V 時間對齊、與我們
**共用 Wan2.2 VAE latent 空間**。Flagship 480B 只列觀察。

## MAGI-2 是兩個不同的東西

| 項目 | MAGI-2 Preview | MAGI-2 flagship |
|---|---|---|
| 發布 | 2026-08-03/05，HF `sand-ai/MAGI-2-preview`、GitHub `SandAI-org/MAGI-2-preview` | 2026-11-24/25 發表 |
| 權重 | **已開放**，約 307 GB | **未開放**（「Not yet」／「盡快」，無日期） |
| 授權 | Apache-2.0（標準全文，無附加 AUP、無輸出用途限制） | 未公布；目前僅 hosted API |
| 參數 | 114B MoE，約 6B active/token | 480B，約 28B active（來源不一，Higgsfield 稱 400B） |
| 架構 | MagiMoE；單流（text/video/audio token 串成一序列）；40 層（36 sparse＋4 dense），寬 3072，每 latent head 各自 router、top-6/256 experts（層數／expert 數僅第三方 ComfyUI Wiki） | Multi-Head Latent MoE、單流（官方貼文轉述） |
| 規格 | 固定 10 秒；512x896 preview → refiner 1080p；100＋5 步 | 4–30 秒，1080p/25fps，最多 20 個參考 |
| 元件 | Video VAE＝**Wan2.2 VAE（取自 Wan2.2-TI2V-5B）**；Audio VAE＝Stable Audio Open 1.0；Text encoder＝Qwen3.5-27B（56 GB）；turbo VAE decoder | 未公開 |
| 推論需求 | **8×Hopper**、torchrun；1080p 下 80 GB 卡也需 roundtrip offload；需 MagiAttention／MagiCompiler | API：約 $4.50–4.80/分鐘 |
| 訓練碼 | **無**（repo 自述為 inference code；無 fine-tune／LoRA 腳本） | 無 |
| AA 排名 | T2V #22（1040）／I2V #30（967），快照 | I2V with audio **#1**（1163±14）、T2V #7（1150） |

## 對自研的逐項判斷

### 1. 當骨幹或 fine-tune 起點：否

- **算力不符**：Preview transformer 單檔 228 GB（bf16），A40 48 GB 連載入都不行；
  Qwen3.5-27B text encoder 56 GB 也超過單卡。全參或 LoRA 微調要多節點 H100，
  遠超 M1/M2 單卡 24–48 GB 預算（RunPod A40 $0.59/hr；8×H100 量級每小時十幾到
  二十幾美元，未查即時價）。
- **沒有訓練碼**：要自己寫 MoE 訓練、MagiAttention 反傳、router 負載平衡。
- **品質不值這個成本**：Preview 在 AA 榜遠弱於 flagship（T2V −110／I2V −196 Elo）。
- **不重做已驗證工作**：M1 50/50、M2 外部 smoke 皆建立在 Wan 2.2 上。

### 2. 當架構參考：是（主要價值）

- **共用 VAE**：MAGI-2 Preview 直接用 Wan2.2-TI2V-5B 的 VAE（stride 16），
  與我們 TI2V-5B 開發骨幹同一 video latent 空間。它的 audio token 如何與這組
  video latent 在時間軸對齊，可直接對照我們的 `model/rope3d.py` 時間縮放。
- **單流 vs 雙流**：MAGI-2 走單流（同 MiniMax H3），我們走 LTX-2.5 式雙流＋
  `CrossModalBridge`。雙流保留凍結 Wan 骨幹、只訓 bridge／audio 分支，才符合
  單卡預算，**維持雙流不變**。單流可作為 M3 以後若有多卡預算時的消融對照。
- **Apache-2.0 可讀碼、可借實作**（附 NOTICE），比 LTX／H3 只能讀論文乾淨得多。

### 3. 當蒸餾教師或合成資料來源：技術上合法，實務上不划算

- Preview 授權無輸出限制，用其輸出訓練不違反 Apache-2.0。
- 但每支 10 秒需 8×Hopper、105 步，品質又在榜外中段；拿 Wan 2.2 自我蒸餾＋
  真實授權 AV 資料更便宜。**列為非優先，若要做需另立預算並經 Tony 書面核准。**
- Flagship 只有 API，條款未公布；在條款明確允許前，**輸出不得進訓練集**（比照
  `license-risks.md` 對閉源 API 的處理）。

### 4. 授權注意事項（待覆核）

- **Preview 不是「全棧 Apache」**：Audio VAE 取自 Stable Audio Open 1.0，
  其權重授權據知為 Stability AI Community License（非 Apache／MIT）；Qwen3.5-27B
  授權亦未逐字核對。若日後借用這兩個元件，需先逐字核對授權再寫入
  `license-risks.md`。我們的 audio codec 維持 DAC（MIT）。

## 建議的下一步（給 Kimi／下個 session）

全部為 CPU／文件工作，不需 GPU：

1. 讀 `SandAI-org/MAGI-2-preview` 推論碼（不下載權重），整理三件事寫回本文件：
   (a) audio latent 與 Wan2.2 VAE latent 的時間對齊與位置編碼方式；
   (b) 是否有 modality-aware CFG 或 audio/video 分開的 guidance；
   (c) text/video/audio token 的序列排列與 attention mask。
   對照 `model/rope3d.py` 與 `CrossModalBridge`，看能否改進 issue #4 的
   loss-scale 問題（目前 CPU fix 為 `norm_video_kv`）。
2. 逐字核對 Stable Audio Open 1.0 VAE 與 Qwen3.5-27B 授權，更新 `license-risks.md`。
3. watchlist 持續追 flagship 開權重與授權；若以 Apache／MIT 釋出，重新評估
   「教師模型」一項（仍不作底座）。
4. 主線順序不變：VBench（50 支）→ issue #4 GPU 重跑 → P1a DPO。

## 來源（2026-11-28 讀取）

- https://huggingface.co/sand-ai/MAGI-2-preview （官方）
- https://github.com/SandAI-org/MAGI-2-preview （官方，含 LICENSE）
- https://comfyui-wiki.com/en/news/2026-08-05-magi-2-preview （第三方；層數／expert 數唯一來源）
- https://chinaaidispatch.substack.com/p/sandais-video-model-leads-by-four （第三方，2026-11-26）
- https://agentspulse.github.io/tutorials/magi-2-ai/ （第三方，2026-11-25）
- https://magi-2-ai.com/magi-2-open-source （第三方商業站，自稱非官方）
- 既有 repo 紀錄：`docs/model-roadmap.md` watchlist 第 24–29 輪

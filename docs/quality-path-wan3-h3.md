# 品質路徑：對齊 Wan 3.0 / MiniMax H3（2026-11-24）

> 目標：Theama Film 的品質 bar 明確定為 **Wan 3.0 / MiniMax H3 同級**。
> 本文件回答：它們現在多強、差在哪、我們怎麼追。

## 一、對標現況（皆有硬出處）

| 模型 | 狀態 | 上限 | 音訊 | 榜單（AA, 2026-10） | 授權 |
|---|---|---|---|---|---|
| **Wan 3.0** | 2026-08-06 beta / 08-24 正式 API | **30s 單次、1080P（無 4K）**、30fps MP4 | 原生一次成型 | T2V #1，Elo ~1175 | 閉源 API-only |
| **MiniMax H3** | 2026-07-31 發布 | 原生 2K（2560×1440）@24fps、15s（Extend ~30s） | 原生立體聲（對話+環境+音樂同 pass） | T2V #4（1158）/ **I2V #1（1140）** / **編輯 #1** | H3-Base 33B 權重開放，但 Community License 排除美歐英韓、$20M cap、禁輸出訓練 |
| Seedance 2.5 | 2026-07-20 | 30s、4K 10-bit | 聯合音畫、lip-sync 強 | 未上榜（T2V） | 閉源 |
| Kling 4.0 | 2026-10 | 30s、4K HDR、10 關鍵幀導引 | 原生立體聲 | T2V #4（Kling 3.0） | 閉源 |
| **Wan 2.2 A14B**（我們的骨幹） | 2025-07-28 | 5s、720P、16fps | 無 | — | **Apache-2.0** |

Wan 3.0 值得注意的事實：**參數量從未公布**（網傳 60B/27B 均為謠言，27B MoE/14B active 是 Wan 2.2 的規格被錯誤轉貼）；且 **1080P 封頂、無 4K** — 4K 是 Theama 明確的差異化空間。

## 二、H3 架構四件（clean-room 參考，授權紅線不變）

1. **Contextual Omni Representation** — 多模態參考上下文從 ~100K token 壓到 ~4K，omni-reference（9 圖/3 影/3 音）才跑得動。
2. **H3-VAE** — 重建的 tokenizer，原生 2K 的基礎。
3. **H3-Omni Transformer** — 理解流與生成分離。
4. **In-Context Regeneration** — base 模型自己把低解析輸出重生到 2K，**不用獨立 SR 模組** → 直接驗證我們 M4「two-stage regenerate 720p→2K→4K」路線是對的。

## 三、差距分析（Wan 2.2 骨幹 vs 品質 bar）

| 維度 | Wan 3.0 / H3 | Wan 2.2 現況 | 差距 | 對策落點 |
|---|---|---|---|---|
| 偏好對齊（盲測 Elo 的主因） | 大規模人類偏好後訓練（H3 Max 對齊 prompt adherence + 美感） | 純 SFT 權重 | 🔴 最大差距 | **P1：video reward model + DPO/RLHF 階段** |
| 資料引擎 | 未公開但推定億級高分質量管線 | 無自有資料管線 | 🔴 | P1：資料管線（爬取→美學/運動/文字過濾→dense caption）|
| 原生音訊 | 單 pass 音畫同生 | M2 並聯 Audio DiT（未訓） | 🟡 | M2 照計畫；Contextual Omni 壓縮可借鑑降 bridge 成本 |
| 時長 | 30s 單次 | 5s（M3 接續未做） | 🟡 | M3 THEAMA-Extend |
| 解析度 | H3 原生 2K；Wan3.0 僅 1080P | 720P 基礎 | 🟡 | M4 已對齊 H3 In-Context Regen 範式；**4K 超越 Wan 3.0** |
| 文字渲染 | H3 廣告級文字/品牌渲染 | Wan 2.2 中等 | 🟡 | M7 繁中 T2I + 影片文字資料強化 |
| Omni-reference | 9–20 個參考輸入 | 無 | 🔴 | P2：Contextual Omni 式壓縮 + reference encoder（M6/M7c 複用） |

## 四、追平路徑（插入既有里程碑，不改 M 編號）

### P1 — 資料引擎 + 偏好後訓練（插在 M1 之後、M2 並行）

- 資料管線：開放授權影片源（Pexels/Pixabay/Mixkit 等 CC0）→ 美學分數、運動量、文字區域過濾 → umt5/Qwen-VL dense caption → data card（授權紅線沿用 `AVDataset.check_source`）。
- Video reward model：先拿現成開源 reward（VideoReward 類）蒸餾，再用自家盲測資料微調。
- DPO/RLHF：對 Wan 2.2 backbone 做 preference fine-tune（LoRA 先，full FT 後）。
- **驗收**：50-prompt 盲測 vs 未後訓練 Wan 2.2 win rate ≥ 65%；VBench 6 維不掉分（±0.01）。

### P2 — Omni-reference 能力（與 M6c/M7c 合流）

- Reference encoder：圖/影/音參考 → 壓縮 token 流注入 DiT（Contextual Omni 概念 clean-room）。
- **驗收**：單參考圖 identity 保持（ArcFace 類相似度 ≥ 阈值）+ 參考影運動遷移 eval 20 組人工檢通過 ≥ 80%。

### P3 — 品質收斂（原 M5 細化）

- 盲測基礎設施：50-prompt 固定集 + 成對 A/B 評分 pipeline（theama-app 加投票頁）。
- **驗收（M5 更新版）**：
  1. 50-prompt 盲測 win/tie ≥ 50% vs **H3 API**（768p 對 768p）；
  2. win/tie ≥ 40% vs **Wan 3.0 API**（720p 對 720p，承認其 30s/音訊優勢）；
  3. VBench 總分進開源前三；
  4. **4K/30s 為 Wan 3.0 做不到的規格，作為差異化交付**。

## 五、誠實範圍（什麼追不了、什麼不用追）

- **資料規模**：MiniMax/阿里資料管線是億級＋自有平台回流，我們做不到也不假裝。追法是「小而精」：高品質子集 + 後訓練效率（DPO 以小博大）+ 繁中/CJK 利基。
- **Wan 3.0 閉源**：永遠不能當骨幹也不能蒸餾其輸出（API ToS）；H3 同理（授權明文禁止輸出訓練）。**盲測對標可以，訓練資料不行。**
- **Wan 3.0 無 4K**：M4 完成時 Theama 在解析度維度直接贏它。
- Wan-Animate-2-14B（2026-08-07，Apache-2.0）已確認發布 — Avatar 線底座從 placeholder 升級為真型號，M6a 動工時核對 HF repo id 後更新 config。

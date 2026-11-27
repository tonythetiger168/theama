# 授權風險與對策

Theama 的紅線：**全棧 Apache-2.0 / MIT，任何環節不混入 community-license 權重、程式碼或生成輸出。**

## 為什麼不用 MiniMax H3 當骨幹

H3 Community License（2026-08-02 生效）的關鍵條款：

1. **地區排除**：美國、歐盟、英國、韓國不在授權適用範圍——這些地區的
   使用者/部署/甚至「使用輸出」都不被授權。開源專案無法接受使用者群被切掉四大市場。
2. **禁止模型改進**：不得用 H3 的權重、衍生品或**輸出**去改進其他模型
   （H3 自身及其合格衍生除外）→ 直接封死「拿 H3 生成資料蒸餾 theama」這條路。
3. **商用條件**：年營收 > $20M 需事先書面授權；商用產品須顯著標示 "MiniMax H3"。
4. **核心模組未開放**：Context-IR（prompt 結構化）與 Regenerate-2K（2K 升解析）
   仍為 API-only——本地只能重現 768p base，完整 2K 管線本來就無法自架。

結論：H3 只作為**架構概念參考**（單流 omni DiT、omni-reference、三模組管線），
權重、程式碼、輸出三者全不碰。

## 為什麼不用 HyperFlow（Video-Rebirth）權重

HyperFlow（`videorebirth/hyperflow`，AA I2V/editing with-audio 榜前十常客）是
MiniMax-H3 的 8-step LoRA（rank 256 / alpha 256，2.8GB），以 data-free flow
self-distillation 把 49→8 NFE（≈3× 加速）。評估結論（2026-11-27）：

1. **權重屬 H3 Model Derivative**——授權隨 H3 Community License 走：地區排除
   （美/歐/英/韓）、禁止以其 Works/Outputs 改進其他模型、AUP。HyperFlow 的
   **權重與其生成輸出同屬 H3 衍生，一律不得進入專案**（與 H3 本體同條紅線）。
2. **程式碼是 Apache-2.0**——可讀、可借鑑、可移植；引用或改寫其程式碼時
   保留其 LICENSE/NOTICE（THIRD_PARTY_NOTICES 登記）。
3. **方法論可合法重現**：two-time conditioning (t, r) 來自 AnyFlow 論文
   （Gu et al., 2026），data-free self-distillation 不依賴 H3 輸出——把同一套
   8-step self-distillation LoRA 訓在 **Wan 2.2（Apache-2.0）** 上，全鏈乾淨。
   列為 P1a 之後的 few-step 加速候選（見 model-roadmap.md Watchlist）。
4. **工程保留**：HyperFlow 僅於 H200（141GB）驗證，80GB 未測——即使不考慮
   授權，消費卡可複現性也未建立。

## 為什麼不用 LTX-2.5 當骨幹

LTX-2.x Community License：年營收 ≥ $10M 的組織需付費協議；fine-tune 轉移
可能另需授權。對「無限制開源」定位不合。同樣僅借架構概念（雙流 DiT、
雙向 cross-attention、two-stage 上 4K），clean-room 自實作。

## Theama 技術棧授權表

| 元件 | 來源 | 授權 | 備註 |
|---|---|---|---|
| Video backbone | Wan 2.2 A14B / TI2V-5B（Alibaba） | Apache-2.0 | 權重使用者自抓，不入庫 |
| Audio DiT 參考實作 | Kandinsky 6.0（Kandinsky Lab） | MIT | 可讀碼對齊 |
| AV MoE 參考 | MAGI-2-preview（Sand AI） | Apache-2.0 | 可讀碼對齊 |
| 長影片參考 | LongCat-Video（Meituan） | MIT | 策略參考 |
| 本專案程式碼 | Theama | Apache-2.0 | SPDX header + NOTICE |

## 義務與合規機制

- 再散布 Wan 2.2 / Kandinsky / MAGI 相關元件時附各自 LICENSE 與 NOTICE。
- 資料集 data card 強制記錄每筆來源與授權；CI 檢查腳本掃描資料清單。
- 任何貢獻者提交涉及第三方模型輸出的內容 → 拒收。

## Fallback 觸發條件

| 觸發 | 動作 |
|---|---|
| Wan 2.2 授權政策變動 | 切 LongCat-Video（MIT）或 MAGI-2（Apache-2.0）為骨幹 |
| 音訊整合卡住 | 退回 Kandinsky 6.0（MIT）直接當 AV 骨幹重構 |
| 法務對任何依賴有疑慮 | 該依賴即刻下架，退回上一個全乾淨里程碑 |

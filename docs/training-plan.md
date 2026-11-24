# Theama 訓練與開發計畫（M1–M5 + M6 avatar 線）

每個里程碑附**可量測驗收條件**；不通過不往下走。

## M1 — 骨幹重現 + 評測骨架（先做）

- Wan 2.2 A14B（TI2V-5B 為 dev 路徑）T2V/I2V 推理跑通，720p/5s/24fps。
- Eval harness：VBench 子集 + 固定 50-prompt eval set 落盤。
- MP4 mux pipeline（目前無音訊，先留 AAC 靜音軌介面）。
- **驗收**：`pytest tests/` 全綠；eval set 全數產出 MP4；VBench 分數與 Wan 官方報告同量級（±2%）。

## M2 — Audio DiT + 聯合微調

- DAC 44.1kHz stereo latent 接入；`CrossModalBridge` 每 3 層一組。
- LoRA 先只訓 bridge 層 → 對齊後 full FT。
- 指標：AV-sync（SyncNet 類偏移量）、音質（PESQ/STOI 對參考集）。
- **驗收**：5s 片段音畫同步偏移 < 40ms；人聽盲測 ≥ 7/10 認為「聲音屬於畫面」。

## M3 — THEAMA-Extend（30s）

- SVI 式接續，anchor_frames=8，audio crossfade 250ms。
- **驗收**：30s 成片；段接縫處 identity drift 指標不超過單段內自然變化的 1.5×；無可見跳切（人工檢查 eval set 全數）。

## M4 — THEAMA-4K（720p→2K→4K）

- Stage 1 in-context regenerate（denoise 0.35）；Stage 2 tiled upscale（denoise 0.25）。
- **驗收**：4K 輸出對 2K 輸入的 LPIPS 不劣化 > 10%；tile 接縫不可見；單片 30s 4K 在 80GB 卡可完成。

## M5 — 品質收斂對標

- 對手：MiniMax H3 API（2K/15s/音訊）、Seedance 2.5 API（30s/4K）固定 eval set 盲測。
- **驗收**：50-prompt 盲測 win/tie rate ≥ 50% vs H3；VBench 總分進開源前三（當時榜單）。

## 風險與對策

| 風險 | 對策 |
|---|---|
| Wan 2.2 無音訊，audio DiT 自行整合難度高 | M2 先用 Kandinsky 6.0（MIT）對照實作降低試錯成本 |
| 30s 接續 drift 失控 | 退守 15s 單次 + 智慧剪輯接口；anchor 幀數與 CFG 排入超參掃描 |
| 4K 訓練資料不足 | Stage 2 可用純影像 4K 資料預訓，再用影片 fine-tune |
| 授權誤用（H3/LTX 輸出混入資料集） | data card 強制記錄來源；CI 加資料來源檢查腳本 |
| 算力不足 | 每里程碑有 CPU/單卡可驗收的最小子集；M3 起才需多卡 |

## M6 — Avatar 互動數字人線（2026-11-24 新增）

### M6a — 離線 talking avatar

- Wan-Animate（Apache-2.0）重現：單圖 + 音訊 → talking avatar clip。
- **驗收**：唇形同步指標（LSE-C/LSE-D 類）達 HunyuanVideo-Avatar 公開報告同量級；eval set 20 組全數產出。

### M6b — Streaming 實時化

- Causal/block-causal attention 改造 + rolling KV cache + replay anchor。
- **驗收**：消費級單卡（24GB 級）720p 串流實時係數 ≥ 1.0（生成速度 ≥ 播放速度）；連續 5 分鐘無可見漂移。

### M6c — 互動層

- VLM agent：語音指令解析、串流中 identity/參考圖熱替換。
- **驗收**：StreamAV-Bench 類評測九項指標全數跑通；「拿起杯子再微笑」類連貫指令狀態保持正確率 ≥ 80%（人工評）。

### 風險

| 風險 | 對策 |
|---|---|
| Wan-Animate 非音訊驅動，lip-sync 需自接 | M6a 先用 LatentSync（Apache-2.0）唇形管線墊底，再原生整合 |
| 實時化 VRAM/延遲不達標 | 退守 540p（Vidu S1 規格）先達實時，再升 720p |
| 人像/聲音權利合規 | 僅用自有或授權肖像；README 加合規聲明 |

# Theama 訓練與開發計畫（M1–M5）

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

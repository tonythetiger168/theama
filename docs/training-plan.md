# Theama 訓練與開發計畫（M1–M5 + M6 avatar 線 + M7 文生圖線）

每個里程碑附**可量測驗收條件**；不通過不往下走。

## M1 — 骨幹重現 + 評測骨架（先做）

- Wan 2.2 A14B（TI2V-5B 為 dev 路徑）T2V/I2V 推理跑通，720p/5s/24fps。
- Eval harness：VBench 子集 + 固定 50-prompt eval set 落盤。
- MP4 mux pipeline（目前無音訊，先留 AAC 靜音軌介面）。
- **驗收**：`pytest tests/` 全綠；eval set 全數產出 MP4；VBench 分數與 Wan 官方報告同量級（±2%）。

### M1 實作落點（2026-11-24）

- 程式：`src/theama/inference/`（config 模型註冊表、t2v/i2v 推理、mux、pipeline CLI）。
- Eval set：`eval/t2v_v1.jsonl`（50 prompts，含 8 條繁中場景）；參考分數 `eval/reference.json`。
- VBench 對標基準：AMD Quark 研究 BF16 reference **0.8681**（832×480、81f@16fps、30 steps、CFG 5.0）；AWS Neuron 卡 6 維平均 67.63%；MLPerf v6.0 reference 70.48。數值隨評測配置浮動，對齊時須複製同一 eval config。
- GPU 驗收腳本：`scripts/gen_eval_set.py`（可續跑，50 支 MP4 全數產出+probe 驗證）、`scripts/run_vbench.py`（6 維子集 ±0.02 gate）。

## P1a — 偏好後訓練（VideoReward + LoRA-DPO，2026-11-24 落地）

品質路徑 P1 的第一刀（見 [`quality-path-wan3-h3.md`](quality-path-wan3-h3.md)）：差距在後訓練不在骨幹，先用 reward-model DPO 把 Wan 2.2 的偏好對齊補上。

- Reward model：**VideoReward（KlingTeam，Apache-2.0）**，Qwen2-VL-2B 多維（視覺品質/時序一致/動態程度/文字對齊）；註冊表 `src/theama/reward/config.py` 強制 Apache/MIT 白名單。
- 流程（全部腳本化、可續跑）：
  1. `scripts/gen_candidates.py` — 每 prompt 4 seeds 產候選（TI2V-5B dev 路徑）
  2. `scripts/score_eval_set.py` — VideoReward 打分 → scores.jsonl
  3. `scripts/make_pref_pairs.py` — best-vs-worst、margin ≥ 0.5 過濾（防 noise-driven drift）→ pairs.jsonl
  4. `scripts/encode_pairs.py` — Wan VAE 預編碼 latents（訓練熱迴圈不背 VAE）
  5. `scripts/train_dpo.py` — Diffusion-DPO flow-matching 版，LoRA rank 32，β=500，凍結 reference
- 配置：`configs/dpo_ti2v_5b.yaml`；數學（flow_match_pair / dpo_loss）CPU 單測覆蓋。
- **驗收**：(1) held-out prompt 盲測 vs 未後訓練 Wan 2.2 win ≥ 65%；(2) VBench 6 維子集對 DPO 前 baseline 不掉超過 ±0.01（防 reward hacking / motion collapse）；(3) dpo_metrics.json 500 steps 收斂、loss 持續下降。
- 5B 全綠後同配方推 A14B（80GB 卡）。

## M2 — Audio DiT + 聯合微調

- DAC 44.1kHz stereo latent 接入；`CrossModalBridge` 每 3 層一組。
- LoRA 先只訓 bridge 層 → 對齊後 full FT。
- 指標：AV-sync（SyncNet 類偏移量）、音質（PESQ/STOI 對參考集）。
- **驗收**：5s 片段音畫同步偏移 < 40ms；人聽盲測 ≥ 7/10 認為「聲音屬於畫面」。

### M2 實作落點（2026-11-24，CPU 核心）

- `src/theama/model/audio_dit.py`：audio self-attention/adaLN blocks + 雙向 `CrossModalBridge`；音訊與影片 loss 都能反傳到 v2a/a2v。
- `src/theama/train/train_joint_av.py`：可實際 forward/backward/optimizer step 的 `JointAVTrainer`。Wan 永遠凍結；`bridge` 階段只更新 bridges，`full` 階段更新整條 audio stream。兩個 stream 各自抽 logit-normal timestep，採 Wan 一致的 `x_t=(1-t)x+t·noise`、`v=noise-x`。
- `src/theama/model/audio_codec.py`：DAC 44.1kHz MIT 規格（hop 512、1024 維 continuous latent、86.13Hz）、mid/side stereo、deterministic CPU mock；DiT 直接吃 2048 維 mid+side tokens。`src/theama/model/dac_codec.py` 隔離官方 `dac.DAC.load` 的 frozen lazy `DACAudioCodec`（上游 `(B,D,T)` ↔ Theama `(B,T,D)`），核心 contract 不需安裝第三方 DAC 套件。
- `src/theama/model/wan_adapter.py`：官方 Diffusers config 對齊（A14B 16 通道、TI2V-5B 48 通道、text 4096）；5D latent ↔ bridge tokens、0–1 flow t → Wan 0–1000 integer timestep、lazy weight loading。
- Video adapter contract：`predict_velocity(video, t, video_cond)`；`video_cond` 是獨立 `(B,L,4096)` 序列文字 embedding，不再拿 pooled audio cond 冒充。外部 latent-grid bridge 先更新 tokens、還原 5D 後送 Wan forward，雙 loss 都接到 cross-modal bridge。
- `src/theama/model/lora.py`：bridge warmup LoRA 注入——只包每個 bridge 的 `v2a`/`a2v` out_proj（`nn.MultiheadAttention` 直接讀 `.weight`/`bias`，`LoRALinear` 暴露 effective-weight 屬性讓梯度照走 A/B），B 零初始化（注入即恆等），alpha/rank 縮放；`merge_lora` 折回 plain Linear，切 `full` phase 時自動合併。trainer 接 `bridge_lora_rank`/`bridge_lora_alpha`（yaml 預設 8/16；0 = dense warmup）。
- 配置：`configs/m2_joint_av.yaml`；46 個 M2 CPU tests 覆蓋 AudioDiT/trainer、DAC contract/wrapper、Wan 規格/布局/forward、5D tiny A/V smoke 與實權重 smoke orchestration，另有 13 個 LoRA tests。
- 實權重入口：`scripts/smoke_m2_real_weights.py dac|wan`，核心在 `src/theama/train/real_weight_smoke.py`。DAC 輸出 JSON＋input/reconstructed WAV；TI2V-5B 跑單步 5D A/V forward/backward/optimizer step，驗證 Wan 凍結、bridge 梯度與參數更新，並記錄 peak CUDA memory。Wan smoke 使用 random latents/text embeddings，只驗證真實權重 contract，不冒充 prompt-to-video 或品質驗收。
- **尚未完成**：真實 DAC/Wan 權重 GPU smoke **執行**、Wan 內部 block 層級 bridge、實際 AV 資料管線與 GPU 訓練；上述 40ms/盲測閘仍未驗收。目前 `latent_pre_backbone` 只是外部 bridge 驗證路徑，不冒充內層整合。

## M3 — THEAMA-Extend（30s）

- SVI 式接續，anchor_frames=8，audio crossfade 250ms。
- **驗收**：30s 成片；段接縫處 identity drift 指標不超過單段內自然變化的 1.5×；無可見跳切（人工檢查 eval set 全數）。

## M4 — THEAMA-4K（720p→2K→4K）

- Stage 1 in-context regenerate（denoise 0.35）；Stage 2 tiled upscale（denoise 0.25）。
- **HDR + AV1 提前至 M4 交付（2026-11-26 拍板，自 M4b 前置移入）**：10-bit HDR10/HLG 封裝＋H.264/HEVC/AV1 三編碼隨 4K 上線，列為獨立賣點——「第一個輸出真 HDR（HDR10/HLG）＋AV1 的開源影片模型」。**修正（2026-11-26）**：Seedance 2.5 已於 2026-11-13 上線原生 4K/30s＋10-bit BT.709（SDR，非 HDR10/HLG；閉源 API 計費，限時 ¥0.15/秒）——4K 與 10-bit 不再是孤峰，Theama 差異化收斂為：真 HDR、AV1/HEVC＋HLS 分發包、開放權重本地部署、8K（M4b）。
- **HLS 分發層（2026-11-26 拍板）**：成片除 MP4 母檔外，輸出 HLS 播放包——fMP4/CMAF segments（AV1/HEVC 在 HLS 不走 MPEG-TS）＋ master .m3u8 自適應階梯（4K HDR AV1 → 1080p HEVC → 720p/480p H.264；HEVC 層為 Safari fallback，非加分項是必需）；同批 segment 可兼出 DASH playlist。
- **驗收**：(1) 4K 輸出對 2K 輸入的 LPIPS 不劣化 > 10%；(2) tile 接縫不可見；(3) 單片 30s 4K 在 80GB 卡可完成；(4) HDR10/HLG 10-bit + HEVC/AV1 mux ＋ ffprobe 驗證通過；(5) HLS master playlist ＋各 variant 全數產出、ffprobe 驗證通過。

## M4b — THEAMA-8K（4K→8K tiled regenerate，2026-11-24 新增）

光環型差異化（Wan 3.0 鎖 1080P、H3 鎖 2K、Seedance 2.5 上限 4K，8K 無人提供）；**加分題不是主線**，M4 4K 穩了才動。

- 方法：M4 Stage 2 直接延伸——4K base → 8K tiled in-context regenerate（~1024² tiles、overlap 128、denoise 0.15–0.25，比 4K 段更保守，只補高頻紋理）。
- 工程前提：tiled VAE decode（8K 整張不進 VAE）；PNG 中介檔 streaming mux（8K 序列 ~21GB/30s 支，不全落盤）；HEVC/AV1 + HDR10 封裝（`mux_mp4(vcodec="hevc"|"av1", hdr="hdr10")` 已就位，hvc1 tag / SVT-AV1；**HDR/AV1 封裝已提前至 M4 交付**，M4b 僅剩 8K 本體）。
- **驗收**：(1) 8K 輸出對 4K 輸入 LPIPS 不劣化 > 10%；(2) tile 接縫人工檢不可見（含文字/人臉特寫偽影檢查）；(3) 單片 30s 8K 在 80GB 卡完成（估數小時/支）；(4) HEVC/AV1 + HDR10 mux + ffprobe 驗證通過。

## M5 — 品質收斂對標（bar = Wan 3.0 / H3，2026-11-24 細化）

- 對手：MiniMax H3 API（2K/15s/音訊）、Wan 3.0 API（30s/1080P/音訊，AA T2V #1）固定 eval set 盲測。
- 前置：P1 資料引擎 + DPO/RLHF 偏好後訓練、P2 omni-reference — 見 [`quality-path-wan3-h3.md`](quality-path-wan3-h3.md)。
- **驗收**：(1) 50-prompt 盲測 win/tie ≥ 50% vs H3（768p 對 768p）；(2) win/tie ≥ 40% vs Wan 3.0（720p 對 720p）；(3) VBench 總分進開源前三（當時榜單）；(4) 4K/30s 交付（對 Wan 3.0 仍是贏點；Seedance 2.5 自 2026-11-13 已有原生 4K/30s——差異化主軸改掛真 HDR＋AV1＋開放權重）。

## 風險與對策

| 風險 | 對策 |
|---|---|
| Wan 2.2 無音訊，audio DiT 自行整合難度高 | M2 先用 Kandinsky 6.0（MIT）對照實作降低試錯成本 |
| 30s 接續 drift 失控 | 退守 15s 單次 + 智慧剪輯接口；anchor 幀數與 CFG 排入超參掃描 |
| 4K 訓練資料不足 | Stage 2 可用純影像 4K 資料預訓，再用影片 fine-tune |
| 授權誤用（H3/LTX 輸出混入資料集） | data card 強制記錄來源；CI 加資料來源檢查腳本 |
| 算力不足 | 每里程碑有 CPU/單卡可驗收的最小子集；M3 起才需多卡 |
| AV1 in HLS 的 Safari 相容缺口 | HLS 階梯內 HEVC 層為 Apple 生態 fallback，必做 |

## M6 — Avatar 互動數字人線（2026-11-24 新增）

### M6a — 離線 talking avatar

- Wan-Animate（Apache-2.0）重現：單圖 + 音訊 → talking avatar clip。
- **驗收**：唇形同步指標（LSE-C/LSE-D 類）達 HunyuanVideo-Avatar 公開報告同量級；eval set 20 組全數產出。

### M6b — Streaming 實時化

- Causal/block-causal attention 改造 + rolling KV cache + replay anchor。
- 分發註記（2026-11-26 拍板）：實時互動**不走標準 HLS**（segment 延遲太高），採 LL-HLS（partial segment）或 WebRTC；標準 HLS 是 Film 成片的分發格式，兩線不混。
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

## M7 — 文生圖線（2026-11-24 新增）

### M7a — 推理重現 + CJK 文字評測

- Qwen-Image-2512（品質）+ Z-Image Turbo（快速）雙路徑推理跑通。
- 自建 CJK 文字渲染評測集（長文、海報、招牌場景），**簡體中文（zh-Hans）優先於繁體（zh-Hant）**——2026-11-26 拍板：主產品流程為 T2I → I2AV，首幀文字以簡中市場為主戰場。seed v1.2 的 100 條須補成簡繁對照結構（同語意簡／繁 prompt 成對），後續新增 prompt 一律先補簡中。
- 簡／繁分開統計：exact hit、char recall、缺字、錯字、簡繁轉換錯（該簡用繁、該繁用簡）、字體替換、標點錯誤（全形／半形與兩岸標點習慣）。
- **驗收**：**zh-Hans 文字準確率達 Qwen-Image 官方報告同量級（放行閘）**；zh-Hant 同時統計、僅作參考不擋放行；兩路徑 eval set 全數產出落盤。

### M7b — 編輯管線

- Qwen-Image-Edit-2511 多參考編輯接入。
- **驗收**：編輯 eval（局部替換/風格轉換/文字修改各 20 組）通過率 ≥ 90%（人工檢）。

### M7c — 三線整合

- KeyframeBridge：T2I 產首末幀 → Film 線 I2V；identity 參考圖 → Avatar IdentityEncoder。
- **驗收**：端到端「prompt → 關鍵幀 → 5s 720p clip」與「prompt → 參考圖 → avatar clip」各跑通 10 組。

### 風險

| 風險 | 對策 |
|---|---|
| 2512 品質離 2.1 有距離 | LoRA 收斂；2.1 權重/輸出授權不乾淨，不碰 |
| 20B 權重 40GB BF16 部署重 | FP8（~20GB）為預設；dev 走 Z-Image Turbo |
| 簡中評測集覆蓋不足 | 評測集版本化（eval/cjk-v1.jsonl），簡中優先持續擴充 |

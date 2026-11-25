# Theama 模型路線圖（Model Roadmap）

模型交付視角的路線圖；里程碑驗收細節見
[`training-plan.md`](training-plan.md)，品質對標邏輯見
[`quality-path-wan3-h3.md`](quality-path-wan3-h3.md)。

## 現況（2026-11-25）

| 階段 | 模型交付物 | 狀態 |
|---|---|---|
| M1 | Wan 2.2 推理重現（TI2V-5B / T2V-A14B / I2V-A14B）+ eval harness + MP4 mux | 程式落地，GPU 驗收待跑 |
| P1a | **theama-ti2v-5b-dpo**（VideoReward 偏好對 + LoRA-DPO） | 程式落地，GPU 驗收待跑 |
| M2 | **theama-audio-dit v1**（並聯音訊 DiT + 雙向 CrossModalBridge） | CPU trainer + DAC contract + 外部 Wan adapter 落地，真實權重 GPU smoke 待跑 |
| M4b（前置） | HEVC / AV1 + HDR10 / HLG 10-bit 封裝 | mux 落地；8K regenerate 本體待 M4 穩定 |
| M7a（前置） | CJK T2I eval seed v1（24 條繁中文字渲染 prompt + expected_text） | 已落地；Qwen-Image-2512 / Z-Image Turbo 推理重現待跑 |

## 近期（下一季）

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| P1a→A14B | **theama-t2v-a14b-dpo** | 5B 全綠後同配方推 A14B（80GB 卡） |
| M2 | **theama-audio-dit v1** | 真實 DAC/Wan 權重 smoke → Wan 內層 bridge、AV 資料管線與 GPU 訓練（bridge warmup LoRA 已落地）；驗收：AV-sync < 40ms |
| M3 | **theama-extend-30s** | SVI 式接續，anchor_frames=8；驗收：identity drift ≤ 單段 1.5× |
| M4 | **theama-4k-regen** | 720p→2K→4K 兩段 in-context 重建；驗收：LPIPS 不劣化 >10% |

## 中期

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| M4b | **theama-8k-regen** | 4K→8K tiled regenerate + HEVC/AV1 + HDR10（光環級，M4 穩定後才動） |
| M5 | 品質收斂版 Film 模型 | 盲測 vs H3 win/tie ≥50%、vs Wan 3.0 ≥40%；VBench 開源前三 |
| M6a | **theama-avatar-offline** | Wan-Animate-2-14B（animation-only）底座 + LatentSync 墊底唇形 |
| M7a | **theama-image 雙路徑** | Qwen-Image-2512（品質）+ Z-Image Turbo（快速）推理重現 + CJK 評測（seed v1 已落地，擴充到 100 prompts 後作為驗收閘） |

## 遠期

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| M6b | **theama-avatar-stream** | causal DiT + rolling KV + replay anchor；24GB 卡 720p 實時係數 ≥1.0 |
| M6c | **theama-avatar-interactive** | VLM agent 互動層：串流中換參考圖、語音指令（Vidu S2 class） |
| M7b/c | 編輯管線 + **KeyframeBridge** | Qwen-Image-Edit-2511；T2I→Film I2V、identity→Avatar 三線整合 |
| P2 | **omni-reference encoder** | 多圖/影片/音訊參考（對標 Wan 3.0 omni-reference；H3 Contextual Omni Rep clean-room） |

## Watchlist（上游釋出追蹤）

最近核查：**2026-11-25**（當日第五輪）。

- **Wan-Animate-2-14B**（Apache-2.0）— M6a 底座候選，已寫入
  `configs/theama_base.yaml`。repo id 已核對：`Wan-AI/Wan2.2-Animate-2-14B`
  （2026-08-07 發布，arXiv 2608.06009），另有
  `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` 與
  `Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`（base + distilled 權重）。
  官方 quick start 只列 character animation；replacement 仍是第一代
  `Wan-AI/Wan2.2-Animate-14B` 的能力，M6a 不預設使用。
- **Wan 3.x 開權重** — 仍 API-only（2026-11-25 再次確認；2026-11-05 多方核查：Wan-AI HF 27 個 repo、
  Wan-Video GitHub 6 個 repo 均無 2.5/2.6/2.7/3.0 權重；官方
  `AlibabaCloud-Official/Wan3.0` 僅 README + license 檔）。骨幹續用 Wan 2.2。
  旁註：新開放控制模型 `Wan-Dancer-14B`（Apache-2.0，2026-07）可留意。
  2026-11-25 第四輪補記：**阿里「下一代影片生成模型」**於 2026 雲棲大會
  （2026-09-22）預告 11 月發布——方向為更長、更可控、更完整、更智慧
  （長時一致性、精準掌控人物／場景／鏡頭、導演級敘事理解）；名稱未公布
  （Wan 3.5／Wan 4 皆外界猜測），是否開權重未知。截至本輪檢索仍未見發布
  （2026-11-09 第三方核查 Model Studio 新模型列表與 Wan-AI HF 均無新模型）。
  若最終開權重且授權相容，列為骨幹再評估觸發點；續盯 Model Studio 新模型
  列表、Wan-AI HF、Wan-Video GitHub。
  2026-11-25 第五輪覆核：11 月窗口進入最後一週，中英文檢索仍無發布
  公告或權重，維持「未發布、續盯」；Wan 3.x API-only 多方來源
  （2026-11 上旬至中旬的獨立核查）一致，無變化。
- **Qwen-Image 新版** — **Qwen-Image-2.1（7B，2026-09-20）統一 T2I+編輯+原生 RGBA，
  但為 Qwen Research License（非商用）→ 紅線確認，不碰**；2.1-Turbo（2026-10-09，8-step）
  同授權。Qwen-Image-3.0 為 hosted-only。Apache-2.0 線最新仍是
  Qwen-Image-2512 與 Qwen-Image-Edit-2511 —— M7a 選型不變。
  2026-11-25 第四輪補記：**Qwen-Image-3.1** 於雲棲大會（2026-09-22）亮相，
  面向電商／創意／設計商用場景、支援圖像精準編輯，官方稱性能逼近 GPT-Image
  系列；迄今未公布權重、授權與定價，型態與 3.0 同為 hosted-only，未證實
  Apache 前不碰。另 Qwen-Image-2.1 Pro／Turbo 已於 2026-10-09 上 Model Studio
  託管 API（¥0.25／¥0.10 每張）——屬託管商用計費，不改變權重的研究授權紅線。
- **開源 video reward 新模型** — awesome-rm-for-video-generation（2026-10 更新）新增
  AesRM（專家標註美學 RM，base + CoT 變體）、UnifiedReward-Flex、GT-SVJ（self-supervised
  judge）等；授權與權重可得性未逐一核實，VideoReward 仍為首選，暫不加第二把尺。
  2026-11-25 補查同清單，新增觀察 **joint audio-video reward** 線：**VA-Judger**
  （VAPref-10K 影音偏好對訓練的 A/V 聯合 judge，直接獎勵跨模態一致性而非手工拼接
  分維指標）與 **AVBench**（人本影音評測器，二元判決信心可轉連續分，可作 RLHF
  reward 與資料過濾）——與 M2 原生音訊路線後續的 A/V 偏好後訓練直接相關；
  授權／權重可得性同樣未核實，列為 P1a 之後的觀察候選，現階段不改動 P1a 配方。
  同輪覆核再補三筆 A/V reward 觀察（同清單 2026-10-06 版）：**AV-GRPO**（2026，
  modality-anchored 解耦擴散 RL——rollout 錨定單模態、凍結另一塔，異構 reward
  分模態歸因，形式直接對應 M2 雙流架構的 A/V 偏好後訓練）、**Joint A/V
  Inference-Time Scaling**（2026，多 verifier Best-of-N + 演化搜尋，Adaptive
  Reward Weighting 融合 VideoReward／JavisScore／ImageBind／AVHScore——可作
  M2/M5 推理期增益與評測交叉驗證）、**Hallo-Live**（2026，fudan-generative-vision
  有開源 repo，HP-DMD reward 重加權蒸餾把串流 A/V avatar 拉向人本偏好——M6b
  avatar-stream 線的 reward 參考）。三者授權／權重可得性均未核實，維持觀察，
  不改動 P1a／M2 配方。
  2026-11-25 第三輪覆核：補 **VA-Judger** hard id——arXiv 2608.18607（2026-08）；
  VAPref-10K 約 9K prompts／10.3K 精細偏好對，五維 rubric（prompt alignment／
  video-audio consistency／audio quality／video quality／completeness & coherence），
  附 VA-Judger-Bench（含 closed-source 模型輸出的 out-of-domain split）；論文以
  LoRA 後訓練驗證其 reward 可提升生成品質，惟 policy 實驗用 LTX-2（紅線模型）——
  僅取 reward 方法論參考，不碰其權重與輸出。同清單再補 **Omni-Judge**（2026，
  以 omni-modal LLM 做 text-conditioned A/V 生成的 CoT judge，覆蓋感知品質與
  跨模態對齊維度），同列觀察，授權／權重可得性未核實。另記上游營運訊號：
  Wan2.2-Animate 官方 HF Space 已 401 下線（2026-11-16 確認；ModelScope Studio
  與社群 ZeroGPU Space 仍在）——M6a 走自架權重路線，不受影響。
  Wan-Animate-2 repo id／Wan 3.x API-only／Qwen-Image 紅線三項本輪覆核無變化。
  2026-11-25 第五輪：新增 **WeMM-Video**（Tencent WeChat Vision，
  arXiv 2611.26409；2026-11-19 技術報告、11-20 權重上架 HF）——2B／4B／9B
  dense + 35B-A3B MoE 的 omni-modal 影音**理解**模型（影格＋原始音軌聯合
  輸入、文字輸出，Instruct／Thinking 雙模式，約 1.2T 多模態 token 預訓練，
  官方稱已部署於微信核心影片服務）。GitHub 程式碼 Apache-2.0，官方說法為
  全系列 checkpoint 適用 Apache-2.0（含第三方元件保留原條款之 carve-out；
  逐卡採用前以各 HF model card 為準）。對本專案的意義：M2 原生音訊路線
  後續的**聯合 A/V judge 底座候選**——相對 Qwen2-VL-2B 系的 VideoReward
  多了原生音軌通道，可做 A/V 一致性評分；亦可用於 P1b 資料引擎的
  A/V 一致性過濾與 dense caption。保留事項：需 `trust_remote_code` 與
  patched vLLM／SGLang（source overlay，上游未合併）、官方 recipe 限 8K
  context／16 frames、量化變體（AWQ／GGUF／NVFP4）自標 unvalidated、
  榜單數字全部官方自報（Video-MME v2 w/ audio 54.9 vs Qwen3.5-Omni-Flash
  36.6，尚無第三方複測；Music／JointAVBench 等子項仍落後）——列觀察，
  不改 P1a／M2 配方。同輪補 **Kandinsky 6.0**（MIT，已列 M2 對照）RL
  配方細節作方法參考：OmniNFT 八 reward 三分支——video（HPSv3／
  VideoAlign）、audio（AudioBox Aesthetics／CLAP／Whisper-ASR）、
  sync（AV-desync／AV-align／LatentSync-gated）；sync 分支的獎項設計直接
  對應 M2「AV-sync < 40ms」驗收維度，可作 M2 後訓練 reward 組合的
  clean-room 參考（OmniNFT 方法源自 LTX-2 社群，僅取方法論、不碰其
  權重與程式碼）。Wan-Animate-2 repo id／Wan 3.x API-only／Qwen-Image
  紅線，本輪覆核無變化。

## 發布策略

- 自研微調權重（DPO LoRA、audio DiT、extend/4K 模組）以 **Apache-2.0** 發布於
  Hugging Face `chienhsinlin` namespace，模型卡遵循 HF model card 規範（訓練資料來源、
  評測分數、限制與合規聲明）。
- 第三方權重（Wan 2.2、Qwen-Image 等）不轉售不重新打包，使用者自行下載
  （`scripts/download_weights.sh`），權重授權各自保留。

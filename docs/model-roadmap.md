# Theama 模型路線圖（Model Roadmap）

模型交付視角的路線圖；里程碑驗收細節見
[`training-plan.md`](training-plan.md)，品質對標邏輯見
[`quality-path-wan3-h3.md`](quality-path-wan3-h3.md)。

## 現況（2026-11-26）

| 階段 | 模型交付物 | 狀態 |
|---|---|---|
| M1 | Wan 2.2 推理重現（TI2V-5B / T2V-A14B / I2V-A14B）+ eval harness + MP4 mux | 程式落地，GPU 驗收待跑 |
| P1a | **theama-ti2v-5b-dpo**（VideoReward 偏好對 + LoRA-DPO） | 程式落地＋全鏈預檢閘（scores→pairs→pair-latents 三段 fail-fast）＋三 cache retry budget＋pairs_stats 剖析器；GPU 驗收待跑 |
| P1b | AV 資料引擎（manifest builder → text/AV latent cache → joint loader → joint training runner） | 程式落地＋三 cache retry budget 對齊＋joint loader skipped 列明確報缺＋joint-cache 預檢閘 check_joint_latents（runbook §5.1 獨立 CPU 閘）；GPU 驗收待跑（issue #4） |
| M2 | **theama-audio-dit v1**（並聯音訊 DiT + 雙向 CrossModalBridge） | CPU trainer + DAC contract + 外部 Wan adapter + CFG 文字條件 dropout 落地，真實權重 GPU smoke 已通過（2026-11-24，RTX 4090，見 docs/status.md） |
| M4b（前置） | HEVC / AV1 + HDR10 / HLG 10-bit 封裝 | mux 落地＋**HLS fMP4/CMAF packager＋DASH manifest 兼出皆落地（11-26，零重編碼、逐 rung ffprobe 閘）**；**2026-11-26 拍板提前至 M4 交付（獨立賣點）**；8K regenerate 本體待 M4 穩定 |
| M7a（前置） | CJK T2I eval seed v1.2（100 條文字渲染 prompt + expected_text，v1.1 48 條再擴一倍；**簡中 zh-Hans 優先**，簡繁對照成對、分開統計） | 已落地、達驗收閘規模；scorer 簡繁分軌（zh-Hans 閘／zh-Hant 並行統計）＋Ming-Image-0.1-Design 第三路徑 diagnostic-only 落地；Qwen-Image-2512 / Z-Image Turbo 推理重現待跑 |

## 近期（下一季）

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| P1a→A14B | **theama-t2v-a14b-dpo** | 5B 全綠後同配方推 A14B（80GB 卡） |
| M2 | **theama-audio-dit v1** | 真實 DAC/Wan 權重 smoke → Wan 內層 bridge、AV 資料管線與 GPU 訓練（bridge warmup LoRA 已落地）；驗收：AV-sync < 40ms |
| M3 | **theama-extend-30s** | SVI 式接續，anchor_frames=8；驗收：identity drift ≤ 單段 1.5× |
| M4 | **theama-4k-regen** | 720p→2K→4K 兩段 in-context 重建＋**HDR10/HLG 10-bit＋HEVC/AV1 輸出（11-26 提前自 M4b）＋HLS fMP4/CMAF 分發包**；驗收：LPIPS 不劣化 >10% |

## 中期

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| M4b | **theama-8k-regen** | 4K→8K tiled regenerate（光環級，M4 穩定後才動；HEVC/AV1＋HDR10 已提前至 M4） |
| M5 | 品質收斂版 Film 模型 | 保底：盲測 vs H3 win/tie ≥50%、vs Wan 3.0 ≥40%、VBench 開源前三；**終局（11-25 拍板）：AA with-audio 三板登頂——T2V/editing 超越 Wan 3.0、I2V 超越 Seedance 2.5** |
| M6a | **theama-avatar-offline** | Wan-Animate-2-14B（animation-only）底座 + LatentSync 墊底唇形 |
| M7a | **theama-image 雙路徑** | Qwen-Image-2512（品質）+ Z-Image Turbo（快速）推理重現 + CJK 評測（**簡中優先**：zh-Hans 準確率為放行閘、zh-Hant 並行統計；seed v1.2 已達 100 條驗收閘規模） |

## 遠期

| 階段 | 模型交付物 | 說明 |
|---|---|---|
| M4b | **theama-8k-regen** | 4K→8K tiled regenerate（光環級，M4 穩定後才動；HEVC/AV1＋HDR10 已提前至 M4） |
| M5 | 品質收斂版 Film 模型 | 保底：盲測 vs H3 win/tie ≥50%、vs Wan 3.0 ≥40%、VBench 開源前三；**終局（11-25 拍板）：AA with-audio 三板登頂——T2V/editing 超越 Wan 3.0、I2V 超越 Seedance 2.5** |
| M6a | **theama-avatar-offline** | Wan-Animate-2-14B（animation-only）底座 + LatentSync 墊底唇形 |
| M7a | **theama-image 雙路徑** | Qwen-Image-2512（品質）+ Z-Image Turbo（快速）推理重現 + CJK 評測（**簡中優先**：zh-Hans 準確率為放行閘、zh-Hant 並行統計；seed v1.2 已達 100 條驗收閘規模） |

## Watchlist（上游釋出追蹤）

最近核查：**2026-11-26**（第十八輪）。

- **Wan-Animate-2-14B**（Apache-2.0）— M6a 底座候選，已寫入
  `configs/theama_base.yaml`。repo id 已核對：`Wan-AI/Wan2.2-Animate-2-14B`
  （2026-08-07 發布，arXiv 2608.06009），另有
  `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` 與
  `Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers`（base + distilled 權重）。
  官方 quick start 只列 character animation；replacement 仍是第一代
  `Wan-AI/Wan2.2-Animate-14B` 的能力，M6a 不預設使用。
  2026-11-26 第十六輪補：GitHub repo id 補登
  `Wan-Video/Wan-Animate-2`（官方 repo，392★）——與 HF 三 id
  併列為 M6a 底座完整索引；base＋distilled 權重／Apache-2.0
  本輪再核無變化。
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
  2026-11-25 第六輪覆核：11 月窗口剩最後五天，中英文檢索仍僅見
  2026-09-22 雲棲預告原文的轉載，無發布公告或權重；Wan 3.x API-only
  維持（2026-11-10／11-12 兩家第三方逐 repo 核查一致：Wan-AI HF 27 個
  repo、Wan-Video GitHub 6 個 repo 均無 3.x 權重，最新仍為 Wan 2.2 與
  Wan-Dancer）。Wan-Animate-2 repo id 本輪覆核無變化。
  2026-11-25 第七輪覆核：11 月窗口剩最後五天，中英文檢索仍無發布
  公告或權重；Wan 3.x API-only 多方來源一致（2026-10 至 11 月中旬的
  逐 repo 獨立核查均確認 Wan-AI HF 無 2.5 以上權重）。Wan-Animate-2
  repo id 本輪覆核無變化。
  2026-11-25 第八輪覆核：11 月窗口進入倒數，中英文檢索仍無發布公告
  或權重；Wan 3.x API-only 維持（2026-11-05 逐 repo 核查：Wan-AI HF
  27 個 repo、Wan-Video GitHub 6 個 repo 均無 2.5 以上權重，最新開放
  權重仍為 Wan2.2-Animate-2 系）。Wan-Animate-2 repo id 本輪覆核無變化。
  2026-11-25 第十輪覆核：11 月窗口剩五天（11-30 關閉），Model Studio
  新模型列表（qwen3.8-flash／wan3.0-video-prime／wan2.6・2.7 flash
  變體／qwen-image-edit-max／kimi-k3）與第三方 11-09／11-11 逐項核查
  仍無下一代影片模型；Wan 3.x API-only 維持，最後開放權重仍 Wan 2.2。
  另駁一筆低權威來源誤報：youart.ai（2026-08-06）稱「Wan 2.7 已開權重、
  權重掛於 Tongyi Wanxiang HF」——與 usefastlane（11-01）、renoise
  （11-11）逐 repo 核查（2.5／2.6／2.7／3.0 皆未開權重）矛盾，歸類
  同 miraflow「2.7 開源 Apache-2.0」誤報，不採信。Wan-Animate-2
  repo id 本輪覆核無變化。
  2026-11-26 第十五輪覆核：11 月窗口 T-4（11-30 關閉），中英文
  檢索仍無阿里下一代影片模型發布公告或權重；Wan 3.x API-only
  維持（atlascloud 2026-08-26 逐 33 repo 核查、evolink 2026-08
  供應路由核查、11-15 第三方榜單快照三方一致——Wan 3.0 無 HF
  checkpoint／GitHub repo／license 文本，最新開放權重仍 Wan 2.2
  系）。Wan-Animate-2 repo id 覆核無變化（base＋distilled 雙
  checkpoint 均在 Wan-AI HF，Apache-2.0）。
  2026-11-26 第十六輪覆核：11 月窗口 T-4，中英文檢索
  仍僅 2026-09-22 雲棲預告（多源一致：下一代影片模型「更長、
  更可控、更完整、更智能」，11 月發布），無發布公告或權重；
  Wan 3.x 日期補精確（2026-08-06 public beta／08-24 正式
  上線，API-only、無 license 文本）＋四代旗艦連閉摘要
  （atlascloud 2026-08-26 逐 33 repo 核查佐證）。骨幹續用
  Wan 2.2。
  2026-11-26 第十七輪覆核（11 月窗口關閉前最後覆核，T-4／11-30 關閉）：
  阿里下一代影片模型**仍未發布**——中英文檢索僅見 2026-09-22 雲棲預告
  及其轉載（搜狐時間線 B 級同文：「更長、更可控、更完整、更智能」、
  11 月發布）；阿里 11 月中旬實際發布為 Qwen4-Max（文本旗艦，非影片
  模型）與雙 11 檔期的 Wan3.0-Video 七折促銷（developer.aliyun.com
  2026-11-19，A 級——仍以 Wan3.0-Video 為影片旗艦介紹），均非下一代
  影片模型。窗口 11-30 關閉後若仍無發布，本項降為每數輪抽查。
  Wan 3.x API-only 維持（熙娛 2026-11-12 前台＋官方列表雙核、
  kenerateai 2026-11-12 逐 repo 核查一致：Wan-AI HF／Wan-Video
  GitHub 無 2.5 以上權重，`AlibabaCloud-Official/Wan3.0` 仍僅
  README＋LICENSE——cinemadrop 2026-11-01 同確認）。Wan-Animate-2
  repo id 覆核無變化。骨幹續用 Wan 2.2。
  2026-11-26 第十八輪（11-30 窗口關閉後首抽，降為每數輪抽查）：
  Wan 3.x 仍 API-only 無權重（kenerateai 2026-11-12 逐 repo 核
  查——Wan-AI HF 27 repos 全為 2.1／2.2 線＋Wan-Dancer-14B，
  Wan-Video GitHub 六 repos 無 3.0；atlascloud 08-26／8frame
  10-02 多源一致），下一代影片模型無發布跡象（11 月阿里實際
  發布仍僅文本線 Qwen3.8-Max／Qwen4-Max preview）。Wan-Animate-2
  repo id 覆核無變化（Wan-Video org 六 repos 名單同前）。
  骨幹續用 Wan 2.2。
- **Qwen-Image 新版** — **Qwen-Image-2.1（7B，2026-09-20）統一 T2I+編輯+原生 RGBA，
  但為 Qwen Research License（非商用）→ 紅線確認，不碰**；2.1-Turbo（2026-10-09，8-step）
  同授權。Qwen-Image-3.0 為 hosted-only。Apache-2.0 線最新仍是
  Qwen-Image-2512 與 Qwen-Image-Edit-2511 —— M7a 選型不變。
  2026-11-25 第四輪補記：**Qwen-Image-3.1** 於雲棲大會（2026-09-22）亮相，
  面向電商／創意／設計商用場景、支援圖像精準編輯，官方稱性能逼近 GPT-Image
  系列；迄今未公布權重、授權與定價，型態與 3.0 同為 hosted-only，未證實
  Apache 前不碰。另 Qwen-Image-2.1 Pro／Turbo 已於 2026-10-09 上 Model Studio
  託管 API（¥0.25／¥0.10 每張）——屬託管商用計費，不改變權重的研究授權紅線。
  2026-11-25 第六輪：2.1／2.1-Turbo 的 Qwen Research License 條文未變
  （10 月多方逐條核查一致，商業授權詢問無官方回覆），Apache 線仍是
  2512／Edit-2511，M7a 選型不變。同輪新增 M7a 觀察候選：
  **Ming-Image-0.1-Design**（螞蟻 inclusionAI，6.15B，**MIT**——
  2026-09-17 上架 HF、09-22 發布，GitHub `inclusionAI/Ming-Image`）——
  專攻 UI／資訊圖／海報等**文字密集型設計**的 T2I：原生 RGBA、12 steps
  CFG 1.0、2048²，另有 Design-Layer 圖層拆解姊妹模型（同 MIT）。
  Artificial Analysis：UI/UX Design 類開權重第一（Elo 1082，全類 17/81），
  Layout 與 Text Rendering 均為開權重第一；總榜中段（約 45/160）。
  對本專案的意義：M7a 除 Qwen-Image-2512（品質）＋ Z-Image Turbo（快速）
  之外的第三條合法授權路徑候選，其強項（文字渲染）正是 cjk_t2i_v1 seed
  的量測目標。保留事項：官方驗證配置為單卡 80 GiB（文字編碼器約 34 GB，
  消費卡需靠 Comfy-Org／Kijai int8／w4a8 量化變體）；**CJK／非英文文字
  渲染未見第三方實測**（外媒同列為首要待驗證項）；榜單為單一來源。列觀察，
  待 M7a 推理重現排程時以 seed 集（v1.2 已達 100 條驗收閘規模）實測後再議是否納入。
  2026-11-25 第七輪：2.1／2.1-Turbo 研究授權、2512／Edit-2511 Apache 線、
  3.0／3.1 hosted-only，本輪覆核均無變化。
  2026-11-25 第八輪：同上四項（2.1 研究授權／2512 Apache 線／3.0・3.1
  hosted-only／Ming-Image 觀察）覆核均無變化。
  2026-11-25 第十輪：四項覆核無變化（2.1／2.1-Turbo 研究授權、
  2512／Edit-2511 Apache 線、3.0／3.1 hosted-only）；HF 社群討論 #39
  （09-23）請願 2.1 回歸 Apache 2.0，迄今無官方回應。補一筆第三方
  授權榜資料（efsora 2026-10-19）：**Ming-Image-0.1-Design 為「排名
  最高的完全寬鬆授權（fully permissive）影像模型」**，與 Qwen-Image-2.1
  差 42 分——支持其作為 M7a 第三條合法路徑的 diagnostic 定位，
  seed 實測結論前不升格。
  2026-11-26 第十五輪：補產品線譜系資料點（juejin 2026-10-08
  同題對比，B 級）——Qwen-Image-2.0（2026-02 架構換代未放權重）
  →3.0（2026-07 純閉源 API）→2.1（2026-09-20 回開源線、7B
  單流 DiT，但 Qwen Research License 非商用）；2512（2025-12-31，
  20.4B MMDiT）為 20B 分支收官版、Apache-2.0 可商用——「要商用
  只有 2512 能用」與本專案紅線劃分一致。紅線不變、M7a 選型不變。
  2026-11-26 第十六輪：2.1／2.1-Turbo 研究授權、
  2512／Edit-2511 Apache 線、3.0／3.1 hosted-only，
  覆核無變化。
  2026-11-26 第十七輪：四項覆核無變化（2.1／2.1-Turbo 研究授權紅線、
  2512／Edit-2511 Apache 線、3.0／3.1 hosted-only）；補資料點——
  Qwen-Image-3.0 Pro 已 2026-08-04 GA on Model Studio（orcarouter
  2026-11-05，B 級：仍無權重、無 model card、無技術報告、無參數量）。
  紅線不變、M7a 選型不變。
  2026-11-26 第十八輪：四項覆核無變化（2.1／2.1-Turbo 研究授權
  紅線、2512／Edit-2511 Apache 線、3.0／3.0 Pro hosted-only）；
  11 月 Qwen feed 無 image 線新版（releasebot 2026-11-18 頁：
  11 月僅 Qwen3.8-Max 文本旗艦與 Qwen4-Max preview）。紅線不變、
  M7a 選型不變。
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
  跨模態對齊維度），同列觀察，授權／權重未核實。另記上游營運訊號：
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
  2026-11-25 第六輪：新增 **Omni-RL Judge**（Adobe，Xingtai Li 等，2026-11；
  ResearchGate 著錄 2026-11-15）——以 verifiable rewards 做 RL 訓練的
  omni-modal 影音 judge，與 VA-Judger／Omni-Judge 同線（M2 後訓練 A/V 偏好
  reward 候選；「可驗證獎勵」路線若成立，比純偏好對訓練的 judge 更抗
  reward hacking）。目前僅檢得 ResearchGate 條目，arXiv hard id、程式碼、
  權重與授權均未核實，列觀察待補。Wan-Animate-2 repo id／Wan 3.x
  API-only／Qwen-Image 紅線，本輪覆核無變化。
  2026-11-25 第七輪：新增三筆 hard-id 已核實的觀察。
  **Omni-RRM**（arXiv 2602.00846，2026-01）——統一 omni-modal reward
  model（text／image／video／audio 四模態）：全自動 rubric-grounded 偏好
  合成（Omni-Preference 資料集，零人工標註）＋ SFT→GRPO 兩階段訓練，
  輸出帶分維 rationale 的結構化判決；官方自報 ShareGPT-V 80.2%、
  Audio-HH-RLHF 66.8%，並驗證 Best-of-N 推理期對齊有效（影像／影片／
  音訊均優於 self-consistency）。對本專案的意義：M2 後訓練 A/V 偏好
  reward 與 M2/M5 推理期 BoN 的統一 judge 候選——一個模型同時覆蓋
  P1a（video）與 M2（audio／sync）的 reward 需求。保留事項：程式碼／
  權重／授權僅見 anonymous 4open 連結，未核實；數字全部官方自報。
  列觀察，不改 P1a／M2 配方。**LongCat-Video-Avatar 1.6**（arXiv
  2611.18822v2，2026-11-18）——frozen 底座上的雙分支後訓練：GRPO
  LoRA（多 reward 提升表情／動作自然度）＋ DMD→OPSD on-policy
  self-distillation LoRA（長影片抗自回歸漂移），推理時兩 adapter 聯合
  啟用、底座不動。「凍結底座＋獨立 LoRA 分支再組合」的形態直接對應
  M6a／M6b 的後訓練與長時穩定需求（亦與本專案 P1a LoRA-DPO 的工程
  路線同構），列方法參考；授權未核實（1.5 自稱 open-source，1.6 待查）。
  **FlowPortrait** 補 hard id——arXiv 2603.00159（JHU，2026-02）：
  MLLM 三維 reward（lip-sync／expressiveness／motion）＋ LPIPS 感知項
  ＋ RAFT 光流 jitter 項的複合 reward（batch 內標準化後加權），附系統性
  reward-hacking ablation——單維優化各自出現特徵性崩壞、純感知項導致
  靜態塌陷（3.3 vs SFT 4.10），結論是 MLLM 高層判決必須配低層感知／
  光流約束。對 M6a 唇形＋表情 reward 組合與防 hacking 設計是直接參考；
  授權／權重未核實，觀察。Omni-RL Judge 本輪再查仍僅 ResearchGate
  條目（2026-11-15），arXiv id 未補上，維持觀察。Wan-Animate-2 repo
  id／Wan 3.x API-only／阿里下一代模型未發布／Qwen-Image 紅線，本輪
  覆核無變化。repo 健康：CI.yml 仍未上傳、issues 0；沙箱執行環境與
  專案記憶 mount 本輪仍不可用，維持 docs-only。
  2026-11-25 第八輪：**VA-Judger repo 狀態更新**——GitHub
  `ShareLab-SII/VA-Judger` 自 2026-08-20 起已釋出 project page、論文、
  基礎程式碼、**checkpoints 與 VA-Judger-Bench**（SFT 訓練程式碼與
  VAPref-10K 資料集仍列 to-do 未釋出）；reward model 本體為
  Qwen3-Omni-30B-A3B-Instruct 全參微調（visual encoder／aligner 凍結，
  8×GPU BF16 + DeepSpeed ZeRO-3），推理經 vLLM。checkpoint 授權於
  repo 檢索結果中未標示——採用前須逐卡核驗 model card；其 policy 驗證
  仍綁 LTX-2（紅線），本專案僅取 reward 方法論參考，維持觀察、不改
  P1a／M2 配方。Omni-RL Judge 仍僅 ResearchGate 條目。Wan-Animate-2
  repo id／Wan 3.x API-only／阿里下一代模型未發布／Qwen-Image 紅線，
  本輪覆核無變化。repo 健康：CI.yml 仍未上傳（無 `.github/` 目錄）、
  issues 0；沙箱執行環境與專案記憶 mount 本輪仍不可用，維持 docs-only。
  2026-11-25 第九輪覆核：新增 **GenRM-Flow**（arXiv 2611.11421，
  2026-11-08，Siming Fu 等）——偏好微調後的影片生成器本身就是
  process-aware RM：從 KL-constrained 最優 policy 的解析解分離 reward
  項、映到 flow-matching surrogate 後，經驗 reward 等於連續時間
  velocity-prediction 誤差的相對差——直接在含噪 latent 空間打分，
  無需外部 reward 網路；並證明 DPO／IPO／RWR 是通往同一最優 policy
  的等價優化路徑、各自誘導生成器的 reward readout（實驗以該 process
  reward 取代外部 VLM scorer 跑 FlowAWR，下游生成有可測提升）。
  對本專案的意義：P1a LoRA-DPO 訓完的 Wan 生成器可直接讀出
  latent-space reward（推理期 BoN、資料過濾免載第二把尺），M2 flow
  後訓練同適用；程式碼／權重／授權未核實，列觀察與方法參考，不改
  P1a／M2 配方。同輪補方法參考 **PAVRM**（arXiv 2511.21541v3，
  CVPR 2026 Findings）：video generation model 直接當 latent reward
  model（對部分去噪樣本於 latent 空間打分、免 decode，stratified
  timestep 評估），與 P1a pair latent cache 形態相容；授權未核實。
  Wan-Animate-2 repo id／Wan 3.x API-only／阿里下一代模型 11 月窗口
  倒數仍無發布／Qwen-Image 紅線，本輪覆核無變化。
  2026-11-25 第十輪：**VA-Judger checkpoint 授權二次核查失敗**——
  HF 頁面（ShareLab-SII/VA-Judger）經沙箱 fetch 僅得空 README、無
  license metadata 可讀；GitHub 下載流程需 hf auth 並接受 gated terms。
  **checkpoint 授權仍無法由沙箱核實，維持「使用者代查 model card
  license tag」待辦**；repo 自 08-20 已釋出 checkpoints＋Bench（SFT
  程式碼與 VAPref-10K 仍 to-do）。其餘觀察條目（GenRM-Flow／PAVRM／
  Omni-RRM 等）本輪無新增。
  2026-11-25 第十四輪覆核：**PAVRM 官方程式碼落地且授權過紅線**——
  騰訊混元官方 repo `Tencent-Hunyuan/HY-Video-PRFL`（GitHub API
  直查）已釋出 PAVRM＋PRFL 完整訓練／推理程式碼，`LICENSE.txt`
  全文為 Apache-2.0（前言明載「不附加額外限制」；GitHub 授權偵測器
  因前言顯示 NOASSERTION，以授權全文為準）。骨幹直接吃
  `Wan-AI/Wan2.1-I2V-14B-720P`（Apache-2.0）、不帶任何第三方受限
  權重；repo 未發布 RM checkpoint（PAVRM reward 頭由使用者自訓），
  無額外權重授權問題。PAVRM 自此由「觀察（授權未核實）」升格為
  **紅線內可直接參考實作**：latent 空間打分免 VAE decode、與 P1a
  pair latent cache 形狀相容、PRFL 較 RGB ReFL 省記憶體且 ≥1.4×
  加速——定位為 VideoReward（首選不變）之外的 process-level
  互補尺，P1a LoRA-DPO 之後的 latent-space ReFL 選項；骨幹
  Wan2.1 與本專案 Wan 2.2 同 Apache 譜系，方法可直接移植。
  固定項本輪覆核無變化：Wan 3.x 仍 API-only 無權重
  （wan30.video 10-16／atlascloud 08-26 再核一致，「Wan 2.7 開源」
  仍屬低權威誤報）；Wan-Animate-2 repo id
  `Wan-AI/Wan2.2-Animate-2-14B`（Apache-2.0）無變化；阿里下一代
  影片模型 11 月窗口 T-5（11-30 關閉）仍無名稱／規格／權重
  （阿里 11-12 云栖官方總結稿亦未提新影片模型）；Qwen-Image
  2.1／2.1-Turbo 仍 Qwen Research License（QwenLM repo 直查），
  3.1 hosted-only「今年稍後推出」，紅線不變；GenRM-Flow 仍僅
  論文（arXiv 2611.11421）、未檢得程式碼。
  2026-11-26 第十五輪：同清單（awesome-rm-for-video-generation
  2026-10-06 版，S 級）補四筆新觀察——**Reward Lightning**
  （latent-space RM 與對抗蒸餾共享骨幹，homologous preference
  distillation 同步對齊＋加速 few-step 生成器，與 P1a LoRA-DPO
  及未來 few-step 需求直接相關）、**GT-SVJ**（把影片生成器改造為
  energy-based judge，self-supervised 於合成退化影片、不依賴外部
  VLM——授權暴露面小於 VLM judge，值得追程式碼落地）、
  **Reward Forcing**（reward-guided DMD 串流蒸餾，以 VLM reward
  把 few-step student 偏向高動態樣本——M6b 串流線方法參考）、
  **RewardDance**（reward scaling 研究：judge 換更大生成式 VLM
  的規模律與抗 hacking 魯棒性）。四者授權／權重可得性均未核實，
  VideoReward 首選不變，維持觀察、不改 P1a 配方。
  2026-11-26 第十六輪：兩筆升級觀察。① **VA-Judger**
  從「授權未核實」升級為「成品已上架」——GitHub
  `ShareLab-SII/VA-Judger`（Fudan ShareLab，2026-08-20 發布
  basic code＋checkpoints＋VA-Judger-Bench；HF
  `ShareLab-SII/VA-Judger` 權重、`ShareLab-SII/VA-Judger-Bench`
  資料集）。論文 hard id 已錄（arXiv 2608.18607）；post-train
  人類偏好率 62.30% vs OmniNFT 27.63% vs 原 LTX-2 10.08%
  （11/13 客觀指標最佳）——紅線僅禁 LTX-2 權重／輸出，judge
  本身授權本輪因 HF API 不可達（HTTP 000）待核，若相容列為
  M2 後訓練 A/V 偏好 reward 首選候選；底模 Qwen3-Omni-30B-A3B
  推測沿用 Qwen 系授權，以 model card 為準。② **IsoVRBench**
  （arXiv 2610.17234，2026-10，S 級）——無人類回饋的 video-RM
  評測基準（自動構造 isomorphic 影片對），現有 RM 排名：
  VideoScore v1.1 60.1／VisionReward 72.6／VideoScore2
  49.4／GPT-4.1 76.0／Qwen3-VL-8B 72.8 vs Human 100——可作
  theama scorer（VideoReward 首選）的交叉驗證參考板，
  授權／程式碼可得性未核實，列觀察。P1a 配方不變。
  2026-11-26 第十七輪：awesome-rm 清單仍 2026-10-06 版無更新；新增
  一筆觀察 **Wan-R1**（2026，task-verifiable decomposed rewards 驅動
  Flow-GRPO——遊戲影片 agent-trajectory 打分、機器人導航的影格相似度／
  排序／端點 verifier，並附 VLM reward hacking 系統分析——可驗證獎勵
  路線對 P1a 防 hacking 設計是直接參考；授權／程式碼未核實，列觀察）。
  **VA-Judger checkpoint 授權第三次核查仍失敗**（HF API 自沙箱持續
  HTTP 000），維持「使用者代查 model card license tag」待辦。
  VideoReward 首選不變，P1a 配方不變。
  2026-11-26 第十八輪：awesome-rm 清單仍 2026-10-06 版無更新；
  新增一筆觀察 **World-R1**（Microsoft Research＋浙江大學，
  arXiv 2604.24764，marktechpost 2026-05-01——Flow-GRPO＋
  3D-aware rewards：以預訓練 3D 基礎模型衍生獎勵＋VLM critic
  post-train Wan 2.1，不動架構、推論成本不變，釋出 World-R1-Small
  （Wan2.1-T2V-1.3B）／World-R1-Large（Wan2.1-T2V-14B）雙變體；
  與 Wan-R1 互補——一個治 reasoning、一個治 3D geometric
  consistency，對 Film 線長鏡頭一致性是方法參考；授權／程式碼
  未核實，列觀察）。**VA-Judger checkpoint 授權第四次核查仍
  失敗**（HF API 自沙箱 HTTP 000），維持使用者代查待辦。
  VideoReward 首選不變，P1a 配方不變。
- **AV leaderboard（Artificial Analysis Video Arena with-audio）現況**
  （2026-11-25 第十一輪補；M5 品質目標的對標板）——**T2V w/audio**
  （11-13 快照）：Wan 3.0 第一（Elo 1175）、Utopai X（H3 post-train）
  第二（1170）、MiniMax H3（768p）第三（1161，**開放權重最高**，
  Kling 3.0 固定錨 1000）；**I2V w/audio**（11-02 快照）：Seedance 2.5
  第一（1153）、H3 第四（1128），前十有六款為 H3 或其 post-train
  （Utopai X／Hyperflow／HeyGen Video 1／P-Video-2-Pro／H3 Max）；
  **editing w/audio**（11-03 快照）：Wan 3.0 第一（1196）、
  Seedance 2.5 第二、Hyperflow H3 第三，Wan 2.7 Video 列第八
  （1074，仍 API-only 無權重）。對本專案的意義：① M5 目標
  （盲測 vs H3 win/tie ≥50%、vs Wan 3.0 ≥40%）的戰場就是 AA
  with-audio 雙板，對手現況如上；② **H3 post-train 生態（六款以上
  前十）證明「強開放底座＋後訓練可逼近 frontier Elo」**——theama
  以 Wan 2.2（Apache-2.0）＋LoRA-DPO 走同構策略，是授權紅線內
  唯一合法的對應路線；③ 開源可商用譜系中 Wan 2.2 仍為唯一頂級
  底座（H3 restricted／LTX community 皆紅線），骨幹不變；
  ④ VBench 對照：Wan2.2-T2V-A14B total 82.61（自報），
  VBench-2.0 早期快照榜首 Veo 3 66.7%——M5「VBench 開源前三」
  目標在 Apache 譜系內等價於「追平並超出 Wan 2.2 官方分」。
  ⑤ 第十二輪（2026-11-25，依使用者指示「**超過 Seedance 2.5 和
  Wan 3.0**」）：上列三榜榜首自此由「對標板」升格為 **M5 終局超越
  對象**——T2V w/audio 須越過 Wan 3.0（1175）、I2V w/audio 須越過
  Seedance 2.5（1153）、editing w/audio 須越過 Wan 3.0（1196）；
  三者皆 closed/API-only，達成即是以 **Apache-2.0 開放權重身份直接
  登頂 with-audio 三榜**（而非僅開源第一）。差距盤點（以開放權重
  現況計）：T2V 差 14 Elo（H3 1161→1175）、I2V 差 25（H3 1128→1153）、
  editing 開放最高僅第三（Hyperflow H3，榜首 Wan 3.0 1196）——
  theama 的追趕路徑全在既有管線上：P1a LoRA-DPO（品味對齊）
  ＋M2 audio DiT（with-audio 評分主戰場）＋資料引擎擴量。
  驗收語意不變：保底閘（盲測 win rate、VBench）照舊，登頂為
  stretch 層，兩者並列於 M5 列。
  2026-11-26 補記（蒸餾合法性盤點，依使用者提問「各榜前五或十可蒸餾？」）：
  **四個榜（T2V／I2V／editing／T2I 總榜）top10 無一可合法蒸餾**——
  Wan 3.0 為 hosted API（Model Studio 條款禁以模型及 Output 訓練競爭
  產品）、Seedance 2.5 受豆包服務協議禁止以服務／生成內容做模型開發
  與微調、MiniMax H3 及其衍生（Utopai X／Hyperflow H3／H3 Max／
  Boreal-H3／Pika H3 Flex 等）受 Community License V.3 明文禁止以
  Works/Outputs 改善其他模型、榜上 **MAGI-2 為 480B flagship 且權重
  尚未釋出**（注意：Apache-2.0 的是 MAGI-2 Preview，單獨排名 T2V #21
  ／I2V #29，114B／10 秒／需 8×Hopper 約 307GB，僅列觀察）。**可蒸餾
  候選全在榜外或 open-weights 榜中段**：Wan 2.2（Apache-2.0，無地區
  ／營收／輸出限制，本專案現行底座＋自我蒸餾起點）、Z-Image Turbo
  （Apache-2.0）、Qwen-Image-2512（Apache-2.0）、Ming-Image-0.1-Design
  （MIT，open-weights T2I 榜上名次最高的完全寬鬆授權）。Hunyuan
  Image／Video 雖開權重，但 Tencent Community License 將蒸餾與合成
  資料訓練納入 Model Derivatives 限制，不算乾淨，維持不碰。結論：
  蒸餾 top10 之路不可行，強化既有路線——Wan 2.2 底座＋自我蒸餾
  （Theama Film 自產 4K/30s 帶音訊樣本回流）＋CJK 簡中優先＋開放
  授權差異化。
  2026-11-26 第二筆補記（交付格式拍板）：**HDR＋AV1 提前至 M4 交付**
  並列為獨立賣點——真 HDR（HDR10/HLG）與 AV1＋HLS 封裝仍無人交付
  （Wan 3.0 鎖 1080P 8-bit、H3 鎖 2K 8-bit）；
  新增 **HLS 分發層**——fMP4/CMAF segment＋master .m3u8 自適應階梯
  （4K HDR AV1 → 1080p HEVC → 720p/480p H.264，HEVC 層為 Safari
  fallback 屬必需，同批 segment 兼出 DASH）；Avatar 實時線（M6b）
  不走標準 HLS，採 LL-HLS 或 WebRTC。
  2026-11-26 第三筆補記（**Seedance 2.5 4K 上線，規格差異化修正**，
  依使用者通報查證）：**即梦於 2026-11-13 正式上線 Seedance 2.5
  4K**——原生 3840×2160、30 秒、**10-bit BT.709（SDR，非 HDR10/HLG）**、
  六種畫幅、480P 樣片→4K 正片 Draft 模式，限時價低至 ¥0.15/秒
  （智東西／騰訊新聞 2026-11-18/19 實測，B 級多源一致）。修正前記：
  「4K＋10-bit 孤峰」說法作廢；仍成立的差異化為 ① **真 HDR**
  （HDR10/HLG——Seedance 4K 僅 BT.709 SDR）② **AV1/HEVC＋HLS
  fMP4/CMAF 分發包** ③ **開放權重本地部署** ④ **8K（M4b）**。M5
  驗收 (4) 語意同步調整（見 training-plan.md）。

## 發布策略

- 自研微調權重（DPO LoRA、audio DiT、extend/4K 模組）以 **Apache-2.0** 發布於
  Hugging Face `chienhsinlin` namespace，模型卡遵循 HF model card 規範（訓練資料來源、
  評測分數、限制與合規聲明）。
- 第三方權重（Wan 2.2、Qwen-Image 等）不轉售不重新打包，使用者自行下載
  （`scripts/download_weights.sh`），權重授權各自保留。

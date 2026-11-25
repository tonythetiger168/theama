---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T08:45:00Z"
head: 6f4799cfb8bd6692b2dfed233b8f2b09eac3fa2b
head_verified: true
last_increment: "② watchlist 第十輪：四固定項覆核——Wan 3.x 仍 API-only（最後開放 Wan 2.2）、阿里下一代影片模型 11 月窗口 T-5 仍無發布（Model Studio 列表＋第三方 11-09/11-11 核查）、Qwen-Image 紅線四項無變化（2.1 回歸 Apache 請願 #39 無回應）、VA-Judger checkpoint 授權二次核查失敗（HF 空 README）維持使用者代查；駁 youart.ai「Wan 2.7 開權重」低權威誤報；補 efsora 資料點（Ming-Image-0.1-Design＝最高排名 fully-permissive 影像模型）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: m2_external_smoke_passed_reviewed   # PR #3 雙跑通過已判讀（pass 18）；M1/P1a/M2 internal 仍待 Claude 跑
blockers: []
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；Claude 若已 push M1/P1a/internal-wan_blocks GPU 結果或 v0.2.0 tag，先消化再增量"
  - "pass 16–19 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十輪（6f4799cf）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra，否則 decode/M7a 測試 skip）"
  - "③ 候選：encode_texts 進一步強化（如 error 重試配額上限）；阿里 11 月窗口 11-30 關閉前續盯下一代影片模型"
  - "model-roadmap.md 已推至 6f4799cf（watchlist 第十輪），後續可正常從本地推送此檔"
  - "GPU 下一棒（Claude，依 status.md）：M1 runpod_m1_eval.sh 5→50 prompts→VBench ±0.02 → P1a（先驗 reward/score.py::_invoke 對真實 VideoReward）→ M2 internal wan_blocks＋LoRA smoke → 正式訓練前換 stable-torch image"
---

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0；**M2 外部 latent-bridge 真實權重 GPU smoke 已通過**（PR #3：
11-24 RTX 4090＋11-25 A40 全 runbook 復跑，詳 docs/status.md）；M1 RunPod
runner 落地待跑（runpod_m1_eval.sh）；P1b 鏈路補齊（雙 cache → joint
loader → runner → runbook 第 5 節），222 CPU tests 全綠；GPU 驗收由
Claude 執行（使用者 11-25 擴權：其他底模＋HDL on RunPod 亦可）；
CI.yml 內容備妥待上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-25 pass 19 @ 6f4799cf — ② watchlist 第十輪覆核（本 session
  增量）：四固定項全查——**Wan 3.x 仍 API-only**（usefastlane 11-01、
  renoise 11-11 逐 repo 核查一致，最後開放權重仍 Wan 2.2）；
  **阿里下一代影片模型 11 月窗口 T-5 仍無發布**（Model Studio 新模型
  列表最新僅 qwen3.8-flash／wan3.0-video-prime／wan2.6・2.7 flash
  變體／qwen-image-edit-max／kimi-k3，第三方 11-09／11-11 核查同）；
  **Qwen-Image 紅線四項無變化**（2.1／2.1-Turbo 研究授權、2512／
  Edit-2511 Apache 線、3.0／3.1 hosted-only；HF 討論 #39 請願回歸
  Apache 無官方回應）；**VA-Judger checkpoint 授權二次核查失敗**
  （HF 頁面沙箱 fetch 僅空 README、無 license metadata；gated 下載），
  維持使用者代查。新增判讀：駁 youart.ai「Wan 2.7 已開權重」低權威
  誤報（與既有逐 repo 核查矛盾，同 miraflow 類）；補 efsora 10-19
  資料點——Ming-Image-0.1-Design 為排名最高 fully-permissive 影像
  模型（差 2.1 四十二分），支持 M7a 第三路徑 diagnostic 定位。
  RESTORE 對帳乾淨（HEAD 仍 0a2a3cd7、無 Claude GPU 新 commit、tags
  空、issues 0、temp 無新上傳）；開工前 mtime 掃描無對岸活動中檔案。
  **SAVE 事故第六實例（長檔轉錄漂移）**：roadmap 首推 d378836b 回傳
  blob 41902464 ≠ 本地 c52928d8——cmp 定位兩處**未編輯舊文**轉錄錯
  （out-of-domain 誤作 out-domain－3 bytes、形態相容誤作形狀相容），
  本地檔始終正確；以本地內容重推 6f4799cf byte-exact 驗證一致，
  壞 commit 留史。教訓加強：**20KB 級長檔推送後必做逐字節 cmp**
  （api.github.com raw 可達，raw.githubusercontent 沙箱連不上）。
  CPU 全套 222 passed（docs-only 重跑確認）。docs/model-roadmap.md
  blob c52928d8 驗證一致。

- 2026-11-25 pass 18 @ a8a8f803 — ① GPU 輸出判讀（本 session 增量）：
  消化 Claude PR #3（08:13:46Z 合併，base d96b7192，10 檔 +447/-16）。
  **M2 外部 latent-bridge 真實權重 smoke 雙跑通過**：11-24 RTX 4090
  （c13d0f2）＋11-25 A40 全 runbook 復跑（4c2d828；三 JSON＋雙 WAV＋
  logs 齊、pod 刪除、network volume theama-m2 保留、< $0.05）——
  DAC 44.1kHz rel-L2 0.0935/0.0942（對照隨機權重 ≈1.0，證實真實權重
  載入）、latent [1,87,2048]；Wan TI2V-5B bf16 凍結骨幹反傳＋bridge
  優化器更新（32 grad tensors／28 updated）、峰值 9.37 GiB 兩卡一致。
  證明範圍僅 contract（真實權重載入、tensor 配置、反傳、記憶體
  餘裕）；不證明 prompt-to-video 品質／感知音質／AV sync——M1 品質閘、
  P1a、M2 internal wan_blocks GPU 驗收仍待跑。**M1 RunPod runner
  落地**：runpod_m1_eval.sh（TI2V-5B 全 repo 入 volume＋容量預檢＋
  M1_EVAL_SMOKE_OK 閘）；配套修正 gen_candidates tuple unpack、
  gen_eval_set --model-path、t2v.py 本地路徑載入＋numpy 幀轉 PIL，
  test_m1 +2。新增 docs/status.md（handoff 單一入口）＋AGENTS.md；
  .gitignore 加 results/（原始 GPU 產物不入庫，存 Tony PC）。
  本地同步：10 檔逐檔 git hash-object 對遠端 sha 全一致（
  runpod_m1_eval.sh 首推比對抓到一處 PR 後修正——disk check 的
  Path(sys.argv[2]) → sys.argv[2]，以遠端為準修齊）；bash -n 通過；
  CPU 全套 **222 passed**（220+2，合併態獨立重跑）。**使用者新指示
  （11-25）：Claude 可做其他底模＋HDL on RunPod**——已記入
  division_of_labor，紅線不變（H3/LTX/Qwen2.1/FLUX3-Klein/Krea2
  權重程式碼輸出皆不碰；其他底模僅限授權相容者）。tags 空、issues 0、
  temp 無新上傳；watchlist 本輪不重查（pass 15/16 已覆核）。

- 2026-11-25 pass 17 @ a7cb7697 — runbook §4.1 Ming 第三路徑驗收段
  （本 session 增量，next_hour 明列的「替對岸補」候選）：`### 4.1 第三
  路徑（diagnostic-only）：Ming-Image-0.1-Design`——gen/score 指令對
  （scorer model-agnostic）、卡片預設 12 steps/CFG 1.0 與 1024/2048
  落桶說明、**單卡 80 GiB 官方驗證配置**（24GB 預期 OOM，有 80 GiB 卡
  再排、不阻塞雙路徑驗收）、diagnostic-only 定位（24 seed 實測前不入
  M7a 驗收閘）。**競態紀律實例五（開工前攔截＋工作中疊加）**：開工前
  mtime 掃描發現 score_cjk_text.py（16:04）／test_m7a.py（16:05）為
  數秒內新檔——對岸正在做 next_hour 明列的 score 強化，依紀律避開
  同題轉 docs；工作中對岸推 `a3067d9b`（08:08:36Z，score provenance
  強化，CPU 220 passed），本 session 推送前對帳（08:07）尚未落地，
  兩次 runbook 推送疊加其上無衝突（不同檔）。**SAVE 事故第五實例
  （尾換行）**：首推 `be140326` payload 漏檔尾換行，blob a1f5f403 ≠
  本地 17239ef3 當場抓到；補尾換行重推 `a7cb7697` byte-exact
  （runbook blob 17239ef3 本地重算一致），壞 commit 留史不強推。
  教訓加強：推送 payload 結尾必帶檔尾換行（create_or_update_file
  不會自動補）。對岸 pass-17 checkpoint 於本 session 推送途中搶先落地
  （sha 325bca2e→6d590445，create_or_update_file 退回 stale），本
  session 以其推送版為基核對——與先前消化之本地版逐字一致、本版為
  嚴格超集——遂以新 sha 重建推送（front matter head→a7cb7697；對岸
  entry 保留於下）。
  交叉驗證：對岸記錄的 score blob 1142893e 與本 session 開工前本地
  重算一致；CPU 全套 220 passed（217+3，本 session 獨立重跑確認）。
  watchlist 本輪不重查（pass 15/16 已覆核，阿里窗口 11-30 才關閉）；
  tags 空、issues 0、遠端無 .github。

- 2026-11-25 pass 17 @ a3067d9b — M7a score 腳本來源標註強化（本
  session 增量，session-state next_hour 明列候選）：score_cjk_text.py
  新增 load_gen_manifest()（讀 gen_cjk_images 寫的 <images>/
  manifest.jsonl；缺檔→None、壞行/缺 id 行 warn-and-skip，provenance
  屬 metadata 永不讓 scoring 失敗）、build_summary() 純函數化（免
  tesseract 可測）。summary sidecar 新增 models/model/repo_ids＋
  no_provenance（有分數但 manifest 無列的圖——同目錄混入他模型舊圖
  的 drift 偵測）；manifest 混多模型時 stderr 警告（summary 會跨模型
  聚合）；新列與 resume 保留列（setdefault 回填）都帶 model 標籤。
  SCORER_VERSION 維持 cjk-text-1——打分語意未變，舊 resume 列全保留。
  test_m7a.py +3：manifest loader（缺檔/壞行/空行）、summary 來源
  標註（rate 不變＋單模型 headline＋無 manifest 不加鍵）、混模型＋
  orphan 旗標。RESTORE 對帳：遠端 HEAD 仍 0e9bceda（pass-16
  checkpoint 修正版），無 Claude GPU commit、tags 空、issues 0、
  temp 無新上傳；開工前 mtime 掃描無對岸新檔。推送前再對帳一次無
  競態；blob SHA 驗證一致（score 1142893e／test 02ec4c3a，本地 git
  hash-object 重算相同）。CPU 全套 220 passed（217+3；沙箱重置又洗
  掉 pytest，已重裝）。watchlist 本輪不重查（pass 15/16 已覆核，
  阿里窗口 11-30 才關閉）。

- 2026-11-25 pass 16 @ 4e1d202f — M7a 第三路徑 Ming-Image-0.1-Design
  （MIT）gen plumbing 預備落地：gen_cjk_images.py 新增
  `--model ming-image-design` → inclusionAI/Ming-Image-0.1-Design；
  EXPERIMENTAL_MODELS 標記 diagnostic-only（選用時印 [note]，CJK
  24 seed 實測前不入 M7a 驗收閘）；turbo/quality if-else 重構為
  MODEL_DEFAULTS 卡片預設表（qwen 30/3.5、z-image 8/0.0、ming
  12 steps/CFG 1.0）；RESOLUTION_BUCKETS＋snap_resolution() 把
  Ming 固定 1024/2048 square 落桶（request/manifest/gen kwargs
  一致用 snapped 值）。事實核查（web 三來源）：MIT、6B、12 steps、
  CFG 1.0、官方驗證配置單卡 80 GiB（24GB 卡預期 OOM，社群量化未驗證）、
  T2I 為純文字路徑不觸及 image-input 才用到的 Qwen-license vision
  tower；卡片 prompt-rewriting 刻意不開（quoted CJK 字串逐字送驗）。
  test_m7a.py +2：模型表 contract（mapping 對齊 theama_base.yaml
  image_backbone/_fast、ming repo id/defaults、experimental 集合不
  含雙正式路徑）＋解析度落桶行為。watchlist 覆核：阿里下一代影片
  模型 11 月窗口（11-30 關閉）仍無發布、Wan 3.x 仍 API-only、
  Qwen-Image 紅線不變。RESTORE 對帳無 Claude GPU commit／tags 空／
  issues 0／temp 兩張 KIMI jpeg 仍為 11-24 RunPod 截圖非新 GPU 輸出。
  CPU 全套 217 passed（215+2）；--help 列出第三選項。推送前後各對帳
  一次無競態；blob SHA 驗證一致（gen 82c4bf02／test bc37dfd1）。

- 2026-11-25 pass 15 @ 5455691 — watchlist 第九輪（本 session 增量）：
  model-roadmap.md 新增 **GenRM-Flow**（arXiv 2611.11421，2026-11-08——
  偏好微調後的生成器即 process-aware RM，在含噪 latent 空間以
  velocity-prediction 誤差相對差打分、免外部 reward 網路；DPO/IPO/RWR
  證為通往同一最優 policy 的等價路徑；對 P1a LoRA-DPO 的 BoN／資料
  過濾與 M2 flow 後訓練直接相關）與 **PAVRM**（arXiv 2511.21541v3，
  CVPR 2026 Findings，latent 空間打分免 decode）觀察條目，授權皆未
  核實、不改 P1a／M2 配方；四固定項覆核無變化（阿里下一代 11 月
  窗口倒數仍無發布）。競態實例三視角補記：本 session 的 runbook
  第 5 節草稿（5a/5b/5c）於對岸推送 58b2bdc 後主動擱置、本地還原，
  兩項獨有改進經對岸 b9c4bb4 採納合併——結案。順帶癒合本地
  model-roadmap.md 的 33-byte echo 漂移：以遠端版 byte-exact 回寫
  （git blob sha 3b607a49 本地重算一致），blocker 移除。推送前後各
  對帳一次（對岸 checkpoint 16ac8e7 於本 session roadmap 推送前
  17 秒落地，內容已消化）。blob SHA 驗證一致（roadmap 3b607a49）。

- 2026-11-25 pass 15 @ b9c4bb4 — runbook 第 5 節（P1b joint training
  驗收段）落地＋競態實例三結案：新檔 `configs/m2_joint_av_dev.yaml`
  （TI2V-5B dev 24GB 版 joint config：video.adapter=ti2v-5b、
  audio.video_dim=48，training/acceptance 與 prod 相同）、
  test_m2_joint_av_runner.py +1 yaml contract 測試（dev↔prod 一致性：
  adapter==dev_adapter、video_dim==48==dev_latent_channels、
  trainer_config/text_dim 相同）、runbook 新增第 5 節（5.0 免權重自檢
  TRAIN_JOINT_AV_SMOKE_OK → 5.1 小 cache E2E dev 路徑：manifest 靜檢→
  decode 探針→雙 cache limit 8→20 步 bridge→full@10→--resume auto 到 30
  → 5.2 正式 A14B run 20k/2k/bf16）。**競態實例三**：本 session 推送
  58b2bdc 後發現對岸把自版第 5 節（5a/5b/5c）append 到本地 runbook
  （未推送）；逐項比對→採納其兩項獨有改進（text cache 以 --encoder
  固定 Wan2.2-T2V-A14B-Diffusers 跨 dev/prod 共用、prod 免重燒 UMT5；
  ckpt_latest 恆於最終步落盤不受 --save-every 影響）以 b9c4bb4 合併
  推送；其餘項（decode 探針、dropped=0 閘、peak memory、AV sync 目檢）
  本版已有。期間本地 runbook 一度被還原成 pass-14 舊版（14:19），
  已從遠端 blob 拉回並驗證一致。watchlist 覆核：Wan 3.x 仍 API-only
  （11-13 多方榜 Wan 3.0 T2V 第一但 closed）；「Wan 2.7 開源
  Apache-2.0」說法經三來源交叉核駁（10-25/11-01/11-10 皆指
  2.5/2.6/2.7 未開權重，最後開權重仍 Wan 2.2 系）——骨幹不變；
  阿里下一代 11 月窗口剩 5 天仍無發布；Qwen-Image 紅線不變。
  RESTORE 對帳無 Claude GPU commit／tags 空／issues 0。CPU 全套
  215 passed（214+1）；CLI smoke 重驗 TRAIN_JOINT_AV_SMOKE_OK。
  blob SHA 驗證一致（dev config a17cfd09／test ff576759／runbook
  合併版 50f79eb7）。

- 2026-11-25 pass 14 @ 8042bff — P1b joint training runner＋薄殼：新檔
  `src/theama/train/joint_av_runner.py`（RunnerConfig＋JointAVRunner：
  dataloader 無限循環、device/dtype 搬運、cond 改為 on-device 由
  audio.cond_pooler（掛 AudioDiT 子模組，bridge phase 凍結/full phase
  訓練、隨 audio state_dict 入 ckpt）對 masked-mean pool 的 video_cond
  投影——pooling 移出 DataLoader collation 保梯度在訓練裝置；
  --bridge-steps 排程切 bridge→full；metrics.jsonl；checkpoint
  精確 resume——先 set_phase（bridge ckpt 重注 LoRA 補齊 strict load
  keys、full ckpt 先 merge 新 LoRA）再載 state、optimizer state 搬
  param device、torch/CUDA RNG 還原；NaN metrics fail-closed 留
  ckpt_nonfinite 緊急檔）、`scripts/train_joint_av.py`（--smoke 合成
  5-D Wan latent 路徑 4 步＋phase 切換＋+2 步 auto-resume，印
  TRAIN_JOINT_AV_SMOKE_OK；real run 接 WanVideoAdapter＋雙 cache）、
  `tests/test_m2_joint_av_runner.py` 15 例（yaml contract、pooler
  phase 行為、loop/log/ckpt、phase 排程、LoRA bridge 與 merged-full
  resume roundtrip、NaN 緊急中止、production planner cache→DataLoader
  →runner E2E、CLI smoke）。watchlist 覆核無變化（Wan 3.0 仍 API-only
  無權重、阿里下一代 11 月窗口最後一週仍無發布、Qwen-Image 紅線不變、
  VA-Judger 仍首選）；RESTORE 對帳無新 commit／無 tag／issues 0。
  CPU 全套 214 passed（199+15）。三檔 blob SHA 驗證一致
  （e270e982／72c69105／bd5022d5）。

- 2026-11-25 pass 13 @ 6a08957 — P1b training-side loader：新檔
  `src/theama/data/joint_latent_loader.py`（join 兩 cache manifest 於
  sample index，fail-closed——僅雙成功列上桌、model 過濾預設取各 cache
  meta.json、AV/text caption 交叉核對補 AV drift guard 刻意不管
  caption 編輯的缺口、Dataset 建構時全量驗 ndim+finite 且 __getitem__
  逐次複查、collate 產 video_tokens/audio_latent/video_cond/text_mask
  並經 masked-mean pool＋cond_pooler 補 cond）、
  `tests/test_m2_joint_latent_loader.py` 15 例（join 交集/丟棄計數、
  caption 漂移排除、model/meta 守衛、NaN/損毀拒絕、mask/pool 正確性、
  DataLoader→train_step E2E）。watchlist 覆核無變化（Wan 3.x 仍
  API-only、阿里下一代 11 月窗口最後一週仍無發布、Qwen-Image 紅線不變、
  VA-Judger 仍首選）；tags 空（v0.2.0 未打）、issues 0、.github 未上。
  CPU 全套 199 passed（184+15）。兩檔 blob SHA 驗證一致
  （39225c0f／8b41f5de）。
- 2026-11-25 pass 12 @ bec7158 — P1b text cache：新檔
  `src/theama/data/text_cache.py`（content-addressed
  `text_{sha256(caption)[:16]}.pt`——caption 編輯結構性失效，舊檔成
  orphan 永不誤用，從根上消除 pair/AV cache 需程序性 drift guard 的
  那類漏洞；resume 驗 2D finite、error 列自動重試、meta.json model
  不符整庫作廢、同 run 重複 caption 只編碼一次、encode 後重驗拒收
  NaN/壞檔/錯 ndim）、`scripts/encode_texts.py`（GPU 薄殼：
  UMT5EncoderModel subfolder、padded (512,D) bf16、seq_len 入列供
  mask 重建、全 cached 不載 encoder）、`tests/test_m2_text_cache.py`
  14 例。RESTORE 對帳發現遠端領先本地五個提交（session-state 系
  pass 5–11：M7a 雙腳本＋seed、GPU runbook、CLAUDE.md、0.2.0 bump）——
  已逐檔拉回本地並驗 blob SHA（10/11 一致；model-roadmap.md 因 echo
  層不可見字元差 33 bytes，以遠端為準不從本地推送，見 blockers）。
  順帶補驗證 pass 9 標記待驗證的 M7a 雙腳本：py_compile 通過、
  scorer 純函數（normalize/char_recall）與無 chi_tra 錯誤路徑
  （exit 1 訊息正確）皆符合預期。CPU 全套 184 passed。三檔 blob
  SHA 驗證一致（ea921edd／f91d8c6f／cf843d4d）。

## RESTORE 協議（每輪第 0 步，依序執行）

1. 讀取本檔 front matter。若 `head` 與遠端 HEAD 不一致，先 `list_commits`
   列出 recorded head 之後的提交，逐一消化後再開始本輪增量。
   **Claude 接手 GPU 後，其結果提交（eval summaries、gpu-results.md、
   可能的 script 修正）都從這一步對帳。**
2. `updated_at` 距今超過 24h 視為 STALE：改以 `docs/model-roadmap.md`
   現況表 + 最近 commits 重建認知，並覆寫本檔。
3. 本檔不存在（首輪或誤刪）：直接依 model-roadmap 現況重建，不留空窗。
4. 恢復順序固定：本檔 → `docs/model-roadmap.md`（現況 + watchlist）→
   `docs/gpu-acceptance-runbook.md`（GPU 進度）→ 使用者當輪貼文。
   專案記憶 mount 可用時亦以本檔為準對帳。

## SAVE 協議（每輪最後一步；僅有進度時執行）

1. 更新 front matter（`updated_at`、`head`、`last_increment`、`blockers`、
   `next_hour`）並在「本輪記錄」頂端加一行：
   `YYYY-MM-DD pass N @ <HEAD 短碼> — 摘要`。
2. 推送後做 blob SHA 驗證：`get_file_contents` 回傳 SHA 須與推送回傳
   一致；不一致立即回報，不得聲稱已存檔。
3. 「本小時無進度」不回寫本檔——`updated_at` 因此恆等於最後有進度時刻。
4. 只記狀態、決策與驗收數據；不貼 log 全文、權重、生成輸出。

## 紅線（任何增量不得削弱）

H3 / LTX-2.x / Qwen-Image-2.1 / FLUX-3-Klein / Krea-2 的權重、程式碼、
生成輸出一律不進本專案；防護為 `AVDataset.check_source` 黑名單 +
reward 註冊表白名單（`src/theama/reward/config.py`）。

---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T07:20:00Z"
head: 4e1d202fa25efb7ae0bfc29aa0b0176d4ee72609
head_verified: true
last_increment: "M7a 第三路徑 plumbing：gen_cjk_images.py 新增 --model ming-image-design（inclusionAI/Ming-Image-0.1-Design，MIT，experimental/diagnostic-only 不入驗收閘；MODEL_DEFAULTS 卡片預設 12 steps/CFG 1.0；RESOLUTION_BUCKETS snap 1024/2048）＋ test_m7a.py +2 contract 測試；CPU 全套 217 passed；watchlist 覆核無變化（阿里下一代 11 月窗口剩 5 天仍無發布）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: none_pending_review            # Claude 接手 GPU 驗收；若其 push 結果，下輪 RESTORE 對帳
blockers: []
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；Claude 若已 push GPU 結果或 v0.2.0 tag，先消化再增量"
  - "M7a 第三路徑 gen plumbing 已於 pass 16 落地（4e1d202f）——RESTORE 到本檔者請勿重複做同題；score 側本就 model-agnostic（吃一個 images dir）"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra，否則 decode/M7a 測試 skip）"
  - "③ 候選：encode/score 腳本強化（如 score_cjk_text summary 從 manifest.jsonl 帶 model 標籤）；watchlist 固定四項覆核（阿里下一代 11 月窗口 11-30 關閉，仍無發布）"
  - "model-roadmap.md 本地已與遠端 byte-exact（pass 15 癒合），後續可正常從本地推送此檔"
---

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0（CPU 側程式全落地、GPU 驗收待跑）；P1b 鏈路補齊：雙 cache →
joint loader → runner＋薄殼 → runbook 第 5 節驗收段（含 dev/prod 雙路徑
config），217 CPU tests 全綠；GPU 驗收由 Claude 依
docs/gpu-acceptance-runbook.md 執行；CI.yml 內容備妥待上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

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
  訓練）對 masked-mean pool 後的 video_cond 產生、--bridge-steps 排程
  切換、metrics.jsonl、checkpoint 精確 resume——先 set_phase 再 strict
  load：bridge ckpt 重注 LoRA 補 keys、full ckpt 先 merge；optimizer
  state 搬 device、RNG 還原；NaN metrics fail-closed 留 ckpt_nonfinite）
  ＋ `scripts/train_joint_av.py`（--smoke 合成 5-D Wan latent 路徑 4 步
  ＋phase 切換＋+2 步 auto-resume，印 TRAIN_JOINT_AV_SMOKE_OK；real run
  接 WanVideoAdapter＋雙 cache）＋ test_m2_joint_av_runner.py 15 例
  （含 production planner cache→DataLoader→runner E2E、CLI smoke）。
  競態實例二結案（train_joint_av 同題撞車，本 session 版為被保留推送
  版）。watchlist：阿里下一代 11 月窗口最後一週仍無發布；Wan 負責人
  张迎亚 10 月中離職。RESTORE 對帳無 Claude GPU commit。CPU 全套
  214 passed（199+15）；CLI smoke 驗證 OK。blob SHA 驗證一致
  （e270e982／72c69105／bd5022d5）。

- 2026-11-25 pass 13 @ 6a08957 — P1b training-side joint latent loader：
  新檔 `src/theama/data/joint_latent_loader.py`（join av_latents.jsonl＋
  text_latents.jsonl 於 sample index；fail-closed：僅雙成功列上桌、
  model 過濾預設取各 cache meta.json、AV/text caption 交叉核對補 AV
  drift guard 缺口；Dataset 建構全量驗 ndim+finite、__getitem__ 複查；
  collate 直出 train_step batch：video_tokens/audio_latent/video_cond/
  text_mask＋cond_pooler 經 masked-mean pool 補 cond）＋
  test_m2_joint_latent_loader.py 15 例（含 DataLoader→train_step E2E
  money test）。競態實例一結案（對岸先推 loader，本 session 重複版
  刪除未推送）。RESTORE 對帳無 Claude GPU commit。CPU 全套 199
  passed（184+15）。blob SHA 驗證一致（39225c0f／8b41f5de）。

- 2026-11-25 pass 12 @ bec7158 — P1b text cache：新檔
  `src/theama/data/text_cache.py`（內容定址 text_{sha256(caption)[:16]}.pt
  ＋meta.json 綁 encoder id）＋ `scripts/encode_texts.py`（GPU 薄殼，
  resume 跳已編碼 caption）＋ test_m2_text_cache.py 14 例。沙箱與記憶
  mount 恢復；model-roadmap 本地副本注意事項記入。CPU 全套 184 passed。
  blob SHA 驗證一致（ea921edd／f91d8c6f／cf843d4d）。

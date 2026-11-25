---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T06:35:00Z"
head: 5455691ec09174196ac6088374e41fdece614bd1
head_verified: true
last_increment: "watchlist 第九輪：GenRM-Flow（arXiv 2611.11421，偏好微調生成器即 process-aware latent RM）＋PAVRM（2511.21541v3）入 model-roadmap 觀察（授權未核實，不改 P1a/M2 配方）；本地 model-roadmap 33-byte 漂移癒合（byte-exact 回寫，blob 3b607a49，blocker 移除）；runbook 第 5 節為對岸 pass-15 增量（58b2bdc＋b9c4bb4 合併本 session 草稿），詳其 checkpoint 16ac8e7"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: none_pending_review            # Claude 接手 GPU 驗收；若其 push 結果，下輪 RESTORE 對帳
blockers: []
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；Claude 若已 push GPU 結果或 v0.2.0 tag，先消化再增量"
  - "runbook 第 5 節已於 pass 15 落地並合併對岸草稿（b9c4bb4）——RESTORE 到本檔者請勿重複做同題"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra，否則 decode/M7a 測試 skip）"
  - "③ 候選：M7a 第三路徑 Ming-Image-0.1-Design（MIT）gen/score plumbing 預備（CJK 實測前不入正式 eval）；watchlist 固定四項覆核（阿里下一代 11 月窗口剩 5 天，仍無發布）"
  - "model-roadmap.md 本地已與遠端 byte-exact（pass 15 癒合），後續可正常從本地推送此檔"
---

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0（CPU 側程式全落地、GPU 驗收待跑）；P1b 鏈路補齊：雙 cache →
joint loader → runner＋薄殼 → runbook 第 5 節驗收段（含 dev/prod 雙路徑
config），215 CPU tests 全綠；GPU 驗收由 Claude 依
docs/gpu-acceptance-runbook.md 執行；CI.yml 內容備妥待上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

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
- 2026-11-25 pass 11 @ 8d19205 — 版本化：pyproject.toml 0.1.0 → 0.2.0
  （0.1.0=scaffold；0.2.0=CPU 側程式全落地 checkpoint：M1 推理/eval/mux、
  P1a DPO 管線＋latent cache drift guard、M2 AudioDiT/JointAVTrainer/
  DAC/Wan contract/AV 資料管線、M6a/b/c 骨架、M7a seed＋gen/score 腳本、
  166 CPU tests）。tag v0.2.0 待打：MCP 工具集無建 tag 能力
  （list/get tag 唯讀），已交付使用者以本機 git tag 或網頁 Release
  建立。blob SHA 驗證一致（67866fe6）。
- 2026-11-25 pass 10 @ a1244b3 — GPU 交接：依使用者指示，GPU 驗收執行面
  switch to Claude；落地 docs/gpu-acceptance-runbook.md（環境前置 → M1 →
  P1a → M2 smoke → M7a 雙路徑的順序 playbook、每節驗收閘、回報格式
  {step, command, exit_code, summary, GPU, 耗時}、紅線重申）＋修簡繁錯字；
  front matter 新增 division_of_labor；下輪起 RESTORE 需對帳 Claude 的
  GPU 結果提交。docs-only。
- 2026-11-25 pass 9 @ 0b7359b — 「我們這邊能做的全做完」批次：
  ① M7a 生成腳本 scripts/gen_cjk_images.py（Qwen-Image-2512／Z-Image
  Turbo 雙路徑、resumable、manifest+summary 閘）；② M7a 評分腳本
  scripts/score_cjk_text.py（tesseract chi_tra/chi_sim OCR、exact_hit +
  char_recall、per-category summary、scorer-version resume）；③ CI.yml
  已備妥但 MCP token 無 workflow scope（403 insufficient scopes），
  內容交付使用者自行上傳；④ 兩腳本均為獨立 script 不被 tests import，
  沙箱掛機期間無法本地驗證，標記為待驗證。blob SHA 驗證一致
  （f0d8d613／c896a9bc）。
- 2026-11-25 pass 8 @ 8657e36 — watchlist 第八輪：VA-Judger repo 狀態
  更新（ShareLab-SII/VA-Judger 自 2026-08-20 釋出基礎程式碼、
  checkpoints 與 VA-Judger-Bench；RM SFT 程式碼與 VAPref-10K 仍 to-do；
  RM 本體為 Qwen3-Omni-30B-A3B 全參微調；checkpoint 授權未標示，
  維持觀察）；Wan-Animate-2 repo id／Wan 3.x API-only／阿里下一代
  未發布／Qwen-Image 紅線覆核無變化；CI.yml 未上傳、issues 0；
  沙箱仍掛，docs-only。

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

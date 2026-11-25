---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T04:35:00Z"
head: 8042bff6fd6dd5e627777db34b6b388b16720422
head_verified: true
last_increment: "P1b joint training runner＋GPU 薄殼：joint_av_runner.py（cond_pooler 掛 audio 隨 phase 凍結/訓練、on-device pooling、bridge→full 排程切換、metrics.jsonl、checkpoint 精確 resume——phase 先還原再 strict load、optimizer state 搬 device、RNG 還原、NaN fail-closed 留緊急 ckpt）＋scripts/train_joint_av.py（--smoke 印 TRAIN_JOINT_AV_SMOKE_OK；real run 接雙 cache＋WanVideoAdapter）＋15 tests（214 passed）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: none_pending_review            # Claude 接手 GPU 驗收；若其 push 結果，下輪 RESTORE 對帳
blockers:
  - "本地 docs/model-roadmap.md 為 echo 逐字轉錄版，與遠端 blob 83cd3626 差 33 bytes（疑為 echo 層不可見字元 artefact）；以遠端為準，本地勿推送此檔"
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–4 節）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；Claude 若已 push GPU 結果或 v0.2.0 tag，先消化再增量"
  - "GPU 側新增可跑項：scripts/train_joint_av.py --smoke（本輪落地，無 cache/無 Wan 權重即可驗 runner 全路徑＋resume）；cache 建好後正式 joint training 直接用同腳本 --av-cache/--text-cache"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra，否則 decode/M7a 測試 skip）"
  - "續做 ③ 候選：runbook 增補第 5 節 joint training 驗收段（train_joint_av.py --smoke → 小 cache E2E）；watchlist 固定四項覆核（阿里下一代 11 月窗口最後一週，仍無發布）"
---

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0（CPU 側程式全落地、GPU 驗收待跑）；本輪新增 P1b joint
training runner＋GPU 薄殼 scripts/train_joint_av.py（雙 cache →
JointAVTrainer 訓練迴路、checkpoint resume、--smoke 自檢，214 CPU
tests 全綠）；GPU 驗收由 Claude 依 docs/gpu-acceptance-runbook.md
執行；CI.yml 內容備妥待上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

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
- 2026-11-25 pass 7 @ 3738703 — watchlist 第七輪：Omni-RRM（arXiv
  2602.00846，四模態統一 RM，P1a＋M2 reward 單一 judge 候選）、
  LongCat-Video-Avatar 1.6（arXiv 2611.18822v2，frozen 底座雙 LoRA
  分支 GRPO＋OPSD，M6a/M6b 方法參考）、FlowPortrait hard id 補齊
  （arXiv 2603.00159，複合 reward＋防 hacking ablation，M6a 參考）；
  Omni-RL Judge 仍無 arXiv id；四大固定項覆核無變化；CI.yml 未上傳、
  issues 0；沙箱仍掛，docs-only。
- 2026-11-25 pass 6 @ 4c2d828 — watchlist 第六輪：Omni-RL Judge（Adobe，
  verifiable-reward omni A/V judge，僅 ResearchGate 條目）＋
  Ming-Image-0.1-Design（MIT 設計 T2I，M7a 第三路徑觀察候選）入觀察；
  docs-only。（本輪原未回寫本檔，pass 7 補記。）
- 2026-11-25 pass 5 @ 07b0a2e — session save/restore 落地：建立
  docs/session-state.md（front matter 狀態 + RESTORE/SAVE 協議 + 滾動
  記錄）；README 加入口；緣由：專案記憶 mount 與沙箱執行環境同時失效，
  repo 成為唯一可靠 checkpoint；docs-only。
- 2026-11-25 pass 5 @ c592a60 — watchlist 第五輪：WeMM-Video 入觀察
  （Apache-2.0 程式碼、逐卡 model card 待核）、Kandinsky 6.0 OmniNFT
  八 reward 配方 sync 分支方法參考、阿里下一代影片模型 11 月窗口仍無
  發布；docs-only。

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

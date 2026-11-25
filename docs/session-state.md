---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T01:28:12Z"
head: 0b7359be61f6cf185b7dc6066488606b74c54b41
head_verified: true
last_increment: "M7a 成雙：gen_cjk_images.py（GPU 雙路徑生成）＋score_cjk_text.py（tesseract OCR 文字準確率評分）；CI.yml 內容備妥但 token 無 workflow scope，待使用者上傳"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: none_pending_review
blockers:
  - "專案記憶 mount（/1a20b091…/areas/theama.md）連續多輪讀取失敗 → 本檔續為唯一可靠 checkpoint"
  - "沙箱執行環境連續多輪不可用（shell/ipython 全掛）→ 本地 pytest 無法跑；M7a 兩支腳本為獨立 script（不被 tests import），語法/執行驗證待沙箱恢復或 CI 上線後補"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；不一致先 list_commits 對帳再增量"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎；補驗證 gen_cjk_images.py / score_cjk_text.py（python -m py_compile + CPU 試跑 scorer）"
  - "沙箱若恢復：跑全量 pytest（預期 166 passed + 新腳本 compile check）；優先 ③ 程式小件續推（M2 audio DiT 骨架 / P1b 資料引擎管線）"
  - "沙箱仍未恢復：續做 ② watchlist（重點：阿里下一代模型 11 月窗口倒數、Omni-RL Judge arXiv id、Omni-RRM／VA-Judger checkpoint 授權）或 ④ repo 健康巡檢"
---

# Session State（每小時 checkpoint）

## 現況一句話

M1 / P1a / M2 程式全部落地、GPU 驗收全部待跑；M7a 成雙落地（24 條 CJK
seed 的生成腳本＋OCR 文字準確率評分腳本，GPU 驗收管線就位）；watchlist
已核查至第八輪；CI.yml 內容備妥待使用者上傳（token scope 限制）；
授權紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 ）

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
2. `updated_at` 距今超過 24h 視為 STALE：改以 `docs/model-roadmap.md`
   現況表 + 最近 commits 重建認知，並覆寫本檔。
3. 本檔不存在（首輪或誤刪）：直接依 model-roadmap 現況重建，不留空窗。
4. 恢復順序固定：本檔 → `docs/model-roadmap.md`（現況 + watchlist）→
   使用者當輪貼文（GPU 輸出等）。專案記憶 mount 可用時亦以本檔為準對帳。

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

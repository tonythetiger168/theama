---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T00:35:00Z"
head: c592a60fa1a85794a8f9477419d071bb7e191312
head_verified: true
last_increment: "watchlist 第五輪：WeMM-Video 入觀察（聯合 A/V judge 底座候選）、Kandinsky OmniNFT sync 分支方法參考"
ci_yml: not_uploaded            # 使用者尚未上傳 .github/workflows/CI.yml；repo 無 .github/ 目錄
issues: 0
gpu_outputs: none_pending_review
blockers:
  - "專案記憶 mount（/1a20b091…/areas/theama.md）本 session 讀取失敗 → 本檔暫為唯一可靠 checkpoint"
  - "沙箱執行環境不可用（shell/ipython 全掛）→ 無法跑 pytest / GPU smoke，本輪僅 docs 級增量"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；不一致先 list_commits 對帳再增量"
  - "沙箱若恢復：補寫專案記憶；優先 ③ 程式小件（M2 audio DiT 骨架 / P1b 資料引擎管線 / encode-score 腳本強化），CI.yml 上傳後跑 pytest"
  - "沙箱仍未恢復：續做 ② watchlist 輕量核查或 ④ repo 健康巡檢（CI.yml、issues）"
---

# Session State（每小時 checkpoint）

## 現況一句話

M1 / P1a / M2 程式全部落地、GPU 驗收全部待跑；watchlist 已核查至
2026-11-25 第五輪（WeMM-Video 列為聯合 A/V judge 觀察候選，不改 P1a/M2
配方）；CI.yml 待使用者上傳；授權紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

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

1. 更新 front matter（`head`、`last_increment`、`blockers`、`next_hour`）
   並在「本輪記錄」頂端加一行：`YYYY-MM-DD pass N @ <HEAD 短碼> — 摘要`。
2. 推送後做 blob SHA 驗證：`get_file_contents` 回傳 SHA 須與推送回傳
   一致；不一致立即回報，不得聲稱已存檔。
3. 「本小時無進度」不回寫本檔——`updated_at` 因此恆等於最後有進度時刻。
4. 只記狀態、決策與驗收數據；不貼 log 全文、權重、生成輸出。

## 紅線（任何增量不得削弱）

H3 / LTX-2.x / Qwen-Image-2.1 / FLUX-3-Klein / Krea-2 的權重、程式碼、
生成輸出一律不進本專案；防護為 `AVDataset.check_source` 黑名單 +
reward 註冊表白名單（`src/theama/reward/config.py`）。

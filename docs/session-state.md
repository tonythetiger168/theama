---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T10:33:00Z"
head: 1e64d0c18784af9249155f0742d82bda665abe75
head_verified: true
last_increment: "③ P1b pair/AV latent cache 對齊 text_cache 同款 retry budget（pair_failures.json／av_failures.json opt-in ledger、--max-retries/--reset-failures、skipped 列不計 encoded/reused/failed、預設行為逐字節不變）；tests +12（test_p1a／test_m2_av_latent_cache 各 6），CPU 253 passed；commit 1e64d0c1 六檔 blob SHA 驗證一致"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: m2_external_smoke_passed_reviewed   # PR #3 雙跑通過已判讀（pass 18）；M1/P1a/M2 internal 仍待 Claude 跑
blockers: []
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head；Claude 若已 push M1/P1a/internal-wan_blocks GPU 結果或 v0.2.0 tag，先消化再增量"
  - "pass 16–22 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十～十二輪（4fad1f38、918001c0+3f3fa0ac）／encode_texts retry budget（e1896295）／build_av_manifest（3f94b385）／pair+AV latent cache retry budget（1e64d0c1）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra，否則 decode/M7a 測試 skip）"
  - "③ 候選：下游消費者對 skipped 列的行為核查＋補強（joint_latent_loader／loader 遇到 skipped: true 列應略過或明確報缺，現無測試覆蓋）；或阿里 11 月窗口 11-30 關閉前最後 watchlist 覆核"
  - "model-roadmap.md 已推至 3f3fa0ac（watchlist 第十二輪：M5 終局目標＝超越 Seedance 2.5／Wan 3.0 登頂 AA with-audio 三榜），後續可正常從本地推送此檔"
  - "GPU 下一棒（Claude，依 status.md）：M1 runpod_m1_eval.sh 5→50 prompts→VBench ±0.02 → P1a（先驗 reward/score.py::_invoke 對真實 VideoReward）→ M2 internal wan_blocks＋LoRA smoke → 正式訓練前換 stable-torch image"
---

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0；**M2 外部 latent-bridge 真實權重 GPU smoke 已通過**（PR #3：
11-24 RTX 4090＋11-25 A40 全 runbook 復跑，詳 docs/status.md）；M1 RunPod
runner 落地待跑（runpod_m1_eval.sh）；P1b 鏈路補齊（manifest builder →
雙 cache → joint loader → runner → runbook 第 5 節），三 cache（text／
pair／AV）retry budget 全數對齊，253 CPU tests
全綠；GPU 驗收由 Claude 執行（使用者 11-25 擴權：其他底模＋HDL on
RunPod 亦可）；CI.yml 內容備妥待上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-25 pass 22 @ 1e64d0c1 — ③ P1b pair/AV latent cache 對齊 retry
  budget（本 session 增量，對岸 pass-20 next_hour 明列候選）：把
  text_cache（e1896295）的 opt-in failure ledger 移植到兩個 media
  cache——latent_cache.py 新增通用 ledger 核心（_ledger_load/_record/
  _clear/_reset：tmp+replace 原子寫、毀損降級 {}）＋pair_failures.json
  綁定 wrappers（key＝pair{index:04d} 檔名 stem，可 grep cache 目錄）；
  av_latent_cache.py 共用同一核心（av_failures.json，key＝
  sample{index:04d}）。EncodePlan/AVEncodePlan 加 skipped 欄、plan/run
  加 max_retries 參數（驗證正整數）；skipped 列寫入 manifest（skipped:
  true／retries／last_error）但不計 encoded/reused/failed（summary 另記
  skipped 數）；成功即清（暫態失誤不累積）；strict 先記錄再 raise；
  ledger 維護 best-effort（OSError 不得搞砸一次好 encode）；預設 None
  行為**逐字節不變**（無限重試、零 ledger I/O）。CLI 雙腳本
  （encode_pairs／encode_av_latents）加 --max-retries/--reset-failures
  （reset 於 planning 前），plan 輸出含 skipped 數、stderr 提示對應
  調整。測試 +12（test_p1a＋test_m2_av_latent_cache 各 6，鏡像
  text_cache 套件：ledger roundtrip 含毀損容忍／三 run 預算耗盡跳過／
  成功清除／無旗標零 ledger／strict 記錄後 reraise／非法 max_retries
  拒絕）。RESTORE 對帳乾淨（HEAD ba7ed628、無 Claude GPU commit、
  tags 空、issues 0、temp 無新上傳）；開工前 mtime 掃描無對岸活動中
  檔案。push_files 六檔一次 commit HEAD 1e64d0c1；**六檔 blob SHA
  全數驗證一致**（latent_cache dc359e64／av_latent_cache fb6cc7ff／
  encode_pairs 8320e3d6／encode_av_latents f8993dc6／test_p1a
  d2429e8a／test_m2_av_latent_cache 8ae8cfa6，本地 git blob 重算
  相同）——首次即 byte-exact。CPU 全套 **253 passed**（241+12）；
  雙 CLI --help smoke 通過；pytest 沙箱重置洗掉已重裝。下一個③候選：
  下游消費者（joint_latent_loader 等）對 skipped 列的行為核查。

- 2026-11-25 pass 22（對帳驗證側）@ 1e64d0c1 — **競態實例九（開工前
  攔截成功，第四次）＋獨立驗證**：本 session 開工即選中同一個③候選
  （pair/AV latent cache retry budget），動手寫碼前 mtime 掃描發現
  latent_cache.py（10:07Z）／av_latent_cache.py（10:09Z）為數分鐘內
  新檔——對岸正在實作同題，依紀律未寫一碼轉獨立驗證（競態實例
  七／八同型，零浪費）。對岸 10:21Z 推 code `1e64d0c1`、10:28Z 推
  checkpoint `23ce3e3b`（session-state blob ed14b929＝本地逐字一致）
  後核驗：六檔 blob SHA 本地 git hash-object 重算全數一致
  （latent_cache dc359e64／av_latent_cache fb6cc7ff／encode_pairs
  8320e3d6／encode_av_latents f8993dc6／test_p1a d2429e8a／
  test_m2_av_latent_cache 8ae8cfa6）；CPU 全套獨立重跑
  **253 passed**（241+12；於對岸推送前以本地最終檔先跑、推送後
  blob 對帳確認同一份內容）；scripts 全目錄對帳順帶確認
  check_av_manifest.py（49ac389a）等無漂移。設計審閱：通用
  _ledger_* 核心＋per-cache 綁定 wrappers、預設 None 零 ledger
  I/O、tmp+replace 原子寫／毀損降級 {}／strict 先記後 raise 皆與
  text_cache 對齊，無異議；觀察一項（非缺陷，不另開工）：skip
  判定純 index-keyed（ledger entry 存 prompt_id/prompt 但未參與
  比對）——pairs.jsonl reorder／編輯後同 index 換成不同 pair 會
  沿用舊 entry 被跳過；recovery 路徑已足（--reset-failures，或
  --no-resume 本就不讀 ledger），operator contract 本就要求
  reorder 後 --no-resume，維持現狀僅記錄。watchlist 第十三輪覆核
  （無 delta）：WeMM-Video 官方 GitHub 再確認程式碼 Apache-2.0、
  權重逐卡以 HF model card 為準（與第五輪記錄一致）；Omni-RRM
  仍僅 anonymous 4open 連結、授權未核實——兩觀察條目維持現狀
  不改 roadmap；四固定項同日第十輪已覆核，本輪不重查。
  RESTORE：HEAD ba7ed628→消化 1e64d0c1＋23ce3e3b；tags 空、
  issues 0、temp 無新上傳、CI.yml 仍待使用者。

- 2026-11-25 pass 21 @ 3f3fa0ac — watchlist 第十二輪（本 session 增量，
  依使用者拍板「**超過 Seedance 2.5 和 Wan 3.0**」）：AA with-audio 三榜
  榜首由對標板升格為 **M5 終局超越對象**——T2V w/audio 越過 Wan 3.0
  （1175）、I2V w/audio 越過 Seedance 2.5（1153）、editing w/audio 越過
  Wan 3.0（1196）；三者皆 closed/API-only，達成即 Apache-2.0 開放權重
  直接登頂三榜。差距盤點（開放權重現況計）：T2V −14 Elo（H3 1161→
  1175）、I2V −25（H3 1128→1153）、editing 開放最高僅第三（Hyperflow
  H3）；追趕路徑＝既有管線（P1a LoRA-DPO＋M2 audio DiT＋資料引擎擴量），
  M5 列改為保底閘（盲測 win rate／VBench）＋終局 stretch 兩層並列。
  model-roadmap.md 三處編輯（M5 列／watchlist 標頭／AV 榜 bullet 加
  ⑤），推送歷程 **SAVE incident #9**：首 push（918001c0）payload 手抄
  漂移一字（line 70「Wan 3.x」誤作「Wan 3.0」，同類 incident #6）——
  size 相等但 blob SHA 不符，byte cmp 定位後以本地正確檔覆推
  （3f3fa0ac），回傳 blob 1e4534ce＝本地 hash-object，curl raw 下載
  byte-identical。對帳：對岸本時段推 build_av_manifest（3f94b385，
  本地 session-state 已記、遠端 checkpoint 尚未推），本輪一併消化。
  RESTORE：HEAD 3f3fa0ac、無 Claude GPU 新 commit、tags 空、issues 0。

- 2026-11-25 pass 20 @ 3f94b385 — ③ P1b build_av_manifest（本 session
  增量）：A/V data-card manifest 此前只能手寫——新腳本
  `scripts/build_av_manifest.py` 掃 staged 目錄（<stem>.mp4/.mov/
  .webm/.mkv 配 <stem>.wav/.mp3/.flac/.m4a，caption 取自 <stem>.txt
  或 <stem>.json sidecar，sidecar 可逐 clip 覆寫 source）產出
  load_manifest 直接可吃的 jsonl（路徑相對 manifest 父目錄，roundtrip
  測試覆蓋）。**紅線建構期即攔**：逐 row 呼叫 dataset.check_source，
  forbidden source（minimax/hailuo/h3-/ltx/lightricks）整批 abort、
  不落盤——壞批次不會拖到 training launch 才炸。確定性：row 依 stem
  排序、副檔名撞車按固定優先級（.mp4>.mov>.webm>.mkv／
  .wav>.mp3>.flac>.m4a）與檔名 tie-break，與目錄列舉順序無關。
  ffprobe best-effort 補 duration/fps/width/height（無 ffprobe 或
  失敗則省略，可 --no-probe）；unpaired／無 caption／無 source 列
  skip reason 不半寫；輸出已存在須 --force。新測試檔
  test_m2_build_av_manifest.py +13（happy path 排序、sidecar 覆寫、
  三類 skip、無 source、forbidden 參數化 ×3、副檔名優先級、壞
  sidecar、main→load_manifest strict roundtrip、覆寫守衛、無
  ffprobe 降級）。**競態實例七（開工前攔截成功）**：開工前 mtime
  掃描發現對岸正在改 encode_texts.py／text_cache.py（09:02Z 數分鐘
  內新檔），依紀律避開同題改做 manifest builder；工作中對岸推
  `e1896295`（encode_texts retry budget）＋其 pass-20 checkpoint
  `f27f2a24`，本 session 推送前對帳消化，全套 241 passed
  （228+13）後推送，無衝突（不同檔）。雙檔 blob SHA 驗證一致
  （script 1c76fafe／test 0bb4887c，本地 git hash-object 重算相同）。
  RESTORE：HEAD adcf6b40 乾淨、無 Claude GPU 新 commit、tags 空、
  issues 0、temp 無新上傳。**順帶修復 checkpoint 檔歷史殘段**：遠端
  f27f2a24 版「## RESTORE 協議」重複出現兩次，且第一份 item 2 混入
  pass-13 entry 截尾（「…改以 `docs0 未打）、issues 0…199 passed…
  39225c0f／8b41f5de」）——應為過往 trim 10 輪時切錯邊界所致；
  本版依 SAVE 協議裁掉 pass-13 entry 後寫回單份乾淨 RESTORE 段。

- 2026-11-25 pass 20 @ e1896295 — ③ P1b encode_texts retry budget
  （本 session 增量，next_hour 明列候選）：theama.data.text_cache 新增
  選配 failure ledger——`--max-retries N` 下每次失敗計入
  text_failures.json（key＝cache 檔名 stem，operator 可直接 grep
  cache 目錄；tmp+replace 原子寫、毀損降級 {}）；達額度 caption 規劃
  為 **skipped**（encoder 永不呼叫、manifest 記 skipped/retries/
  last_error、summary 加 skipped 計數且不佔 failed/encoded）；
  成功即 clear（暫態失誤不累積）；strict 先記錄 ledger 再 raise；
  ledger 維護 best-effort（OSError 不得搞砸一次好 encode）。預設
  max_retries=None 行為**逐字節不變**（無限重試、零 ledger I/O）。
  CLI encode_texts.py 加 --max-retries/--reset-failures，plan 輸出
  含 skipped 數、stderr 提示對應調整。test_m2_text_cache.py +6：
  ledger roundtrip（含毀損容忍）、三 run 預算耗盡跳過、成功清除、
  無旗標零 ledger、strict 記錄後 reraise、非法 max_retries 拒絕。
  中途修一處設計（ledger key 由裸 digest 改檔名 stem，測試即規格）。
  RESTORE 對帳乾淨（HEAD adcf6b40、無對岸新 commit、temp 無新上傳）；
  推送前 mtime 掃描無對岸活動中檔案。push_files 三檔一次 commit
  HEAD e1896295；**三檔 blob SHA 驗證一致**（text_cache 4143a1a5／
  encode_texts 52a79957／test d7b0c038，本地 git hash-object 重算
  相同）——首次即 byte-exact。CPU 全套 **228 passed**（222+6）；
  py_compile＋--help smoke 通過。

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
  **第十一輪補（同輪加單，依使用者指示「根據 AV model leaderboard」）**：
  AA Video Arena with-audio 三榜快照入 roadmap 新條目（commit
  `4fad1f38`，blob 616597e8 推送回值＝本地、逐字節一致一次過）——
  T2V：Wan 3.0 1175 第一／H3 1161 第三（開放權重最高）／Kling 3.0
  錨 1000；I2V：Seedance 2.5 1153 第一／前十六款 H3 post-train；
  editing：Wan 3.0 1196 第一／Wan 2.7 Video 1074 第八（仍 API-only）。
  四點結論：M5 對標板＝AA with-audio 雙板；H3 post-train 生態證
  「強開放底座＋後訓練可近 frontier」→ theama 的 Wan 2.2＋LoRA-DPO
  為紅線內同構對應；Apache 譜系唯一頂級底座仍 Wan 2.2；VBench 開源
  前三在 Apache 線內等價追平並超出 Wan 2.2 自報 82.61。

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

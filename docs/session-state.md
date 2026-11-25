---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T13:50:00Z"
head: 6f478fd5c0c8e738222b9eb6af37033680f35384
head_verified: true
last_increment: "③ P1b 資料引擎：scripts/manifest_stats.py manifest 剖析器（row 計數、caption 覆蓋與字長分佈、duration_s 五桶、fps／解析度分佈、source mix、紅線能見度計數；deterministic JSON＋--out；可讀即 exit 0）＋9 tests；code commit 1b896e72，雙檔 blob c0f7cf31／f1968f66 首次即 byte-exact，CPU 264 passed"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 0
gpu_outputs: m2_external_smoke_passed_reviewed   # PR #3 雙跑通過已判讀（pass 18）；M1/P1a/M2 internal 仍待 Claude 跑
blockers: []   # pass-25 eval/roadmap 漂移已由本 session 覆正（ea97af39 eval aa87fac9／6f478fd5 roadmap 77e790e4，皆一次 byte-exact）
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head（pass-25 四連推：對岸 1596c716→abb440d2 兩連漂、本 session ea97af39／6f478fd5 覆正，最後皆已消化）；Claude 若已 push M1/P1a/internal-wan_blocks GPU 結果或 v0.2.0 tag，先消化再增量"
  - "pass 16–25 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十～十四輪（4fad1f38、918001c0+3f3fa0ac、129d5d67 PAVRM Apache 升格）／encode_texts retry budget（e1896295）／build_av_manifest（3f94b385）／pair+AV latent cache retry budget（1e64d0c1）／joint loader skipped 列明確報缺（cc85d330）／M7a eval seed 24→48（153be48c＋9d83f6c3）／M7a seed 48→100 v1.2（1596c716，漂移已由 ea97af39／6f478fd5 覆正）／manifest_stats 剖析器（1b896e72）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra，否則 decode/M7a 測試 skip）"
  - "③ 候選：阿里 11 月窗口 11-30 關閉前最後 watchlist 覆核（第十四輪已核 PAVRM，固定項無 delta）；或 M2 audio DiT 骨架／P1b 資料引擎下一小件（對齊 docs/model-roadmap.md 現況表）；M7a seed 已達 100 閘，下一小件改為 gen/score 管線對 100 列的實跑準備"
  - "GPU 下一棒（Claude，依 status.md）：M1 runpod_m1_eval.sh 5→50 prompts→VBench ±0.02 → P1a（先驗 reward/score.py::_invoke 對真實 VideoReward）→ M2 internal wan_blocks＋LoRA smoke → 正式訓練前換 stable-torch image"
---

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0；**M2 外部 latent-bridge 真實權重 GPU smoke 已通過**（PR #3：
11-24 RTX 4090＋11-25 A40 全 runbook 復跑，詳 docs/status.md）；M1 RunPod
runner 落地待跑（runpod_m1_eval.sh）；P1b 鏈路補齊（manifest builder →
雙 cache → joint loader → runner → runbook 第 5 節）＋ manifest_stats
剖析器（pass 25），三 cache（text／pair／AV）retry budget 全數對齊＋
下游 joint loader 對 skipped 列明確報缺；**M7a CJK eval seed 已達 100 條
驗收閘**（v1.2，對岸 pass-25 `1596c716`；對岸首推／自修兩連漂，
由本 session 以本地驗證版覆正——eval `ea97af39`／roadmap
`6f478fd5` 皆一次 byte-exact，詳 pass-25 記錄；對岸同日 watchlist
第十四輪將 PAVRM 官方碼核實 Apache-2.0 升格紅線內可參考），264
CPU tests 全綠；GPU 驗收由 Claude 執行（使用者 11-25 擴權：其他
底模＋HDL on RunPod 亦可）；CI.yml 內容備妥待上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-25 pass 25 @ 1596c716 — ③ M7a CJK T2I eval seed 48→100
  （v1.2，達 100 條驗收閘；pass-24 v1.1 的續擴，本 session 增量）。
  新 52 列（T2I-CJK-049…100）：十六類配比收斂（poster／sign 各
  11、ui／menu 各 8、product／book_cover／newspaper／ticket／
  packaging／receipt 各 6、sticker／banner 各 5、calendar／
  label／coupon／map 各 4）；難度梯度續推——民國年號
  （民國116年）、冒號（今日定食：照燒鮭魚／例行賽：龍象大戰）、
  直排（感恩回饋全面五折起／芒果雪花冰）、3 字短目標
  （營業中）、CJK×英數混排（維他命C發泡錠／主峰步道 3.2K）；
  本地生成即逐列機器驗證 expected_text⊆prompt、id 唯一、類別
  總數（並當場抓修三處草稿疊字）。test_m7a 計數 48→100＋類別
  覆蓋斷言九類→全十六類（expected_text⊆prompt 契約不變）；
  roadmap M7a 現況／中期行與 Ming diagnostic 註同步「達驗收閘
  規模」。RESTORE 對帳乾淨（開工 HEAD a4cc2948、無 Claude GPU
  commit、tags 空、issues 0、temp 無新上傳、mtime 掃描無對岸
  活動）；推送前再對帳無競態。CPU 全套 255 passed（當時樹）。
  **SAVE 事故第十二實例（本 session 側視角：同一檔連續兩次壞
  payload，依新紀律停手）**：首推 1596c716 eval 6 處疊字
  （004／010／038／046／066／071）＋roadmap 3 處（67／141／150），
  blob 驗證攔下（test 45837407 首次即 byte-exact）；fetch→diff
  定位後第二次推送 abb440d2 又帶新疊字（eval 7 處 021／045／
  052／071／076／090／096＋roadmap 3 處 121／141／297，其中
  071／141 為修復標的本身再疊）——兩敗觸發事故十一新紀律：
  停手不第三抄；fix-set 已在磁碟證明完備（遠端＋10 處單段刪除
  ＝本地 byte-exact），交對岸以共享本地乾淨檔覆推（blockers
  明列）。**補充教訓：本 session 對 24KB+ 全 CJK payload 的
  疊字率約每遍 6～10 處、與是否 fetch 錨定無關；對岸今日多次
  一次 byte-exact——長 CJK 檔推送宜常態分工給對岸，本 session
  做本地產出＋驗證。** 對岸同小時增量（競態實例十二對側，
  相鄰選題和平收場）：刻意避開 next_hour 熱門項、落地
  manifest_stats 剖析器（1b896e72，13:17:30Z，與 1596c716 差
  16 秒、檔案零交集），雙檔 blob c0f7cf31／f1968f66 首次即
  byte-exact；其 checkpoint（13:40Z 本地版）已記事故十二與
  修復處方，本 entry 合併入同一 checkpoint（遠端 checkpoint
  仍 5b881fdb，本版為合併版首推）。合併後全套（含對岸
  9 tests）獨立重跑 **264 passed**。

- 2026-11-25 pass 25 @ 1b896e72 — ③ P1b 資料引擎小件：
  `scripts/manifest_stats.py` manifest 剖析器＋9 tests（code commit
  `1b896e72`，雙檔 blob **c0f7cf31**／**f1968f66** 首次即
  byte-exact，目錄 listing 驗證）：check_av_manifest 是 pass/fail
  閘，此工具是 GPU encode 前的資料集側寫——row 計數、caption
  覆蓋率與字長分佈、probed 媒體 metadata（duration_s 五桶直方圖
  ＋總量／min／median／max、fps 與解析度分佈，讀
  build_av_manifest 的 ffprobe 選配鍵）、source mix、紅線能見度
  計數（複用 FORBIDDEN_SOURCES）；只讀 manifest 文字、不開媒體
  檔；容錯解析＋bad_json_lines 計數；deterministic JSON
  （sorted keys、固定 rounding）＋ `--out`；可讀即 exit 0
  （描述不判決）。CPU 全套 **264 passed**（255+9，連跑兩次
  穩定）。**競態實例十二（同小時相鄰選題，零交集和平收場）**：
  開工前 mtime 掃描乾淨（對岸最後活動 12:26Z），本 session
  刻意避開 next_hour 熱門項 M7a 48→100、選未列名的
  manifest_stats；工作中 13:09–13:11Z 偵測到對岸正在寫
  eval jsonl／test_m7a／roadmap（M7a 48→100）；雙方檔案零交集，
  對岸 13:17:14Z 推 `1596c716`、本 session 13:17:30Z 推
  `1b896e72`（16 秒差，自動接龍無衝突）。**SAVE 事故第十二
  實例（首例在對岸推送側，由本 session 驗證抓到，兩連漂未收斂）**：對岸
  1596c716 的遠端 eval jsonl blob 6076f987（25039B）≠ 其本地
  驗證版 aa87fac9（24997B）——fetch 比對定位五處疊字（row
  004「各燈籠各寫」、010「新增一一筆」、038「遊標示「遊
  戲區」」、066「書名圓，書名圓體」、071「門票特特寫」，
  合計約 42B；對岸自修時共記錄六處——多 row 046「感字字體」，
  本 session 掃描漏一）；roadmap 573a97ee（24367B）≠ 本地
  77e790e4（24352B，+15B）同推同機制，對岸記錄其 drift 三處
  （或或／核核實／授察，授權）；test_m7a 45837407（近 ASCII）
  byte-exact。對岸 13:24:47Z 推修復 `abb440d2`——**修復
  payload 再漂**：遠端 eval 8d551f93（25036B）／roadmap
  f6356bd1（24373B）皆 ≠ 目標 aa87fac9／77e790e4（本 session
  目錄 listing 核驗），whack-a-mole 兩連敗，與事故十一同型
  （全檔重抄每遍錯誤率獨立非零）。本地檔自檢乾淨：六處皆
  單寫、100 列 expected_text⊆prompt 契約成立、全套 264 兩次
  穩定（含對岸 v1.2 檔）、對岸修復期間未動本地檔。**收場
  （本 session）**：對岸依紀律停手並在合併 checkpoint 點名
  交棒（「交對岸以共享本地乾淨檔覆推」），本 session 接手
  各試一次——eval `ea97af39`（13:36:44Z，blob aa87fac9
  一次 byte-exact）、roadmap `6f478fd5`（13:40:04Z，
  77e790e4 一次 byte-exact），**事件收斂結案**。本日長 CJK
  檔推送戰績：對岸 2 敗後停手、本 session 2/2 一次過——
  與 pass-24（本 session 4 連敗、對岸救援一次過）正好互換，
  再次證明疊字率與 session 無關、純屬單次生成噪音；分工
  教訓修正為「**先看誰當日手順，連敗方停手交棒**」。
  **教訓（實例十二）：next_hour 明列的熱門項刻意避開、改做
  未列名小件，是本 session 首次主動避讓成功（非撞上才撤退）。**
  RESTORE 對帳乾淨（開工 HEAD a4cc2948、無 Claude GPU commit、
  tags 空、issues 0、temp 無新上傳）；推送前再對帳一次。
  pytest 沙箱重置洗掉已重裝；全套曾現兩次環境性零星 flake
  （p1a compile／m7a 各一次，重跑即綠，隔離重跑 10/10 綠，
  FUSE 讀取 8 次 md5 一致——記錄觀察，非程式缺陷）。

- 2026-11-25 pass 24 @ 9d83f6c3 — ③ M7a CJK T2I eval seed 24→48
  （v1.1；roadmap 明列「擴充到 100 prompts 後作為驗收閘」的第一步）。
  新 24 列（T2I-CJK-025…048）：補齊薄弱類別（newspaper／ticket／
  receipt／banner 各 +2、sticker／packaging／menu 補量）＋新增
  calendar／label／coupon／map 四類；難度梯度——數字混排
  （2027年1月）、冒號標點（今日例湯：苦瓜排骨）、直排（特製豚骨
  拉麵／霧中車站）、3 字短目標（美食區）。test_m7a 計數 24→48
  ＋新契約斷言「expected_text 必須逐字出現在 prompt」（48 列全數
  成立，防 scorer 目標與生成指令漂移）＋類別覆蓋斷言擴至九類。
  roadmap M7a 行與 Ming diagnostic 註同步更新。RESTORE 對帳乾淨
  （開工 HEAD 01926a10、無 Claude GPU commit、tags 空、issues 0、
  temp 無新上傳、開工前 mtime 掃描無對岸活動中檔案）；推送前再
  對帳無競態。CPU 全套 255 passed。push_files 三檔一次 commit
  153be48c；test／roadmap blob 首次即 byte-exact（9432fb21／
  d3870969）。
  **SAVE 事故第十一實例（最嚴重：同一檔連續四次壞 payload，
  對岸救援收場）**：eval jsonl（48 行、12KB 全 CJK）首推
  153be48c 轉錄疊字於 row 034（「紅色布前的紅色布條」，blob
  驗證當場攔下）；三次修復推送各自引入新隨機疊字（fc32f727
  壞 008/013/022/045 → c3b09ca8 壞 010/048 → 42b3462d 壞
  006）——每次 fetch-diff 定位皆正確，但全檔重抄的每遍錯誤率
  獨立非零（1～4 列／遍），whack-a-mole 不收斂。對岸 12:20:52Z
  以正確本地檔覆推 9d83f6c3（eval blob c96b7cbb＝本地
  hash-object）收尾；本 session 第五次 payload（又帶三處新疊字）
  恰因 stale-SHA 被拒——**stale-SHA 閘這次反向保護了正確版本**。
  **教訓（新）：同一長檔連續兩次 blob 驗證失敗即停手——第三次起
  改用「fetch 遠端→本地 diff→只信任單行修補」或等待對岸乾淨副本，
  禁止無限全檔重抄；CJK 長檔全量重抄錯誤率不隨謹慎歸零。**
  對岸同分鐘增量：129d5d67（12:14:14Z）watchlist 第十四輪——
  PAVRM 官方 repo Tencent-Hunyuan/HY-Video-PRFL 授權核實
  Apache-2.0，由觀察升格紅線內可直接參考實作（latent 空間打分
  免 VAE decode、PRFL ≥1.4× 加速、骨幹 Wan2.1 Apache 譜系）；
  本地 roadmap 早已含該段（對岸工作期間寫入），本 session 推送
  的 roadmap 版本 d3870969 與之合體無衝突。三壞 commit 留史
  不強推。HEAD 現為 9d83f6c3。

- 2026-11-25 pass 23 @ cc85d330 — ③ P1b 下游消費者對 skipped 列的
  行為核查＋補強（本 session 增量，pass-22 next_hour 明列候選）：
  核查結論——三 cache planner 對 skipped 列皆安全（reuse 一律要求
  磁碟上 .pt 驗證通過，skipped 列無 .pt 永不誤用；manifest compaction
  會覆寫舊 skipped 列），唯一缺口在下游：joint_latent_loader 的
  join 把 skipped 列（無 latent keys）掉進誤導性的 missing_keys
  桶。補強：join 新增明確 drop reason skipped_av_row／
  skipped_text_row（先於 caption/missing_keys 判定）——fail-closed
  serving 行為不變（skipped 列本就不可能上桌），只改回報真因；
  train_joint_av.py 本來就印 dropped dict，真因自動進 training log
  與 runbook dropped=0 閘。測試 +2（test_m2_joint_latent_loader）：
  雙 manifest 各帶一條 skipped 列（預算 1、兩 run 耗盡）→ join
  records [0,3]、dropped＝{skipped_av_row:1, skipped_text_row:1}，
  且斷言 skipped 列無 latent keys、非 error 列；lifecycle——
  reset_failure_ledger 雙側＋重燒成功 → records [0,1,2,3]、
  dropped {}。RESTORE 對帳乾淨（HEAD c1d93362＝pass-22 雙 entry
  checkpoint、無 Claude GPU commit、tags 空、issues 0、temp 無新
  上傳）；開工前 mtime 掃描無對岸活動中檔案（對岸最後活動 10:33Z），
  推送前再對帳無競態。push_files 雙檔一次 commit HEAD cc85d330；
  **雙檔 blob SHA 首次即 byte-exact**（joint_latent_loader 723a2506／
  test 91981524，本地 git hash-object 重算相同）。CPU 全套
  **255 passed**（253+2）；pytest 沙箱重置洗掉已重裝。watchlist
  本輪不重查（第十三輪今早無 delta，阿里窗口 11-30 才關閉）。
  下一個③候選：阿里 11-30 窗口關閉前最後覆核；或 M2 audio DiT
  骨架小件。

- 2026-11-25 pass 23 @ cc85d330（對帳驗證側）— **競態實例十（工作
  中途撞車，本 session 主動撤退、零浪費收場）**：RESTORE 乾淨
  （HEAD c1d93362、無 Claude GPU commit、tags 空、issues 0、temp
  無新上傳、開工前 mtime 掃描無對岸活動）後，本 session 獨立選中
  同一個③候選（pass-22 next_hour 明列的 skipped 列下游核查），
  完成缺口分析（三 cache planner 對 skipped 列皆安全——reuse 恆
  要求磁碟 .pt 驗證；唯一缺口＝join 誤報 missing_keys）並開始
  實作：docstring 第一處 edit 落地（11:08Z）後，第二處 edit 失敗
  ——檔案已含對岸 11:09:55Z 寫入的 skipped 處理碼（對岸工作中，
  其 test 檔 11:10:49Z 亦落地）。依紀律立即停止自版：11:11:30Z
  精準撤回自己的 docstring hunk（檔案還原至對岸內容）、不碰
  對岸檔、不在範圍不明時擴散至 dataset.py，轉獨立驗證。對岸
  11:15:17Z 推 code `cc85d330`（雙檔、parent c1d93362 乾淨、
  +15/−1 與 +74/−1 無夾帶）、11:20:49Z 推 pass-23 checkpoint
  `00a90fdb` 後核驗：雙檔 blob 本地 hash-object 重算一致
  （loader 723a2506／test 91981524，14008／14416 bytes），
  checkpoint blob 64059201 一致（size 26445）；CPU 全套獨立
  重跑 **255 passed**；本地樹無其他 drift。設計審閱無異議：
  skipped 判定先於 caption／missing_keys 順序正確、fail-closed
  serving 不變僅改回報真因；兩新測試斷言不空洞（先驗 manifest
  確帶 skipped 列、無 latent keys、非 error 列；lifecycle 覆蓋
  reset→重燒→records 全回補 dropped {}）。補充核查（關閉審計
  迴路，結論無需動碼）：data-card 側（dataset.py／loader.py）
  無 skipped 列生產者（build_av_manifest skip 不半寫），silent
  -accept 僅理論風險；summarize_encoding 直接餵 skipped 列會誤計
  encoded，但三處 call site 皆先濾 plan.skipped。**教訓（實例十）：
  熱門待辦即使開工前掃描乾淨，對岸仍可能同分鐘開工——長讀檔後
  動手前可再掃一次 mtime；edit 失敗（anchor 不符）本身就是撞車
  訊號，應先讀檔確認再決定撤退或續行。**

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

- 2026-11-25 pass 17 @ a7cb7697 —（trim：pass 17 雙則與 pass 16
  全文移出，事實存於專案記憶第十六／十七輪存檔：score manifest
  provenance a3067d9b 220 passed、Ming 第三路徑 plumbing
  4e1d202f 217 passed、runbook §4.1 a7cb7697、SAVE 事故第五實例
  尾換行教訓、競態實例五開工前攔截＋工作中疊加。）

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

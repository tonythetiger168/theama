---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-25T22:10:00Z"
head: 3ecce283ec4ff807ed663327790a1a49161ab6d1
head_verified: true
last_increment: "本輪＝② watchlist 第十六輪：Wan-Animate-2 GitHub repo id 補登（Wan-Video/Wan-Animate-2，392★）＋Wan 3.x 日期補精確（08-06 beta／08-24 上線，API-only）＋阿里下一代影片模型 11 月窗口 T-4 仍無發布＋Qwen-Image 四項無變化＋VA-Judger 成品上架升級觀察（62.30% 偏好率，授權待核）＋IsoVRBench 交叉驗證板觀察。SAVE 事故第十七實例隨本檔覆推收斂（本地乾淨版取代遠端 d34f18c9 三處「腦本」錯字版）；事故第十八實例（31KB roadmap）：首推 b00eff38 兩處漂移、覆推 3ecce283 byte-exact（blob b5c676d1）。（前輪：對帳驗證側消化對岸 P1b joint-cache 預檢閘 c3756605——三檔 blob byte-exact、CPU 387；roadmap 現況表同步 a54b2e3c 由本側代推）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 1   # #4 RunPod: M2 internal wan_blocks + LoRA real-weight smoke（Claude GPU 任務書，11-25 17:20Z 開立，runbook §0 入口）
gpu_outputs: m2_external_smoke_passed_reviewed   # PR #3 雙跑通過已判讀（pass 18）；M1/P1a/M2 internal 仍待 Claude 跑
blockers: []   # pass-25 eval/roadmap 漂移已由本 session 覆正（ea97af39 eval aa87fac9／6f478fd5 roadmap 77e790e4，皆一次 byte-exact）
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head（pass-36：② watchlist 第十六輪 3ecce283——CPU 387）；repo 已轉 public（匿名 api.github.com 200），拉檔可走 curl raw 零轉錄路線；對岸若推新 checkpoint 或 v1.3 簡繁對照 seed，先消化再增量；Claude 若回報 issue #4 GPU 結果或 v0.2.0 tag，先消化再增量"
  - "pass 16–33 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十～十五輪（4fad1f38、918001c0+3f3fa0ac、129d5d67 PAVRM Apache 升格、42ec2a64）／encode_texts retry budget（e1896295）／build_av_manifest（3f94b385）／pair+AV latent cache retry budget（1e64d0c1）／joint loader skipped 列明確報缺（cc85d330）／M7a eval seed 24→48（153be48c＋9d83f6c3）／M7a seed 48→100 v1.2（1596c716，漂移已由 ea97af39／6f478fd5 覆正）／manifest_stats 剖析器（1b896e72）／M7a 跨模型 summary 比較器（d4e8d0e3＋773f61e9）／pairs_stats 偏好對剖析器（19881f0a）／pair-latent 預檢閘＋train_dpo fail-fast（afde66af）／score_eval_set.collect() 測試（59163737）／scores 預檢閘＋make_pref_pairs fail-fast（3840eb88）／M7a scorer 簡繁分軌（5d1d26a5）／M4 HLS fMP4/CMAF 分發包 packager（79191bc3）／pairs.jsonl 預檢閘＋encode_pairs fail-fast（39295734＋9b41f3bb）／M2 CFG 文字條件 dropout（82d18e37）／M4 DASH manifest 兼出（12a5eabe）／status.md handoff 入口補齊 pass 19–33（d1038099）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra＋chi-sim，否則 decode/M7a 測試 skip）"
  - "③ 候選：P1b 資料引擎下一小件（對齊 docs/model-roadmap.md 現況表）／M7a v1.3 簡繁對照 seed 擴充（100 條補同語意簡中成對＋script 標籤——對岸可能已在進行，開工前必查 mtime 與遠端）；M2 CFG 訓練半邊已落地（82d18e37），推理期 CFG sampler 待 GPU 訓練出帶 null 列的 checkpoint 再議；M4 分發層 HLS＋DASH 皆落地（79191bc3／12a5eabe）；阿里 11 月窗口 11-30 關閉——最近覆核為第十五輪（2026-11-26，無 delta），窗口過後降為每數輪抽查"
  - "GPU 下一棒（Claude，依 status.md＋runbook §0）：issue #4 M2 internal wan_blocks＋LoRA real-weight smoke → M1 runpod_m1_eval.sh 5→50 prompts→VBench ±0.02 → P1a（先驗 reward/score.py::_invoke 對真實 VideoReward）→ 正式訓練前換 stable-torch image"

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0；**M2 外部 latent-bridge 真實權重 GPU smoke 已通過**（PR #3：
11-24 RTX 4090＋11-25 A40 全 runbook 復跑，詳 docs/status.md）；M1 RunPod
runner 落地待跑（runpod_m1_eval.sh）；P1b 鏈路補齊（manifest builder →
雙 cache → joint loader → runner → runbook 第 5 節）＋ manifest_stats
剖析器（pass 25），三 cache（text／pair／AV）retry budget 全數對齊＋
下游 joint loader 對 skipped 列明確報缺；P1a 側 pairs_stats 剖析器
（pass 27）＋ pair-latent 預檢閘 check_pair_latents＋train_dpo
fail-fast 接線（pass 28）＋ score_eval_set.collect() 測試八項
（pass 28）＋ scores.jsonl 預檢閘 check_scores＋make_pref_pairs
fail-fast 接線（pass 30）＋ pairs.jsonl 預檢閘 check_pairs＋
encode_pairs fail-fast（pass 32：不丟列續跑——pair index 為
cache 鍵，壞檔拒絕啟動由 make_pref_pairs 重建）＋ **M7a scorer
簡繁分軌——seed script
標籤 loader＋逐列 tesseract pack＋缺 pack pre-flight＋summary
per_script 拆帳（pass 31，承接 11-26 拍板：M7a 驗收閘改 zh-Hans
優先、zh-Hant 並行統計不擋放行）**；**M7a CJK eval seed 已達 100 條
驗收閘規模**（v1.2，對岸 pass-25 `1596c716`，詳 pass-25 記錄；
v1.3 簡繁對照擴充待排）；**11-26 交付格式拍板：HDR10/HLG＋
HEVC/AV1 提前至 M4 交付（獨立賣點）＋ HLS fMP4/CMAF 分發層
（M6b 實時線走 LL-HLS/WebRTC）**＋ HLS 分發層 packager 落地
（pass 31 本 session：inference/hls.py＋scripts/package_hls.py，
逐 rung ffprobe 驗證不過不交付）＋ **M2 CFG 文字條件 dropout
落地（pass 33 本 session：null 列訓練機制＋雙 yaml 0.1——推理期
modality-aware CFG 的訓練半邊）**＋ **M4 DASH manifest 兼出落地
（pass 33 對帳驗證側：inference/dash.py＋scripts/package_dash.py——
同批 fMP4/CMAF segment 兼出 DASH、零重編碼、manifest 本體 ffprobe
驗證不過不交付）**；375 CPU tests 全綠；GPU 驗收由
Claude 執行（runbook §0 開 pod SOP 落地；issue #4＝M2 internal
wan_blocks＋LoRA real-weight smoke 任務書）；CI.yml 內容備妥待
上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-26 pass 36 @ 3ecce283（本 session）— ② watchlist 第十六輪
  落地：Wan-Animate-2 GitHub repo id 補登 `Wan-Video/Wan-Animate-2`
  （官方 repo 392★，與 HF 三 id 併列 M6a 底座完整索引）；Wan 3.x
  日期補精確（2026-08-06 public beta／08-24 正式上線，API-only、
  無 license 文本，atlascloud 08-26 逐 33 repo 佐證）＋阿里下一代
  影片模型 11 月窗口 T-4 仍僅 09-22 雲棲預告無發布；Qwen-Image
  四項覆核無變化（2.1 研究授權紅線／2512 Apache／3.0・3.1
  hosted-only）；reward 線兩筆升級觀察——VA-Judger 成品已上架
  （GitHub ShareLab-SII/VA-Judger＋HF 權重與 Bench，post-train
  人類偏好率 62.30% vs OmniNFT 27.63%，授權因 HF API 不可達
  待核，若相容列 M2 A/V reward 首選候選）＋IsoVRBench（arXiv
  2610.17234，無人類回饋 RM 評測板，VideoScore v1.1 60.1／
  VisionReward 72.6／GPT-4.1 76.0 vs Human 100）列 scorer 交叉
  驗證觀察。CPU 全套 **387 passed** 重跑（docs-only 輪）；RESTORE
  乾淨（HEAD 4a251576＝記憶、無對岸活動、temp 無新 GPU 輸出、
  issue #4 無留言、tags 空）。**SAVE 事故第十八實例（31KB roadmap
  長檔）**：首推 b00eff38 兩處漂移（249 行「目號錄」多字／377 行
  分號全半形），覆推 3ecce283 byte-exact（blob b5c676d1）。本檔
  推送同時收斂**事故第十七實例**（遠端 d34f18c9 三處「腦本」
  錯字版由本地乾淨版取代）。trim：pass-28@afde66af 一條移出
  （事實存專案記憶），維持 11 entries。trim 錨點碰撞小事故：舊 entry
  內文恰含「## RESTORE 協議`」字樣，index 提前命中致段重複——已以
  section 全標題「（每輪第 0 步，依序執行）」修復；紀律新增：trim／
  切除錨一律用 section 全標題，禁用子串。

- 2026-11-26 pass 35 @ c3756605（對帳驗證側，本 session）— 對岸增量
  `c3756605`（P1b joint-cache 預檢閘）獨立驗證全過＋③ roadmap 現況表
  本地同步。開工 RESTORE 發現對岸 21:17Z 剛落地 code：三新檔——
  `src/theama/data/joint_cache_check.py`（純函數 profile_manifest
  七類分類，precedence 逐項 mirror latent_cache.load_reusable_rows；
  zero_filled_dropped 六 reason 零填充含 pass-23 的 skipped_*_row）＋
  `scripts/check_joint_latents.py`（runbook §5.1「joint cache: N
  samples (dropped: 0)」驗收閘的獨立 CPU 版：served=0 永遠 exit 1、
  --allow-dropped 降級、--validate-files 實載每個 .pt 抓
  corrupt/non-finite、exit 1 不寫 json-out）＋
  `tests/test_m2_check_joint_latents.py` +12（純函數四＋CLI 七——
  含 validate-files 壞 .pt 定位到 index:audio 列、stdout/json-out
  連跑 byte-identical）。驗證：三檔 curl raw 回寫本地 blob
  **a1e1d4f4**／**25185c09**／**8d453c0c** byte-exact；CPU 全套
  **387 passed**（375+12）獨立重跑通過；設計審閱無異議——
  load_reusable_rows 逐行對照 precedence 完全一致（bool index
  拒絕亦一致）、MANIFEST_NAME 常數正確；觀察一項（非缺陷僅
  記錄）：profile 的 kept_rows 計行數、loader 對 duplicate index
  後者勝，兩者資訊粒度不同非契約衝突。**競態實例十七（接龍
  零浪費）**：對岸佔走 next_hour 明列的 P1b 熱門題（其本地樹
  05:10Z 批次檔案＝本 session 共享樹目擊），本 session 未動一碼
  轉獨立驗證。③ roadmap 現況表本地同步（不推，交對岸）：日期
  →11-26、P1a 行補預檢閘鏈＋retry budget＋pairs_stats、**新增
  P1b 行**（資料引擎全鏈＋joint-cache 閘）、M2 行補 CFG dropout、
  M4b 行補 HLS/DASH packager 落地、M7a 行補 scorer 簡繁分軌＋
  Ming 第三路徑；本地 blob **185f22b4**（29306B），與遠端
  28615B 版 diff 僅現況表段兩 hunk、六 heading 完整。RESTORE
  其他：遠端 session-state 仍 3c371a6b（對岸 checkpoint 未推）、
  issue #4 無留言、tags 空、無 .github、temp 無新 GPU 輸出。
  trim：pass-27@b87fa1c9 一條移出（事實存專案記憶第二十七輪
  存檔），維持 11 entries。

- 2026-11-26 pass 34 @ d1038099（本 session）— ④ repo 健康／docs 小件：
  status.md handoff 入口補齊（code/docs commit `d1038099`，blob
  **5c871010**＝本地 hash-object、10142B byte-exact，curl raw＋cmp
  雙驗）。缺口——status.md 為 AGENTS.md 指明的新接手入口，卻停在
  pass-18（PR #3）快照：milestone 表無 P1b／M4／M7a 行、P1a 行無
  預檢閘鏈、M2 行無 CFG dropout。更新：milestone 表加 P1b 資料引擎
  （manifest→雙 cache→joint loader→runner＋runbook §5）／M4 分發層
  （HLS fMP4/CMAF＋DASH 兼出＋HDR/HEVC/AV1 mux）／M7a（seed v1.2
  達閘、scorer 簡繁分軌、Ming 第三路徑 diagnostic-only）三行，P1a
  行補 check_scores→make_pref_pairs／check_pairs→encode_pairs／
  check_pair_latents→train_dpo 三段 fail-fast 鏈，M2 行補 CFG
  dropout；表後補 375 tests／repo public／M5 north-star 三句；
  next steps 第 3 項補 fail-fast 閘、第 4 項掛 issue #4。歷史
  smoke 段（A40／4090 雙表、pod SOP）逐字不動。RESTORE 對帳乾淨
  （開工 HEAD 9dc80a6c＝記憶、session-state blob 3c371a6b＝本地
  一致、issue #4 無留言、tags 空、遠端無 .github、temp 無新 GPU
  輸出、mtime 第二閘無對岸活動）；CPU 全套 **375 passed** 重跑
  確認（docs-only 輪，測試數基線第三網無對岸活動）。watchlist
  本輪不重查（第十五輪無 delta，降每數輪抽查）。
  **SAVE 事故第十六實例（顯示層誤導＋diff 方向誤判，四推收斂）**：
  首推 `32abcfcc`／覆推 `8297ef21`／三推 `771a2ba5` 皆帶同一處
  +8B——4090 表 line 60 payload 寫 `| Image / torch |`，磁碟原文
  實為 `| Image |`（原版 7fe2554d 兩表欄名本就不同：A40 表
  `Image / torch`、4090 表 `Image`）；**推送前 read_file 顯示層
  將磁碟的 `| Image |` 顯示為 `| Image / torch |`**（同 pass-31
  roadmap ／顯示為、事件），payload 照顯示抄入；首推 cmp 報差時
  誤判方向以為本地錯，二推照原 payload 再錯，三推意圖修正但
  payload 沿襲舊文未落實，四推 `d1038099` 以 sed 直讀磁碟核對後
  byte-exact 收斂。壞 commit 三枚留史不強推。**教訓：payload
  對帳以 shell（sed/grep）直讀磁碟為準、read_file 顯示層可能與
  磁碟不符；cmp 報差先查磁碟真值再判方向；意圖修正須在 payload
  中逐字落實並於推後 grep 目標行驗證。**
  trim：pass-27@19881f0a（pairs_stats）一條移出（事實存專案記憶
  第二十七輪存檔），維持 11 entries。

- 2026-11-26 pass 33 @ 12a5eabe（對帳驗證側，本 session）— ③ M4 分發層
  小件：DASH manifest 兼出落地（code commit `12a5eabe`）。缺口——
  11-26 拍板「同批 segment 兼出 DASH」，但 HLS packager 只產
  master.m3u8，DASH 播放器無 manifest 可吃。`src/theama/inference/
  dash.py`（parse_variant_playlist／parse_master_playlist 嚴格只認
  package_hls 形狀、EXTINF 數必須等於 segment URI 數；render_mpd
  deterministic 靜態 MPD——per-codec AdaptationSet 依首現序、
  Representation 層 SegmentTemplate 帶 rung 前綴 URI＋EXTINF 重建的
  精確 SegmentTimeline（1e6 ticks）；muxed segment 的 codecs 沿用
  video+AAC-LC 同 HLS；package_dash 四重對帳 master 宣告 vs
  variant playlist vs 磁碟檔案 vs 逐 rung ffprobe，寫 manifest.mpd
  後 ffprobe 再驗 manifest 本體，不過即 MuxError 不半交付；零重編碼、
  HLS bundle 唯讀）＋`scripts/package_dash.py`（--json-out
  deterministic；失敗 exit 1 不寫檔）＋tests/test_m4_dash.py +12
  （純函數九＋ffmpeg-gated 三：720p 雙 rung bundle E2E 且 HLS master
  不動、CLI --json-out 連跑一致、缺 master exit 1 不寫檔）。
  **三檔同 commit 首次即 byte-exact**：dash.py **7576ce4b**／
  package_dash.py **f32e4490**／test **b2135e1a**（inference／
  scripts／tests 三目錄 listing 全量對帳無夾帶）；CPU 全套
  **375 passed**（363+12）。競態實例十六（接龍模式再現）：開工
  RESTORE 對帳乾淨（HEAD 2e6b4f49），動手前 mtime 第二閘抓到
  對岸 04:06Z 活動中（M2 CFG dropout 五檔批次）——題目零交集續做；
  推送前再對帳對岸已落地 `82d18e37`＋`8fe9899d`＋`a03e9cfe`，
  本 commit 落於 `a03e9cfe` 之上。對岸 82d18e37 獨立驗證：CPU
  375＝363（其 commit 記錄值）+12、其五檔屬本地樹 04:06Z 批次、
  設計審閱無異議（p=0 短路不碰 RNG、null token 零向量不增
  checkpoint 參數、預設 0.0 逐位元不變）。**checkpoint 合體：對岸
  SAVE 事故十五兩推兩漂停手、本地乾淨版 2c9fff79 交棒——本 session
  以其為基 prepend 本 entry＋front matter head→12a5eabe＋trim
  pass-26 一條（事實存專案記憶第二十六輪存檔），維持 11 entries，
  推送之。** RESTORE 其他：issue #4 仍 OPEN 無留言（Claude 未回報）、
  tags 空、遠端仍無 .github（CI.yml 待使用者）、temp 無新 GPU
  輸出；watchlist 第十五輪（11-25 19:09Z）無 delta 本輪不重查。

- 2026-11-26 pass 33 @ 82d18e37（本 session）— ③ M2 audio DiT 骨架
  小件：**classifier-free guidance 文字條件 dropout 落地**（code
  commit `82d18e37`）。缺口：audio_dit.py 架構註記自始即列
  「modality-aware CFG」為設計一環，但全 repo 無任何 CFG 機制——
  訓練端若不打 null 條件列，推理期永遠無法做 audio 流 CFG（M2
  驗收與 M5 品質閘都需要）；GPU 正式訓練開跑前這是最後窗口。
  三件套：`train/train_joint_av.py` 純函數雙件——
  sample_cfg_drop_mask（Bernoulli mask；p=0／p=1 短路不碰 RNG，
  p=0 時訓練流與 pre-CFG 逐位元一致）＋ apply_cfg_dropout
  （masked 列置零向量 null token——state-free 不增 checkpoint
  參數；可傳自訂 null_cond (D,)／(1,D)；全 False mask 原物件
  回傳零拷貝零漂移；形狀／dtype fail-fast）＋ JointAVTrainer
  接線（新 config cfg_dropout 預設 0.0、建構期範圍驗證；batch
  可帶 bool cfg_drop_mask 覆寫採樣——對齊 audio_noise／
  video_noise 的 deterministic replay 慣例；mask 在加 time
  embedding 前套用，無條件列仍收各自的 timestep；metrics 新增
  cfg_dropped；wan_blocks 路徑只 drop audio 流 cond、video_cond
  來自 text cache 不動——正是 modality-aware 的意義）＋雙 yaml
  cfg_dropout: 0.1（trainer_config 直通，dev↔prod parity 契約
  不破）＋ runner 文件字串一行。tests/test_m2_cfg_dropout.py
  +13（純函數七＋trainer 接線五＋yaml 契約一）。CPU 全套
  **363 passed**（350+13，差額全為己出——測試數基線第三網
  無對岸活動）。**五檔同 commit 首次即 byte-exact**：trainer
  **d6e67651**／runner **5e5af6a8**／prod yaml **3fb2a513**／
  dev yaml **9aefd9e2**／test **9d9adae8**（curl raw＋cmp 雙驗
  零轉錄；全檔 payload 皆來自 read_file 全檔複製——SAVE 事故
  十四教訓落實）。RESTORE 對帳乾淨：開工 HEAD 2e6b4f49＝記憶、
  遠端 session-state blob 7da95962 一致（本地 66a05dc7 為含本
  session pass-32 entry 版——本輪由本 session 覆推收斂，執行
  pass-32 記錄的「下輪以覆推收斂」）、issue #4 OPEN 無新回報、
  tags 空、遠端無 .github、temp 無新 GPU 輸出、mtime 掃描無
  對岸活動；推送前再對帳無競態。watchlist 本輪不重查（第十五輪
  2026-11-26 無 delta；阿里窗口 11-30 關閉後降每數輪抽查）。
  trim：pass-25 M7a seed v1.2 一條移出（事實存專案記憶第二十五
  輪存檔），維持 11 entries。
  **SAVE 事故第十五實例（checkpoint 兩連漂，whack-a-mole 第二次
  實證，停手交棒）**：首推 `8fe9899e` 一處疊字（pass-31 entry 內
  「分發分發層」，+6B）——blob 驗證攔下；覆推 `a03e9cfe` 修掉
  該處但同遍長兩處新疊字（line 3 註解「設計原設計原則」＋SAVE
  協議標題「協議（每協議（每」，共 +21B，遠端 36909B）。diff
  確認僅此兩行有異：**front matter（head 82d18e37／last_increment
  ／next_hour）與 11 條 entries、RESTORE 協議錨標題逐字完好，
  RESTORE／trim 功能不受影響**。依事故十一／十三紀律兩敗停手
  不第三抄；本地乾淨版 blob **24cbaf74**（36888B）即本檔，對岸
  下一輪 checkpoint 以本檔為基覆寫即自然收斂；壞 commit 兩枚
  留史不強推。

- 2026-11-25 pass 32 @ 9b41f3bb（本 session）— ③ P1a 資料引擎小件：
  **pairs.jsonl 預檢閘＋encode_pairs fail-fast 落地**（code commits
  `392957343`＋修復 `9b41f3bb`）。缺口（pass-27 對岸 entry 已點名：
  encode_pairs 對壞行 JSONDecodeError／缺 key KeyError 層級崩——
  P1a 鏈最後一處裸 comprehension）：encode_pairs 裸解析 pairs.jsonl
  後位置式餵入 latent cache，缺 key／NaN/±Inf 分數（json 原生接受
  字面量）／倒挂 margin／self-pair／clip 缺檔的列穿過規劃，到 GPU
  encode 迴圈才炸或空燒 retry budget。三件套：
  `src/theama/reward/pairs_check.py` 純驗證器（stdlib only、injected
  exists 免磁碟；root-cause-first：missing_keys 含 bool 分數拒絕／
  non_finite_score 含選配 margin／self_pair／non_positive_margin／
  missing_video_file；deterministic report 五 reason 零填充＋
  summarize）＋ `scripts/check_pairs.py`（check_pair_latents 的輸入
  側對應閘：JSON summary、--show-errors、--no-files schema-only；
  unusable 列或 bad JSON 即 exit 1）＋ encode_pairs 接線（規劃前
  驗證、逐列原因印 stderr、任一 unusable 即拒絕啟動；缺檔由
  traceback 改為乾淨 exit 1；全合法檔行為逐字節不變）。**關鍵設計
  分歧（相對 scores／latents 閘）：不可丟列續跑**——latent cache 以
  pair index 為鍵，丟列令後續 index 全數位移，prompt_id 漂移閘擋
  不住同 prompt 兩 pair 的位移（舊 .pt 被錯配複用）——拒絕啟動＋
  以 make_pref_pairs 重建才是安全契約。tests/test_p1a_pairs_check.py
  +12（純函數七＋CLI 四＋接線一：encode_pairs 在 latent_cache
  import 前拒絕壞檔）。CPU 全套 **350 passed**（338+12）。
  **SAVE 事故第十四實例（首推四檔其三 byte-exact、test 檔
  payload 截斷＋一處 typo）**：392957343 的 test payload 只剩
  前半且帶 `[0"]` 語法錯——blob 驗證攔下（遠端 153c47d7≠本地
  7b7fd653）；module f841fb1d／script 97777442／encode_pairs
  8640c0e1 三檔首次即 byte-exact（reward／scripts／tests 三目錄
  listing 全量對帳無夾帶）；以本地完整檔覆推 `9b41f3bb` 一次
  byte-exact（**7b7fd653**）。**教訓：多檔 push 的每一檔 payload
  都必須來自 read_file 全檔複製——本例前段憑 write_file 記憶、
  後段只讀了 offset 100 起，接縫截斷；「憑記憶重組」之害不限於
  長 CJK 檔。** RESTORE 對帳：開工 HEAD c1005ea4＝記憶、blob
  ce1b9456 一致、issue #4 OPEN 無新回報、tags 空、遠端無
  .github、temp 無新 GPU 輸出；工作中對岸落地 watchlist 第十五輪
  `42ec2a64`（19:09Z，docs-only 零交集——roadmap 本地即
  byte-exact 無需回寫；② 本輪已由對岸完成，不重查）＋其
  pass-32 checkpoint `2e6b4f49`（session-state blob 7da95962＝
  本地 hash-object）——競態實例十五（工作中途對岸落地，零交集
  接龍，同實例十四模式）。trim：pass-25 對岸 manifest_stats 一條
  移出（事實存專案記憶第二十五輪存檔），維持 11 entries。
  checkpoint 依 pass-30 先例留對岸覆推（本地檔已更新並
  hash-object 驗證）。

- 2026-11-25 pass 32 @ 42ec2a64（本 session）— ② watchlist 第十五輪
  覆核落地（docs commit `42ec2a64`，roadmap blob **74cfefbe** 首次
  即 byte-exact——curl raw＋cmp 雙驗零轉錄）：四固定項——Wan 3.x
  仍 API-only（atlascloud 2026-08-26 逐 33 repo 核查、evolink
  供應路由核查、11-15 第三方榜單快照三方一致；Wan 3.0 無 HF
  checkpoint／GitHub repo／license 文本，最新開放權重仍 Wan 2.2
  系）；Wan-Animate-2 repo id 無變化（base＋distilled 皆在 Wan-AI
  HF，Apache-2.0）；阿里下一代影片模型 11 月窗口 T-4（11-30
  關閉）中英文檢索仍無發布公告或權重；Qwen-Image 補產品線譜系
  資料點（juejin 2026-10-08 同題對比，B 級：2.0 架構換代未放
  權重→3.0 純閉源 API→2.1 回開源線但 Qwen Research License
  非商用；2512 為 20B 分支收官版、Apache-2.0 唯一可商用線）——
  紅線不變、M7a 選型不變。video reward 新觀察四筆（awesome-rm
  清單 2026-10-06 版，S 級）：**Reward Lightning**（latent RM
  與對抗蒸餾共享骨幹，同步對齊＋加速 few-step）、**GT-SVJ**
  （生成器即 energy-based judge，self-supervised 不依賴外部
  VLM，授權暴露面小）、**Reward Forcing**（reward-guided DMD
  串流蒸餾，M6b 串流線方法參考）、**RewardDance**（reward
  scaling 規模律與抗 hacking）——授權／權重均未核實，
  VideoReward 首選不變，維持觀察。RESTORE 對帳乾淨（開工 HEAD
  c1005ea4＝記憶，session-state blob ce1b9456＝本地 hash-object；
  issue #4 OPEN 無新回報、tags 空、遠端無 .github、temp 無新
  GPU 輸出、mtime 掃描無對岸活動）；推送前再對帳無競態。CPU
  全套 **338 passed** 重跑確認（docs-only 輪，測試數基線第三網
  無對岸活動）。trim pass-24 一條移出（事實存專案記憶第二十四
  輪存檔），維持 11 entries。

- 2026-11-25 pass 31 @ 79191bc3（對帳驗證側，本 session）— **③ M4
  小件：HLS fMP4/CMAF 分發包 packager 落地（承接 11-26 交付格式
  拍板）＋對岸 pass-31（5d1d26a5）獨立驗證全過**：RESTORE 開工
  HEAD 3840eb88＝記憶，消化對岸三 docs commit（9db4c982 M7a 簡中
  優先／8a84f7cc HDR+AV1 提前至 M4＋HLS 分發層／8250bb22 runbook
  §0 開 pod SOP）＋pass-31 code 5d1d26a5。**新發現：repo 已轉
  public**（匿名 curl api.github.com/repos 200＋Accept raw 直拉
  成功——本地 docs 同步自此改走 curl raw 零轉錄風險路線；roadmap
  首抄 939575f5 差 3B，靠 raw 對比定位兩處：「線、」實為「線／」
  （MCP 顯示層誤導，兩次 fetch 皆顯示 、）＋漏字「不改動」；
  training-plan 70b8580e／runbook 09fc9852 轉錄一次過）。issue
  #4 已讀（Claude GPU 任務書：M2 internal wan_blocks＋LoRA
  smoke；指令旗標 --bridge-location wan_blocks／--bridge-lora-rank
  現有程式碼全支援，CPU 側無缺口）；tags 空、遠端無 .github、
  temp 無新 GPU 輸出。對岸 5d1d26a5 驗證：三檔 blob 本地
  hash-object 重算＝其 commit message 記錄值（1c00f852／
  02b3589a／856d9771）、CPU 全套獨立重跑 **326 passed**；其
  session-state 本地更新（099c31c1，含 pass-30 trim 重複段修復）
  明確把 checkpoint 推送交棒本 session。增量選題刻意避開對岸
  next_hour 熱區（M7a v1.3 seed／M2 骨架）：
  `src/theama/inference/hls.py`——LADDER 階梯表（2160p AV1→
  1080p HEVC→720p/480p H.264，HEVC 為 Safari fallback 必需層）＋
  build_ladder 純函數（高於源解析度的 rung 丟棄永不升頻、sub-480p
  單 H.264 rung 鉗位源高、寬等比且偶數、HDR10/HLG 只標頂 rung、
  複用 mux.HDR_FORMATS）＋ RFC 6381 codecs_string＋deterministic
  render_master_playlist＋package_hls（probe→逐 rung ffmpeg fMP4
  分段→逐 rung ffprobe 驗證 codec/height/audio/segments，驗證不過
  即 MuxError——驗證不過的包不交付）。**ffmpeg 5.x 陷阱（首輪
  測試抓到）：相對 -hls_segment_filename 對 CWD 解析而 playlist
  URI 照字面**——rung encode 改以 cwd=variant_dir 跑，playlist
  保留可攜相對 URI。scripts/package_hls.py CLI（--hdr hdr10/hlg
  只標頂 rung／--hls-time／--json-out deterministic JSON；任何
  失敗 exit 1 且不寫檔）。tests/test_m4_hls.py +12（純函數七：
  4K SDR/HDR 階梯形、rung 丟棄、sub-480 鉗位不升頻、等比偶數寬、
  bandwidth＋codec 字串、非法輸入拒絕、master golden render＋
  determinism；ffmpeg-gated 五：360p 單 rung E2E 驗證、720p 雙
  rung 相對 URI、CLI --json-out 連跑一致、缺檔 exit 1 不寫檔）。
  三檔同 commit `79191bc3` 首次即 byte-exact：hls.py **7357267d**
  ／package_hls.py **b446a755**／test **49a7b6d0**（inference＋
  scripts＋tests 三目錄 listing 全量對帳無夾帶）；CPU 全套
  **338 passed**（326+12）。推前 mtime 雙閘＋list_commits 再對帳
  無競態；順手清掉首輪除錯遺留的 repo 根 seg_00000.m4s（未推送）。
  checkpoint 由本 session 推送（對岸交棒）：本地檔＝對岸修復版
  099c31c1＋本 entry＋front matter head→79191bc3＋trim pass-23
  code 側一條（事實存專案記憶第二十三輪存檔），維持 11 entries。

- 2026-11-25 pass 31 @ 5d1d26a5（本 session）— ③ score 腳本強化：
  **M7a scorer 簡繁分軌（zh-Hans／zh-Hant script split）**——承接對岸
  17:49Z 拍板 docs（9db4c982：M7a 驗收閘改 zh-Hans 達 Qwen-Image 官方
  同量級、zh-Hant 並行統計不擋放行、seed 須簡繁對照）。缺口：scorer
  整場只選一支 tesseract pack（chi_tra 優先）——簡中列用 chi_tra OCR
  必然誤讀，且 summary 無簡繁分開統計，放行閘數字算不出來。三件套：
  `src/theama/image/eval.py` T2IEvalPrompt 加可選 script 標籤
  （"hans"／"hant"，載入時 fail-fast 驗證——打錯值即 raise 不靜默
  丟棄；v1.2 無標籤列照舊合法）＋ score_cjk_text.py（標籤列改用對應
  pack chi_sim／chi_tra+eng；開跑前 pre-flight 缺 pack 即 exit 1 不寫
  任何檔；resume 規則——標籤列前次用錯 pack 的分數丟棄重打
  （keep_scored_row），無標籤列行為不變；summary 新增 per_script
  區塊——hans／hant 零填充 fixed keys（scored／exact_hit_rate／
  mean_char_recall）＋部分標籤過渡期 untagged 計數；無標籤 seed
  完全不產 per_script key，v1.2 summary 逐字節不變）＋ docstring
  驗收句更新為 11-26 拍板。tests/test_m7a_script_split.py 新檔 +11
  （loader 標籤解析／拒絕、pack pre-flight、語言映射、resume 過濾、
  per_script 拆帳／零填充／untagged、summary 整合含 v1.2
  byte-stability、CLI 三——缺 pack 開跑前中止不寫檔／錯 pack 舊列
  以 chi_sim 重打／對 pack 舊列保留不 OCR）；刻意新開測試檔不碰
  test_m7a.py（對岸 v1.3 seed 工作零交集）。三檔同 commit
  `5d1d26a5` 首次即 byte-exact：eval.py **1c00f852**／scorer
  **02b3589a**／test **856d9771**（src/image＋scripts＋tests 三目錄
  listing 全量對帳無夾帶）；CPU 全套 **326 passed**（315+11）。
  **RESTORE 大事件：對岸三 docs commit 消化**——9db4c982（M7a 簡中
  優先）／8a84f7cc（HDR+AV1 提前至 M4＋HLS fMP4/CMAF 分發層，驗收
  +2 條 ffprobe 閘；M6b 走 LL-HLS/WebRTC）／8250bb22（runbook §0
  RunPod 開 pod SOP，issue 驅動）；三檔以遠端為準回寫本地 byte-exact
  （runbook 09fc9852 一次過／training-plan 70b8580e 首抄誤植一處
  「簡繁轉換、」hash 對帳攔下即修／roadmap ac716a95 一次過）。
  **repo 健康新事項：issue #4 OPEN（RunPod: M2 internal wan_blocks
  ＋ LoRA real-weight smoke——Claude GPU 任務書，runbook §0 入口）**；
  tags 仍空、遠端仍無 .github、temp 無新 GPU 輸出。**本地
  session-state 結構修復（未推送、本地攔修）**：本地檔在 pass-30
  trim 時踩中 pass-27 同型錨點事故——pass-26…pass-23 五條 entry
  整段重複（逐塊比對 byte-identical）＋pass-23 驗證側 entry 尾部帶
  207B orphan 殘段（「## RESTORE 協議` 為錨…」），檔案腫脹至
  52403B；程式化重建＝保留第一副本去重＋補做 pass-30 未竟的
  pass-22 trim（事實存專案記憶第二十二輪存檔）＋本輪 trim
  pass-23 驗證側一條（事實存第二十三輪存檔），grep 結構掃描驗證
  11 entries／四 section 齊／無殘段無重複。checkpoint 依分工仍留
  對岸推（遠端仍 b529a70c 五疊字版——front matter 完好、RESTORE
  功能正常）。

- 2026-11-25 pass 30 @ 3840eb88（本 session）— ③ score 腳本強化：
  **scores.jsonl 預檢閘＋make_pref_pairs fail-fast**（對岸 pass-28
  pair-latent 閘 afde66af 的 scores 側對應件，P1a 鏈最後一處裸
  comprehension）。缺口：score_eval_set 把失敗列（{error}）寫進
  scores.jsonl 供 resume 重試，make_pref_pairs 裸解析——一列
  KeyError 啟動即炸（build_pairs／coverage_report 皆直接索引
  prompt_id／overall）；NaN/±Inf overall（Python json 原生接受
  字面量）過 isinstance、靜默污染 margin 過濾與候選排序；混用
  兩個 reward model 的 scores 檔無任何阻擋（跨 scorer margin
  無意義）。三件套：`src/theama/reward/score_check.py` 純驗證器
  （stdlib only、無磁碟檢查——scores 列只引用路徑，encode_pairs
  自有 retry budget；root-cause-first：score_failed／missing_keys
  含 bool overall 拒絕／non_finite_score／reward_mismatch 可選
  閘——無 reward 標籤列依 load_successful_scores 慣例計
  videoreward；deterministic report 四 reason 零填充＋rewards
  直方圖排序輸出，過濾關閉時混 scorer 仍可見）＋
  `scripts/check_scores.py`（check_pair_latents 的 scores 側
  對應閘：JSON summary、--show-errors 逐列、--reward 單一 scorer
  強制；有 unusable 列或 bad JSON 即 exit 1）＋make_pref_pairs
  接線（build_pairs 前驗證，unusable 列帶原因印 stderr、全無
  可用拒絕啟動、--reward passthrough；全可用檔行為逐字節不變、
  檔序保留）。tests/test_p1a_score_check.py +12（純函數七＋
  CLI 四＋接線一：混檔丟 error 列仍成對、全壞檔拒寫輸出）。
  RESTORE 對帳乾淨（HEAD 3000b64e＝記憶、tags 空、issues 0、
  遠端無 .github、temp 無新 GPU 輸出、mtime 雙閘皆無對岸活動——
  對岸最後活動仍 pass-28 00:24）；刻意再避 next_hour 熱門項
  （M2 audio DiT 骨架／P1b 小件）。推送前再對帳無競態；四檔
  同 commit `3840eb88` blob **18db8d4f／7e665ad5／6ff5d2b1／
  a56d0be1 首次即 byte-exact**（reward／scripts／tests 三目錄
  listing 全量對帳無夾帶）。CPU 全套 **315 passed**（303+12；
  pytest 沙箱重置洗掉已重裝）。checkpoint 依實例十二／十三
  分工不自行推送（33KB+ 長 CJK 檔交對岸），本地檔已更新並
  hash-object 驗證。trim：pass-22 一條移出（事實存專案記憶
  第二十二輪存檔）。

- 2026-11-25 pass 28 @ 59163737（對帳驗證側，本 session）—
  **競態實例十四（同領域相鄰選題，工作中途對岸落地，零交集各自
  推送）＋本 session 增量＋afde66af 獨立驗證全過**：開工 RESTORE
  乾淨（16:03Z，HEAD f0e16afd＝記憶、session-state blob
  079e998b＝本地 hash-object、tags 空、issues 0、temp 無新
  上傳、mtime 掃描僅 pass-27 檔案無對岸活動）；刻意避開
  next_hour 熱門項（阿里覆核／M2 audio DiT 骨架／P1b 小件），
  選中未列名缺口 score_eval_set.collect()——P1a 評分鏈唯一
  未測的 CPU 邏輯（GPU pod 打分前的 clip→prompt 配對，
  gen_candidates 的 <out>/<prompt_id>/cand_<k>.mp4 佈局＋
  flat stem fallback）。動手前第二閘 mtime 掃描（16:09Z）
  乾淨；寫碼＋測試期間對岸於 16:06Z 落地 pair_latent_check
  四檔（同 P1a 領域相鄰題：train_dpo 上 GPU 前 fail-fast
  預檢閘）——全套跑出 303 passed（預期 295+8）時 +12 差額
  才偵測到對岸新 test 檔。檔案零交集（本 session 僅
  tests/test_p1a.py；對岸 module／script／train_dpo／新 test
  四檔），依實例十二先例各自推送：對岸 code `afde66af`
  （16:09Z）＋checkpoint `379433a4`（16:13Z）、本 session
  code `59163737`（16:19Z）。**教訓（實例十四）：mtime 雙閘
  之外的第三偵測網——全套測試數與預期基線的差額本身即是
  對岸落地訊號（283→303 中 +12 非己出）；零交集時不必撤退，
  比照實例十二接龍推送、checkpoint 由後到者合體。**
  afde66af 獨立驗證全過：四檔 blob 本地 hash-object 重算
  （c559d826／ba3f0f4a／755232b3／7ae3d4ee）＝其 commit
  message 記錄值；對岸 checkpoint 的 session-state blob
  **45e75505**＝本地 hash-object（33125B）；CPU 全套獨立
  重跑 **303 passed**（295+8）；設計審閱無異議——root-cause
  -first drop reason 與 joint loader 同款 fail-closed 契約、
  全可用 manifest 行為不變、--no-files schema-only 模式合
  CI 場景。本 session 增量（③ score 腳本強化）：collect()
  測試 +8——per-prompt dir 佈局／flat stem 佈局／nested
  fallback／unmatched＋non-mp4＋deep skip／duplicate stem
  父目錄優先／deterministic 排序連跑一致／empty dir／CLI
  無匹配 exit 1（於 scorer import 前返回，CPU 可驗）。首推
  payload 自查抓一處轉錄漂移（`"RuntimeError: boom"` 字串內
  冒號誤植字串外，未送驗前修正——GitHub 500 同時退回該次，
  重送即 byte-exact）：blob **a37ed40c**＝本地 hash-object、
  27045B 首次成功即 byte-exact。HEAD 現為 59163737。trim：
  pass-22 對帳驗證側一條移出（競態實例九＋1e64d0c1 驗證，
  事實存專案記憶第二十二輪存檔），維持 11 entries。

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

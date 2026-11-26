---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-26T13:18:54Z"
head: 210358e063904288f39c9089dff0dcac7b5c91d4
head_verified: true
last_increment: "本輪＝④ checkpoint 覆推解僵局：遠端 session-state 自 1e6bba46（pass-36 版）落後十六輪（37–52）一次補齊——front matter head→210358e0、issues 2→1（PR #5 已於 pass-48 使用者合併 06b122d4）、現況一句話事實更新（M1 5-prompt smoke 已過／套件數 454→570／issue #4 紅旗修復已落地待 GPU 重跑）、本輪記錄補登 pass 46–52 七則、trim pass 36–42（事實存專案記憶各輪存檔）維持 10 entries；CPU 570 passed docs-only 重跑確認。（前輪：pass-51 雙增量——watchlist 第十九輪四線無 delta 115c513b＋status.md follow-through 210358e0）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 1   # #4 RunPod M2 internal smoke（紅旗修復 b5321003…d455e3dc 已落地，Claude 重跑待回報——紅旗清除＝audio loss／grad norm 回個位數，過則解 bridge-warmup 阻塞）；PR #5 已於 pass-48 由使用者合併（06b122d4）
gpu_outputs: m1_5clip_and_m2_internal_smoke_passed_reviewed   # 2026-11-26 A40：M1 5/5 probe-valid（mean 842s/clip、peak 33.7GB）＋issue #4 internal wan_blocks+LoRA PASS(contract) 但 audio loss 79/grad 94 紅旗（bridge input 需 norm/scale 重跑才准訓練）；pass-41 已判讀＋PR #5 評審留言；VBench/50-clip/P1a 仍待 Claude
blockers: []   # pass-25 eval/roadmap 漂移已由本 session 覆正（ea97af39 eval aa87fac9／6f478fd5 roadmap 77e790e4，皆一次 byte-exact）
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head（pass-52＝④ checkpoint 覆推解僵局，遠端自 1e6bba46 落後十六輪 37–52 一次補齊）；issue #4 Claude 紅旗重跑結果（過則解 bridge-warmup 阻塞）／v0.2.0 tag／CI.yml，先消化再增量"
  - "pass 16–51 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十～十五輪（4fad1f38、918001c0+3f3fa0ac、129d5d67 PAVRM Apache 升格、42ec2a64）／encode_texts retry budget（e1896295）／build_av_manifest（3f94b385）／pair+AV latent cache retry budget（1e64d0c1）／joint loader skipped 列明確報缺（cc85d330）／M7a eval seed 24→48（153be48c＋9d83f6c3）／M7a seed 48→100 v1.2（1596c716，漂移已由 ea97af39／6f478fd5 覆正）／manifest_stats 剖析器（1b896e72）／M7a 跨模型 summary 比較器（d4e8d0e3＋773f61e9）／pairs_stats 偏好對剖析器（19881f0a）／pair-latent 預檢閘＋train_dpo fail-fast（afde66af）／score_eval_set.collect() 測試（59163737）／scores 預檢閘＋make_pref_pairs fail-fast（3840eb88）／M7a scorer 簡繁分軌（5d1d26a5）／M4 HLS fMP4/CMAF 分發包 packager（79191bc3）／pairs.jsonl 預檢閘＋encode_pairs fail-fast（39295734＋9b41f3bb）／M2 CFG 文字條件 dropout（82d18e37）／M4 DASH manifest 兼出（12a5eabe）／status.md handoff 入口補齊 pass 19–33（d1038099）／joint-cache 預檢閘 check_joint_latents（c3756605）／watchlist 第十六輪（3ecce283）／runbook §5.1/§5.2 預檢閘接線＋status.md P1b 行（7a6446e0）／scores.jsonl 剖析器 scores_stats（0bf42cd7）／M1 VBench 閘純函數化＋14 tests（22ca23aa）／runbook §2 P1a 預檢閘鏈接線＋status.md 合體（41e99d60＋2e165956）／gen_eval_set probe_ok＋resume 閘測試（cc00b918）＋status.md M1 行 follow-through（254e83b4）／M2 norm_video_kv 紅旗修復六 commit（b5321003…d455e3dc，＋9 tests，CPU 431）／watchlist 第十七輪（d2001b4a）／P1b cache 剖析器 joint_latents_stats（4f90cde5）／P1a score_eval_set CLI 測試＋resume 強化（867071d1）＋status.md follow-through（c424b897）／train_dpo CLI 測試（bef21e22）／train_joint_av _real CLI 測試（2256980f）／M7a score_cjk_text CLI 測試（4d926f6c）＋status.md（99b0a4e1）／make_pref_pairs CLI 測試（c3b661ca）／check_av_manifest CLI 測試（a216a3d1）／gate↔profiler 一致性測試（efd1af65）／watchlist 第十八輪（pass-48 抽查無 delta）／第十九輪（115c513b）／status.md follow-through（210358e0）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra＋chi-sim，否則 decode/M7a 測試 skip）"
  - "③ 候選：M7a v1.3 簡繁對照 seed 對岸進行中（pass-52 偵測 13:16Z 雙檔未推）——勿碰、推送後消化驗證（預期全套 579）／其他未列名 docs・資料面小件；**encode_av_latents CLI 測試仍為對岸明列項勿碰**；scripts 主鏈 CLI 契約測試已全線閉合（gen／encode／score／pairs／train 皆齊），剩 smoke_decode 等 GPU 殼腳本屬 Claude 道；watchlist 下次抽查 pass-54±（阿里 11 月窗口已關閉，每數輪抽查）"
  - "GPU 下一棒（Claude，依 status.md＋runbook §0）：**issue #4 紅旗修復重跑（HEAD≥d455e3dc，同一 smoke 命令免改 CLI，紅旗清除＝audio loss／grad norm 回個位數才解 bridge-warmup 阻塞）** → M1 50 prompts（~12h，先試 offload=False／volume 僅 ~5GB 須擴容或清舊檔）→ VBench ±0.02 → P1a（先驗 reward/score.py::_invoke 對真實 VideoReward）；v0.2.0 tag、CI.yml 仍待使用者"

# Session State（每小時 checkpoint）

## 現況一句話

版本 0.2.0；**M2 外部 latent-bridge 真實權重 GPU smoke 已通過**（PR #3：
11-24 RTX 4090＋11-25 A40 全 runbook 復跑，詳 docs/status.md）；M1 RunPod 5-prompt generation smoke 已通過（0bf42cd：5/5 probe-valid、mean 842s/clip、peak 33.7GB；50-prompt＋VBench 待跑）；P1b 鏈路補齊（manifest builder →
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
v1.3 簡繁對照對岸進行中——pass-52 偵測雙檔落地未推）；**11-26 交付格式拍板：HDR10/HLG＋
HEVC/AV1 提前至 M4 交付（獨立賣點）＋ HLS fMP4/CMAF 分發層
（M6b 實時線走 LL-HLS/WebRTC）**＋ HLS 分發層 packager 落地
（pass 31 本 session：inference/hls.py＋scripts/package_hls.py，
逐 rung ffprobe 驗證不過不交付）＋ **M2 CFG 文字條件 dropout
落地（pass 33 本 session：null 列訓練機制＋雙 yaml 0.1——推理期
modality-aware CFG 的訓練半邊）**＋ **M4 DASH manifest 兼出落地
（pass 33 對帳驗證側：inference/dash.py＋scripts/package_dash.py——
同批 fMP4/CMAF segment 兼出 DASH、零重編碼、manifest 本體 ffprobe
驗證不過不交付）**＋ **runbook §5.1/§5.2 接入 joint-cache
預檢閘 check_joint_latents（pass 37 本 session：燒 GPU 前攔
dropped 列／壞 .pt）**＋ **scores.jsonl 剖析器 scores_stats
落地（pass 38 本 session：P1a profiler 三件套補齊——配對產出
pairable／single／all_failed 預覽）**＋ **run_vbench VBench ±0.02 閘邏輯純函數化＋14 tests（pass 39 本 session：M1 GPU 驗收終點閘改 CPU 可驗）**＋ **P1b cache 剖析器 joint_latents_stats 落地（pass 43 本 session：閘之外的分佈剖析——shape mix／caption 共享／index 覆蓋／join 視圖）**；570 CPU tests 全綠；GPU 驗收由
Claude 執行（runbook §0 開 pod SOP 落地；issue #4 internal smoke 契約 PASS 但 loss-scale 紅旗——CPU 修復 norm_video_kv 已落地（b5321003…d455e3dc）、GPU 重跑待 Claude，過則解 bridge-warmup 阻塞）＋ **P1a／P1b／M7a 主鏈 CLI 契約測試全線閉合（pass 46–51：score_eval_set 867071d1／train_dpo bef21e22／train_joint_av 2256980f／score_cjk_text 4d926f6c／make_pref_pairs c3b661ca／check_av_manifest a216a3d1＋gate↔stats 一致性 efd1af65）**＋ watchlist 第十八／十九輪無 delta（pass-48／115c513b，阿里窗口 11-30 已關閉）；CI.yml 內容備妥待
上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-26 pass 52 @ 210358e0（本 session）— ④ checkpoint 覆推解僵局：
  **遠端 session-state 自 1e6bba46（pass-36 版）落後十六輪（37–52）一次
  補齊**——互讓僵局第七輪後，本 session 依分工（hourly_agent 職掌
  session checkpoint）主動收斂：front matter head→210358e0、issues
  2→1（PR #5 已於 pass-48 使用者合併 06b122d4）、現況一句話事實性
  更新（M1 5-prompt smoke 已過、套件數 454→570、issue #4 紅旗修復
  已落地待 GPU 重跑）、本輪記錄補登 pass 46–52 七則（細節存專案記憶
  各輪補記）、trim pass 36–42 七則移出（事實存專案記憶各輪存檔）
  維持 10 entries。RESTORE 對帳乾淨：開工遠端 HEAD 210358e0＝記憶
  pass-51；issue #4 仍 2 comments 無新（Claude 紅旗重跑未回報——
  bridge-warmup 仍阻塞）；PR open 列表空；tags 空（v0.2.0 未打）；
  遠端無 .github（CI.yml 待使用者）；temp 無新 GPU 輸出（兩張 KIMI
  jpeg 11-24 舊照＋pass45_checkpoint.py 舊 helper）；mtime 雙閘
  開工時無對岸活動（對岸最後推送 12:14Z）。**工作中偵測（測試數
  第三網，實例三十一）**：全套跑出 579≠預期 570——對岸 13:16Z
  落地 M7a v1.3 簡繁對照 seed 雙檔（eval/cjk_t2i_v1_3.jsonl＋
  tests/test_m7a_seed_v1_3.py +9，未推送、零交集）；依實例二十
  先例不碰對岸進行中檔、不搶推、不代記為已落地，其增量由對岸
  推送後下輪消化驗證。本側 CPU 基線 570 passed 於 pass-51 合體
  樹已驗；本輪 docs-only。授權紅線未碰。推送紀律：全程 sed 分段
  直讀磁碟重組 payload、寫後即建 hash-object 基線、推後 API
  blob 驗證。

- 2026-11-26 pass 51 @ 210358e0（雙邊補登）— ② watchlist 第十九輪
  四線無 delta（Wan 3.x 仍 API-only——api.github.com 直核 Wan-Video
  org 無 3.x repo／Wan-Animate-2 repo id 無變化／Qwen-Image 無 11 月
  新版／awesome-rm 仍 2026-10-06 版；commit 115c513b，roadmap blob
  5ffd4472 36975B byte-exact）＋ status.md follow-through
  （210358e0，blob 6a9dd7d9 16182B，套件數 546→570）。**推送事故
  三推收斂**：fc941127 誤送 `$(cat ...)` 字面值、61aaf11 誤送截斷
  stub，兩度污染遠端皆由 blob SHA 閘即時攔截——教訓：push_files
  content 是純字串非 shell，大檔送出前驗 payload 位元組數。競態
  實例三十：雙 session 同題撞車（watchlist 第十九輪）結構化結論
  全數一致、零交集收斂。CPU 570 passed。

- 2026-11-26 pass 50 @ efd1af65（雙邊補登）— ③ P1b 資料引擎雙件：
  check_av_manifest CLI 契約測試 +8（a216a3d1，test blob 7f8317cc
  5699B，554→562——manifest-gate 的 exit code／--show-errors 能見度／
  forbidden-source 紅線訊息）＋ gate↔profiler 跨工具一致性測試 +8
  （efd1af65，test blob 78994c67 10767B，562→570——同一 manifest
  pair 同餵閘與剖析器，kept＝success＋skipped 關係式與兩個文件化
  分叉鎖定）。競態實例二十九（測試數第三網偵測擦肩，零交集）。

- 2026-11-26 pass 49 @ c3b661ca（補登）— ③ make_pref_pairs main()
  層 CLI 契約測試 +8（c3b661ca，test blob aa347e95 7894B 首次即
  byte-exact，546→554）——P1a 鏈 gen→score→pairs→encode→train
  CLI 測試全線閉合。競態實例二十八：status.md follow-through 同題
  撞車（對岸 99b0a4e1 先落地），edit_file 全數未命中即時偵測轉題
  零浪費。

- 2026-11-26 pass 48 @ 4d926f6c（雙邊補登）— ③ score_cjk_text
  main() 層 CLI 契約測試 +13（4d926f6c，test blob 57f71a82，
  533→546——M7a CLI 契約最後缺口閉合）＋ status.md follow-through
  （99b0a4e1，blob 12d060e3 15728B，套件數 503→546）＋ RESTORE
  消化使用者合併 PR #5（06b122d4，status.md blob bf5403f6 15232B
  curl raw pinned-SHA 回寫 byte-exact）。② watchlist 第十八輪抽查
  無 delta（VA-Judger code LICENSE＝MIT 確認，HF 權重 license tag
  維持使用者代查）。

- 2026-11-26 pass 47 @ 2256980f（雙邊補登）— ③ train 側 CLI 缺口
  閉合：train_dpo CLI 契約測試 +8（bef21e22，test blob 04a66206，
  516→524——P1a 鏈最後一棒 preflight fail-fast 先於 trainer
  import）＋ train_joint_av _real 接線測試 +9（2256980f，test blob
  7d79c860 12215B，524→533——bridge-warmup 入口的引數 gating／
  YAML config 載入／凍結 Wan 構造參數全釘）。

- 2026-11-26 pass 46 @ 867071d1（雙邊補登）— ③ score_eval_set CLI
  契約測試 +13＋腳本 resume 強化（867071d1，script blob 73b6712c
  5665B／test blob de554e63 14594B 雙檔同 commit 首次即
  byte-exact，503→516）＋ status.md follow-through（c424b897）。
  競態實例二十五（零交集接龍）。

- 2026-11-26 pass 45 @ 075f8964（本 session）— ③ encode 腳本強化：
  **encode_pairs CLI 契約 CPU 測試落地（+18，純測試增量零行為變更）**。
  RESTORE 對帳乾淨：開工遠端 HEAD 654443e2＝記憶（pass-44 雙增量
  合體）；遠端 session-state 仍 1e6bba46（落後八輪，覆推以本地最新
  檔為準）；issue #4 兩 comments 無新進展（Claude 紅旗修復重跑未
  回報——bridge-warmup 仍阻塞）、PR #5 head b3461ceb 仍 open/draft
  無 rebase、tags 空（v0.2.0 未打）、遠端無 .github（CI.yml 待使用
  者）、temp 無新 GPU 輸出（兩張 KIMI jpeg 仍 11-24 舊照）；mtime
  雙閘＋測試數基線（485 重跑確認）三網皆無對岸活動；推送前第二閘
  HEAD 未動無競態。**缺口**：encode_pairs 是 P1a DPO 鏈的
  pair-latent 預編碼段（pairs.jsonl→Wan VAE latents→train_dpo），
  pass-32 只加了 refuse-before-cache-import 預檢接線一測；腳本自身
  的 resume 經濟（全 cache 時 diffusers/torch 完全不 import、
  encode_fn=None）、encode_video 正規化契約、--smoke/--limit 截斷、
  旗標直通、失敗列 exit 0 契約全無 main() 層覆蓋。
  tests/test_p1a_encode_pairs.py +18（fake latent_cache/config/
  diffusers/torch/numpy 注入，仿 test_m2_encode_texts；preflight 跑
  真 pairs_check＋tmp_path fixture）：fresh run 管線載入
  （repo/bf16/VAE cuda fp32 hoist）＋win 先 lose 後＋
  manifest/summary 落盤；**encode_video 逐 op 契約**（tagged fake
  tensor：NCHW permute＋batch dim＋/127.5-1＋cuda fp32 →
  (latent−latents_mean)×(1/latents_std) 統計取自 vae.config
  view (1,z_dim,1,1,1) → cpu → save_latent_atomic）；lazy-load
  sys.modules None 絆線（torch/diffusers/numpy）＋encode_fn=None；
  --limit 0「no pairs to encode」exit 1 先於 planner、缺 pairs 檔
  exit 1 先於 preflight、unusable 列拒絕啟動先於 planner
  （missing_keys＋refusing to start stderr）；--no-resume／
  --max-retries／--strict 直通；--reset-failures 先於 plan 清
  ledger（call order 鎖定）；失敗列 exit 0＋stderr 提示（有
  budget 時提示帶 --reset-failures）；cache-invalidated／
  budget-exhausted 計數訊息；--model 同時決定 planner cache
  namespace 與 pipeline repo。**首次推送即 byte-exact**（blob
  **6f5c90d6** 21455B＝本地 hash-object；tests/ listing 39 檔對帳
  無夾帶）。CPU 全套 **503 passed**（485+18）。

- 2026-11-26 pass 44 @ 654443e2（雙邊零交集合體，補記）— ③ 雙增量：
  **encode_texts CLI 契約測試 +17**（`dad92980`，test blob 72d60588
  首次即 byte-exact——lazy-load 絆線／--smoke/--limit／resume／失敗
  列 exit 0 契約；腳本零行為變更）＋ **M7a gen_cjk_images resume 閘
  測試 +14＋per-prompt keep-mining**（`654443e2`，script blob
  ee08bb1b／test blob f46cb7d1 byte-exact——單 prompt 例外不再炸掉
  100-prompt GPU run，失敗列刪殘檔 continue、final gate 列 failed
  ＋exit 1）。**競態實例二十三（零交集接龍）**：一側 encode_texts
  雙檔 05:07–05:09Z 落共享樹（對側全套 485≠預期 471 第三網偵測），
  雙邊各自只推己檔——dad92980 05:12:34Z、654443e2 05:13:35Z 自動
  叠上（linear history 保持）；**新觀察：push_files 容忍推送窗口內
  對岸先推**——不報 non-fast-forward、以最新 main 為父落 commit
  （檔案零交集前提下；有交集後果未驗證，推送前對帳仍為最後一道
  閘）。合體樹 CPU 全套 **485 passed**（454+17+14）雙邊獨立重跑
  皆過。

- 2026-11-26 pass 43 @ 4f90cde5（本 session）— ③ P1b 資料引擎小件：
  **joint_latents_stats cache 剖析器落地**。RESTORE 對帳乾淨：開工遠端
  HEAD 56fd4abd＝記憶（pass-42 status.md 覆推）；遠端 session-state 仍
  1e6bba46（落後七輪，覆推以本地最新檔為準）；issue #4 無新 comment
  （Claude 紅旗修復重跑待回報）、PR #5 仍 open/draft 無 rebase 動向、
  tags 空（v0.2.0 未打）、遠端無 .github（CI.yml 仍待使用者上傳）、
  temp 無新 GPU 輸出、mtime 雙閘無對岸活動；推送前第二閘乾淨無競態。
  **③ 缺口＋落地**：P1a 有 profiler 三件套（manifest_stats／scores_stats／
  pairs_stats），P1b 只有 check_joint_latents 判決閘——雙 cache 上盤後
  到 train_joint_av 燒 GPU 前無法預覽分佈。scripts/joint_latents_stats.py
  （純函數 profile_caches：AV/text 列分類沿用 joint_cache_check 優先序
  ——bad JSON→error→skipped→success，計數與閘一致；video/audio shape
  mix 直方圖、caption 共享計數——同 caption 共享 content-addressed
  text embedding 值得能見、index 覆蓋 gaps／duplicate、reused cache-hit
  計數、per-model 列分佈、join 視圖——served／av-only／text-only／
  caption drift；只讀 manifest 文字不開 .pt；deterministic sorted-keys
  JSON＋--out；描述不判決，可讀 exit 0／不可讀 exit 1）＋
  tests/test_m2_joint_latents_stats.py +10（純函數七＋CLI 三）。
  **雙檔同 commit 首次即 byte-exact**（script blob **3c9f516b** 8931B／
  test blob **88780702** 8723B——API contents SHA＝本地 hash-object 雙驗；
  raw 分支 ref 首抓 CDN 滯後 cmp 報差，改 pinned-SHA 重抓 byte-exact——
  同 pass-32 的 GitHub raw 滯後現象，以 API blob SHA 為權威）；目錄樹
  listing 掃描 130 檔無夾帶。CPU 全套 **441 passed**（431+10，差額全為
  己出）。**競態實例二十二（收尾接龍，零交集）**：本 session
  推送後對岸 04:14Z／04:16Z 落地 `ae3d8046`＋`33e55d6a`——gen_candidates
  lazy pipeline load（全 cache resume 不再付模型載入成本，對齊
  gen_eval_set resume-gate 經濟）＋test_p1a_gen_candidates.py +13
  （其 commit message 明寫「includes parallel session 4f90cde5 +10,
  zero overlap」）。消化：gen_candidates.py 對岸已在共享樹改好（patch
  套用時報 already applied——共享樹即時性又一實證），test 檔 curl raw
  pinned-SHA 回寫，雙 blob byte-exact（ccbb1b2b／33a69f24）；CPU 全套
  **454 passed**（441+13）獨立重跑；設計審閱無異議（pipe=None lazy
  載入、行為不變、resume 全跳過不載 pipeline）。匿名 API 撞 60/hr
  上限一次（消化途中 403／raw SSL 35），改 MCP get_commit full_patch
  取 patch——MCP 驗證額度與匿名 REST 分池的備援路徑確立。status.md
  follow-through（P1b 行補 profiler＋套件數 454）由本 session 補推
  `2fb1d931`，blob **f39640d7**（11954B）首次即 byte-exact（API
  contents SHA＝本地 hash-object）。watchlist 本輪不重查（第十七輪
  pass-42 剛落地；阿里窗口 11-30 關閉後降每數輪抽查）。trim：
  pass-33@12a5eabe 一條移出（事實存專案記憶第三十三輪存檔），維持
  11 entries。

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

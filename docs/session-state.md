---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-27T01:20:00Z"
head: 62ed3c686c2e3f5559da6e8cda17b67375a63bea
head_verified: true
last_increment: "本輪＝③ P1b manifest 閘↔剖析器交叉一致性測試：tests/test_m2_manifest_gate_stats_consistency.py +9（純 ASCII，9898B）——check_av_manifest（閘）× manifest_stats（profiler）同一 manifest 雙跑釘定：總帳不變量 samples+errors+duplicates==rows+bad_json_lines==非空行數、兩文件化分歧（非 dict JSON 行：閘 schema／profiler bad_json_line；forbidden+schema 雙缺陷行：閘 schema-first、profiler 子串照計，forbidden_sources>=forbidden_rejected 且 schema-valid 時等號）、clean manifest 雙邊 source 分佈全等、角色差異（閘 exit 1／profiler exit 0）、雙 CLI 重跑 byte-identical、缺檔雙邊 exit 1；CPU 全套 642→651 passed；commit 62ed3c68，test blob 0e2db02c 首次即 byte-exact；同輪 SAVE prepend pass-59 雙 entry＋60、trim pass-50/51（事實存專案記憶存檔）、head→62ed3c68。（前輪：pass-59 對岸 status.md 8ce3bf5f＋watchlist 二十二輪 bfdb81dc）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 1   # #4 RunPod M2 internal smoke（紅旗修復 b5321003…d455e3dc 已落地，Claude 重跑待回報——紅旗清除＝audio loss／grad norm 回個位數，過則解 bridge-warmup 阻塞）；PR #5 已於 pass-48 由使用者合併（06b122d4）
gpu_outputs: m1_5clip_and_m2_internal_smoke_passed_reviewed   # 2026-11-26 A40：M1 5/5 probe-valid（mean 842s/clip、peak 33.7GB）＋issue #4 internal wan_blocks+LoRA PASS(contract) 但 audio loss 79/grad 94 紅旗（bridge input 需 norm/scale 重跑才准訓練）；pass-41 已判讀＋PR #5 評審留言；VBench/50-clip/P1a 仍待 Claude
blockers: []   # pass-25 eval/roadmap 漂移已由本 session 覆正（ea97af39 eval aa87fac9／6f478fd5 roadmap 77e790e4，皆一次 byte-exact）
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head（pass-60＝③ P1b manifest 閘↔剖析器一致性測試 62ed3c68，blob 0e2db02c byte-exact）；issue #4 Claude 紅旗重跑結果（過則解 bridge-warmup 阻塞）／v0.2.0 tag／CI.yml，先消化再增量"
  - "pass 16–58 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十～十五輪（4fad1f38、918001c0+3f3fa0ac、129d5d67 PAVRM Apache 升格、42ec2a64）／encode_texts retry budget（e1896295）／build_av_manifest（3f94b385）／pair+AV latent cache retry budget（1e64d0c1）／joint loader skipped 列明確報缺（cc85d330）／M7a eval seed 24→48（153be48c＋9d83f6c3）／M7a seed 48→100 v1.2（1596c716，漂移已由 ea97af39／6f478fd5 覆正）／manifest_stats 剖析器（1b896e72）／M7a 跨模型 summary 比較器（d4e8d0e3＋773f61e9）／pairs_stats 偏好對剖析器（19881f0a）／pair-latent 預檢閘＋train_dpo fail-fast（afde66af）／score_eval_set.collect() 測試（59163737）／scores 預檢閘＋make_pref_pairs fail-fast（3840eb88）／M7a scorer 簡繁分軌（5d1d26a5）／M4 HLS fMP4/CMAF 分發包 packager（79191bc3）／pairs.jsonl 預檢閘＋encode_pairs fail-fast（39295734＋9b41f3bb）／M2 CFG 文字條件 dropout（82d18e37）／M4 DASH manifest 兼出（12a5eabe）／status.md handoff 入口補齊 pass 19–33（d1038099）／joint-cache 預檢閘 check_joint_latents（c3756605）／watchlist 第十六輪（3ecce283）／runbook §5.1/§5.2 預檢閘接線＋status.md P1b 行（7a6446e0）／scores.jsonl 剖析器 scores_stats（0bf42cd7）／M1 VBench 閘純函數化＋14 tests（22ca23aa）／runbook §2 P1a 預檢閘鏈接線＋status.md 合體（41e99d60＋2e165956）／gen_eval_set probe_ok＋resume 閘測試（cc00b918）＋status.md M1 行 follow-through（254e83b4）／M2 norm_video_kv 紅旗修復六 commit（b5321003…d455e3dc，＋9 tests，CPU 431）／watchlist 第十七輪（d2001b4a）／P1b cache 剖析器 joint_latents_stats（4f90cde5）／P1a score_eval_set CLI 測試＋resume 強化（867071d1）＋status.md follow-through（c424b897）／train_dpo CLI 測試（bef21e22）／train_joint_av _real CLI 測試（2256980f）／M7a score_cjk_text CLI 測試（4d926f6c）＋status.md（99b0a4e1）／make_pref_pairs CLI 測試（c3b661ca）／check_av_manifest CLI 測試（a216a3d1）／gate↔profiler 一致性測試（efd1af65）／watchlist 第十八輪（pass-48 抽查無 delta）／第十九輪（115c513b）／status.md follow-through（210358e0）／checkpoint 覆推解僵局（55efe9fe）／M7a v1.3 簡繁對照 seed（27300882+658a67f+a7b99262）＋status.md v1.3 行（08ccf824）／watchlist 第二十輪（880f9da7，修推 cede601a／072e3d18 留史）／M6 avatar 骨架契約測試（5f4bb7a0）／M1 pipeline CLI 測試（30e0bd2d）／M3/M4/M7c 骨架契約測試（2b46e53f）／status.md follow-through（4f7e4993，修推 a1e9cbd1 留史）／M1 DiT wrapper 契約測試（293fbf3f）／status.md follow-through（8ce3bf5f，4aaa9bec 留史）／watchlist 第二十二輪（bfdb81dc）／P1b manifest 閘↔剖析器一致性測試（62ed3c68）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra＋chi-sim，否則 decode/M7a 測試 skip）"
  - "③ 候選：其他未列名 docs・資料面小件（src 全模組契約釘定已閉合——dit.py 薄 wrapper pass-58 293fbf3f 落地；manifest 閘↔profiler 一致性 pass-60 62ed3c68 落地）；**encode_av_latents CLI 測試仍為對岸明列項勿碰**；scripts 主鏈 CLI 契約測試已全線閉合（gen／encode／score／pairs／train 皆齊），剩 smoke_decode 等 GPU 殼腳本屬 Claude 道；watchlist 下次抽查 pass-63±（第二十二輪 bfdb81dc 已落地勿重做；阿里 11 月窗口已關閉，每數輪抽查；GenVidCritic／RewardVerse 已入冊弱訊號，隨 reward 線覆核 pushed_at）"
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
v1.3 簡繁對照 200 列已落地（pass-52 對岸 27300882+658a67f+a7b99262））；**11-26 交付格式拍板：HDR10/HLG＋
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
pairable／single／all_failed 預覽）**＋ **run_vbench VBench ±0.02 閘邏輯純函數化＋14 tests（pass 39 本 session：M1 GPU 驗收終點閘改 CPU 可驗）**＋ **P1b cache 剖析器 joint_latents_stats 落地（pass 43 本 session：閘之外的分佈剖析——shape mix／caption 共享／index 覆蓋／join 視圖）**；651 CPU tests 全綠；GPU 驗收由
Claude 執行（runbook §0 開 pod SOP 落地；issue #4 internal smoke 契約 PASS 但 loss-scale 紅旗——CPU 修復 norm_video_kv 已落地（b5321003…d455e3dc）、GPU 重跑待 Claude，過則解 bridge-warmup 阻塞）＋ **P1a／P1b／M7a 主鏈 CLI 契約測試全線閉合（pass 46–51：score_eval_set 867071d1／train_dpo bef21e22／train_joint_av 2256980f／score_cjk_text 4d926f6c／make_pref_pairs c3b661ca／check_av_manifest a216a3d1＋gate↔stats 一致性 efd1af65）**＋ watchlist 第十八～二十輪無 delta（pass-48／115c513b／880f9da7，阿里窗口 11-30 已關閉）＋ **M6 Avatar 骨架契約測試落地（pass 55 本 session：+17，src 零覆蓋可用邏輯模組閉合）**＋ **M1 pipeline 一鍵產片入口 CLI 契約測試（pass 55 對岸：+9，主鏈入口全線閉合）＋ M3/M4/M7c 骨架契約測試（pass 56 本 session：+19，src 宣告式契約全數釘定）＋ status.md follow-through（pass 57 對岸：M1／M6／M3+ 行補齊→624）＋ M1 DiT backbone wrapper 契約測試（pass 58 本 session：+18，src/theama/model 最後零覆蓋模組閉合→642）**；CI.yml 內容備妥待
上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-27 pass 60 @ 62ed3c68（本 session）— ③ P1b manifest 閘↔剖析器
  交叉一致性測試：tests/test_m2_manifest_gate_stats_consistency.py +9
  （純 ASCII，9898B）。缺口——joint-cache 層早有
  test_m2_joint_gate_stats_consistency（閘×profiler 一對 fixture 雙跑），
  其上一層 raw manifest 層 check_av_manifest×manifest_stats 各自有
  獨立測試但無交叉釘定，重構任一側會靜默分歧、第一見證是 GPU pod 上
  困惑的操作者。鎖定：總帳不變量（samples+errors+duplicates ==
  rows+bad_json_lines == 非空行數，7 行混合 fixture 全分類覆蓋）＋
  兩文件化分歧（非 dict JSON 行：閘記 schema error／profiler 記
  bad_json_line；forbidden+schema 雙缺陷行：閘 schema 優先記 schema、
  profiler 紅線子串照計——forbidden_sources >= forbidden_rejected，
  schema-valid 時等號）＋ clean manifest 雙邊 source 分佈全等＋
  角色差異（閘 exit 1 的 manifest profiler 照 exit 0）＋雙 CLI 重跑
  byte-identical＋缺檔雙邊 exit 1。測試自身 bug 一輪修掉
  （missing-manifest 測試變數名 typo）。9/9 綠；CPU 全套
  642→651 passed（66.6s）；test blob 0e2db02c 首次推送即 byte-exact，
  get_commit 確認單檔 +228 無夾帶。RESTORE 乾淨：開工 HEAD
  bfdb81dc＝記憶（4aaa9bec 壞 commit 留史）；issue #4 無新
  （Claude 紅旗重跑未回報——bridge-warmup 仍阻塞）；PR open 空；
  tags 空（v0.2.0 未打）；遠端無 .github；temp 無新 GPU 輸出；
  mtime 雙閘＋推送前第二閘皆乾淨。環境事件：沙箱再重置洗掉
  pytest＋theama（第四度同型），pip 復位後基線 642 確認動工。
  watchlist 未到點未查（下次 pass-63±）。授權紅線未碰。

- 2026-11-27 pass 59 @ bfdb81dc（本 session 補登）— ② watchlist 第二十二輪：
  四線全無 delta（Wan 3.x 仍 API-only；Wan-Animate-2 repo id 無變化；
  Qwen-Image 無 11 月新版；awesome-rm 仍 10-06 版，VA-Judger 第八次
  HF 核查 HTTP 000）；GenVidCritic（11-08，1★）／RewardVerse（09-19，
  6★）裁定入冊弱訊號觀察項（無論文 hard id 無權重，每數輪隨 reward
  線覆核 pushed_at），不列正式觀察、不改 P1a 配方。roadmap blob
  53467d8f 42195B 首次即 byte-exact（四段行界重組 payload——長 CJK
  全檔重推連續第二次零事故）；get_commit 僅 roadmap 一檔 +33/−1。
  競態實例三十九：工作中偵測對岸首推 4aaa9bec 轉錄漂移
  （48824e2e≠17797700），依 5 分鐘自修觀察窗紀律不介入，對岸 75 秒
  自修 8ce3bf5f byte-exact——觀察窗紀律第二次主動生效、零互修賽。

- 2026-11-27 pass 59 @ 8ce3bf5f（對岸補登）— ③ status.md follow-through：
  M1 行補 dit wrapper 契約測試子句＋套件數 624→642（合體樹重跑實測
  69.4s）。首推 4aaa9bec 鄰近段落相似句張冠李戴（+56B，「identical
  memory to the 4090 run」被誤植為 issue #4 段句子＋多插 Loss 81.5
  行——長檔轉錄植錯第七度，新型態非 stutter），blob 閘即時攔截，
  diff 收斂法一輪定點修復 8ce3bf5f byte-exact（status blob 17797700
  17466B；4aaa9bec 留史）。

- 2026-11-26 pass 58 @ 293fbf3f（本 session）— ③ M1 DiT backbone
  wrapper 契約測試：tests/test_m1_dit_wrapper.py +18（純 ASCII，
  7639B）——src/theama/model 最後零覆蓋模組（dit.py 薄 wrapper）。
  鎖定：常數二（A14B／TI2V-5B repo id）＋ BackboneConfig 二（預設
  repo／bf16／expert_boundary 0.875、custom 存放）＋ wiring 三
  （None→預設 config、custom config＋dev 存放、_pipe 初始 None
  lazy）＋ load_t2v 四（dev→TI2V-5B／full→A14B repo 選擇、VAE
  float32＋vae subfolder、pipe 吃 config.dtype＋VAE 注入、_pipe
  快取——sys.modules 假 diffusers 免 GPU）＋ generate 七（預設契約
  prompt／negative_prompt 空／1280×720／num_frames=5*24+1=121 Wan
  8n+1／40 steps／seed None→generator None、(W,H)→width/height
  映射、custom 2s@8fps→17、negative_prompt 傳遞、seed→
  deterministic torch.Generator、回傳 frames[0]、no_grad）。18/18
  首輪即綠；CPU 全套 624→642 passed（70.8s）；test blob d7eb89a6
  首次推送即 byte-exact。RESTORE 乾淨：開工 HEAD 4f7e4993＝記憶、
  session-state 1ed3ea38／roadmap 4aa1f02e／status d9e83558 皆
  byte-exact；issue #4 兩 comments 無新（Claude 紅旗重跑未回報——
  bridge-warmup 仍阻塞）；PR open 空；tags 空（v0.2.0 未打）；
  遠端無 .github（CI.yml 待使用者）；temp 無新 GPU 輸出；沙箱存活
  免重裝（基線 624 確認後動工）。授權紅線未碰。

- 2026-11-26 pass 57 @ 4f7e4993（對岸補登）— ③ status.md follow-through：
  M1 行補 pipeline CLI 測試（30e0bd2d +9）＋新增 M6 行（avatar 骨架
  5f4bb7a0 +17）＋M3+ 行補骨架契約測試（2b46e53f +19）＋套件數
  579→624。首推 a1e9cbd1 於歷史行植錯「a 4096 run」（應 4090），
  blob 閘攔截，diff 收斂單字修復 4f7e4993 byte-exact（status blob
  d9e83558 17216B；a1e9cbd1 留史）。本側同題撞車轉消化驗證（實例
  三十七）；對岸推送後 5 分鐘自修觀察窗新經驗確立。

- 2026-11-26 pass 56 @ 2b46e53f（本 session）— ③ M3/M4/M7c 骨架契約測試：
  tests/test_m3_m4_m7c_skeletons.py +19（純 ASCII，7095B）——src 最後一批
  未釘宣告式契約（具可用邏輯模組已全覆蓋；smoke_* GPU 殼屬 Claude 道）。
  鎖定：extend_30s 七（ExtendConfig 預設 5s/30s/8 anchor/250ms、custom
  存放、None→預設 config＋backbone 同一性、num_segments 預設 6、ceil
  31/5→7・29/5→6・30/4→8、max(1,…) 下限 0/3/5→1、extend()
  NotImplementedError M3）＋ upscale_4k 七（DEFAULT_STAGES 順序
  regen2k→regen4k、參數 2560×1440@0.35 untiled→3840×2160@0.25 tiled、
  順序契約 resolution 嚴格升・denoise 嚴格降・僅 4K tiled、UpscaleStage
  tiled 預設 False、None→DEFAULT_STAGES 同一性、custom stages 尊重、
  upscale() NotImplementedError M4）＋ keyframe 五（KeyframeSpec 預設
  1280×720／style_lock・seed None、custom 存放、image backbone 同一性、
  first_last_frames／identity_reference 皆 NotImplementedError M7c）。
  19/19 首輪即綠；CPU 全套 605→624 passed（65.6s）；test blob e30b5392
  首次推送即 byte-exact，get_commit 僅一檔 +177 無夾帶。環境事件：沙箱
  重置洗掉 pytest＋theama 安裝，pip 復位後基線 605 確認動工；純 ASCII
  預檢閘攔下四個破折號改「--」後才推送。RESTORE 乾淨：開工 HEAD
  ea91d1c8＝記憶；issue #4 兩 comments 無新（Claude 紅旗重跑未回報——
  bridge-warmup 仍阻塞）；PR open 空；tags 空（v0.2.0 未打）；遠端無
  .github（CI.yml 待使用者）；temp 無新 GPU 輸出；mtime 雙閘＋推送前
  第二閘皆乾淨。授權紅線未碰。watchlist 未到點未查（下次 pass-57±）。
- 2026-11-26 pass 55 @ 5f4bb7a0（本 session）— ③ M6 Avatar 骨架契約測試：
  tests/test_m6_avatar_skeletons.py +17——theama.avatar 三模組（identity／
  audio_drive／streaming）為 src 最後零覆蓋且具可用邏輯的模組。鎖定：
  IdentityBank 六（default/None 空表、register→swap 同物件 roundtrip、
  unknown KeyError 帶名、overwrite、entries 獨立）＋IdentityEncoder 二
  （預設 512→1024 shape、custom dim＋float64 dtype 跟隨）＋AudioDrive 五
  （forward (B,Tv,Dm)、token 數跟 video grid 不跟 Ta、window 屬性、
  frame_queries expand＋dtype/device 跟隨＋基底向量同源、eval  determinism）
  ＋StreamConfig/session 四（720p/25fps/chunk5/KV60/anchor50 預設、deque
  maxlen 接線、custom config、M6b/M6c 三通道 NotImplementedError 契約釘定）。
  CPU 全套 579→596 passed；test blob 6d6221c4（6087B）首次推送即
  byte-exact，tests/ 目錄 48 檔對帳無夾帶。
- 2026-11-26 pass 54 @ 880f9da7（雙邊補登）— ② watchlist 第二十輪四線
  無 delta（Wan 3.x 仍 API-only／Wan-Animate-2 repo id 無變化／Qwen-Image
  無 11 月新版／awesome-rm 仍 2026-10-06 版；VA-Judger HF 第六次 HTTP 000，
  使用者代查維持）；同題撞車實例三十四，雙邊結構化結論全數一致；推送事故
  三推收斂——cede601a 誤植「公告或」stutter、072e3d18 又植「未未」二連，
  皆 blob 閘自攔自修（壞 commit 留史不修史），880f9da7 roadmap blob
  a2bfa8c9 byte-exact；CPU 579 passed。
- 2026-11-26 pass 53 @ 08ccf824（雙邊補登）— ③ status.md follow-through
  （M7a 行補 v1.3 簡繁對照 seed 子句＋套件數 570→579，blob 8a7da146
  byte-exact）；同題撞車實例三十三——assert-unique 拒寫立功，零浪費轉
  消化驗證（三件套全過）；CPU 579 passed。
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

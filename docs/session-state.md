---
# Theama session checkpoint — hourly agent runs save/restore THIS file.
# 設計原則：沙箱會重置、專案記憶 mount 可能失效，唯 GitHub repo 恆在。
schema: 1
updated_at: "2026-11-27T07:30:00Z"
head: ee42be8cdefae9553c4025c60f419bb0bbb87e9f
head_verified: true
last_increment: "本輪＝② watchlist 同題撞車消化驗證＋SAVE 補登（docs-only 輪，無 code 增量）：對岸 07:16Z 推 ee42be8c 落地 watchlist 第二十四輪（roadmap +33/−1，blob 96cf6a61 首推即 byte-exact），本側同小時獨立覆核逐項一致——同題撞車實例四十三（工作中誤讀對岸熱寫的共享樹 本地檔為「早已落地」，遠端歷史 ref 對帳還原真相）；SAVE 補登 pass-64/65/對岸 round-24 三輪＋head→ee42be8c＋next_hour 排程 pass-66±→pass-69±（第二十四輪已落地勿重做）＋status.md 651→662 已落地 9782f049 勿重做；prepend pass-66、trim pass-55 維持 10 entries。（前輪：pass-65 對岸 status.md follow-through 9782f049、pass-64 AV 榜更新 ad53eef9、pass-66 對岸 watchlist 第二十四輪 ee42be8c）"
ci_yml: content_ready_pending_user_upload   # MCP token 推 .github/workflows 被 403（insufficient scopes）；內容已交付使用者
issues: 1   # #4 RunPod M2 internal smoke（紅旗修復 b5321003…d455e3dc 已落地，Claude 重跑待回報——紅旗清除＝audio loss／grad norm 回個位數，過則解 bridge-warmup 阻塞）；PR #5 已於 pass-48 由使用者合併（06b122d4）
gpu_outputs: m1_5clip_and_m2_internal_smoke_passed_reviewed   # 2026-11-26 A40：M1 5/5 probe-valid（mean 842s/clip、peak 33.7GB）＋issue #4 internal wan_blocks+LoRA PASS(contract) 但 audio loss 79/grad 94 紅旗（bridge input 需 norm/scale 重跑才准訓練）；pass-41 已判讀＋PR #5 評審留言；VBench/50-clip/P1a 仍待 Claude
blockers: []   # pass-25 eval/roadmap 漂移已由本 session 覆正（ea97af39 eval aa87fac9／6f478fd5 roadmap 77e790e4，皆一次 byte-exact）
division_of_labor:
  hourly_agent: "watchlist 核查、docs、CPU 側程式準備、repo 健康、session checkpoint"
  claude: "GPU 驗收執行（runbook 第 1–5 節＋status.md next steps）、CI.yml 上傳（人工作業）、結果回報/落盤、git tag v0.2.0（人工作業）；2026-11-25 使用者擴權：可做其他底模＋HDL on RunPod（紅線不變：H3/LTX/Qwen2.1/FLUX3-Klein/Krea2 不碰）"
next_hour:
  - "RESTORE：核對 HEAD 與本檔 head（pass-66＝本 SAVE；對岸 watchlist 第二十四輪 ee42be8c／roadmap 96cf6a61、pass-65 status.md follow-through 9782f049、pass-64 AV 榜更新 ad53eef9 皆已消化勿重做）；issue #4 Claude 紅旗重跑結果（過則解 bridge-warmup 阻塞）／v0.2.0 tag／CI.yml，先消化再增量"
  - "pass 16–62 增量皆已落地請勿重做：Ming gen 第三路徑（4e1d202f）／score provenance（a3067d9b）／runbook §4.1（a7cb7697）／PR #3 判讀＋同步（0a2a3cd7）／watchlist 第十～十五輪（4fad1f38、918001c0+3f3fa0ac、129d5d67 PAVRM Apache 升格、42ec2a64）／encode_texts retry budget（e1896295）／build_av_manifest（3f94b385）／pair+AV latent cache retry budget（1e64d0c1）／joint loader skipped 列明確報缺（cc85d330）／M7a eval seed 24→48（153be48c＋9d83f6c3）／M7a seed 48→100 v1.2（1596c716，漂移已由 ea97af39／6f478fd5 覆正）／manifest_stats 剖析器（1b896e72）／M7a 跨模型 summary 比較器（d4e8d0e3＋773f61e9）／pairs_stats 偏好對剖析器（19881f0a）／pair-latent 預檢閘＋train_dpo fail-fast（afde66af）／score_eval_set.collect() 測試（59163737）／scores 預檢閘＋make_pref_pairs fail-fast（3840eb88）／M7a scorer 簡繁分軌（5d1d26a5）／M4 HLS fMP4/CMAF 分發包 packager（79191bc3）／pairs.jsonl 預檢閘＋encode_pairs fail-fast（39295734＋9b41f3bb）／M2 CFG 文字條件 dropout（82d18e37）／M4 DASH manifest 兼出（12a5eabe）／status.md handoff 入口補齊 pass 19–33（d1038099）／joint-cache 預檢閘 check_joint_latents（c3756605）／watchlist 第十六輪（3ecce283）／runbook §5.1/§5.2 預檢閘接線＋status.md P1b 行（7a6446e0）／scores.jsonl 剖析器 scores_stats（0bf42cd7）／M1 VBench 閘純函數化＋14 tests（22ca23aa）／runbook §2 P1a 預檢閘鏈接線＋status.md 合體（41e99d60＋2e165956）／gen_eval_set probe_ok＋resume 閘測試（cc00b918）＋status.md M1 行 follow-through（254e83b4）／M2 norm_video_kv 紅旗修復六 commit（b5321003…d455e3dc，＋9 tests，CPU 431）／watchlist 第十七輪（d2001b4a）／P1b cache 剖析器 joint_latents_stats（4f90cde5）／P1a score_eval_set CLI 測試＋resume 強化（867071d1）＋status.md follow-through（c424b897）／train_dpo CLI 測試（bef21e22）／train_joint_av _real CLI 測試（2256980f）／M7a score_cjk_text CLI 測試（4d926f6c）＋status.md（99b0a4e1）／make_pref_pairs CLI 測試（c3b661ca）／check_av_manifest CLI 測試（a216a3d1）／gate↔profiler 一致性測試（efd1af65）／watchlist 第十八輪（pass-48 抽查無 delta）／第十九輪（115c513b）／status.md follow-through（210358e0）／checkpoint 覆推解僵局（55efe9fe）／M7a v1.3 簡繁對照 seed（27300882+658a67f+a7b99262）＋status.md v1.3 行（08ccf824）／watchlist 第二十輪（880f9da7，修推 cede601a／072e3d18 留史）／M6 avatar 骨架契約測試（5f4bb7a0）／M1 pipeline CLI 測試（30e0bd2d）／M3/M4/M7c 骨架契約測試（2b46e53f）／status.md follow-through（4f7e4993，修推 a1e9cbd1 留史）／M1 DiT wrapper 契約測試（293fbf3f）／status.md follow-through（8ce3bf5f，4aaa9bec 留史）／watchlist 第二十二輪（bfdb81dc）／P1b manifest 閘↔剖析器一致性測試（62ed3c68）／status.md follow-through（cff0ffc8）／RunPod shell 腳本 repo-health 閘（a9eec28e）／docs 結構 repo-health 閘（7fb2f20c）；agent 入口改為 docs/status.md＋AGENTS.md"
  - "使用者上傳 CI.yml 後：Actions 即為驗證引擎（CI image 需 ffmpeg + tesseract chi-tra＋chi-sim，否則 decode/M7a 測試 skip）"
  - "③ 候選：其他未列名 docs・資料面小件（src 全模組契約釘定已閉合；manifest 雙層閘↔profiler 一致性已閉合；shell 入口腳本閘 a9eec28e＋docs 結構閘 7fb2f20c 皆已落地勿重做；status.md follow-through 651→662 已落地 9782f049 勿重做）；**encode_av_latents CLI 測試互讓僵局待使用者裁示，勿碰**；watchlist 下次抽查 pass-69±（第二十四輪對岸 ee42be8c 已落地勿重做——pass-66 本 session 獨立覆核逐項一致；MAGI-2 開源動向隨覆核；GenVidCritic／RewardVerse 已入冊弱訊號，隨 reward 線覆核 pushed_at）"
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
pairable／single／all_failed 預覽）**＋ **run_vbench VBench ±0.02 閘邏輯純函數化＋14 tests（pass 39 本 session：M1 GPU 驗收終點閘改 CPU 可驗）**＋ **P1b cache 剖析器 joint_latents_stats 落地（pass 43 本 session：閘之外的分佈剖析——shape mix／caption 共享／index 覆蓋／join 視圖）**；662 CPU tests 全綠；GPU 驗收由
Claude 執行（runbook §0 開 pod SOP 落地；issue #4 internal smoke 契約 PASS 但 loss-scale 紅旗——CPU 修復 norm_video_kv 已落地（b5321003…d455e3dc）、GPU 重跑待 Claude，過則解 bridge-warmup 阻塞）＋ **P1a／P1b／M7a 主鏈 CLI 契約測試全線閉合（pass 46–51：score_eval_set 867071d1／train_dpo bef21e22／train_joint_av 2256980f／score_cjk_text 4d926f6c／make_pref_pairs c3b661ca／check_av_manifest a216a3d1＋gate↔stats 一致性 efd1af65）**＋ watchlist 第十八～二十輪無 delta（pass-48／115c513b／880f9da7，阿里窗口 11-30 已關閉）＋ **M6 Avatar 骨架契約測試落地（pass 55 本 session：+17，src 零覆蓋可用邏輯模組閉合）**＋ **M1 pipeline 一鍵產片入口 CLI 契約測試（pass 55 對岸：+9，主鏈入口全線閉合）＋ M3/M4/M7c 骨架契約測試（pass 56 本 session：+19，src 宣告式契約全數釘定）＋ status.md follow-through（pass 57 對岸：M1／M6／M3+ 行補齊→624）＋ M1 DiT backbone wrapper 契約測試（pass 58 本 session：+18，src/theama/model 最後零覆蓋模組閉合→642）＋ P1b manifest 閘↔profiler 一致性（pass 60：+9→651）＋ RunPod shell 腳本 repo-health 閘（pass 62 本 session：+5→656）＋ docs 結構 repo-health 閘（pass 63 本 session：+6→662）＋ watchlist 第二十四輪落地（pass-66 對岸 ee42be8c：四線無 delta、I2V 榜首 MAGI-2 持穩；本 session 同題撞車轉消化驗證逐項一致）**；CI.yml 內容備妥待
上傳；紅線不變。

## 本輪記錄（新輪加在最上面，保留最近 10 輪）

- 2026-11-27 pass 66 @ ee42be8c（本 session）— ② watchlist 第二十四輪
  同題撞車轉消化驗證（競態實例四十三）＋SAVE 補登（docs-only 輪，
  無 code 增量）：開工 RESTORE 全乾淨（遠端 HEAD 9782f049＝記憶、
  四檔 blob byte-exact、issue #4 兩 comments 無新、tags 空、遠端無
  .github、temp 無新 GPU 輸出），watchlist pass-66± 到點遂開工四線
  覆核——GitHub API 直核＋web 線查訖，結論全無 delta。**工作中誤判
  與還原**：讀共享樹本地 roadmap 見第二十四輪條目已在，誤判為
  「pass-64 推送鏈早已內含、雙邊記憶漏記」——實為對岸同小時同題
  開工、其熱寫本地檔早於推送被本側讀到；遠端歷史對帳（ad53eef9
  版仍第二十三輪）還原真相。對岸 07:16:37Z 推 ee42be8c 正式落地
  第二十四輪（roadmap +33/−1，blob 96cf6a61 首推即 byte-exact），
  內容與本側獨立覆核逐項一致（Wan-Animate-2 393★／Apache-2.0／
  pushed_at 2026-08-08；Wan-Video org 六 repos 名單；Wan3.0
  contents 僅 LICENSE＋README.md；awesome-rm 仍 2026-10-06；
  GenVidCritic／RewardVerse 無後續；VA-Judger 第十次 HF 核查失敗；
  Qwen-Image 無 image 線新貨、紅線不變；MAGI-2 開源動向
  Higgsfield 11-25「open weights coming soon」＋elsewhere.news
  11-05 Series A+ 旁證；I2V 榜首 MAGI-2 1163±14／5,943 票持穩、
  Seedance 2.5 仍第二）。依實例四十／四十二先例不碰不搶推不代推、
  零浪費轉消化驗證。**新教訓入冊：共享樹 mtime 掃描只於開工做一次
  不夠——工作中讀到的本地檔內容若與 RESTORE 時點 hash 矛盾，先以
  遠端歷史 ref 拉舊版 diff 再下判斷（本側 hash d8009827 後未重掃
  mtime，對岸 07:0x 熱寫即從此縫隙流入誤判）**。**AA 即時頁本側
  拉取失敗**（web_open_url internal error×2、curl HTTP 000），I2V
  穩定性以對岸落地版入冊為準，下次隨第二十五輪覆核。SAVE 主增量：
  head 7fb2f20c→ee42be8c（補登 pass-64 AV 榜更新／pass-65
  status.md／pass-66 對岸 round-24 三輪）、next_hour 排程
  pass-66±→pass-69±＋status.md 651→662 已落地標記勿重做；
  prepend 本 entry、trim pass-55 維持 10 entries。授權紅線未碰
  （H3／LTX／Qwen2.1／FLUX3-Klein／Krea2 不碰）。

- 2026-11-27 pass 63 @ 7fb2f20c（本 session）— ④ repo 健康小件：docs
  結構 repo-health 閘 tests/test_repo_docs_structure.py +6（純 ASCII，
  6197B）。缺口——session-state.md／status.md／model-roadmap.md 三份
  長 CJK 檔每輪全檔重推，RESTORE/SAVE 依賴的骨架不變量過去全靠手工
  掃描（每次 SAVE 後的五 heading 檢查、status.md follow-through 的
  表格 pipe 分佈比對），滑手要等下輪 RESTORE 才暴露。鎖定：已知六
  文件存在閘（glob 不靜默縮零）＋衝突標記掃描（<{7}/={7}/>{7}）＋
  表格 pipe 數逐塊一致（code fence 外）＋fence 計數成對＋
  session-state 標頭區塊（首行 ---、schema／updated_at／head／
  last_increment／next_hour 五鍵、head 為 40-hex、updated_at 可
  fromisoformat）＋SAVE 協議五 heading 齊（CJK 標題以 \u 跳脫
  構建保測試檔純 ASCII）。只鎖骨架不鎖內容（套件數／ledger／
  排程每輪皆變）。6/6 首輪即綠；CPU 全套 656→662 passed（69.9s，
  差額全為己出——測試數第三網乾淨）；純 ASCII 預檢 0 non-ascii；
  推送前第二閘 HEAD 仍 76b363b7 無競態；test blob 64656fca 首次
  推送即 byte-exact；get_commit 確認單檔 +172 無夾帶。
- RESTORE 乾淨（開工對帳）：HEAD 76b363b7＝記憶（04a94e44／
  bd5f13b7／13588fd2 三壞 commit 留史勿驚）；五檔 blob 全
  byte-exact；issue #4 兩 comments 無新（Claude 紅旗重跑未回報，
  bridge-warmup 仍阻塞）；PR open 空；tags 空（v0.2.0 未打）；
  遠端無 .github（CI.yml 待使用者）；temp 無新 GPU 輸出。對岸
  活動訊號：egg-info 04:02Z 再生（環境復位）但全程零後續檔案
  寫入——status.md 651→662 follow-through 屬其高機率候選，本側
  主動避讓（實例四十前鑑）改取零交集的 docs 結構閘。pytest
  環境存活免重裝；基線 656（67.4s）確認後動工。watchlist 未到點
  未查（下次 pass-66±——本 SAVE 已修正 next_hour 陳舊排程
  pass-63± 陷阱）。授權紅線未碰（H3／LTX／Qwen2.1／FLUX3-Klein／
  Krea2 不碰）。
- 2026-11-27 pass 62 @ a9eec28e（本 session）— ③ repo 健康小件：RunPod
  shell 入口腳本 repo-health 閘 tests/test_repo_shell_scripts.py +5
  （純 ASCII，2757B）。缺口——scripts/*.sh 五支（download_weights／
  run_inference／runpod_bootstrap／runpod_m1_eval／runpod_m2_smoke）是
  付費 GPU pod 入口，語法滑手或引用改名後的 python 入口要等到燒 pod
  時間才暴露，tests/ 對 .sh 零引用。鎖定：已知五支存在閘（glob 不
  靜默縮零）＋ bash -n 語法閘（shutil.which 缺 bash 則 skip）＋表頭
  契約（env-bash shebang／SPDX 行／set -euo pipefail）＋ wiring 閘
  二（腳本內所有 scripts/*.py|sh token——含 usage 註解——皆須存在；
  python -m theama.* 模組引用映射 src/ 實檔）。5/5 首輪即綠；CPU
  全套 651→656 passed（77.7s，差額全為己出——測試數第三網無對岸
  殘留）；test blob 12c0f165 首次推送即 byte-exact，get_commit 確認
  單檔 +83 無夾帶。**競態實例四十一（工作中偵測，零交集）**：推送前
  mtime 掃描抓到對岸 03:10:21Z 熱寫 docs/model-roadmap.md
  （53467d8f 42195B→ce0670a3 44180B 未推——疑似 watchlist／roadmap
  增量進行中）；依紀律不碰不搶推不代推，本側僅推己檔，其落地後
  下輪消化（驗 blob＋獨立重跑）。RESTORE 乾淨：開工 HEAD
  cff0ffc8＝記憶；五檔 blob 全 byte-exact；issue #4 兩 comments
  無新（Claude 紅旗重跑未回報——bridge-warmup 仍阻塞）；PR open
  空；tags 空（v0.2.0 未打）；遠端無 .github（CI.yml 待使用者）；
  temp 無新 GPU 輸出。環境事件：沙箱再重置洗掉 pytest＋theama
  （第六度同型），pip 復位後基線 651（104.3s）確認動工。watchlist
  未到點未查（下次 pass-63±）。授權紅線未碰。

- 2026-11-27 pass 61 @ cff0ffc8（對岸補登）— ③ status.md follow-through：
  P1b 行補 raw-manifest 閘↔剖析器一致性子句（check_av_manifest×
  manifest_stats：總帳不變量／文件化分歧釘定／clean manifest source
  分佈全等／剖析器對閘拒 manifest 仍 exit 0）＋套件數 642→651＋日期
  2026-11-26→27。status blob abd99d5a 17948B 首次即 byte-exact
  （+2/−2 單檔無夾帶，含 CJK 段全檔轉錄零事故）。同小時雙 cron
  同題撞車（實例四十）——另一 session 工作中偵測其改寫，依觀察窗
  紀律轉消化驗證零推送收斂。

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
  bridge-warmup 仍阻塞）；PR open 空；tags 空（v0.2.0 未打）；
  遠端無 .github（CI.yml 待使用者）；temp 無新 GPU 輸出；mtime 雙閘＋推送前
  第二閘皆乾淨。授權紅線未碰。watchlist 未到點未查（下次 pass-57±）。

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

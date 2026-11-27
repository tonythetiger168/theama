# GPU 驗收 Runbook（Claude 交接版）

GPU 執行面由 Claude 接手；本檔是唯一權威 runbook。每支腳本都是
resumable，中斷直接重跑同一條命令即可。回報格式見文末；只回報
summary 與驗收數據，不貼 log 全文。

## 0. RunPod 開 pod SOP（每次 GPU 任務的入口）

任務書看對應 GitHub issue（例如 M2 internal wan_blocks smoke = issue #4）；
本節是「怎麼開 pod」的固定流程，所有 GPU 任務共用。

1. RunPod console → **Secure Cloud** → 地區選 **CA-MTL-1**（network
   volume 所在區）。
2. GPU 過濾：**VRAM ≥ 24GB、時價 ≤ $1/hr**（RTX 4090 / A40 優先；
   A14B prod 路徑或 Ming-Image diagnostic 才需要 80GB 卡，另行指定）。
3. Template 沿用前次 smoke 的 PyTorch + CUDA 映像。
4. **掛上既有 network volume `theama-m2`**（/workspace 指向它）——
   權重、cache、smoke_out 都在上面，不重新下載。
5. Deploy → 等 Running → SSH 或 web terminal 進去。
6. **長任務（無人值守 >30 分鐘）一律帶 `THEAMA_SELF_TERMINATE=1`**
   （2026-11-27 拍板）：腳本 EXIT trap 會先把結果打包到 volume 的
   `$WORKSPACE/pub/m1_out.tgz`、再 `runpodctl remove pod $RUNPOD_POD_ID`
   刪除自己——不再依賴本機控制器在線。教訓來源：2026-11-26 M1 50 支
   run（生成 ~10.6h ≈ $6.27 在 $8 上限內），因 trinity 離線強制停止未執行，
   pod 閒置 ~16h，總帳 $15.73。配套：開 pod 的 start command 不要在
   container restart 時自動重跑任務（會覆寫 log／重跑 resume）。
   注意：現行實作「歸檔失敗仍刪 pod」的 fail-unsafe 邊界已在 PR #6
   review 標記（tar 成功才刪 pod），正式依賴前確認該修復落地。
7. 收尾鐵律：任務結束（成功或失敗皆同）→ 打包結果到 volume →
   回報 → **Delete pod（不是 Stop）** → console 確認 pod 消失 →
   **volume 保留不刪**（帶 `THEAMA_SELF_TERMINATE=1` 時 6 已自動完成
   打包＋刪 pod，本步改為驗證 volume 上 tar 存在）。
8. 費用紀律：開 pod 後立刻設手動提醒；閒置超過 30 分鐘未跑任務
   就先 delete，要用再開。

不得把 RunPod API key、SSH 密碼或任何 credential 寫進 repo、issue、
PR 或聊天回報。

## 0.1 環境前置（pod 內一次）

```bash
cd /workspace
git clone https://github.com/tonythetiger168/theama.git  # 或 git pull 至最新 main
cd theama
pip install -e ".[train]"          # torch + diffusers + DAC
# 系統層：ffmpeg（decode/mux tests 與資料管線）、tesseract-ocr + chi-tra（M7a 評分）
bash scripts/download_weights.sh   # 權重各自保留原授權，不入庫
```

硬體參考：TI2V-5B dev 路徑 24GB 可跑；A14B 建議 80GB 或 enable offload。

## 1. M1 推理重現（先跑，後面全依賴它）

```bash
python scripts/gen_eval_set.py --model ti2v-5b --out eval_out/ti2v-5b
python scripts/run_vbench.py --videos eval_out/ti2v-5b   # gate: 6 維 ±0.02 vs eval/reference.json
```

**驗收**：50 支 MP4 probe 全過（exit 0）；VBench 6 維與 reference 同量級。

## 2. P1a DPO 管線（M1 綠後）

```bash
python scripts/gen_candidates.py --out dpo_out/candidates           # 每 prompt 4 seeds
python scripts/score_eval_set.py --videos dpo_out/candidates        # → dpo_out/candidates/scores.jsonl
python scripts/check_scores.py --scores dpo_out/candidates/scores.jsonl   # 閘 1：壞列/混 scorer exit 1
python scripts/scores_stats.py --scores dpo_out/candidates/scores.jsonl   # 配對產出預覽（pairable / all_failed）
python scripts/make_pref_pairs.py --scores dpo_out/candidates/scores.jsonl --out dpo_out/pairs.jsonl   # best-vs-worst, margin >= 0.5
python scripts/check_pairs.py --pairs dpo_out/pairs.jsonl           # 閘 2：任一 unusable exit 1（壞檔由 make_pref_pairs 重建，不丟列續跑）
python scripts/pairs_stats.py --pairs dpo_out/pairs.jsonl           # 可選：margin 分佈 / winner 集中度預覽
python scripts/encode_pairs.py --pairs dpo_out/pairs.jsonl --model ti2v-5b --out dpo_out/latents
python scripts/check_pair_latents.py --manifest dpo_out/latents/pairs_latents.jsonl   # 閘 3：dropped/缺檔 exit 1
python scripts/train_dpo.py --config configs/dpo_ti2v_5b.yaml
```

**驗收**：三段預檢閘皆 exit 0 才進下一步（閘 2 不通過時以 make_pref_pairs
重建 pairs.jsonl——pair index 為 latent cache 鍵，不可丟列續跑）；
held-out 盲測 win ≥ 65%；VBench 對 DPO 前 baseline 不掉 ±0.01；
dpo_metrics.json 500 steps 收斂。

## 3. M2 真實權重 contract smoke（DAC → Wan）

```bash
python scripts/smoke_m2_real_weights.py dac
python scripts/smoke_m2_real_weights.py wan
```

**驗收**：兩支 exit 0；Wan 凍結、bridge 有梯度、peak CUDA memory 落盤。
通過後才排 AV 資料管線 GPU 訓練（`scripts/encode_av_latents.py` →
joint AV training，見 training-plan M2）。

## 4. M7a 雙路徑 T2I（可與 2、3 平行，不占同卡）

```bash
python scripts/gen_cjk_images.py --model qwen-image-2512 --out eval_out/m7a/qwen
python scripts/gen_cjk_images.py --model z-image-turbo --out eval_out/m7a/zimage
python scripts/score_cjk_text.py --images eval_out/m7a/qwen
python scripts/score_cjk_text.py --images eval_out/m7a/zimage
```

**驗收**：各 24/24 PNG（exit 0）；記下兩路的 exact_hit_rate 與
mean_char_recall 當 M7a 基線（第一輪無對標值，如實記錄）。

### 4.1 第三路徑（diagnostic-only）：Ming-Image-0.1-Design

```bash
python scripts/gen_cjk_images.py --model ming-image-design --out eval_out/m7a/ming
python scripts/score_cjk_text.py --images eval_out/m7a/ming   # scorer 本就 model-agnostic，吃一個 images dir
```

- `inclusionAI/Ming-Image-0.1-Design`（MIT，6B 文字密集設計 T2I）；卡片
  預設 12 steps／CFG 1.0 已內建於 gen 腳本 MODEL_DEFAULTS，解析度自動
  吸附 1024/2048 bucket。prompt verbatim 直送 sampler（不過官方
  rewriting 層）；純 T2I 路徑不觸及卡片的影像輸入 vision tower。
- **硬體：官方驗證配置為單卡 80 GiB**——24GB 卡預期 OOM（社群量化未
  驗證）。有 80 GiB 卡再排，不阻塞本節雙路徑驗收。
- **定位 diagnostic-only**（gen 腳本 EXPERIMENTAL_MODELS 標記，選用時印
  [note]）：24 條 seed 實測計分前不入 M7a 驗收閘。跑出結果如實記錄
  exact_hit_rate／mean_char_recall 與雙路徑基線並列；是否轉正由
  hourly agent 依數據提議。

## 5. P1b joint training（M2 主訓練迴路；3 綠後再排，與 4 可平行但同卡注意 VRAM）

### 5.0 免權重自檢（先跑，約 1 分鐘，無 cache、無 Wan 權重，GPU/CPU 皆可）

```bash
python scripts/train_joint_av.py --smoke --out runs/joint_smoke
```

**驗收**：exit 0 且印 `TRAIN_JOINT_AV_SMOKE_OK`；`runs/joint_smoke/` 內有
metrics.jsonl 與 ckpt_*。覆蓋合成 5-D Wan latent 路徑 4 步＋bridge→full
排程切換＋auto-resume（4→6 步）。此步不過就不用往下走。

### 5.1 小 cache E2E（dev 路徑 TI2V-5B，24GB）

前置：manifest 為 jsonl data-card（來源須過 `AVDataset.check_source`
黑名單）。先靜態檢查，再 decode 探針，再建雙 cache；所有 encode 皆
resumable，中斷重跑同一條命令即可。text cache 與 video backbone 無關，
**用 `--encoder` 固定成 A14B repo 建一份、dev/prod 兩路共用**（cache 以
meta.json 綁 encoder id；不帶 `--encoder` 時隨 `--model` 解析，兩路會
各建一份、prod 白白重燒一次 UMT5）。

```bash
python scripts/check_av_manifest.py --manifest data/av/manifest.jsonl
python scripts/smoke_decode_to_latents.py --manifest data/av/manifest.jsonl \
    --out data/av/probe --limit 4            # decode→latent 探針
python scripts/encode_av_latents.py --manifest data/av/manifest.jsonl \
    --out data/av/latents_dev --model ti2v-5b --limit 8
python scripts/encode_texts.py --manifest data/av/manifest.jsonl \
    --out data/av/texts --encoder Wan-AI/Wan2.2-T2V-A14B-Diffusers --limit 8
python scripts/check_joint_latents.py --av-cache data/av/latents_dev \
    --text-cache data/av/texts --validate-files  # joint cache 預檢閘：燒 GPU 前攔 dropped／壞 .pt
python scripts/train_joint_av.py --config configs/m2_joint_av_dev.yaml \
    --av-cache data/av/latents_dev --text-cache data/av/texts \
    --out runs/m2_e2e --steps 20 --bridge-steps 10 --batch-size 1 --dtype bf16
python scripts/train_joint_av.py --config configs/m2_joint_av_dev.yaml \
    --av-cache data/av/latents_dev --text-cache data/av/texts \
    --out runs/m2_e2e --steps 30 --bridge-steps 10 --batch-size 1 --dtype bf16 \
    --resume auto                            # 驗精確 resume（phase/optimizer/RNG）
```

**驗收**：check_joint_latents 先過——exit 0 且 summary served=8、
dropped 全零（--validate-files 會實載每個 .pt 抓 corrupt/non-finite）；
預檢閘不過就修 cache，不要進 trainer。trainer 載入時印
`joint cache: 8 samples (dropped: 0)`——dropped 非 0
先查 AV/text caption 交叉核對與 cache model 守衛；metrics.jsonl 每列
finite、audio/video loss 俱在且整體下降；step 10 切 full phase 後繼續
出數；resume 從 step 20 接到 30（summary resumed_from=20、
steps_completed=30）不重頭。回報 peak CUDA memory。

### 5.2 正式 run（A14B prod 路徑，80GB 或 enable offload）

text cache 沿用 5.1 建的共享庫——同 encoder 下 encode_texts 以 resume
增量補齊全量 caption，已編碼的不重燒。

```bash
python scripts/encode_av_latents.py --manifest data/av/manifest_full.jsonl \
    --out data/av/latents_a14b --model t2v-a14b
python scripts/encode_texts.py --manifest data/av/manifest_full.jsonl \
    --out data/av/texts --encoder Wan-AI/Wan2.2-T2V-A14B-Diffusers
python scripts/check_joint_latents.py --av-cache data/av/latents_a14b \
    --text-cache data/av/texts --validate-files  # 正式 run 前同一道閘：dropped≠0 即 exit 1
python scripts/train_joint_av.py --config configs/m2_joint_av.yaml \
    --av-cache data/av/latents_a14b --text-cache data/av/texts \
    --out runs/m2_run1 --steps 20000 --bridge-steps 2000 --batch-size 2 \
    --dtype bf16 --save-every 500
```

**驗收**：bridge warmup（前 2000 步）loss 穩定下降、step 2000 切 full
後無 NaN；metrics.jsonl 全程 finite；ckpt_latest 可 `--resume auto`
接續（checkpoint 恆於最終步落盤，不受 --save-every 影響）。任何 NaN
會 fail-closed 留 ckpt_nonfinite 並中止——回報該 checkpoint 路徑，
勿刪檔重跑。定期用最新 ckpt 做一次短推理目檢 AV sync（gate：offset
< 40ms，見 config acceptance）。

## 回報格式（貼回 hourly agent 或寫 docs/gpu-results.md）

每完成一節回報：`{step, command, exit_code, summary.json 內容, GPU 型號/VRAM, 耗時}`。
失敗時回報 exit code + 最後一行錯誤 + 已完成的範圍，重跑即可，不要清空輸出目錄。

## 紅線（Claude 同樣適用）

H3 / LTX-2.x / Qwen-Image-2.1 / FLUX-3-Klein / Krea-2 的權重、程式碼、
生成輸出一律不碰；資料來源過 `AVDataset.check_source` 黑名單；
第三方輸出永不進訓練集（configs `forbid_third_party_outputs: true`）。

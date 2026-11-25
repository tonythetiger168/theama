# GPU 驗收 Runbook（Claude 交接版）

GPU 執行面由 Claude 接手；本檔是唯一權威 runbook。每支腳本都是
resumable，中斷直接重跑同一條命令即可。回報格式見文末；只回報
summary 與驗收數據，不貼 log 全文。

## 0. 環境前置（一次）

```bash
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
python scripts/gen_candidates.py            # 每 prompt 4 seeds
python scripts/score_eval_set.py --videos dpo_out/candidates
python scripts/make_pref_pairs.py           # best-vs-worst, margin >= 0.5
python scripts/encode_pairs.py --pairs dpo_out/pairs.jsonl --model ti2v-5b --out dpo_out/latents
python scripts/train_dpo.py --config configs/dpo_ti2v_5b.yaml
```

**驗收**：held-out 盲測 win ≥ 65%；VBench 對 DPO 前 baseline 不掉 ±0.01；
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

## 回報格式（貼回 hourly agent 或寫 docs/gpu-results.md）

每完成一節回報：`{step, command, exit_code, summary.json 內容, GPU 型號/VRAM, 耗時}`。
失敗時回報 exit code + 最後一行錯誤 + 已完成的範圍，重跑即可，不要清空輸出目錄。

## 紅線（Claude 同樣適用）

H3 / LTX-2.x / Qwen-Image-2.1 / FLUX-3-Klein / Krea-2 的權重、程式碼、
生成輸出一律不碰；資料來源過 `AVDataset.check_source` 黑名單；
第三方輸出永不進訓練集（configs `forbid_third_party_outputs: true`）。

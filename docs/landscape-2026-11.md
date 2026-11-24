# 全球影片生成技術盤點（2026-11）

目標規格：**4K、30 秒、原生音訊、MP4 電影級短片**（Wan 3.0 / MiniMax H3 級品質）。
資料日期：2026-11-24。來源：各模型官方 model card / GitHub repo / 授權文本。

## 盤點表

| 專案 | 定位 | 關鍵規格（hard numbers） | 授權 | 對 theama 的核心啟發（要偷什麼） |
|---|---|---|---|---|
| **Wan 2.2** A14B / TI2V-5B（Alibaba，2025-07，repo 18.9k★） | 開源影片 DiT 主力 | A14B = 27B MoE（14B active/step）；TI2V-5B；5s@24fps；480p–720p；無音訊 | **Apache-2.0** | **骨幹**。高低噪聲雙專家 MoE、3D causal VAE、umt5 編碼器；生態與工具鏈最成熟 |
| **MiniMax H3**（2026-07-31 API / 08-03 開權重，12.2k★） | 開放權重 omni-modal 標竿 | 33B dense 單流 Omni Transformer（~13B 在 AdaLN 分支、inference 可 cache 不載）；Qwen3-VL-32B；3D multimodal RoPE；4–15s@24fps；本地 768p（2K 的 H3-Regenerate-2K **未開放**）；32kHz 立體聲同生；11 語言；最多 12 個 omni-reference（9 圖/3 影/3 音）；FL2VA + Ref2VA 兩 checkpoint | H3 Community License：**排除美/歐/英/韓**本地部署、$20M/yr cap、商用須標示 "MiniMax H3"、**禁止用輸出改進其他模型** | 架構參考：單流 AV 聯合生成、omni-reference 任務切分、三模組管線（Context-IR → Base → Regenerate）。**權重、程式碼、輸出全不碰** |
| **LTX-2.5**（Lightricks，2026-08-11，repo 11.5k★） | 開放權重 AV 原生 4K | 22B 非對稱雙流 DiT，雙向 cross-attention 聯合 AV + modality-aware CFG；Gemma 4 12B 文本編碼器；6–20s；720p→原生 4K HDR；原生多鏡頭；DFR（Diffusion Fidelity Rendering） | LTX-2.x Community License（<$10M ARR 免費，以上付費） | 架構參考：雙流 DiT + 雙向跨模態注意力（theama 音訊分支的主要範式）、two-stage 上 4K 的工程路徑。**不碰權重** |
| **Kandinsky 6.0**（Kandinsky Lab，2026-09/10） | MIT 授權 AV | 3B Lite / 29B Pro；44.1kHz 音訊 + lip-sync；5s@24fps | **MIT** | 音訊分支可對照/借用的**乾淨授權**實作 |
| **MAGI-2-preview**（Sand AI，2026-08） | Apache-2.0 大 MoE | 114B MoE / 6B active；有音訊 | **Apache-2.0** | 乾淨授權的 AV MoE 參考實作 |
| **LongCat-Video**（Meituan，2025-10） | MIT 長影片方向 | 13.6B | **MIT** | 長影片延續策略參考 |
| HunyuanVideo 1.5（Tencent，2025-11，4.6k★） | 輕量 1080p | 8.3B | Tencent Community License（排除歐/英/韓） | 僅參考，授權不乾淨 |
| Seedance 2.5 / Kling 4.0 / Wan 3.0 API / Veo 3.1 | 閉源前沿 | Seedance 2.5：30s、4K、50 個多模態參考、音訊同生；Kling 4.0：30s | API only | 品質對標線（M5 eval set 的對手），不偷實作 |

## 差距分析（theama 目標 vs 全球現況）

| 前沿指標 | 最佳開源現況 | 差距 | 對策 |
|---|---|---|---|
| 30s 單次生成 | 開源最長 15–20s（H3 15s、LTX-2.5 20s）；30s 僅閉源（Seedance 2.5、Kling 4.0） | 🔴 | THEAMA-Extend：SVI 式接續，5s × 6 段 → 30s |
| 原生 4K | LTX-2.5 有但授權不乾淨；H3 的 2K stage 未開放（任何路線都得自建 HR stage） | 🔴 | THEAMA-4K：自建 two-stage regenerate（720p→2K→4K） |
| 原生音訊同生 | H3 / LTX-2.5 / Kandinsky 6.0 / MAGI-2 已驗證可行 | 🟡 | Wan 2.2 上加並聯 Audio DiT（雙流 cross-attention） |
| 全棧乾淨授權 + 4K + 30s + AV | **無人同時滿足** | — | **theama 的定位與賣點** |

## 誠實範圍

- Wan 2.2 骨幹（720p/5s）離 H3（768p/15s/音訊）有不小差距：theama 的貢獻在「整合 + 延長 + 升解析 + 加音訊」，不是重新發明 base model。
- 4K/30s 的 VRAM 與訓練成本是硬約束：M1–M2 在單卡 24–80GB 驗證，M3 之後需要多卡。
- H3/LTX-2.5 只讀 paper、model card 與公開架構描述做 clean-room 參考；**不逆向權重、不用其輸出**。

## 加掛：Avatar 互動數字人線盤點（2026-11-24 補）

對標：Vidu S2-Avatar 級實時互動數字人。

| 專案 | 定位 | 關鍵規格 | 授權 | 對 theama-avatar 的啟發 |
|---|---|---|---|---|
| **Vidu S2-Avatar**（ShengShu × 清華，2026-09-15，arXiv 2609.11638） | 實時互動數字人標竿 | 720p 實時（S1 為 540p）@25–42fps；串流中可動態換參考圖（VLM Agent）；大動作（跳舞）指令跟隨；Self-Replay Forcing 防長流漂移；消費級 GPU 實時；StreamAV-Bench 九項第一 | 閉源 API | **品質對標**；SRF 概念 clean-room 參考 |
| **Wan-Animate**（Alibaba） | 開源 avatar 動畫 | 圖生影片角色動畫，可自架；Wan 家族同門 | **Apache-2.0** | **Avatar 線首選底座**——與主線 Wan 2.2 同家族，整合成本最低 |
| **LongCat-Video-Avatar**（Meituan） | 開源單圖+音訊說話人 | 單照片 + 音訊 → talking avatar | **MIT** | 備選底座 / 對照實作 |
| HunyuanVideo-Avatar（Tencent，2.2k★） | 開源情緒可控多角色 | 情緒控制、多角色 talking video | Tencent Community（排除歐/英/韓） | 僅架構參考 |
| OmniHuman 2.0（ByteDance，2026-10-15） | 閉源長片 avatar | 單圖 + 最長 5 分鐘音訊一鏡到底；720p；多人/動物/動漫 | 閉源 API（$0.07/s） | 長片 avatar 對標線 |
| SadTalker / Hallo / LatentSync | 早期開源 talking-head | 唇形同步管線 | Apache-2.0 / MIT / Apache-2.0 | eval baseline 與唇形管線參考 |
| Tavus Phoenix-4 / HeyGen Avatar IV | 閉源商用實時 avatar | sub-600ms 延遲（Tavus）；177+ 語言（HeyGen） | 閉源 API | 商用延遲標竿 |

### Avatar 線差距分析

| 前沿指標 | 開源現況 | 差距 | 對策 |
|---|---|---|---|
| 實時串流生成（720p、消費級 GPU） | 開源多為離線 clip 生成；實時僅 Vidu S2 / Tavus 等閉源 | 🔴 | causal DiT 改造 + rolling KV cache + replay anchor（SRF 概念自建） |
| 串流中動態換參考圖 | 僅 Vidu S2 | 🔴 | VLM agent + identity embedding 熱替換介面（M6c） |
| 單圖+音訊 talking avatar（離線） | Wan-Animate / LongCat-Video-Avatar 已可自架 | 🟢 | 直接以 Wan-Animate 起步（M6a） |

## 加掛：文生圖線盤點（2026-11-24 補）

對標：開源 T2I 最強品質 + 授權乾淨；CJK 文字渲染為差異化重點。
榜單錨點：LMArena text-to-image（2026-11-22）與 Artificial Analysis 開放權重榜（2026-11-02）。

| 專案 | 定位 | 關鍵規格 | 授權 | 對 theama-image 的啟發 |
|---|---|---|---|---|
| Qwen-Image 2.1 / 2.1-Turbo（Alibaba，2026-09/10） | 開放權重品質頂點之一 | LMArena 1227（開放權重最高）；7B 生成 + 8B 文本編碼器；生成/編輯/透明圖一體 | **Qwen Research License（非商用）** | 品質對標；**不能當底座** |
| FLUX 3 Image Klein（BFL，2026-10） | 開放權重品質並列頂點 | 5.6B；AA Elo 1043；10 參考圖 + bbox 版面引導；INT8 8GB 可跑 | FLUX Kommunity（<$5M、輸出限定） | 品質對標；授權不乾淨不當底座 |
| Fibo 2.0（Bria） | 控制導向 | AA Elo 1039 | CC BY-NC 4.0 | 僅參考 |
| **Qwen-Image-2512 / Edit-2511**（Alibaba，2025-12） | **乾淨授權品質擔當** | 20B；多語文字渲染最強檔；生成 + 編輯 | **Apache-2.0** | **品質底座**（M7a/M7b） |
| **Z-Image Turbo / base**（Alibaba Tongyi，2025-11 / 2026-01） | 乾淨授權速度擔當 | 6B；8 NFE；16GB；雙語文字渲染；HF LoRA 838+146 | **Apache-2.0** | **快速/dev 路徑**；LoRA 生態現成 |
| FLUX.2 [klein] 4B（BFL，2026-01） | 生成+編輯一體小模型 | 4B；多參考編輯；~13GB | **Apache-2.0** | 編輯管線對照 |
| HiDream-O1-Image（HiDream，2026-05） | MIT 統一架構 | 8B；生成/編輯/個人化一體；原生 2K；無獨立 VAE | **MIT** | 架構對照（pixel-space 無 VAE） |
| GLM-Image（Z.ai，2026-01） | AR+DiT 混合 | 9B AR + 7B decoder；雙語文字最強檔 | **MIT** | 文字渲染對照 |
| LongCat-Image / MOSS-Image / ERNIE-Image | 乾淨授權中堅 | 6B / 5B / 8B | Apache-2.0 | 備選與對照 |
| 閉源前沿（GPT Image 2.5、Nano Banana 2、Seedream 5.0 Pro 等） | 品質天花板 | LMArena 1247–1429 | API only | M7 對標線；Seedream 5.0 Pro 繁中渲染弱點 = 我方切入點 |

### 文生圖線差距分析

| 前沿指標 | 開源現況 | 差距 | 對策 |
|---|---|---|---|
| 頂尖整體品質 | Qwen-Image 2.1 / FLUX 3 Klein 並列，但授權皆非全乾淨 | 🟡 | Qwen-Image-2512（Apache-2.0）+ LoRA/蒸餾收斂（遵守各自授權） |
| CJK / 繁中文字渲染 | Qwen-Image 系最強；閉源 Seedream 5.0 Pro 繁中常錯 | 🟢 | 主攻點：100-prompt 繁中文字評測集 |
| 生成+編輯+多參考一體 | Qwen-Image-Edit-2511、FLUX.2 klein 4B 已驗證 | 🟢 | M7b 直接接上 |
| 與影片/avatar 線整合 | 無現成開源全棧 | — | **theama 定位**：keyframe → Film I2V、identity ref → Avatar |

# RECAP の再現実験（smolVLA、LIBERO-10 Task 0）の結果

RLinf の RECAP（価値モデル → advantage → advantage 条件付き SFT）を、`lerobot/smolvla_libero` で、LIBERO-10 の Task 0 に適用した実験の全記録。期間は 2026-10-01〜10-04。
GPU は、主に RTX 6000 Ada（48GB）。一部の評価と実験 S は、別環境の RTX A5000（24GB、コンテナのメモリ上限 25GB）で実行した。

**結論（先に）**: 試した範囲では、ベースライン（`success_once = 0.40`）を、統計的に有意に上回る設定は見つからなかった。
- 価値モデルは、評価データで順位をよく予測した（value_spearman 0.96）。しかし、そこから作った **フレーム単位の advantage は、成功と失敗をほとんど分けていない**（AUC 0.565）。
- **"Advantage: positive" のプロンプトの有無で、成功率が変わらない**（条件づけが、ほとんど学ばれていない）。
- **成功エピソードのフレームだけを positive にした強いラベル（実験 S）でも、改善しなかった**（0.44）。
- 最も低かったのは、positive を上位 10% に絞った実験 R10（0.32）。ただし、これもベースラインとの差は有意ではない。
- 評価の誤差が大きく（50 試行で標準誤差 約 0.07）、約 ±0.1 以内の差は、検出できない。

関連: STEAM の結果は `steam_smolvla_libero10_task0.md`。方法の背景は `docs/pi06_RECAP_ja.md`、引き継ぎは `docs/handoff_step2_onward.md`。

---

## 1. 評価の方法

- 対象: LIBERO-10 の Task 0（"put both the alphabet soup and the tomato sauce in the basket"）、固定初期状態 50 個を 1 回ずつ（50 試行）。
- 評価コマンド: `RLinf/evaluations/run_eval.sh libero libero_10_smolvla_eval`（`+env.eval.task_id_filter=[0]`）。`advantage_prompt=true` のとき、指示文に "\nAdvantage: positive" を付ける。
- 評価の並列数が、実験によって違う。
  - **10 環境**（`total_num_envs=10`、`max_steps_per_rollout_epoch=2600` = 520 × 5 周）: ベースラインの測定、B と C の最初の評価、R10。
  - **2 環境 × 25 周**（`total_num_envs=2`、`max_steps_per_rollout_epoch=13000`）: プロンプトの有無の評価（B・C）と S。メモリ上限 25GB の環境で、10 環境が Ray の OOM で落ちるため。
- 初期状態は同じ 50 個のはずだが、並列数によって乱数の使われ方が変わる可能性がある（推測）。評価の `noise_seed` は null で、毎回、VLA のノイズが異なる。
- 信頼区間は Wilson の 95% 区間。p 値は、ベースライン（20/50）との Fisher の正確検定（両側）。

## 2. 成功率の全一覧

| 実験 | 条件 | 評価の環境 | 成功数/試行数 | success_once | 95% 区間 | Fisher p（vs ベースライン） | success_at_end | return（平均） |
|---|---|---|---|---|---|---|---|---|
| **ベースライン** | `smolvla_libero`（学習なし） | 10 環境 | 20/50 | 0.40 | [0.28, 0.54] | — | 0.40 | 2.0 |
| B: global_step_1000 | q30、sft 50:50、プロンプトあり | 10 環境 | 23/50 | 0.46 | [0.33, 0.60] | 0.69 | 0.44 | 2.3 |
| B | 同上、プロンプトあり | 2 環境 | 20/50 | 0.40 | [0.28, 0.54] | 1.00 | 0.38 | 2.0 |
| B | 同上、プロンプトなし | 2 環境 | 21/50 | 0.42 | [0.29, 0.56] | 1.00 | 0.42 | 2.1 |
| C: global_step_2000 | B から 1000 ステップ追加（重みなし、q30）、プロンプトあり | 10 環境 | 22/50 | 0.44 | [0.31, 0.58] | 0.84 | 0.44 | 2.2 |
| C | 同上、プロンプトあり | 2 環境 | 24/50 | 0.48 | [0.35, 0.61] | 0.55 | 0.46 | 2.4 |
| C | 同上、プロンプトなし | 2 環境 | 25/50 | 0.50 | [0.37, 0.63] | 0.42 | 0.48 | 2.5 |
| **R10**: global_step_1450 | B から 450 ステップ追加（重みなし、**上位 10%**）、プロンプトあり、1 回目 / 2 回目 | 10 環境 | 16/50、16/50（計 32/100） | 0.32、0.32 | [0.24, 0.42]（計） | 0.37 | 0.30、0.32 | 1.6、1.6 |
| R10 | 同上、プロンプトなし、1 回目 / 2 回目 | 10 環境 | 18/50、17/50（計 35/100） | 0.36、0.34 | [0.26, 0.45]（計） | 0.59 | 0.36、0.34 | 1.8、1.7 |
| **S**: global_step_1450 | B から 450 ステップ追加（**成功エピソードのフレームを positive**）、プロンプトあり、1 回目 / 2 回目 | 2 環境 | 22/50、22/50（計 44/100） | 0.44、0.44 | [0.35, 0.54]（計） | 0.73 | 0.42、0.44 | 2.2、2.2 |

プロンプトの有無の比較（Fisher の正確検定）: B で p = 1.0、C で p = 1.0、R10 で p = 0.76。**差は、いずれも有意ではない。**

- ベースラインは、50 試行を 1 回測っただけ（`tmp/baseline_eval_libero.log`）。同じ手順で、繰り返し測っていない。
- B と C の評価は、10 環境では 0.46 > 0.44、2 環境（プロンプトあり）では 0.40 < 0.48 と、大小が入れ替わっている。**B と C の優劣は、50 試行では決められない。**

## 3. データ

- **旧データは破棄した。** 旧データは、成功しても 520 ステップまで続けていたため、return が「何ステップ目か」と「成否」だけで決まっていた（価値 V の分散の約 92% を説明）。成功で打ち切る設定（`ignore_terminations=False`、チャンク境界で打ち切り）で、再収集した。
- **train（rollout）**: 4,122 本、1,757,810 フレーム、成功 1,689 本（41.0%）。成功の長さは平均 292（230〜520）、失敗は常に 520。
  - 3 環境 × 4 ラウンドで収集し、LeRobot v2.1（動画）に変換、元データと照合済み（PSNR 34.6 dB 以上）。
  - 11 個（`libero10_task0_train_v2_r{R}_m{M}`、R1: m0〜m2、R2: m1・m2、R3: m0〜m2、R4: m0〜m2）を、**ファイルレベルで 1 つにマージ**（`libero10_task0_train_v2_merged`）。Step4 のローダは、データセットごとに同じ重みを割り当てる（`balance_dataset_weights`）ため。
  - 10 フレームしかない偽エピソード 3 本（初期状態プールの境界で、環境がリセットされずに出る収集のアーティファクト）は、変換時に除外した（`--min-frames 100`）。
- **sft**: Task 0 の人のデモ 3 本（836 フレーム、20fps、`libero10_task0_sft_target3`）。元は全 10 タスク × 3 本の 30 本（8,005 フレーム、`libero10_task0_sft`）で、π0.5 版のデータセットから取得した。Step2 は全 30 本を使い、Step3 以降は Task 0 の 3 本だけを使った（元の RECAP は全タスクの 30 本を使う）。
- **eval 用**（Step2 の監視のみ。学習には使わない）: `libero10_task0_eval_v2_m{0,1,2}`、計 64 本（22 + 21 + 21、成功率 36〜38%）。
- LeRobot v3.0 版（Step4 用）: `data/smolvla_recap/v30/` に、sft_target3、train_v2_merged がある。

## 4. Step1: return

- 1 ステップあたり −1（報酬）。失敗エピソードは、終端に −300（`fail300`）。gamma = 1.0。範囲 [−819, 0]。
- 成功は、先頭が −(n−1)、末尾が 0。失敗は、先頭が −(n−1)−300、末尾が −300。
- 全 4,122 本で、ルールとの不一致は 0。ファイル: 各データセットの `meta/returns_fail300.parquet`。

## 5. Step2: 価値モデル（4,000 ステップ）

- モデル: SigLIP2 + Gemma3-270M + Critic Expert。201 個のビン（return を正規化した範囲）への分類として学習。
- データ: sft（全 30 本）+ train 11 個（各 `weight: 1.0`、`type: rollout`）。eval は 3 個（各 300 サンプル）。
- 設定: global batch 256、micro batch 16、`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、`runner.max_steps=4000`、300 ステップごとに保存と eval。
- 時間: 2026-10-02 08:13 〜 10-03 04:54（20 時間 38 分）。約 17〜20 秒/ステップ（GPU のサーマルスロットリングで、クロックが 465〜2295 MHz に変動）。エラー・メモリ不足・再起動なし。
- 最終 checkpoint: `RLinf/logs/recap_value_model_sft-20261002-08:13:11/step2_value_sft_v2/checkpoints/global_step_4000`（6.8GB）。
- 最終 eval（3 セット平均）: **value_spearman 0.963、loss 2.298、cat_acc_best 0.325、cat_acc_neighbor 0.456、mae 0.066**。セットごとの value_spearman は、0.985（m0）、0.982（m1）、0.921（m2）。
- 最終の train: loss 2.55、value_spearman 0.861。

eval の推移（TensorBoard、3 セット平均）:

| ステップ | loss | value_spearman | cat_acc_best | cat_acc_neighbor | mae |
|---|---|---|---|---|---|
| 300 | 3.476 | 0.929 | 0.183 | 0.311 | 0.211 |
| 600 | 2.803 | 0.944 | 0.208 | 0.362 | 0.147 |
| 900 | 2.484 | 0.966 | 0.258 | 0.446 | 0.042 |
| 1200 | 2.464 | 0.957 | 0.251 | 0.416 | 0.022 |
| 1500 | 2.433 | 0.977 | 0.258 | 0.393 | 0.089 |
| 1800 | 2.389 | 0.916 | 0.270 | 0.430 | 0.027 |
| 2100 | 2.366 | 0.942 | 0.261 | 0.425 | 0.015 |
| 2400 | 2.232 | 0.952 | 0.304 | 0.476 | 0.023 |
| 2700 | 2.321 | 0.964 | 0.305 | 0.448 | 0.031 |
| 3000 | 2.274 | 0.948 | 0.303 | 0.471 | 0.026 |
| 3300 | 2.257 | 0.936 | 0.337 | 0.475 | 0.016 |
| 3600 | 2.180 | 0.971 | 0.346 | 0.482 | 0.005 |
| 3900 | 2.257 | 0.972 | 0.304 | 0.452 | 0.012 |
| 4000 | 2.298 | 0.963 | 0.325 | 0.456 | 0.066 |

- value_spearman は、300 ステップですでに 0.93 あり、その後は 0.92〜0.98 で推移。loss は、2,400 ステップ以降、2.18〜2.32 でほぼ横ばい。
- **公式は、約 30,000 ステップの設定で、18,000 ステップの checkpoint を使う**（タグ `ckpt18000`）。今回は 4,000 ステップで、約 1/4。
- cat_acc_best は、ビンが 201 個と細かいため、低く出る（ランダムは約 0.5%）。

## 6. Step3: advantage の計算

- 入力: Step2 の `global_step_4000`、sft_target3（836 フレーム、sft タイプで全 positive）、train_v2_merged の先頭 622,430 フレーム（`+advantage.max_samples=622430`、r1_m0・r1_m1・r1_m2・r2_m1 の 1,458 本）。計 623,266 サンプル。
- 設定: 先読み幅 N = 10、`advantage.positive_quantile=0.3`、`returns_tag=fail300`、タグ `fail300_N10_q30`、batch 64、DataLoader のワーカー 4 個。
- 時間: 2026-10-03 05:13 〜 09:22（4 時間 9 分）、約 40〜60 サンプル/秒（スロットリングで変動）。
- 結果: advantage の平均 0.0012、標準偏差 0.058、範囲 [−0.66, 0.63]。**閾値（70 パーセンタイル）0.0058**。positive は合計 186,980 サンプル（rollout の positive 率 29.98%）。
- 残りの約 113 万フレーム（全 4,122 本のうちの 2,664 本）には advantage がなく、Step4 のローダが警告のうえ学習から除外する。
- 閾値のみの再計算（relabel）: `relabel_advantages.py` で、上位 20%（`fail300_N10_q20`）と 10%（`fail300_N10_q10`）を作成（各 約 4 分、GPU 不要）。relabel は sft を区別しないため、sft_target3 の advantage は、手作業で全 True に直した（デモは常に positive、という論文の扱い）。
- 成功エピソードに基づくラベル（`fail300_outcome`、実験 S 用）: 成功エピソード 594 本のフレームを positive（173,150 フレーム、27.82%）、sft は全 positive。

### 診断: advantage のフレーム単位のラベルは、成功と失敗をあまり分けない
先頭 1,458 本（成功 594 本、失敗 864 本）の集計:

| | フレーム数 | positive の割合 | advantage の平均 | advantage の標準偏差 |
|---|---|---|---|---|
| 成功エピソードのフレーム | 173,150 | **37.8%** | +0.0081 | 0.077 |
| 失敗エピソードのフレーム | 449,280 | **27.0%** | −0.0016 | 0.048 |

- フレーム単位の AUC（成功 vs 失敗）: **0.565**。エピソード平均 advantage の AUC: **0.995**。
- positive の約 65% は、失敗エピソードのフレーム。positive 内の成功フレームの比率は 35%（全体の基準 27.8%）。
- value_current と return の相関（フレーム単位）: 0.887。
- positive の割合を絞ったときの、positive 内の成功比率（基準 27.8%）:

| 上位 | 閾値で絞った positive 内の成功比率 | 基準比 | positive フレーム数 |
|---|---|---|---|
| 30% | 0.351 | 1.26 倍 | 186,729 |
| 20% | 0.397 | 1.43 倍 | 124,486 |
| 10% | 0.514 | 1.85 倍 | 62,243 |
| 5% | 0.598 | 2.15 倍 | 31,122 |

- フレーム位置別の positive 率（成功 / 失敗）: 0〜50 で 46.1% / 20.8%、51〜100 で 49.8% / 20.3%、101〜200 で 35.2% / 25.3%、201〜300 で 25.9% / 33.7%、301〜400 で 39.4% / 26.5%、401〜520 で 48.8% / 28.5%。成功と失敗の差は、時間位置で向きが変わり、一様ではない。

## 7. Step4: advantage 条件付き SFT

### 共通の設定
- ベース: `models/smolvla_libero`。fp32、`use_orig_params=true`、FSDP `no_shard`、gradient checkpointing なし。
- global batch 256（micro batch は GPU による: 48GB は 16、24GB は 8）、AdamW（beta 0.9 / 0.95、eps 1e-8、weight decay 1e-10）、勾配クリップ 10。
- 学習率 1e-5。**warmup が終わった後は一定**（`warmup_style: cosine` は、warmup の形のみ。減衰はしない）。
- action chunk 50。positive のフレームには "\nAdvantage: positive" を付け、10% の確率で外す（`unconditional_prob: 0.1`）。negative には付けない。
- データ: sft_target3（836 フレーム）+ train_v2_merged（advantage のある先頭 622,430 フレーム）。
- DataLoader のワーカー 4 個（コンテナのメモリ 25GB 対策）、`HF_HUB_OFFLINE=1`、シード 42。
- micro batch の探索: 32 はメモリ不足、16 が動作（48GB）。

### 抽出の重み（sft と rollout の混合比）の注意
- **元の RLinf（`ReCapMixtureDataset`）**: `balance_dataset_weights=true` のとき、データセットの選ばれる確率が `weight × フレーム数` に比例（フレーム均等）。公式設定（`true` + 両方 `weight: 1.0`）では、sft は約 0.5%。
- **このプロジェクトの smolVLA 用ローダ（`smolvla_cfg.py`、パッチ追加分）は、`true` と `false` の意味が逆**: `true` で `weight ÷ フレーム数`（データセット均等、sft と rollout が 50:50）。`false` がフレーム均等（元の RLinf の `true` と同じ挙動）。
- そのため、前半の 1〜1,000 ステップ（`true`）は、公式から外れ、sft（836 フレーム）が全体の 50%（1 フレームあたり約 300 回の繰り返し）になっていた。後半（`false`）は、公式に近く、sft は約 0.13%。
- 元論文は、デモと自律データの比率を規定せず、全データを一様に扱う（"the dataset consists of all of the data collected so far"）。デモの improvement indicator は True に固定。

### 7.1 前半（1〜1,000 ステップ、sft 50:50、q30）→ B
- 実行: 2026-10-03 09:38 〜 19:12（9 時間 34 分、約 34 秒/ステップ。開始直後は 19 秒、後に 40 秒まで低下）。`global_step_1000` で止めた（評価のため）。
- 損失（100 ステップごとの平均、grad_norm）: 1〜100: 0.328（3.12）、101〜200: 0.128（0.94）、301〜400: 0.107（0.87）、501〜600: 0.098（0.79）、701〜800: 0.093（0.81）、901〜1000: **0.090**（0.77）。
- checkpoint: `RLinf/logs/20261003-09:38:15-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_1000`（と `global_step_900`）。
- 評価: 0.46（23/50、10 環境）。

### 7.2 C（1,001〜2,000 ステップ、重みなし、q30）
- `global_step_1000` から再開（`+runner.resume_dir`）。`balance_dataset_weights=false`、`runner.max_steps=2000`、`total_training_steps=2000`、`lr_warmup_steps=100`、`save_interval=100`。
- 実行: 10-03 20:10 〜 10-04 07:01（10 時間 47 分、約 35〜43 秒/ステップ）。wandb run id `8pdouya0`（project `libero_RL`、実験名 `step4_recap_noweight_resume1000`）。
- 損失: 1001〜1100: 0.148（grad_norm 0.82）、1401〜1500: 0.139、1801〜1900: 0.133、1901〜2000: **0.131**（0.75）。重みなしに切り替えると、sft の暗記が減り、損失が 0.09 から 0.148 に上がった（推測）。
- checkpoint: `RLinf/logs/20261003-20:10:02-libero10_task0_recap_smolvla/step4_recap_noweight_resume1000/checkpoints/global_step_2000`（と `global_step_1900`）。
- 評価: 0.44（22/50、10 環境）。

### 7.3 R10（1,001〜1,450 ステップ、重みなし、上位 10%）
- `global_step_1000` から再開、advantage は `fail300_N10_q10`（rollout の positive 10.0%、sft は全 positive）。micro batch 16、`SAVE=150`。
- 実行: 10-04 08:22 〜 12:45（4 時間 19 分、34.65 秒/ステップ）。wandb run id `h40npem7`（実験名 `step4_expR10_q10_1450`）。
- 損失（50 ステップごと）: 1001〜1050: 0.149、1151〜1200: 0.143、1301〜1350: 0.138、1401〜1450: 0.139。最終 0.134。
- 評価: プロンプトあり 0.32、0.32（計 32/100）、プロンプトなし 0.36、0.34（計 35/100）。
- checkpoint: `RLinf/logs/20261004-08:22:11-libero10_task0_recap_smolvla/step4_expR10_q10_1450/checkpoints/global_step_1450`。

### 7.4 S（1,001〜1,450 ステップ、重みなし、成功エピソードのフレームが positive）
- 実行環境: RTX A5000（24GB）、micro batch 8。`global_step_1000` から再開、advantage は `fail300_outcome`（rollout の positive 27.8% = 成功エピソードのフレーム、sft は全 positive）。
- 実行: 10-04 09:55 〜 14:53（4 時間 53 分、39.1 秒/ステップ）。wandb run id `49lx4tdp`（実験名 `step4_expS_outcome_1450`）。エラー・再開なし。
- 損失（50 ステップごと）: 1001〜1050: 0.149、1151〜1200: 0.143、1301〜1350: 0.138、1401〜1450: 0.140。最終 0.134、grad_norm 0.73。
- **R10 と S の損失の曲線は、ほぼ同じ**（約 0.149 → 0.139）。ラベルが大きく違うのに、損失が変わらない。損失は、ラベルの質を反映していない。
- checkpoint: `RLinf/logs/20261004-09:55:21-libero10_task0_recap_smolvla/step4_expS_outcome_1450/checkpoints/global_step_1450`（と 1350）。
- 自動評価は、10 環境で OOM になって失敗した（`OOM kill reason: Memory usage 23.78GB exceeded threshold of 23.75GB (95.0% of 25GB)`）。スクリプトは、失敗を検出せず、結果が空のまま「ALL DONE」と出力した。評価は、`scripts/recap_eval/expS_eval_2envs.sh`（2 環境 × 25 周）で、別途やり直した。
- 評価: 0.44、0.44（計 44/100、95% 区間 [0.35, 0.54]）。2 回の評価で、各試行の成否は一致しない（別のサンプル）。

## 8. 診断実験: プロンプトの有無（B・C）

B（global_step_1000）と C（global_step_2000）を、`advantage_prompt=true` と `false` の両方で評価した（2 環境 × 25 周、各 50 試行）。評価スクリプトは `scripts/recap_eval/prompt_eval_2envs.sh`。

| checkpoint | `advantage_prompt` | 成功数/試行数 | success_once | 95% 区間 |
|---|---|---|---|---|
| B | true | 20/50 | 0.40 | [0.28, 0.54] |
| B | false | 21/50 | 0.42 | [0.29, 0.56] |
| C | true | 24/50 | 0.48 | [0.35, 0.61] |
| C | false | 25/50 | 0.50 | [0.37, 0.63] |

- プロンプトありとなしの差は、B で 1 件、C で 1 件（いずれも「あり」の方が低い）。R10 でも、「あり」0.32 に対し「なし」0.35。**"Advantage: positive" の条件づけが行動を変えている証拠はない。** ただし、50 試行では、小さな効果は検出できない。

## 9. 考察

1. **改善は確認できていない。** 全実験で、ベースラインとの差は、Fisher の p = 0.37〜1.0。95% 区間は、すべてベースラインの値（0.40）を含む。
2. **条件づけが学ばれていない。** プロンプトの有無で、成功率が変わらない（3 つの checkpoint で一貫）。
3. **ラベルの質は、ボトルネックと言い切れない。** 成功エピソードの成否という、ほぼ完全なシグナル（S）でも、改善しなかった。ただし、S も、条件づけが効かない状態で学習したため、「ラベルの強さ」と「条件づけの効き」は、分けて判断できない。
4. **損失は、成功率の指標になっていない。** 全実験で 0.13 前後に収束し、R10 と S で曲線が同じ。
5. **positive を絞りすぎると悪化する可能性がある**（R10 0.32）。ただし、有意ではない。
6. **学習量が、公式の約 1/15。** 価値モデルは約 1/4（4,000 ステップ）、方策は追加 450〜1,000 ステップ（合計 1,450〜2,000）。
7. **原因の候補（検証していない）**: (a) 学習量の不足、(b) `smolvla_libero` が、この rollout を集めた方策そのもので、伸びしろが小さい、(c) ベースラインの値（0.40）が、50 試行の 1 回の測定で、真の値とずれている、(d) 評価の誤差が大きい。

## 10. 元の RECAP との違い

| 項目 | 元の RECAP（π0.5、公式） | 今回 |
|---|---|---|
| sft と rollout の混合 | フレーム均等（sft 約 0.5%） | 前半 50:50（逆の意味のローダ）、後半フレーム均等（sft 約 0.13%） |
| sft | 全タスクの 30 本 | Task 0 の 3 本（Step2 は全 30 本） |
| 価値モデルの学習量 | 約 30,000 ステップの設定、18,000 ステップの checkpoint を使用 | 4,000 ステップ |
| 方策の学習量 | 30,000 ステップ、batch 512、warmup 5,000 | 追加 450〜1,000 ステップ（合計 1,450〜2,000）、batch 256、warmup 100 |
| 使う rollout | 約 4,096 本 | 4,122 本を収集、advantage を付けたのは 1,458 本（全体の約 35%） |
| ベースライン → 結果 | 48.8% → 66.5%（π0.5、few-shot SFT のベースライン） | 0.40 → 0.32〜0.50（有意差なし） |
| デモの advantage | 常に positive | 同じ |

## 11. 未実施・結果待ち
- 別環境で、実験 P（positive のみ、プロンプトなしで評価）と T（negative に "Advantage: negative" も付ける）が、STEAM のラベルで実行された。結果は `steam_smolvla_libero10_task0.md` を参照（P 0.52、T 0.36）。RECAP のラベルでは、未実施。
- ベースラインを、同じ手順で複数回測る（100〜200 試行）。
- 学習量を大きく増やす実験（例: batch 64 で 8,000 更新、学習率の減衰を入れる）。
- 価値モデルの追加学習（18,000 ステップ前後）、先読み幅 N の変更（20 など）、全 4,122 本の advantage の計算。
- advantage のフレーム単位の精度を上げる方法の検討。

## 12. 運用上の記録（再現時の注意）
- **環境ごとの準備**（`scripts/preflight_collect_env.sh`）: `/root` の `uv` の Python（`uv python install 3.12.11`）、`~/.libero/config.yaml`（LIBERO が入力待ちで落ちる）、システムのライブラリ `libegl1` と `libopengl0`（`eglQueryString` の AttributeError）。
- **メモリ上限 25GB の環境**: 評価の 10 並列は OOM で落ちる。`EVAL_ENVS=2 EVAL_STEPS=13000` を使う（`scripts/resume_step4_noweight.sh`）。
- **GPU のサーマルスロットリング**: RTX 6000 Ada が 89〜93℃で、クロックが最大 3105MHz の 15〜74% に下がり、速度が 17 → 43 秒/ステップに変動した。
- **自動実行スクリプトの不具合（修正済み）**:
  1. Step2 の supervisor が、checkpoint の保存先を `step2_value_sft` と誤り（実際は `step2_value_sft_v2`）、整理と、落ちたときの自動再開が効いていなかった（約 1,050 ステップ時点で発見、学習は止めずに引き継いだ）。
  2. Step4 の再開で、実験名を変えたため保存先が変わり、同じ種類の不具合が再発。さらに、検索パスを広くしたため、**他の実験（STEAM）の checkpoint 4 個（global_step_200 / 400 / 600 / 800）を削除した**。復元できない。以降、検索対象は、その実験の保存先だけ。
  3. 複数環境で、同じ `tmp/` を共有すると、ログのファイル名が衝突した。ファイル名に実験名を入れて解消した。
  4. 評価が失敗しても、「ALL DONE」と出力する（失敗を検出しない）。未修正。
- プロセスの停止で、`ps | grep` のパターンが自分のシェルのコマンド文字列に一致し、シェルごと落ちる事故が 1 回あった（PID を `awk` で厳密に指定して回避。CLAUDE.md の `pkill` 禁止と同じ種類の問題）。
- 学習の再開（`+runner.resume_dir`）は、`global_step_1000` からの再開で動作を確認済み。落ちたときの自動再開は、実際には発動していない。
- wandb: project `libero_RL`。再開分から TensorBoard と並行して記録（前半の 1〜1,000 ステップは TensorBoard のみ）。API キーは `~/.bashrc` の `WANDB_API_KEY` を読む。

## 13. 再現手順と成果物
1. Step2: `scripts/run_step2_v2.sh` と `scripts/step2_v2_supervisor.sh`（4,000 ステップ、300 ステップごとに保存、直近 2 つを保持）。
2. Step3: `tmp/run_step3_v2.sh`（`compute_advantages.py`、`+advantage.max_samples=622430`）。
3. relabel: `RLinf/examples/offline_rl/advantage_labeling/recap/process/relabel_advantages.py`（`--source_tag fail300_N10_q30 --new_tag ... --positive_quantile ...`）。
4. Step4: `scripts/run_step4_onward.sh`（初回）、`scripts/resume_step4_noweight.sh`（再開。環境変数 `BALANCE`、`ADV_TAG`、`EXP_NAME`、`MAX_STEPS`、`SAVE`、`MB`、`N_EVAL`、`EVAL_ENVS`、`EVAL_STEPS`、`BASE_CKPT`）。
5. 評価: `RLinf/evaluations/run_eval.sh libero libero_10_smolvla_eval ...`、2 環境版は `scripts/recap_eval/`。
6. 学習の条件づけオプション（`data.positive_only`、`data.negative_prompt`）: `patches/rlinf/smolvla-w-condition-options.patch`。

ログ: `tmp/step4_events_<実験名>.log`、`tmp/step4_<実験名>_attempt<N>.log`、`tmp/step9_eval_*.log`、`tmp/recap_prompt_eval/`、`tmp/recap_expS_eval/`、`tmp/step3_v2.log`、`tmp/relabel_q10.log`、`tmp/relabel_q20.log`。

# STEAM の再現実験（smolVLA、LIBERO-10 Task 0）の結果

RLinf の STEAM（Self-supervised Temporal Ensemble Advantage Modeling、arXiv:2606.29834）を、smolVLA（`lerobot/smolvla_libero`）で、LIBERO-10 の Task 0 に適用した実験の記録。
期間は 2026-10-01 〜 10-04。GPU は RTX A5000（24GB、コンテナのメモリ上限 25GB）。論文の日本語訳は `docs/steam_paper_ja.md`。
**全実験の詳細（条件、数値、時系列、障害）は `results/steam_smolvla_libero10_task0_details.md`**。

**結論（先に）**: critic（時間オフセット予測器）は、評価データで成功と失敗をよく分離できた（エピソード単位 AUC 1.00、フレーム単位 0.91）。
しかし、その advantage で方策を追加学習しても、成功率は、ベースライン 0.40 から、**統計的に区別できる変化を示していない**（0.36〜0.52、50 試行の標準誤差は約 0.07）。
最高値は「positive のフレームだけで学習」した実験 P の 0.52（26/50）だが、ベースラインとの差は誤差の範囲内で、再現性は未確認。

## 1. 成功率の一覧（Task 0、固定初期状態 50 個を 1 回ずつ、`success_once`）

| 実験 | 設定 | 成功数 | success_once | 95% 信頼区間（Wilson） | success_at_end |
|---|---|---|---|---|---|
| ベースライン | 学習なし（`smolvla_libero`） | 20/50 | 0.40 | [0.28, 0.54] | — |
| 旧 Step 4 ① | critic step 600 のラベル、sft 1:1、600 step、プロンプトあり | 21/50 | 0.42 | [0.29, 0.56] | 0.40 |
| 旧 Step 4 ② | critic step 2000 のラベル、sft 1:1、600 step、プロンプトあり | 21/50 | 0.42 | [0.29, 0.56] | 0.42 |
| 旧 Step 4 ②（同じ checkpoint） | プロンプトなしで評価 | 24/50 | 0.48 | [0.35, 0.61] | 0.46 |
| 新 Step 4（補正 D・重みなし）の step 1000 | プロンプトあり | 18/50 | 0.36 | [0.24, 0.50] | 0.34 |
| **実験 P** | 新 Step 4 の step 1000 から 450 step、positive のみで学習、プロンプトなしで評価 | **26/50** | **0.52** | [0.39, 0.65] | 0.44 |
| 実験 T | 同、negative にも `Advantage: negative` を付けて学習、positive プロンプトで評価 | 18/50 | 0.36 | [0.24, 0.50] | 0.34 |

- 評価は、2 環境 × 25 周（`total_num_envs=2`、`max_steps_per_rollout_epoch=13000`）。ベースライン（README）は、10 環境 × 5 周で測ったもの。同じ 50 個の初期状態を 1 回ずつ回ると考えているが、2 環境での網羅は確認していない（軌跡数は 50）。
- **2 つの評価の差の 95% の幅は、約 ±0.19**。上の表の差は、どれも、これより小さい。
- 参考（RECAP 側、別の環境の記録。私は内容を検証していない）: `results/recap_prompt_ablation_and_expS.md` に、RECAP の step 1000 が 0.40（あり）/0.42（なし）、step 2000 が 0.48 / 0.50、実験 S（成功エピソードのフレームのみ positive）が 0.44（100 試行）とある。

## 2. パイプラインと設定

| 段階 | 内容 |
|---|---|
| 1. critic の学習 | SigLIP-so400m + Gemma3-270M に分類ヘッド。フレームペア `(t, t+k)`（`k=32`）の符号付き間隔を、**長さ正規化**（`L_max/L_ep`、`L_max=380`）して 32 個の bin に分類（`num_bins=32`）。1 メンバー、global batch 64（micro 8）、学習率 5e-5、warmup 50 step、約 15 秒/step |
| 2. advantage の計算 | 各フレーム `t` で `(t, t+32)` を critic に通し、期待 bin を `[-1,1]` に正規化した値（`advantage_continuous`）。rollout は全 rollout をまとめた上位 30%、sft は上位 80% を positive（`label_mode=quantile`）。約 17 フレーム/秒（A5000、batch 16） |
| 3. 方策の学習（Step 4） | smolVLA 全パラメータを fp32 で追加学習。positive のフレームに `"\nAdvantage: positive"` を付ける（確率 0.9）。global batch 256（micro 8）、学習率 1e-5（warmup 30 step、その後一定）、約 40 秒/step |
| 4. 評価 | Task 0、50 初期状態、`success_once` |

### 学習データ
- **rollout**: ベース方策自身が Task 0 で動いた結果を再収集したデータ（成功で打ち切り、10fps と記録、実際の制御は 20Hz）。`data/smolvla_recap/libero10_task0_train_v2_r{1..4}_m{0..2}`。統合版 `v30/libero10_task0_train_v2_merged`（4,122 エピソード、1,757,810 フレーム）。
  **advantage を計算したのは、先頭の 1,458 エピソード（r1 の 3 つ + r2_m1、622,430 フレーム、成功 594 本）のみ**。
- **sft**: 人間デモのうち、Task 0 の 3 本（836 フレーム、`data/smolvla_recap/libero10_task0_sft_target3`）。元の `libero10_task0_sft`（10 タスク × 3 本）から取り出した（元は変更していない）。
- critic の学習データは、sft_target3（重み 4.0）と、全 rollout 11 個の**成功エピソードのみ**（1,689 本）。論文は専門家デモのみで学習するが、専門家が 3 本しかないため、成功した rollout を擬似的な専門家として使った（意図的な逸脱）。

## 3. critic の学習

| 実験 | 設定 | 結果 |
|---|---|---|
| 初回 | sft（全 30 本）+ 旧 rollout（失敗を含む）、N=32、500 step、長さ正規化なし | loss 3.47 付近で横ばい（ランダム = ln32 = 3.47）、accuracy 約 0.05 |
| sft のみ | sft（全 30 本）、N=32、300 step | 学習せず（loss 3.47 付近） |
| N=2 | sft のみ、N=2、300 step | **学習した**（loss 0.69 → 0.13、accuracy 1.0）。実装自体は正常 |
| A | sft 全 30 本 + 新データの成功 rollout、N=32、約 320 step（途中で停止） | 学習せず（loss 3.42〜3.49） |
| **D** | sft_target3 + 成功 rollout、**N=8、長さ正規化あり**、300 step | **学習した**（loss 2.10 → 1.42、accuracy 0.09 → 0.49、neighbor 0.78）。ランダム相当は loss 2.08 |
| E | 同、**N=32、長さ正規化あり**、300 step | loss 3.41 → 3.10、accuracy 0.24 |
| E2 | E を step 600 まで | loss 2.57、accuracy 0.30、neighbor 0.40 |
| **F（最終）** | E2 を step 2000 まで（別の GPU で実行） | **loss 2.18、accuracy 0.35、neighbor 0.57**。まだ下がり続けていた |

- **重要な発見**: RLinf の STEAM では、**長さ正規化（`length_scale_enabled`）が、既定で無効**（コード上の既定が `False`、yaml にキーなし）。有効にして初めて、N=8 と N=32 の critic が学習できた（`+data.length_scale_enabled=true`）。
- 損失の評価は、学習データ上の値。最終版（step 2000、N=32）を、以降の advantage 計算に使った。

### 評価データでの critic の質（`eval_v2_m{0,1,2}`、64 エピソード = 成功 24 + 失敗 40、27,880 フレーム。学習に未使用）

| critic | エピソード単位 AUC | フレーム単位 AUC（成功フレーム vs 失敗フレーム） | 成功エピソードの advantage 平均 | 失敗エピソードの平均 | 失敗フレームの positive 率 | 成功フレームの positive 率 |
|---|---|---|---|---|---|---|
| D（N=8、300 step） | 1.000 | 0.820 | 0.771 | 0.205 | 15.8% | 71.7% |
| N=32、step 600 | 1.000 | 0.864 | 0.765 | 0.095 | 14.1% | 76.6% |
| **N=32、step 2000** | 1.000 | **0.912** | 0.893 | 0.165 | 13.0% | 80.0% |

- 失敗エピソードの advantage は、中盤以降 0 付近まで下がる（論文の図 4 と同じ傾向）。成功と失敗の分離は、step が進むほど良くなった。
- ただし、**フレーム単位の正解ラベルはなく**、「各フレームの動作が良いか」は検証できていない。

## 4. advantage ラベルの分析（step 2000 の critic、先頭 1,458 エピソード）

### 4.1 positive の分布（補正 D 後、610,766 フレーム中 183,230 が positive = 30.0%）

| | 成功エピソード（594 本） | 失敗エピソード（864 本） |
|---|---|---|
| フレーム数 | 168,398 | 442,368 |
| positive | 137,002（81.4%） | 46,228（10.5%） |
| positive の内訳に占める割合 | 74.8% | 25.2% |
| エピソード内の位置 | **全区間に、ほぼ均等**（各 10% 区間で 7〜11%） | **序盤に集中**（最初の 10% が 43.8%、最初の 40% までで 86.8%） |

- 成功エピソードでは、**短い（速く終わった）ものほど positive が多い**: 260 フレーム以下 98.9%、261〜300 で 93.6%、301〜350 で 66.4%、350 超で 34.4%。
- 同じ時刻のフレーム同士で、positive と negative の行動の平均を比べると、効果量 d が 0.3〜1.0 程度の差がある（特に回転 z で −1.0 前後、グリッパー +0.5〜+0.9）。**データ側には、条件づけの信号がある**。

### 4.2 エピソード終端の人工的な低下と補正 D
- 各フレームで `(t, t+32)` を使うため、終端の 31 フレームは `t+32` が終端に切り詰められ、間隔が短い分、advantage が小さくなる（critic は、間隔に比例した値を出す。成功エピソードで実測：実際の値 ÷（d/32）≈ 1.0〜1.2）。
- 補正前は、成功エピソードの最後の 10 フレームの positive が 0%（1 件のみ）、11〜31 フレーム前で 33%。
- **補正 D**: 終端までの距離 `d` が 8〜31 のフレームの advantage に `32/d` を掛け、最後の 8 フレーム（`d < 8`）を除外。ラベルは、5 データセットをまとめた分位数で付け直した。
- 補正後の成功エピソードの終端付近: 8〜10 フレーム前で positive 72.9%、11〜31 フレーム前で 88.7%（内部は 79〜91%）。失敗エピソードの終端が不自然に上がることはない（5.7〜10.2%）。
- 除外は 11,664 フレーム（1.9%）。ラベルが変わったのは 22,135 行（3.6%）。除外した最後の 8 フレームは、成功エピソードでは、物を置いたあとのグリッパーが開いた状態（開の指令が 89〜96%）。
- 補正の `32/d` は近似で、成功エピソードでの比例関係に基づく。成功率への効果は未検証。

## 5. 方策の学習（Step 4）の詳細

| 実験 | 学習 | 損失（平均）| 備考 |
|---|---|---|---|
| 旧 ①② | 600 step、sft 1:1（smolVLA 版の意味で）、約 6.7 時間 | 0.098（最終）| sft の割合 50%（3 本を約 315 回繰り返す）|
| 新（補正 D、重みなし） | 2000 step 計画、step 1182 まで | 0.136（step 1000〜1200）| sft の割合 0.133%。step 1000 の checkpoint で評価 |
| 実験 P | step 1000 から 450 step（約 5.1 時間）、positive の 183,230 フレームのみ、プロンプトなし | 約 0.064 | 追加学習。評価はプロンプトなし |
| 実験 T | 同、negative に `Advantage: negative`、positive に positive を常に付ける | 約 0.12 | 評価は positive プロンプトで |

- 損失は、学習データの再現精度で、成功率とは別の指標。新しい学習の損失（0.136）は、旧（0.098）より高いが、データ構成が違う（旧は 3 本のデモの繰り返しを含み、記憶しやすかった）ため、同じ尺度では比べられない。
- 実験 P・T の追加学習は、両方とも step 1000 の checkpoint から。条件の違いは、`data.positive_only` / `data.negative_prompt`（パッチ `smolvla-w-condition-options.patch`）。

## 6. 見つかった問題と対処

| 問題 | 内容 | 対処 |
|---|---|---|
| 長さ正規化が既定で無効 | RLinf の STEAM の既定は `length_scale_enabled=False` | 有効化して critic を学習 |
| データ読み込みが遅い | lerobot 0.6.1 の `_query_hf_dataset` が、1 サンプルごとに全フレームの列（175 万行）を torch に変換（7.3 秒/サンプル、GPU 稼働率 0%）| パッチで行指定に変更（出力は同一、0.29 秒/サンプル）。1 step 300 秒 → 38 秒。`smolvla-v-fast-hf-query.patch` |
| 終端の advantage 低下 | 4.2 のとおり | 補正 D |
| 重みづけの意味の取り違え | 元の RLinf（`ReCapMixtureDataset`）では、`balance_dataset_weights=true` で**件数に比例**（sft 8,005 vs rollout 166 万 → sft 0.48%）。**smolVLA 版（`smolvla_cfg.py`）は逆**で、`true` で重みの比（1:1）になる | 旧 Step 4 は 1:1（元と違う挙動）。以降は `false`（件数比例、元の挙動と一致） |
| critic 学習での sft の割合 | sft_target3（重み 4.0）の割合は、約 0.67%（4×1,666 / 約 99 万）。**実質、成功 rollout のみで学習** | 記録のみ |
| 他の環境の整理による checkpoint の削除 | 他の環境の Step 4 の監視スクリプトが、私の実験の checkpoint（step 200〜800）を削除 | 保存済み checkpoint をハードリンクで別の場所に保持 |
| wandb の動画 | 評価の run に動画を追加した際、step 0 が無視され、ep00 が欠ける（私の 2 つの run と、別の環境の run で同じ） | 未修正 |

## 7. 元の RLinf・論文との違い（確認できた範囲）

| 項目 | 論文・元の RLinf | 私たち |
|---|---|---|
| critic の学習データ | 専門家のみ（論文 3.1、ドキュメントの Step 1） | sft 3 本（0.67%）+ 成功 rollout |
| critic のアンサンブル | M=3（最小値で集約） | 1 メンバー（アンサンブルの効果は未検証） |
| critic の学習量 | 論文は未記載、RLinf は 16,000 step・batch 512 | 2,000 step・batch 64 |
| advantage の式 | 期待 bin から基準 bin を引く（論文 式 4） | 基準を引かない（RLinf の実装）。ラベルは分位数なので、影響しない |
| 方策学習の混合 | 和集合から一様（件数比例と読める） | 件数比例（重みなし） |
| 推論時のガイダンス | 条件あり／なしの外挿（`w = 2.5`） | **未実装**（プロンプトを付けるだけ、`cfgrl_guidance_scale=1.0`） |
| 方策学習の量 | 論文は 30,000 step・batch 512 | 600〜2,000 step・batch 256（A5000 は 40 秒/step） |
| データ量 | 専門家 50〜240 本、rollout 63〜594 本 | 専門家 3 本、rollout 1,458 本 |

## 8. 解釈（推測を含む）と未確認事項
- 評価の誤差（標準誤差 約 0.07）が大きく、**全実験が、ベースライン 0.40 と区別できない**。P の 0.52 も同様。
- 条件づけ（`Advantage: positive`）の効果は確認できなかった（旧 ②：あり 0.42、なし 0.48。実験 T：0.36）。方策のプロンプトに対する反応（有無での行動の差）は、**直接測っていない**。
- P（positive のみ）が最良だったのは、条件づけに頼らず、良い動作を直接真似ることが、現状では向いている可能性を示唆するが、再現性は未確認。
- 学習量は、元の RECAP 再現（30,000 step × batch 256、全 4,096 本）の約 15 分の 1、データは 3 分の 1。
- 未実施: P の再評価（別 seed）、ガイダンスの実装、critic のアンサンブル化、全データ（4,122 本）への advantage 計算、専門家デモの増加。

## 9. 成果物の場所

| 内容 | 場所 |
|---|---|
| 論文の日本語訳 | `docs/steam_paper_ja.md` |
| 追加したパッチ（commit 済み） | `patches/rlinf/smolvla-v-fast-hf-query.patch`、`smolvla-w-condition-options.patch` |
| critic の最終 checkpoint（step 2000） | `RLinf/logs/steam_sft/steam_value_model_sft-20261002-20:05:34/steam_expF_member0/checkpoints/global_step_2000` |
| advantage の出力（5 データセット） | `tmp/steam/shadow/<データセット>/meta/advantages_n32e2000.parquet`（`tmp/` は git 管理外） |
| Step 4 用のラベル（補正 D、統合データの番号） | `tmp/steam/step4/adv_train_merged_n32e2000_endfixD_q30.parquet`、`adv_sft_target3_n32e2000_endfixD_q30.parquet` |
| 新 Step 4 の step 1000 | `tmp/steam/step4D_safe/global_step_1000` |
| 実験 P の checkpoint（1050 / 1200 / 1350 / 1450） | `RLinf/logs/20261004-08:08:44-libero10_task0_recap_smolvla/step4_expP_posonly_1450/checkpoints/` |
| 実験 T の checkpoint（同） | `RLinf/logs/20261004-08:16:06-libero10_task0_recap_smolvla/step4_expT_twosided_1450/checkpoints/` |
| 旧 Step 4 ②（step 2000 の critic）の step 600 | `RLinf/logs/20261003-09:51:31-libero10_task0_recap_smolvla/step4_steam_n32e2000_600/checkpoints/global_step_600` |
| スクリプト | `tmp/steam/`（`run_exp.sh`、`eval_ckpt.sh`、`adv_queue.sh` など） |
| wandb（`libero_RL`） | 評価: `eval_step4_steam_n32e600_600`（run `qusbnjkp`）、`eval_step4_steam_n32e2000_600`（`bx19m3pg`）、`eval_step4_steam_n32e2000_endfixD_step1000`（`zua80urf`）、`eval_step4_expP_posonly_1450`（`014zeyfp`）、`eval_step4_expT_twosided_1450`（`cdpyso8k`） |

## 10. 訂正（過去の報告の誤り）
- critic の学習で、「sft がサンプルの約 27%（4/15）を占める」と報告したが、**誤り**。元の `ReCapMixtureDataset` の計算式では、約 0.67%。
- README の「`balance_dataset_weights: true`（sft と rollout を件数によらず 1:1 で混ぜる。RLinf の CFG 学習と同じ）」は、smolVLA 版の挙動としては正しいが、**「RLinf の CFG 学習と同じ」は誤り**（元は件数比例）。README は、まだ修正していない。

# RECAP の再現実験（smolVLA、LIBERO-10 Task 0）の結果

2026-10-01〜10-04。RLinf の RECAP（価値モデル → advantage → advantage 条件付き SFT）を、`lerobot/smolvla_libero` で再現した記録。
**結論: 試した範囲では、ベースライン（success_once = 0.40）を有意に上回る設定は見つかっていない。**
追加の診断（プロンプトの有無、成功エピソードだけを positive にした実験 S）は `recap_prompt_ablation_and_expS.md` に詳しい。

## 1. 成功率の一覧（Task 0、固定初期状態 50 個、`success_once`）

| 実験 | 条件 | 成功数/試行数 | 成功率 |
|---|---|---|---|
| ベースライン | `models/smolvla_libero`（学習なし） | 20/50 | 0.40 |
| B: global_step_1000 | q30、sft 50:50、プロンプトあり（10 環境） | 23/50 | 0.46 |
| B | 同上（2 環境、プロンプトあり / なし） | 20/50、21/50 | 0.40、0.42 |
| C: global_step_2000 | B から 1000 ステップ追加（重みなし、q30）、プロンプトあり（10 環境） | 22/50 | 0.44 |
| C | 同上（2 環境、プロンプトあり / なし） | 24/50、25/50 | 0.48、0.50 |
| R10: global_step_1450 | B から 450 ステップ追加（重みなし、**上位 10%**）、プロンプトあり（50 × 2 回） | 32/100 | 0.32 |
| R10 | 同上、プロンプトなし（50 × 2 回） | 35/100 | 0.35 |
| S: global_step_1450 | B から 450 ステップ追加（**成功エピソードのフレームを positive**）、プロンプトあり | 44/100 | 0.44 |

- 評価の標準誤差は、50 試行で約 0.07、100 試行で約 0.05。**どの差も、誤差の範囲に近く、有意とは言えない。**
- ベースラインは 50 試行を 1 回測っただけ。同じ手順で繰り返し測っていない。
- 評価の並列数が違う（10 環境と 2 環境 × 25 パス）。初期状態は同じ 50 個だが、手順が揃っていない。
- 評価は `evaluations/run_eval.sh libero libero_10_smolvla_eval`（Task 0、`env.eval.task_id_filter=[0]`）。`advantage_prompt=true` のとき、指示文に "\nAdvantage: positive" を付ける。

## 2. データ

- 旧データ（成功後も 520 ステップ続行）は、return が「何ステップ目か」と「成否」だけで決まっていたため破棄し、成功で打ち切る設定で再収集した。
- train: 4,122 本、1,757,810 フレーム、成功 1,689 本（41.0%）。11 個のデータセットを、ファイルレベルで 1 つにマージした（`libero10_task0_train_v2_merged`）。
- sft: Task 0 の人のデモ 3 本（836 フレーム、`libero10_task0_sft_target3`）。全 10 タスク入りの 30 本は使っていない（元の RECAP は全タスクの 30 本を使う）。
- eval 用（Step2 の監視のみ）: 3 個、計 64 本。
- return（Step1）: 1 ステップあたり −1、失敗時の終端に −300（`fail300`、gamma = 1.0、範囲 [−819, 0]）。

## 3. Step2: 価値モデル（4,000 ステップ）

- 設定: global batch 256、micro batch 16、sft + train 11 個（重みは各 1.0）、約 20.6 時間。
- 最終 eval（各 300 サンプル × 3 セットの平均）: value_spearman 0.963、loss 2.298、cat_acc_best 0.325、cat_acc_neighbor 0.456、mae 0.066。
- セットごとの value_spearman: 0.985、0.982、0.921。
- eval の loss は、300 ステップで 3.48、2,400 ステップ以降は 2.18〜2.32 で、ほぼ横ばい。
- checkpoint: `RLinf/logs/recap_value_model_sft-20261002-08:13:11/step2_value_sft_v2/checkpoints/global_step_4000`。
- 公式は約 30,000 ステップの設定で、18,000 ステップの checkpoint を使う。今回は 4,000 ステップで、約 1/4。

## 4. Step3: advantage（先頭 622,430 フレームのみ）

- 対象: `train_v2_merged` の先頭 1,458 本（622,430 フレーム）と、sft_target3（836 フレーム）。計 623,266 サンプル、約 4 時間 9 分。
- 先読み幅 N = 10、上位 30% を positive（タグ `fail300_N10_q30`）。閾値 0.0058（70 パーセンタイル）。rollout の positive は 29.98%。sft は全部 positive。
- advantage の分布: 平均 0.0012、標準偏差 0.058、範囲 [−0.66, 0.63]。
- 残りの約 113 万フレームは、advantage がなく、Step4 の学習から除外される（ローダが警告を出して除外する）。

### 診断: advantage が成功と失敗をあまり分けていない
- 成功エピソード 594 本のフレーム（173,150）の positive 率は 37.8%、失敗エピソード 864 本のフレーム（449,280）は 27.0%。
- フレーム単位の AUC（成功 vs 失敗）は 0.565。**エピソードごとの平均 advantage だと 0.995**。
- positive の約 65% は、失敗エピソードのフレーム。positive 内の成功フレームの比率は 35%（全体の基準は 27.8%）。
- positive の割合を絞ったときの、positive 内の成功比率（基準 27.8%）:

| 上位 | 成功比率 | 基準比 | positive フレーム数 |
|---|---|---|---|
| 30% | 0.351 | 1.26 倍 | 186,729 |
| 20% | 0.397 | 1.43 倍 | 124,486 |
| 10% | 0.514 | 1.85 倍 | 62,243 |
| 5% | 0.598 | 2.15 倍 | 31,122 |

## 5. Step4: advantage 条件付き SFT

共通: `models/smolvla_libero` から開始、fp32、global batch 256、AdamW（beta 0.9 / 0.95、weight decay 1e-10）、勾配クリップ 10、学習率 1e-5（**一定。cosine の減衰は効いていない**）、action chunk 50、positive には "Advantage: positive" を付け 10% の確率で外す。negative には付けない。

| 実験 | ステップ | sft と rollout の抽出 | positive | 損失の推移 | 速度 |
|---|---|---|---|---|---|
| 前半 | 1〜1000 | **50:50**（`balance=true`、1 フレーム約 300 回） | 上位 30% | 0.33 → 0.090 | 約 37 秒 |
| C（後半） | 1001〜2000 | 重みなし（フレーム均等、sft 約 0.13%） | 上位 30% | 0.148 → 0.131 | 約 35〜43 秒 |
| R10 | 1001〜1450 | 重みなし | 上位 10% | 約 0.146 → 0.134 | 約 35 秒 |
| S | 1001〜1450 | 重みなし | 成功エピソードのフレーム | 0.148〜0.169 → 0.134 | 39 秒（A5000） |

- 損失は、すべての実験で 0.13 前後に収束しており、**成功率とは結びついていない**。
- 前半の損失の低さ（0.09）は、sft の 836 フレームの暗記を含む可能性がある（推測）。重みなしに切り替えると、損失が 0.148 に上がった。
- 学習量: 今回は追加 450〜1,000 ステップ（合計 1,450〜2,000）。公式の約 3 万ステップ（batch 512、warmup 5,000）の 1/15 程度。

## 6. わかったこと
1. **プロンプトの有無で、成功率が変わらない。** B で +2 ポイント、C で +2 ポイント、R10 で +3 ポイント（いずれも「なし」の方が高い、または同等）。条件づけがほとんど学ばれていない。
2. **強いラベル（S）でも改善しない。** ただし、S も条件づけが効かない状態で学習しており、ラベルの強さと、条件づけの効きは、分けて判断できない。
3. **positive を絞りすぎると悪化する可能性がある**（R10 は 0.32）。ただしベースラインとの差は有意ではない。
4. **評価の誤差が大きい**（50 試行で ±0.07）。約 ±0.1 以内の差は、今の評価では検出できない。

## 7. 元の RECAP との違い・訂正
- **sft と rollout の混合比:** 元の RLinf（`ReCapMixtureDataset`）は、`balance_dataset_weights=true` のとき、選ばれる確率が `weight × フレーム数` に比例し（フレーム均等）、sft は約 0.5%。**このプロジェクトの smolVLA 用ローダ（`smolvla_cfg.py`）は、`true` と `false` の意味が逆**（`true` = データセット均等で 50:50）。前半の 1,000 ステップは、そのため公式から外れていた。後半（重みなし）は、公式に近い。元論文は、デモと自律データの比率を規定せず、全データを一様に扱う。
- sft は Task 0 の 3 本のみ（公式は全タスクの 30 本）。
- 価値モデルは 4,000 ステップ（公式の 18,000 ステップ前後の checkpoint の約 1/4）、方策は 1/15 の学習量。
- デモの advantage は常に positive（論文と同じ）。

## 8. 未実施・結果待ち
- 別環境で、実験 P（positive のみ、プロンプトなし）と T（両側タグ）が実行されている（STEAM のラベル）。結果は未反映。
- ベースラインを、同じ手順で、100〜200 試行測っていない。
- 学習量を大きく増やす実験（例: batch 64 で 8,000 更新）、価値モデルの追加学習（18,000 ステップ前後）、先読み幅 N の変更、全 4,122 本の使用は、未実施。

## 9. 運用上の記録（再現時の注意）
- 環境ごとに、`/root`（`uv` の Python、`~/.libero/config.yaml`）と、システムのライブラリ（`libegl1`、`libopengl0`）の準備が必要（`scripts/preflight_collect_env.sh`）。
- メモリ上限 25GB の環境では、評価を 10 並列にすると OOM で落ちる（2 環境 × 25 パスにする）。
- checkpoint の保存先は、`experiment_name` で変わる。自動実行スクリプトの検索パスを間違えて、(a) 整理と自動再開が効かなかった、(b) **他の実験（STEAM）の checkpoint 4 個（global_step_200 / 400 / 600 / 800）を削除した**。後者は、復元できない。修正済み（検索対象をその実験の保存先だけに限定）。
- 学習の再開（`+runner.resume_dir`）は動作を確認済み（global_step_1000 から）。落ちたときの自動再開は、実際には発動していない。
- wandb: project `libero_RL`。記録は、TensorBoard と並行（再開分から）。

## 10. 再現手順（要点）
1. Step2: `scripts/run_step2_v2.sh`（`scripts/step2_v2_supervisor.sh` で 4,000 ステップ、300 ステップごとに保存）。
2. Step3: `tmp/run_step3_v2.sh`（`compute_advantages.py`、`+advantage.max_samples=622430`）。
3. Step4: `scripts/run_step4_onward.sh`（初回）、`scripts/resume_step4_noweight.sh`（再開。`BALANCE`、`ADV_TAG`、`EXP_NAME`、`MAX_STEPS`、`EVAL_ENVS` などを環境変数で指定）。
4. relabel（閾値の変更）: `RLinf/examples/offline_rl/advantage_labeling/recap/process/relabel_advantages.py`（sft は全て positive に直す）。
5. 評価: `RLinf/evaluations/run_eval.sh libero libero_10_smolvla_eval ...`。

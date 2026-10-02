# 引き継ぎ: データ再収集の完了と、Step2 以降の方針

作成: 2026-10-02。他の環境の Claude Code に、これまでの経緯と今後の方針を渡すための文書です。
`/workspace` は共有で、同じリポジトリ（`/workspace/libero_RL`）と同じデータが見えます。
先に `CLAUDE.md`、`.claude/skills/recap-smolvla-status/SKILL.md`（既知のハマりどころ）を読んでください。
SKILL.md の「現在の進捗」は 2026-10-01 時点の古い内容（旧データの Step2/3）なので、この文書を優先してください。

## 1. 目的

RLinf の RECAP を smolVLA（`lerobot/smolvla_libero`）で再現する。対象は LIBERO-10 の Task 0。
パイプラインは、Step1（return）→ Step2（価値モデル SFT）→ Step3（advantage 計算）→ Step4（advantage 条件付き SFT）→ Step9（シミュレータ評価）。
ベースラインは `success_once=0.40`（20/50、固定初期状態 50 個）で、これを上回ることがゴール。

## 2. 作業ルール（必ず守る）

- RLinf（`RLinf/`）は直接編集しない。変更は `patches/rlinf/*.patch` にし、`bash patches/apply_rlinf_patches.sh` で適用する。
- 一時的な作業は `tmp/` をカレントディレクトリにして行う（`/tmp` は使わない）。
- 作業が一区切りしたら commit する。
- **プロセスの停止に `pkill` / `pkill -f` / `killall` を使わない**。`pgrep -af` と `ps` で PID を確認して `kill <PID>` する。
  また、`pgrep -f <パターン>` は、自分のシェルのコマンド文字列にも一致する場合がある。PID の確認は `ps -eo pid,cmd | grep <パターン> | grep -v grep` を使う。
- データやログの削除は、ユーザーの明示的な指示があるときだけ行う。削除コマンドは環境の権限設定で拒否されることがあり、その場合はユーザーがコマンドを実行する。
- ディスクのクォータは 600GB。現在は約 240GB（`collected_v2/` の元 PNG は削除済み。他の環境の RL token 用の特徴量が `tmp/rlt_stage1/` に約 90GB）。
- コンテナのメモリ上限は 25GB（cgroup）。DataLoader のワーカーを増やすと OOM になる。

## 3. 再収集したデータ（完了）

旧データは、成功しても 520 ステップまで続けていたため、return が「何ステップ目か」と「成否」だけで決まっていた（V の分散の約 92% を説明）。
そのため、成功で打ち切る設定（`ignore_terminations=False`、チャンク境界で打ち切り）で再収集した。

- 3 環境 × 4 ラウンドで収集し、**有効な train は 4,122 本、1,757,810 フレーム、成功 1,689 本（41.0%）**。
  成功の長さは平均 292（230〜520）、失敗は常に 520。
- LeRobot v2.1（動画）に変換済みで、元データとの照合も済み（PSNR 34.6 dB 以上）。
  - train（11 個）: `data/smolvla_recap/libero10_task0_train_v2_r{R}_m{M}`（R1: m0,m1,m2 / R2: m1,m2 / R3: m0,m1,m2 / R4: m0,m1,m2）
  - eval（3 個、計 64 本）: `data/smolvla_recap/libero10_task0_eval_v2_m{0,1,2}`（22+21+21 本、成功率 36〜38%）
  - sft（人の収集、30 本、全 LIBERO 10 タスク、20fps）: `data/smolvla_recap/libero10_task0_sft`
- **Step1（return）は全データセットに付与済み**。`meta/returns_fail300.parquet`（gamma=1.0、失敗時の終端報酬 -300、タグ `fail300`）。
  全 4,122 本でルールとの不一致 0（成功は先頭 -(n-1)・末尾 0、失敗は先頭 -(n-1)-300・末尾 -300、範囲 [-819, 0]）。
- 変換時に、10 フレームしかない偽エピソード 3 本を除外した（初期状態プールの境界で、環境がリセットされずに出る収集のアーティファクト）。除外は `--min-frames 100`。
- 元の PNG（`collected_v2/`）は、2026-10-02 にユーザーの指示で削除済み。以降の処理（Step2〜9、RL token、STEAM）は、変換後の動画（`data/smolvla_recap/`）を使う。検査・照合のスクリプト（`verify_collected_integrity.py` など）は、元データが無いので再実行できない。
- 旧データ（`libero10_task0_train` など）と旧 Step2/3/4 のログ・チェックポイントは、ユーザーが削除済み。旧パイプラインの結果は使えない。
- 詳細は `docs/data_recollection_plan.md`。

## 4. Step2（価値モデルの再学習）— 次の作業、GPU の良い環境で実行

ユーザーは、**性能の良い GPU の環境で Step2 を実行する**と決めた。ステップ数は **4,000**（ユーザー指定）。
**1 時間おきに checkpoint を保存する**（落ちても再開できるように）。

### 4.1 これまでの経緯

- 24GB の RTX A5000 では、旧版（L40S 46GB、micro batch 16、8,000 ステップで 25 時間、約 11 秒/ステップ）の設定がそのままでは動かない。
  - micro batch 16 は GPU メモリ不足。
  - micro batch 8 は、GPU メモリの断片化で 48 秒/ステップ（8,000 ステップで約 106 時間）。
  - micro batch 8 に `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を付けると 24 秒/ステップ（4,000 ステップで約 27 時間）。
  - 測定結果: micro 4 は 30.7 秒、micro 2 は 45 秒、gradient checkpointing は効果なし。micro 12 は global batch 256 で割り切れず不可。
- A5000 で本番を始め、約 40 ステップ（損失 5.8）で止めた（チェックポイントなし。引き継ぎ対象の成果物はない）。
- データの読み込みは律速ではなく、GPU 側の問題。

### 4.2 実行方法（このリポジトリのスクリプト）

- `scripts/run_step2_v2.sh MAX_STEPS [hydra の追加引数]`：sft + train 11 個 + eval 3 個を指定して、Step2 を起動する。
  環境変数 `MB`（micro batch、既定 8）、`SAVE`、`VAL`（既定 560）、`EVS`（eval の各データセットのサンプル数、既定 3000）で調整する。
- `scripts/step2_v2_supervisor.sh`：4,000 ステップ、150 ステップごとに checkpoint と eval を行い、プロセスが落ちたら最新の checkpoint から再開する（最大 6 回）。
  直近 2 個と 600 の倍数のステップの checkpoint を残して、古いものを削除する。
  イベントは `tmp/step2_v2_events.log` に記録される。**カレントディレクトリは `/workspace/libero_RL`、内部で `tmp/` に移動して起動する**。
  - 150 ステップ = 約 1 時間は、A5000 の速度（24 秒/ステップ）に合わせた値。**GPU が速い環境では、SAVE を「1 時間分のステップ数」に合わせて変更する**（例: 11 秒/ステップなら約 330）。
  - **`save_interval` は `val_check_interval` で割り切れる必要がある**（そうでないと assertion で落ちる）。また、`global_batch_size` は `micro_batch_size × GPU 数` で割り切れる必要がある。
- GPU が大きい環境では、`MB=16`（または既定の 32）で動くか、先に 2〜3 ステップで確認する。動けば `expandable_segments` が不要な場合もある。
  GPU の名前と空きメモリを確認してから、`MB` を決める。
- 学習データは、sft（weight 1.0）と train 11 個（すべて `type: rollout`）。eval は 3 個。`data.tag=fail300`。
  siglip・gemma3 のパスは `models/` 以下（`siglip2-so400m-patch14-224`、`gemma-3-270m`）。
- チェックポイントは `RLinf/logs/value_sft/recap_value_model_sft-<日時>/step2_value_sft/checkpoints/global_step_<N>`。
  1 個あたりのサイズは未確認（旧版では 2 個で約 14GB、つまり 1 個約 7GB が目安）。
- 完了後、`eval/value_spearman` などの指標を確認する。旧版の最終値は `train/value_spearman=0.946`、`eval/cat_acc_best=0.287`（旧データの値で、直接の比較には使えない）。

## 5. Step3 以降の方針

- **Step3（advantage 計算）**: Step2 の最終 checkpoint で、**sft + 新 train 11 個を、1 つのプロセスで一括処理**する
  （上位 30% を positive とする閾値は、全データセットの advantage を結合して 1 つ決めるため。分けて実行すると閾値がずれる）。
  `data.return_min` などの設定は不要（return の正規化範囲 [-819, 0] は各 `stats.json` から自動算出）。
  - 旧版の実行スクリプトは `tmp/run_step3_advantages.sh`（旧データ用の例。未追跡）。新データ用に、データセットの一覧を差し替える。
  - 途中再開パッチ（`patches/rlinf/advantage-resume-checkpoint.patch`）が適用済み。結果は `tmp/step3_state/` に保存される。完了後は削除する。
  - メモリ: DataLoader のワーカーは 4 個まで（12 個だと 25GB で OOM した）。
  - 処理量は約 177 万サンプル。速度は GPU に依存する（A5000 で約 38 サンプル/秒）。
- **Step4（advantage 条件付き SFT）**: 新データを LeRobot v3.0 に変換（`scripts/convert_rlinf_to_lerobot_v30.sh`）してから実行する。
  v3.0 の sft は `data/smolvla_recap/v30/libero10_task0_sft` にある（旧版の train/eval の v3.0 は削除済みなので、新データの v3.0 は新規に作る）。
  目標は 30,000 ステップ（README/RLinf docs で確認済み）。24GB の GPU では、fp32・micro batch 32 が載るか未確認なので、本番前に確認する。
  プロンプトは positive に "Advantage: positive"（学習時の dropout 0.1）、negative には付けない。
- **Step9（評価）**: `eval/success_once` が 0.40 を上回るかで、RECAP の効果を判定する。

## 6. 注意（過去の失敗）

- 長時間の処理は、ユーザーが監視しやすい形で、バックグラウンド（`nohup` など）で実行し、ログを `tmp/` に出す。
- Ray やメモリのしきい値は `RAY_memory_usage_threshold=0.98` を使ってきた（収集時）。
- 一部のスクリプトは環境依存の準備が必要（`scripts/preflight_collect_env.sh`）。
- 報告は、完了と問題を中心に。ユーザーが「現状を教えて」と聞いたときは、現状を答える。
- ユーザーへの質問は、判断が必要なものに限る。推測は推測と明記し、確認できていないことを確認済みと書かない。

## 7. 決定事項（2026-10-02）

- **Step3 以降の sft は、Task 0 の 3 本だけを使う**（ユーザー決定）。
  - `libero10_task0_sft`（30 本、10 タスク × 3 本）のうち、Task 0（"put both the alphabet soup and the tomato sauce in the basket"、836 フレーム）の 3 本のみ。
  - 理由: Step4 は `balance_dataset_weights: true` で sft と rollout を 1:1 で混ぜるため、全タスク入りだと学習の約半分が Task 0 以外になる。
  - Step3 の前に、Task 0 の 3 本だけの sft データセット（v2.1）を新規に作る（元の `libero10_task0_sft` は変更しない）。作り方の詳細はまだ決めていない。v3.0 変換は、その後に行う。
  - 実行中の Step2 は、全タスク入りの sft（学習データの 0.5% 未満）のまま続ける。

## 8. Step4 用のデータ準備（完了、2026-10-02）

Step4 のデータローダは、`balance_dataset_weights: true` のとき、各データセットに同じ重みを割り当てる（件数は無関係）。train が 11 個のままだと、sft と合わせた 12 個が約 1/12 ずつになり、意図した sft : rollout = 1 : 1 にならない。そのため、train を 1 つにマージした。

- **マージ（v2.1）**: `data/smolvla_recap/libero10_task0_train_v2_merged`（4,122 本、1,757,810 フレーム、成功 1,689 本）。
  `scripts/merge_v21_datasets.py` で、11 個をファイルレベルで結合した（R1 m0,m1,m2 → R2 m1,m2 → R3 m0,m1,m2 → R4 m0,m1,m2 の順）。
  parquet は `episode_index`（0〜4121）と `index`（通しのフレーム番号）だけを振り直し、mp4 はコピー（再エンコードなし、ハードリンクなし）。元の 11 個は変更していない。
  Step1（return）は、マージ後のデータセットで `compute_returns.py` を再実行した。
  `scripts/verify_merged_v21.py` で、全エピソードの parquet の列、全 mp4（8,244 個、バイト単位）、`episodes.jsonl`/`episodes_stats.jsonl`、return を、元と照合し、全て一致した。
  return は、元の 11 個の sidecar を番号を振り直して結合したものと全 1,757,810 行が一致し、ルール違反は 0、範囲は [-819, 0]。
- **sft（Task 0 の 3 本）**: `data/smolvla_recap/libero10_task0_sft_target3`（3 本、836 フレーム、20fps）。
  Step1 を付与し、元の `libero10_task0_sft` の Task 0 の 3 本（episode 15〜17）の return と、836 行の全てで一致（全て成功、先頭 -(n-1)・末尾 0）。
- **v3.0 版**: `data/smolvla_recap/v30/libero10_task0_train_v2_merged`、`data/smolvla_recap/v30/libero10_task0_sft_target3`。
  LeRobot 0.6 系（`.venv-smolvla`）で読み込めることを確認した（エピソード数・フレーム数が一致、画像・状態・行動が取得できる）。
  v3.0 の `meta/` には、`returns_*.parquet` は含まれない（変換ツールが落とす。既存の `v30/libero10_task0_sft` も同じ）。Step4 は、advantage を `advantages_path` で、v2.1 側の `meta/advantages_*.parquet` から読む。
- **Step3 は、`libero10_task0_train_v2_merged`（v2.1）と `libero10_task0_sft_target3`（v2.1）に対して実行する**（番号がマージ前と変わっているので、元の 11 個の train には実行しない）。
  Step4 の設定は、v3.0 の 2 個、weight は両方 1.0 の予定。
- 増えたディスク量: 約 22GB（v2.1 マージ 11GB、v3.0 変換 11GB）。

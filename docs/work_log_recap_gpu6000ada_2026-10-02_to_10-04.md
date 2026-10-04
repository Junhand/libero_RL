# 作業ログ: RECAP の Step2〜Step4・評価・比較動画（RTX 6000 Ada の環境、2026-10-02〜10-04）

他の Claude Code が参照するための記録。RLinf の RECAP を smolVLA（LIBERO-10 Task 0）で再現するプロジェクトのうち、**RTX 6000 Ada（48GB）の環境を担当した Claude Code が、実際に実施したこと**を、時系列で全部書く。
結果の数字と考察は `results/recap_smolvla_libero10_task0.md` にある。この文書は、「何をしたか、どこに何があるか、何を間違えたか、次に何をすべきか」の記録。
先に読むもの: `CLAUDE.md`、`docs/handoff_step2_onward.md`、`results/recap_smolvla_libero10_task0.md`。

## 1. 要約

- Step2（価値モデル、4,000 ステップ）→ Step3（advantage、先頭 622,430 フレーム）→ Step4（方策の学習、前半 1,000 ステップ + 後半 1,000 ステップ、および追加の実験 R10）→ 評価、まで実施した。
- **結果: ベースライン（success_once 0.40）を、有意に上回る設定は見つからなかった。** プロンプト（"Advantage: positive"）の有無でも、成功率が変わらない。詳細は結果文書。
- 実験 S（成功エピソードを positive）と、プロンプトの有無の評価は、別環境（A5000）の Claude が担当した。実験 P・T（STEAM のラベル）、RLT、STEAM、スライドも、別の Claude の担当。この文書は、それらの内容を詳しく書かない（ポインタのみ）。
- 比較動画（初期状態 14、ベース失敗 / RECAP 成功）を、wandb にアップロードした。
- 作業中に、**他の実験（STEAM）の checkpoint 4 個を誤って削除した**（§7）。

## 2. 担当と環境

| 項目 | 内容 |
|---|---|
| 環境 | RTX 6000 Ada（48GB）、コンテナのメモリ上限は 94GB（preflight の表示）。ホストのメモリは 755GB |
| 担当した作業 | Step2、Step3、Step4（前半・C・R10）、ベースライン等の評価、結果文書、比較動画 |
| 別環境（A5000、24GB、メモリ上限 25GB）の Claude | 実験 S、プロンプトの有無の評価（B・C）、`results/recap_prompt_ablation_and_expS.md`（統合して削除済み）、評価の 2 環境版スクリプト |
| その他の Claude | 実験 P・T（STEAM のラベル）、STEAM の critic 学習と advantage、RL Token、スライド、`make_init_state_clip.py` |
| ユーザー | 方針の決定（§8 に一覧）。wandb は、ユーザーのアカウント `junhand`、`WANDB_API_KEY` は `~/.bashrc` |

- `/workspace`（共有ストレージ）は、全環境で同じ。`/root`、`/tmp`、システムのパッケージ（apt）、GPU は、環境ごとに別。
- この環境の GPU は、サーマルスロットリングが強く、89〜93℃、クロックが 465〜2295MHz に変動した（最大 3105MHz）。速度は、17〜43 秒/ステップ（Step2 と Step4 の micro batch 16）で変動した。

## 3. 時系列の作業ログ

時刻は、環境の時計（`date`）。

### 3.1 Step2（価値モデル、10-02 07:55 〜 10-03 04:54）
1. GPU を確認（RTX 6000 Ada、48GB）。micro batch の探索（3 ステップ）: **32 はメモリ不足、16 が動作**（11.5 秒/ステップ。冷えていたときの値）。
2. `scripts/step2_v2_supervisor.sh` を `MB=16 SAVE=300 VAL=300 EVS=1000` に変更して起動（commit 済み）。
3. 07:55 に起動後、設定変更のため 2 回、起動し直した（約 30 ステップで止めた。損失は 6.0 前後）。**最終の実行は 08:13:11 開始。**
   - checkpoint を直近 2 つだけ残す変更（600 の倍数を残す処理を削除、`d5136b5`）。再起動（約 12 ステップ）。
   - eval のサンプル数を 1000 → 300 に削減（eval が約 160 秒 → 約 26 秒、`9d1870a`）。再起動（約 12 ステップ）。
4. 13:24（約 1,050 ステップ）: **supervisor が checkpoint の保存先を誤っていた**ことを発見（§7-1）。学習は止めずに、`ADOPT_PID` で引き継ぎ（`b6d9cd9`）。
5. ユーザーの指示（「上の 6 個を削除してよい」）で、過去の試運転の checkpoint 6 個（global_step 3〜20、各 6.8GB、約 40GB）を削除した（§6）。
6. 10-03 04:54: **`DONE step 4000`**。最終 eval: value_spearman 0.963、loss 2.298、cat_acc_best 0.325、cat_acc_neighbor 0.456。
7. 途中で、cat_acc の読み方（201 ビンの分類精度、ランダム 0.5%）、速度の低下（サーマルスロットリング）、wandb の要望への対応を、ユーザーに説明した。

### 3.2 Step3（advantage、10-03 05:13 〜 09:22）
1. `tmp/run_step3_advantages.sh`（旧データ用）を元に `tmp/run_step3_v2.sh` を作成: `sft_target3` と `train_v2_merged` の 2 個、`value_checkpoint=global_step_4000`、`advantage.batch_size=64`、`num_dataloader_workers_per_gpu=4`、`advantage.resume_dir=tmp/step3_state_v2`。
2. **`+advantage.max_samples=622430`**（先頭 1,458 本だけを処理）。`+` を付けないと Hydra が `Key 'max_samples' is not in struct` で落ちる（最初の起動はこれで失敗）。
3. 約 4 時間 9 分で完了（約 40〜60 サンプル/秒）。閾値 0.0058、rollout の positive 29.98%。

### 3.3 Step4（方策の学習）
1. **自動実行スクリプト** `scripts/run_step4_onward.sh`: Step3 の完了待ち → micro batch の探索（32 → 16 → 8）→ Step4 → 評価。10-03 05:18 に待機を開始した。
   - ユーザーの指示: Step4 を **30,000 → 2,000 ステップ**に変更（lr のスケジュールの総ステップも 2,000、warmup は 100）。
   - 最初の Step4 の起動は、`.venv-smolvla` の Python（`/root/.local/share/uv/python/cpython-3.12.11`）がこの環境になく、`No module named 'hydra'` で、3 つの probe がすべて失敗した。`uv python install 3.12.11` で解決（§5）。
   - probe の結果: micro batch 32 はメモリ不足、**16 が動作**。09:38 に本番を開始（約 19〜43 秒/ステップ）。
2. **前半（1〜1,000 ステップ、sft 50:50）**: 10-03 09:38 〜 19:12。ユーザーの指示で、`global_step_1000` の保存の直後に止め、評価（B: 0.46）。
3. 評価が、環境の不足で 2 回落ちた（§5）。直して再実行。
4. ユーザーの指示（「`global_step_1000` から再開して、重みなしに切り替える」）: `scripts/resume_step4_noweight.sh` を作成。**後半（1,001〜2,000 ステップ、`balance_dataset_weights=false`）**: 10-03 20:10 〜 10-04 07:01（C: 0.44）。
   - wandb（TensorBoard と並行）を有効にするため、一度止めて再開した（約 8 ステップを失った）。run id `8pdouya0`。
   - 実験名を変えたため checkpoint の保存先が変わり、スクリプトの検索パスが外れた（§7-2）。直したが、**広すぎた検索パスで、他の実験（STEAM）の checkpoint を削除した**（§7-3）。
5. **実験 R10**（`global_step_1000` から 450 ステップ追加、上位 10% のラベル）: 10-04 08:22 〜 12:45。評価はプロンプトあり 0.32（32/100）、なし 0.35（35/100）。wandb run id `h40npem7`。
   - relabel（`relabel_advantages.py`）で、上位 10%（`fail300_N10_q10`）と 20%（`fail300_N10_q20`）の advantage を作成した。relabel は sft を区別しないので、sft_target3 の advantage は、手作業で全 True に直した。
   - 成功エピソードに基づくラベル（`fail300_outcome`、実験 S 用）も作成した（別環境の Claude が学習に使用）。

### 3.4 診断・調査
- advantage の診断（§ results の 6 節）: 成功 / 失敗の positive 率、AUC（フレーム 0.565、エピソード 0.995）、positive 内の成功比率、閾値を絞ったときの比率。
- **sft と rollout の混合比の調査**: RLinf 本体（`ReCapMixtureDataset`）は、`balance=true` で `weight × フレーム数`（フレーム均等）。smolVLA 用ローダ（`smolvla_cfg.py`）は、`weight ÷ フレーム数`で**意味が逆**。ユーザーの指示（「戻して」）で、`BALANCE` の既定を `false` に戻し、README と引き継ぎ文書を訂正した。元論文（arXiv 2511.14759）も確認: デモと自律データの比率は規定されていない。
- 学習率の減衰: `warmup_style: cosine` は warmup の形のみで、**warmup 後は 1e-5 で一定**（ログで確認。当初の説明を訂正した）。
- 他の依頼文: 別の Claude 向けの依頼文を 3 回作成した（評価の増加、実験 R20 → S への差し替え、データ準備・マージ）。

### 3.5 結果文書・動画（10-04）
- `results/recap_smolvla_libero10_task0.md` を作成・詳細化（`430022d`、`5729e5f`）。別環境の文書（`recap_prompt_ablation_and_expS.md`）の内容を統合して、その文書を削除した（ユーザーの指示）。
- **比較動画**（初期状態 14）: ベースを 2 環境で再測定（`noise_seed=0`、18/50 = 0.36）し、0.50 の評価（C、プロンプトなし、2 環境）と、同じ trial id で比べた。`scripts/make_init_state_clip.py` で左側を切り出し、wandb にアップロード（`https://wandb.ai/junhand/libero_RL/runs/30cagwmg`）。RLT の動画と同じ初期状態であることを、最初のフレームの画素差で確認した（`fb4417e`）。

## 4. 作成・変更したファイル

### スクリプト（`scripts/`）
| ファイル | 内容 |
|---|---|
| `step2_v2_supervisor.sh` | Step2 の 4,000 ステップを、checkpoint 保存・整理（直近 2 つ）・自動再開つきで実行。`ADOPT_PID` と `START` で、実行中の学習を引き継げる |
| `run_step4_onward.sh` | Step3 の完了待ち → micro batch 探索 → Step4 → 評価（初回のみ使用。30,000 → 2,000 ステップに変更済み） |
| `resume_step4_noweight.sh` | Step4 の再開・追加学習・評価。環境変数: `BALANCE`（既定 `false`）、`ADV_TAG`、`EXP_NAME`、`MAX_STEPS`、`SAVE`、`MB`、`N_EVAL`、`EVAL_ENVS`、`EVAL_STEPS`、`BASE_CKPT`、`RUN_DIR`、`ADOPT_PID`、`START` |
| `recap_eval/eval_base_2envs.sh` | ベースラインの 2 環境評価（`noise_seed=0`、動画あり） |
| `recap_eval/upload_compare_video_wandb.py` | 比較動画を wandb にアップロード |

### 文書
- `results/recap_smolvla_libero10_task0.md`（結果）、この文書、`docs/handoff_step2_onward.md` の §7〜§9（Step3 以降の方針、混合比の訂正）。
- README の Step4 の `balance_dataset_weights` の説明を訂正。

### データ（`data/smolvla_recap/`、新しく作成したもの）
- `libero10_task0_train_v2_merged`（v2.1）と `v30/` 版は、別の Claude が作成（マージ）。私は確認のみ。
- advantage のファイル（`meta/advantages_<tag>.parquet`）: `fail300_N10_q30`（Step3）、`fail300_N10_q20`・`fail300_N10_q10`（relabel）、`fail300_outcome`（成功エピソード基準）。**`train_v2_merged` と `libero10_task0_sft_target3` の両方にある。** q20 は、学習に使っていない（実験 R20 は、S に差し替えた）。
- sft の advantage（q10・q20・outcome）は、全 True に直してある。

### tmp/（git の管理外）
- `tmp/run_step3_v2.sh`、`tmp/step3_v2.log`、`tmp/step3_state_v2/`、`tmp/relabel_q10.log`、`tmp/relabel_q20.log`。
- ログ: `tmp/step2_v2_events.log`、`tmp/step2_v2_attempt1.log`、`tmp/step4_attempt1.log`、`tmp/step4_resume_attempt1.log`（R10）、`tmp/step4_events_*.log`、`tmp/step9_eval_*.log`。
- 動画: `tmp/recap_video/`（クリップ、評価ログ、アップロードのスクリプト）。

## 5. この環境に入れたもの・設定したもの（`/root` とシステム。共有されない）

| 内容 | 理由 |
|---|---|
| `uv python install 3.12.11` | `.venv-smolvla` の Python が `/root/.local/share/uv/python/` にあり、この環境になかった（`No module named 'hydra'`） |
| `~/.libero/config.yaml` | LIBERO が、設定ファイルがないと入力待ちになり、`EOFError` で落ちる。README の手順で作成（パスは `.venv-smolvla/lib/python3.12/site-packages/libero/libero/` 以下） |
| `apt-get install libegl1 libopengl0` | MuJoCo の描画（`eglQueryString` の `AttributeError`） |
| `HF_HUB_OFFLINE=1` と、評価の `rollout.model.smolvla.vlm_model_name=models/SmolVLM2-500M-Video-Instruct` | Hub への接続を避けるため |

別の環境で作業するときは、`bash scripts/preflight_collect_env.sh`（読み取りのみ）で、足りないものを確認する。

## 6. checkpoint と、削除したもの

### 現存する主な checkpoint（`RLinf/logs/` 以下）
| 実験 | パス |
|---|---|
| Step2 価値モデル（最終） | `value_sft/recap_value_model_sft-20261002-08:13:11/step2_value_sft_v2/checkpoints/global_step_4000`（と 3900） |
| 前半（B: 1000 ステップ） | `20261003-09:38:15-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_1000`（と 900） |
| 後半（C: 2000 ステップ） | `20261003-20:10:02-libero10_task0_recap_smolvla/step4_recap_noweight_resume1000/checkpoints/global_step_2000`（と 1900） |
| R10 | `20261004-08:22:11-libero10_task0_recap_smolvla/step4_expR10_q10_1450/checkpoints/global_step_1450`（と 1350、1200） |
| S（別環境） | `20261004-09:55:21-libero10_task0_recap_smolvla/step4_expS_outcome_1450/checkpoints/global_step_1450`（と 1350） |

各 6.8GB（Step2）または 1.8GB の重み（Step4 の `actor/model_state_dict/full_weights.pt`）。

### 削除したもの
1. **ユーザーが明示的に指示して削除**（10-02）: 過去の試運転の checkpoint 6 個（`recap_value_model_sft-20261002-05:23:21`〜`07:49:58` の `global_step_3 / 6 / 20`、約 40GB）。
2. **自動整理（直近 2 つのみ残す）**で削除: Step2 の途中の checkpoint（300〜3600）、Step4 の途中の checkpoint。
3. **誤って削除した（他の実験のもの）**: §7-3。

## 7. 失敗・事故・訂正（再発防止のため、隠さず記録）

1. **Step2 の supervisor の保存先の誤り**: checkpoint は `.../step2_value_sft_v2/checkpoints/` だが、supervisor は `step2_value_sft` を探していた。整理も、落ちたときの自動再開（0 ステップに戻るところだった）も効いていなかった。約 1,050 ステップで発見。`ls` の glob と、`sort -t_ -k4` が、パスの `_` の位置が同じで効かない問題も同時に修正（数値ソートに変更）。
2. **Step4 の保存先の変更**: wandb の有効化で `experiment_name` を変えると、保存先が変わる。スクリプトの検索パスを、実験名つきに直した（`f1d9f1f`）。
3. **他の実験（STEAM）の checkpoint 4 個を削除した**（`step4_steam_n32e2000_endfixD_2000/checkpoints/global_step_200 / 400 / 600 / 800`）。2. の修正で、検索パスを `*/checkpoints/` と広くしたところ、同じ共有ストレージの他の実験の保存先まで対象になり、整理（直近 2 つ以外を削除）で消した（10-03 22:33〜10-04 03:05）。復元できない。`5446bce` で、その実験の保存先だけに限定した。**教訓: 共有ストレージで `rm` を伴う処理の対象は、実験名まで含めて厳密に限定する。**
4. **シェルごと落ちた事故**（10-03 08:28）: `ps | grep "[b]ash scripts/..."` のパターンが、自分のシェルのコマンド文字列（同じ文字列を含む）に一致し、`kill` で自分のシェルを落とした。`ps -eo pid,args | awk '$2=="bash" && $3=="..."'` のように、フィールドを厳密に指定する。
5. **評価の失敗を検出しない**: 自動評価（`resume_step4_noweight.sh`）が、10 環境で OOM になっても（A5000 環境）、「ALL DONE」と出力する。未修正。`EVAL_ENVS=2 EVAL_STEPS=13000` を指定すれば動く。
6. **複数環境で `tmp/` を共有**すると、ログのファイル名が衝突する（別の Claude の指摘で発見）。ファイル名に実験名を入れて解消した（`f8fc577`）。
7. **説明の訂正**: (a) 前半の sft 50:50 は、元の RLinf と逆の意味だった（§3.4）。(b) 学習率は cosine で減衰しない。(c) 「マージ済みのデータセットは使えない」という私の最初の説明は誤り（`advantage.max_samples` で先頭だけ処理でき、Step4 のローダは、advantage のないフレームを除外する）。
8. **relabel が sft を区別しない**: relabel が sft を区別せず、sft_target3 の 73% を negative にしていた（手作業で全 True に修正してから学習）。

## 8. ユーザーの決定事項（時系列）
- Step2 を 4,000 ステップ、約 1 時間おきに checkpoint。eval のサンプルを 300 に削減。checkpoint は直近 2 つのみ。
- Step3 以降の sft は、Task 0 の 3 本のみ（`libero10_task0_sft_target3`）。
- Step3 完了後は、Step4 以降に自動で進める。Step4 は 2,000 ステップ。
- 1000 ステップの checkpoint が保存されたら、次の 1 ステップの後で学習を止め、評価。
- 重みなし（`balance_dataset_weights=false`）で `global_step_1000` から再開。wandb も有効にする。
- 公式の挙動（フレーム均等）に合わせて、混合比の既定を `false` に戻す。
- R20（上位 20%）の代わりに、実験 S（成功エピソードを positive）を別環境で実施。
- RECAP の結果を `results/` に詳細に記載し、不要な別文書は削除。
- 比較動画（初期状態 14）を wandb にアップロード。

## 9. 未完了・次の候補
- ベースラインの、同じ手順での複数回の測定（現状: 10 環境で 20/50、2 環境（`noise_seed=0`）で 18/50）。
- 学習量を増やす実験（例: batch 64 で 8,000 更新、学習率の減衰）、価値モデルの追加学習（18,000 ステップ前後）、先読み幅 N の変更、全 4,122 本の advantage（Step3 の追加）。
- `resume_step4_noweight.sh` が、評価の失敗を検出するようにする。
- `smolvla_cfg.py` の `balance_dataset_weights` の意味を、RLinf 本体と揃えるパッチ（過去の実行の再現性に注意）。
- 実験 R20（`fail300_N10_q20`）は、advantage のファイルだけがあり、学習は未実施。

## 10. 別の Claude への申し送り
- 作業の前に `CLAUDE.md` を読む。**`pkill` / `killall` は使わない。** PID は `awk` などで厳密に確認する。
- **共有ストレージで削除を伴う処理は、対象の実験名まで含めて限定する**（§7-3）。他の実験の `RLinf/logs/` 以下は、触らない。
- 環境ごとの準備（§5）を、`scripts/preflight_collect_env.sh` で確認する。
- RLinf/ のファイルは直接編集しない（`patches/rlinf/*.patch` にして `bash patches/apply_rlinf_patches.sh`）。一時的な作業は `tmp/`（`/tmp` は使わない）。
- 評価は、2 環境 × 25 周（`env.eval.total_num_envs=2`、`max_steps_per_rollout_epoch=13000`）で、動画の切り出し（`make_init_state_clip.py`）と整合する。メモリ上限 25GB の環境で 10 環境にすると OOM になる。
- 数字の確認が必要なとき: 結果は `results/recap_smolvla_libero10_task0.md`、ログは `tmp/` と `RLinf/logs/<日時>-.../run_embodiment.log`。

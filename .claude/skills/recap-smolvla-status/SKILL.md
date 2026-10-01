---
name: recap-smolvla-status
description: libero_RL リポジトリで進行中の「smolVLA (lerobot/smolvla_libero) で RLinf の RECAP を再現する」実験の現在の進捗状況・既知のハマりどころ・標準運用ルールをまとめたオリエンテーションスキル。新しいセッションの開始時、コンテキスト圧縮の直後、「現状を教えて」「ここまで何をした？」「Step1〜4や評価は終わった？」「次は何をすればいい？」「なぜこのエラーが起きた？」のように聞かれたら、README.md を読む前に必ずこのスキルを使うこと。RLinf 側の既知バグとその修正方法・適用済みパッチの一覧、各 Step の完了状況、長時間学習を監視する際の Monitor/run_in_background のタイムアウト制限と回避策も含む。
---

# libero_RL: smolVLA 版 RECAP 再現 — 現状スキル

## このプロジェクトの目的

RLinf の RECAP（advantage-conditioned SFT）手法を、smolVLA（`lerobot/smolvla_libero`）で
LIBERO-10 Task 0 上に再現する実験。手順の全詳細は `README.md` の
「RECAP を smolVLA（lerobot/smolvla_libero）で行う」セクション（226 行目以降）にある。
このスキルは「今どこまで進んでいるか」「何に気をつけるべきか」を、ゼロから探り直さずに
把握するためのもの。README.md とあわせて読むこと。README.md の方が一次情報源であり、
このスキルは状況が進むたびに更新する前提の要約レイヤー。

## 全体パイプライン

Step1（return 計算） → Step2（価値モデル SFT） → Step3（advantage 計算） →
Step4（advantage 条件付き SFT） → Step9（LIBERO シミュレータでの評価）

## 現在の進捗（2026-10-01 時点）

- ベースライン評価: `lerobot/smolvla_libero` で `success_once=0.40`（20/50、固定初期状態 50 個）。
  `smolvla_libero_plus` は 0.96 と高すぎたため不採用（アドバンテージ信号が出にくい）。
- rollout 収集: train split 目安 4,096 episodes → 完了（途中で破損した 2 episodes は除外）。
- **Step1（return 計算）**: sft / train / train_extra / eval の 4 データセット全てでフルスケール
  完了。`meta/returns_fail300.parquet` 生成済み。
- **Step2（価値モデル SFT）**: フルスケール完了。8000/8000 step、所要 25 時間 2 分。
  最終 checkpoint:
  `RLinf/logs/value_sft/recap_value_model_sft-20260929-23:35:58/step2_value_sft/checkpoints/global_step_8000`
  最終指標: `train/value_spearman=0.946`, `train/loss=2.09`, `eval/cat_acc_best=0.287`,
  `eval/cat_acc_neighbor=0.442`。
- **Step3（advantage 計算）**: 未実行。次のアクション。上記 checkpoint を
  `advantage.value_checkpoint` に指定して、sft/train/train_extra の実データに対して実行する。
- **Step4（advantage 条件付き SFT）**: 未実行。Step3 完了後、30000 step（README/RLinf docs
  で確認済みの目標値）で実行する。
- **Step9（評価）**: 未実行。ゴールは `eval/success_once` がベースライン 0.40 を上回ることの
  確認（RECAP のアドバンテージ学習に効果があったかどうかの最終判定）。

## 既知のハマりどころ（再発見しなくていいように）

1. **lerobot `datasets` ライブラリの List 型非互換**: v2.1→v3.0 変換は学習と同じ
   `.venv-smolvla` 内で実行すること（別 pixi env だと `datasets>=4.8` 要求で `datasets==3.6.0`
   前提のコードと衝突してコケる）。`scripts/convert_rlinf_to_lerobot_v30.sh` に実装済み。
2. **`convert_rlinf_collected_to_v21.py` の OOM**: 修正済み（`ProcessPoolExecutor` を
   bounded pipeline 化、1 parquet ファイル = 1 episode の逐次読み込みに変更）。
3. **Step4 の FSDP クラッシュ**: `actor.fsdp_config.use_orig_params=true` と
   `actor.model.precision=fp32` の **両方** が必須（片方だけでは `FlatParamHandle` の
   dtype/`requires_grad` 不一致で落ちる）。`patches/rlinf/smolvla-recap.patch` の
   デフォルト値として反映済み。
4. **`run_compute_returns.sh` / `run_compute_advantages.sh` の `eval $CMD` が Hydra の
   リスト型 override を bash の brace 展開で壊す**（`[{a:1,b:2}]` が `a:1 b:2` に化ける）:
   ラッパーの shell script を使わず、配下の `.py` を直接呼ぶこと。
5. **`SFTRunner.set_max_steps()` の epoch 計算バグ**: `max_epochs` がデフォルトの 30000 の
   ままだと、データセット規模によっては実質無限ステップ（数億ステップ）になる。
   `runner.max_steps=8000`（Step2）のように明示指定すること。
6. **`save_interval % val_check_interval == 0` の assert**（`rlinf/utils/runner_utils.py`）:
   両方を同じ値にする（例: 560/560）。ずれていると最初の backward pass 直後にクラッシュし、
   `[Collator Verification]` ログが繰り返されるため一見ハングに見える（実際はクラッシュ）。
7. **動画デコードエラーでの学習クラッシュ**: `libero10_task0_train` の episode_002599 など、
   ごく一部の動画ファイルが壊れている（`ffprobe` の軽い形式チェックでは検出できない）。
   `patches/rlinf/value-dataset-retry-on-decode-error.patch` で、`ValueDataset.__getitem__`
   がデコード失敗時に詳細（dataset path・episode index・video path）をログした上で次の
   サンプルにリトライするよう修正済み。**Step2（value model）用のデータセットクラスのみ
   対応済み**。Step4 の SFT 用データセットクラスで同様のクラッシュが起きた場合は、同じ
   パターンで別途パッチが必要。
8. **`resume_dir` は Hydra で `+runner.resume_dir=...`（先頭に `+`）が必要**: 素の
   `runner.resume_dir=...` だと `Key 'resume_dir' is not in struct` エラーになる
   （base config に存在しないキーのため、追加キーとして渡す必要がある）。
9. **cgroup v1 の `memory.usage_in_bytes` は RSS ではなくキャッシュ込みの合計**: メモリ監視は
   `memory.stat` の `rss` フィールドを見ること。`usage_in_bytes` だけ見て OOM と早合点しない。
   真の OOM は `memory.oom_control` の `oom_kill` カウンタで確認する。

## 長時間学習を監視する際の注意

- `Monitor` ツールは最大 30 分で失効し、`Bash` の `run_in_background` も実測で約 10 分前後で
  強制終了される。25 時間規模の学習を見張るには、これらに頼らず **完全にセッションから
  デタッチした watchdog スクリプト**（`nohup bash tmp/xxx_watchdog.sh > tmp/xxx_watchdog.log
  2>&1 & disown`）を `tmp/` に書いて起動し、状況確認のたびにそのログファイルを `tail` する
  運用にすること。
- チェックポイントは 2 時間おきに保存し、常に最新 2 つだけ保持（watchdog で自動 prune）。
  wandb には metrics のみアップロードし、モデル重みは上げない。
- ユーザーから「確認頻度が多すぎる」「Claude の使用量を抑えて」という指示が出ている。
  進捗報告は実際のイベント（完了・クラッシュ・ディスク逼迫などの閾値超え）が起きたときのみ
  行い、ルーチンのポーリング結果（「まだ動いています」等）は逐一報告しない。
- ディスク容量 300GB クォータに常に注意（2026-10-01 時点で約 140GB 使用）。

## 標準運用ルール（CLAUDE.md より）

- 一時的な作業は `tmp/` で行う（`/tmp` は使わない）。`tmp/` は git 管理外。
- RLinf（submodule）のファイルは直接編集しない。変更は `patches/rlinf/*.patch` にまとめ、
  `bash patches/apply_rlinf_patches.sh` で適用する（`--revert` で解除）。
- 作業が一区切りついたら、都度 commit まで実行する（ユーザーへの確認は不要）。

## 次のアクション

Step3（advantage 計算）を、Step2 の `global_step_8000` checkpoint を使って実行する。
README.md の「6. Step1〜3（価値モデルとアドバンテージ）」セクション参照。

## 更新履歴

2026-10-01、Step2 フルスケール完走直後に作成。Step3/4/9 が進んだら本スキルも更新すること。

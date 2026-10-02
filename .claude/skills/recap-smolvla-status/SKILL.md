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

## 現在の進捗（2026-10-02 時点）

**旧データ（成功後も 520 ステップ続行）での Step1〜3 は、無効になった。** 旧データと、旧 Step2/3/4 のログ・チェックポイントは、ユーザーが削除済み（旧版の記録: ベースライン `success_once=0.40`、Step2 は L40S で 8,000 ステップ・25 時間）。
旧データは return が「フレーム番号と成否」だけで決まっていたため、成功で打ち切る設定で再収集した。**次の作業は、`docs/handoff_step2_onward.md` が一次情報（他環境の Claude への引き継ぎ文書）。**

- ベースライン評価: `lerobot/smolvla_libero` で `success_once=0.40`（20/50、固定初期状態 50 個）。
  `smolvla_libero_plus` は 0.96 と高すぎたため不採用。
- **再収集（完了）**: 成功で打ち切る設定（`ignore_terminations=False`）、3 環境 × 4 ラウンド、2 並列。
  有効な train は **4,122 本・1,757,810 フレーム、成功 1,689 本（41.0%）**、成功の長さは平均 292、失敗は常に 520。
  LeRobot v2.1（動画）に変換して検証済み。
  - train: `data/smolvla_recap/libero10_task0_train_v2_r{R}_m{M}`（11 個）
  - eval: `data/smolvla_recap/libero10_task0_eval_v2_m{0,1,2}`（計 64 本）
  - sft（人の収集、30 本、20fps）: `data/smolvla_recap/libero10_task0_sft`（v3.0 版は `data/smolvla_recap/v30/libero10_task0_sft`）
  - 元の PNG は `collected_v2/`（削除しない）。偽エピソード 3 本（10 フレーム）は変換時に除外。
  - 詳細は `docs/data_recollection_plan.md`。
- **Step1（return 計算）**: 新 train 11 個と eval 3 個に付与済み（`meta/returns_fail300.parquet`、
  gamma=1.0、失敗時の終端報酬 -300）。全 4,122 本でルールとの不一致 0。範囲は [-819, 0]。
  sft の return も既存。
- **Step2（価値モデル SFT、新データ）**: **未実行（GPU の良い環境で実行する予定）**。4,000 ステップ、1 時間おきに checkpoint。
  A5000（24GB）では、micro batch 8 + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` で 24 秒/ステップ（約 27 時間）。
  起動は `scripts/run_step2_v2.sh`、再開付きは `scripts/step2_v2_supervisor.sh`。速度測定と注意は、引き継ぎ文書の §4。
- **Step3（advantage 計算）**: 未実行。sft + 新 train 11 個を **1 プロセスで一括処理**する（positive 閾値、上位 30%、は全データセットを結合して決めるため）。
  途中再開は `advantage-resume-checkpoint.patch`（`tmp/step3_state/`）。DataLoader のワーカーは 4 個まで。旧スクリプト `tmp/run_step3_advantages.sh` が例。
  return の正規化範囲 [-819, 0] は各 `stats.json` から自動算出されるので指定不要。
- **Step4（advantage 条件付き SFT）**: 未実行。新データを v3.0 に変換（`scripts/convert_rlinf_to_lerobot_v30.sh`）してから、30000 ステップ（README/RLinf docs の目標値）。
  24GB の GPU で fp32・micro batch 32 が載るかは未確認。
- **Step9（評価）**: 未実行。ゴールは `eval/success_once` がベースライン 0.40 を上回ること。

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
   サンプルにリトライするよう修正済み。Step2 用に加え、Step3（`compute_advantages.py`）と Step4（`SmolVLARecapDataset`、advantage 欠損フレームは
   除外）にも `advantage-decode-error-handling.patch` で対応済み。さらに根本対策として、壊れた episode 2599 を
   除いた `libero10_task0_train_clean` を作成した（壊れているのは 2599 のみ）。
8. **base config にないキーは Hydra で先頭に `+` が必要**（`+runner.resume_dir=...`、
   Step3 の `+advantage.max_samples=N` など）。`resume_dir` の例: 素の
   `runner.resume_dir=...` だと `Key 'resume_dir' is not in struct` エラーになる
   （base config に存在しないキーのため、追加キーとして渡す必要がある）。
9. **cgroup v1 の `memory.usage_in_bytes` は RSS ではなくキャッシュ込みの合計**: メモリ監視は
   `memory.stat` の `rss` フィールドを見ること。`usage_in_bytes` だけ見て OOM と早合点しない。
   真の OOM は `memory.oom_control` の `oom_kill` カウンタで確認する。
10. **Step3 の `recap_compute_advantages.yaml` 既定値 `advantage.batch_size: 1024` は 48GB GPU で
    CUDA OOM になる**（Gemma3 の SDPA で 3.78GiB の確保に失敗）。最初のバッチで即落ちる。
    `advantage.batch_size=16` を指定すること（GPU 律速なので大きくしても速くならない）。
    `advantage.max_samples` で件数を絞る場合は base config にないキーなので `+advantage.max_samples=N`。
    計算結果は各データセットの `meta/` に書かれるので、試し実行は `meta/` だけコピーした影
    データセット（data/videos はシンボリックリンク）で行う。

11. **コンテナのメモリ上限は 25GB（cgroup v1）**。長時間の DataLoader 処理で OOM kill されうる
    （`memory.oom_control` の `oom_kill` で確認）。`pkill -f compute_advantages.py` は自分のシェルの
    コマンドラインにも一致して自分を落とすので、PID を指定して kill すること。

12. **Step2 は GPU メモリの断片化で遅くなる（24GB の A5000）**。micro batch 8 で 48 秒/ステップだったのが、
    `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` で 24 秒/ステップになる。micro 16 は 24GB ではメモリ不足
    （gradient checkpointing も効果なし）。`save_interval` は `val_check_interval` で割り切れる必要があり、
    `global_batch_size` は `micro_batch_size × GPU 数` で割り切れる必要がある（Hydra の assertion で落ちる）。
13. **`pkill` / `pkill -f` / `killall` は使わない**（CLAUDE.md の「禁止事項」）。シェルごと落ちる事故が複数回起きた。
    `ps -eo pid,cmd | grep <パターン> | grep -v grep` で PID を確認し、`kill <PID>` を使う。`pgrep -f` も、自分のシェルのコマンド文字列に一致して落ちることがある。
14. **収集の偽エピソード（10 フレーム）**: 初期状態プールの境界で、環境がリセットされずに出る。変換時に `--min-frames 100` で除外する（RLinf 側は修正しない方針）。

## GPU 環境の変更

2026-10-02 時点のこの環境の GPU は **RTX A5000（24GB）1 枚**。旧版のベースライン評価〜Step2 は L40S（46GB）で
実行していた。Step2（新データ）は、ユーザーの指示で、性能の良い GPU の別環境で実行する。README の VRAM・速度の記述は L40S 前提なので、Step4（fp32、micro batch 32）が
24GB に収まるかは本番前に要確認。

## 長時間学習を監視する際の注意

- `Monitor` ツールは最大 30 分で失効し、`Bash` の `run_in_background` も実測で約 10 分前後で
  強制終了される。25 時間規模の学習を見張るには、これらに頼らず **完全にセッションから
  デタッチした watchdog スクリプト**（`nohup bash tmp/xxx_watchdog.sh > tmp/xxx_watchdog.log
  2>&1 & disown`）を `tmp/` に書いて起動し、状況確認のたびにそのログファイルを `tail` する
  運用にすること。
- チェックポイントは定期的に保存し、最新 2 つ（と、節目のステップ）だけ保持（watchdog で自動 prune）。
  wandb には metrics のみアップロードし、モデル重みは上げない。
- ユーザーから「確認頻度が多すぎる」「Claude の使用量を抑えて」という指示が出ている。
  進捗報告は実際のイベント（完了・クラッシュ・ディスク逼迫などの閾値超え）が起きたときのみ
  行い、ルーチンのポーリング結果（「まだ動いています」等）は逐一報告しない。
- ディスク容量のクォータは **600GB**（2026-10-01 にユーザーが 300GB から引き上げ）。同日の `/workspace` 使用量は約 310GB（再収集の PNG を含む）。`df` の使用率は、基になる共有ストレージ全体の値で、クォータの状況ではない（`du -sh /workspace/libero_RL /workspace/.cache` で測る）。

## 標準運用ルール（CLAUDE.md より）

- 一時的な作業は `tmp/` で行う（`/tmp` は使わない）。`tmp/` は git 管理外。
- RLinf（submodule）のファイルは直接編集しない。変更は `patches/rlinf/*.patch` にまとめ、
  `bash patches/apply_rlinf_patches.sh` で適用する（`--revert` で解除）。
- 作業が一区切りついたら、都度 commit まで実行する（ユーザーへの確認は不要）。

## 次のアクション

1. Step2（新データ、4,000 ステップ、1 時間おきに checkpoint）を、GPU の良い環境で実行する（手順は `docs/handoff_step2_onward.md`）。
2. Step3（一括処理）→ Step4（v3.0 変換後に SFT）→ Step9（評価）。各 Step の開始は、ユーザーの指示を待つ。

## 更新履歴

2026-10-01、Step2 フルスケール完走直後に作成。2026-10-02、データ再収集の完了（旧データの Step1〜3 は無効）と、Step2 の速度測定・引き継ぎを反映。Step2〜9 が進んだら本スキルも更新すること。

# 引き継ぎ: smolVLA の強化学習（LIBERO-10 Task 0）— 2026-10-09 時点

この文書は、新しいセッション（人でも Claude でも）が、続きを進められるように、現状・方針・実装・結果・運用ルール・未決事項をまとめたものです。数値の出所（ログ・ファイル）を併記し、**確認済み**と**推測**を区別しています。

## 0. 目的と現在の方針
- 目的: 凍結でない smolVLA（`lerobot/smolvla_libero`、約 450M）を、LIBERO-10 の Task 0（"put both the alphabet soup and the tomato sauce in the basket"）で、強化学習により改善する。基準のベースの成功率は、約 0.42〜0.44（50 エピソード、標準誤差 約 0.07）。
- **完了済み（ユーザー決定、2026-10-07）:** STEAM、RL Token（最良 H: 0.74）。追加実験は提案しない。
- **中止:** FPO（2026-10-08、ユーザーが「やめます」）。
- **現在の方針（2026-10-08、ユーザー）:** **素の Flow-SDE PPO で学習する。共学習（PPO+SFT）は保留。** 報酬は **π0.5 の設計と同じ「成功のみ」に戻す**（段階報酬は、「掴んで落とす」だけで報酬が入る穴が見つかったため、やめる。2026-10-09）。**サンプル（1 更新のエピソード数）を増やす**方向。
- 方針は二転三転した（PPOのみ → 共学習 → PPOのみ）。ユーザーの最新の指示を優先し、共学習を再提案する前に確認する（Claude の記憶ファイル `project-ppo-only-direction.md` は、元の環境でも失われた。内容はこの節のとおり）。

## 1. 学習アルゴリズム（Flow-SDE PPO、πRL 方式）
- 論文: πRL（arXiv:2510.25889）。RLinf の π0.5 の PPO（`rlinf/models/embodiment/openpi/{tasks/rl.py,sampling/rl_sampler.py}`）を、smolVLA に移植した。
- 方策: デノイズ **10 ステップ**（smolVLA の既定。学習中に 5 ステップにしていた時期がある。いまの既定は 10）の Euler 法。学習時だけ、無作為な 1 ステップを SDE（標準偏差 `noise_level 0.5 × √(t/(1−t)) × √δ`）にし、他は ODE。そのステップのガウス対数確率が、行動チャンク全体の厳密な尤度。評価時は全ステップ ODE。
- 対数確率の形: 先頭 10 ステップ × 7 次元 = 70 要素を、チャンク単位（`logprob_type: chunk_level`）で和をとり、比率を 1 つ作る。
- 価値（critic）: VLM の出力の平均（960 次元）+ 状態（8 次元）→ MLP（512, 256）→ スカラー。学習率 1e-4。
- GAE: γ=0.99、λ=0.95（チャンクあたり）、アドバンテージは全体で正規化。打ち切りは終端扱い（verl-vla も同じ）。
- PPO: clip 0.2、dual clip 3.0、value_clip 0.2、Huber(δ=10)、update_epoch 4、`target_kl 0.1`（エポックの平均 KL(k3) が超えたら、残りのエポックを止める。RLinf にはなかったので追加）。
- 学習する範囲: action expert（`lm_expert`）、行動の入出力の層、時刻の層、価値ヘッド（約 100.5M / 450.7M）。VLM は凍結。
- 精度: 学習対象は fp32。**凍結した VLM のプレフィックスだけ bf16**（`sde.prefix_bf16: True`）で、KV キャッシュを fp32 に戻す（学習 約 2.7 倍速、ロールアウト 約 1.5 倍速）。全体の bf16 / TF32 は、小さな重みの変化が丸め誤差に埋もれ、PPO の比率が雑音になるので不可（測定済み）。古い対数確率は、学習前に学習側で再計算する（`algorithm.recompute_logprobs: True`、バッチの形による bf16 の丸めの差を消す）。
- 1 更新の流れ（詳細は会話で説明済み。要点）: ロールアウト（環境数 × `rollout_epoch` エピソード、最大 480 ステップ = 48 チャンク）→ チャンク報酬（10 ステップの合計）→ GAE（逆順、1 更新に 1 回、ロールアウト時の価値で計算）→ 学習（ミニバッチごとに 1 回重みを更新、最大 4 エポック。1 更新目だけ最初の 24 回は価値ヘッドのみ）→ 重みを同期。

## 2. 報酬
- **π0.5 の設計（RLinf の 4 スイートの PPO 設定）:** 成功のときだけ。`reward_coef 1.0 × terminations`、相対報酬（`use_rel_reward: True`、成功の最初のステップで +1、1 エピソードに 1 回）、ステップペナルティなし、チャンク報酬は 10 ステップの合計。中間報酬はない。
- **現在の既定:** `env.train.grasp_reward.enabled: False`（成功のみ）。
- 段階報酬（実装済み・既定オフ）: 把持（接触+2cm 持ち上げ）+0.2（1 つ目、2 つ目）、1 つ目をカゴに入れて離す +0.3、成功 +1.0。**穴:** 2cm 持ち上げて落とすだけで +0.2、保持の継続を見ない、順序の制約なし。成功に結びつかない把持が約 15%。やめた。
- 過去の誤り: `use_rel_reward: False` のとき、成功状態が続く間、毎ステップ +1 が足された（`returns_max=10`）。2 回目の PPO 以降は修正済み。FPO と最初の PPO の結果は、この誤りの影響がある。

## 3. 結果の一覧（学習時の success_once は各更新 15〜48 エピソードの平均。評価は 50 エピソード、シード 950001）
| 実験 | 設定 | 学習時の全体平均 | 評価 |
|---|---|---|---|
| 対照（ベースの重み、5 ステップ、fp32 ラッパー） | – | – | **0.44** |
| PPO、学習率 5e-6、報酬の誤りあり | 最初の PPO | 悪化（終盤 約 0.11） | – |
| PPO、5e-6、報酬修正、5 ステップ | | 更新 10 から 0 に崩壊（13 更新で停止） | – |
| 共学習 β=5、5e-6 | | 0.29 | 0.34 |
| 共学習 + 把持報酬、5e-6（環境 1、c1e1） | | 0.48 | 0.48 |
| 共学習 β=20、5e-6（環境 2、c1e2） | | 0.30 | – |
| 共学習 β=5、5e-7 | 20 更新で停止 | 0.41 | – |
| PPO のみ、5e-7（環境 1、l1e1） | | 0.43（崩壊せず、改善もせず） | 未取得 |
| 段階報酬 A（環境 1）、5e-7、15 エピソード | | 0.36 | 0.44 |
| 段階報酬 C（環境 2）、1e-6、15 エピソード | | 0.35 | 0.38 |
| **段階報酬 B（私）**、5e-7、**48 エピソード**（2 環境 × 24 エポック） | 30 更新完了 | **0.41（後半 0.46〜0.47）** | **0.42**（success_at_end 0.40。`tmp/hybrid_fpo/eval_B_stage_48ep_seed950001.log`、2026-10-09 02:22 完了） |
- **結論（暫定）:** 学習率が決定的（5e-6 は崩壊、5e-7 は崩壊しない）。1 更新のエピソードが 15 だと改善せず、48 だと学習時の成功率が上がる（B）。**ただし B の評価は 0.42 で、ベース（0.44）と同じ。48 エピソード × 30 更新では、評価での改善は確認できなかった。**
- 測定: 各重みを勾配の向きに 5e-6 動かすだけで、1 チャンクの対数確率が約 13 動く（clip 半径 0.2）。方策の対数確率は、極端に敏感。
- FPO（参考）: 学習率 3e-6 で崩壊せず、学習時 0.54 まで上がったが、評価（標準経路、一部 bf16）は 0.46、0.42 でベースと同じ。
- RLinf の π0.5（ページ）: LIBERO-10（Long）43.9% → 93.0%。ただし 1 更新 **512 エピソード**（64 環境 × 8 エポック）。こちらの 15〜48 エピソードとは桁が違う。

## 4. 制約と、サンプル数を増やす方法
- **ホストメモリ（最大の制約）:** コンテナの上限 約 25GB。LIBERO の環境プロセス 1 つが約 3.3GB（シミュレータ本体 約 2.4GB）。3 環境で 23〜24GB、**データ量を増やすと、受け取りと並べ替えの時点で、ピークが上限に達し、アクターが落ちる**（3 環境 × 15 エポック、3 環境 × 8 エポックで落ちた）。
- 2 環境にすると、約 3.3GB 空き、**2 環境 × 24 エポック（48 エピソード）が、ピーク 24.7GB で動いた**（B）。2 環境のまま増やせるのは、約 55 エピソードまで（推測）。1 環境なら 72〜96 エピソード（未検証、1 更新 約 36〜48 分）。1 エポックの時間は、環境数にほぼよらない（約 27 秒）。
- **変更しないこと（ユーザー指示）:** 環境プロセスの起動方式（spawn → fork）などの、メモリ削減の改変。試して戻した（fork は効果が小さく、リセットごとのシミュレータ再構築でメモリが増える、という観察は得た）。
- **3 GPU（3 コンテナ）の活用:** (1) 3 コンテナを 1 つの Ray クラスタにつなぐ（RLinf の複数ノード構成）と、ロールアウトが約 1/3、ホストメモリが 3 台分になる。ただし、コンテナ間の通信（Ray、NCCL）が通るか**未確認**。自分のコンテナの IP は `172.21.0.2`（eth0）と `10.0.94.39`（podnet1）。次の一手は、通信の確認（私のコンテナで待ち受けを立て、他の 2 環境から接続を試す）。(2) 通らなければ、3 本の実験を並行する（いまの運用）。

## 5. 次にやること（優先順）
1. **B の評価は 0.42 で完了（対照 0.44 と同じ。基準の 0.57 に届かず）。** 10 ステップの対照（`eval_control10_seed950001.log`）は**未実行**（B の後に続けて回す予定だったが、ログがない。コンテナの再起動の可能性）。必要なら `./eval_sde.sh control10 950001`（重みなし = ベース）で回す。
2. **報酬を成功のみに戻した実験を回す:** 私の GPU で、2 環境 × 24 エポック（48 エピソード）、学習率 5e-7、`grasp_reward` なし（B との違いは報酬だけ）。環境 1 で、1 環境 × 72 エポック（72 エピソード）。起動コマンドの例（私）: `cd tmp/hybrid_fpo; EXP=ppo_success_only_48ep CONFIG=libero_10_smolvla_ppo_sde EPOCHS=30 nohup ./run_cfg_wandb.sh runner.save_interval=30 actor.optim.lr=5.0e-7 env.train.total_num_envs=2 env.train.rollout_epoch=24 actor.global_batch_size=384 actor.micro_batch_size=12 > /dev/null 2>&1 &`（`grasp_reward.enabled` の既定は False。環境 1 に出す指示文は、前の `S1-E1` の書式を踏襲し、`env.train.grasp_reward.enabled=True` を外す）。
3. 3 GPU の複数ノード化の可否を、通信の確認で判断する。
4. （保留）価値ヘッドのオフライン事前学習（ベースのロールアウト 4,122 エピソード `data/smolvla_recap/libero10_task0_train_v2_merged` がある）、混在する初期状態への絞り込み（50 状態のうち、成功確率 0.1〜0.9 が 24、0.1 未満が 18）、エピソード上限の短縮（480 → 360）。

## 6. 実装の場所
- パッチ（RLinf のサブモジュールは直接編集しない。`bash patches/apply_rlinf_patches.sh` で適用）: **`patches/rlinf/smolvla-zzz-fpo.patch`**（名前は、ほかのパッチの後に適用するため）。クリーンな RLinf に全パッチを順に当てて、開発用コピーと一致することを確認してある。
- 開発用コピー（実際に実行しているコード）: `tmp/rlinf_dev/`。環境ごとの自分用コピー: `tmp/rlinf_dev_<tag>/`。
- 主なファイル: `rlinf/models/embodiment/smolvla/{sde.py,smolvla_sde_model.py,smolvla_fpo_model.py,fpo.py}`、設定 `examples/embodiment/config/libero_10_smolvla_ppo_sde.yaml`、共学習 `rlinf/data/datasets/smolvla_sft.py` と `embodied_fsdp_actor_worker.py`、段階報酬 `rlinf/envs/sim/libero/{libero_env.py,venv.py}`、ロールアウト動画 `rlinf/utils/rollout_video.py`、保存時のメモリ対策 `fsdp_config.save_training_state: False`（重みだけ保存）。
- スクリプト: `tmp/hybrid_fpo/` に実行用、`scripts/fpo/` にコピー。`run_cfg.sh`（私）、`run_cfg_env.sh`（環境ごと: `DEV=<dev copy> TAG=<tag> CONFIG=<config>`）、`run_cfg_wandb.sh` / `run_cfg_env_wandb.sh`（wandb と動画）、`eval_sde.sh`（50 エピソード評価、既定 10 ステップ）、`memwatch.sh`（メモリ記録）、`test_sde_model.py` ほか。
- 設計書: `docs/flow_sde_ppo_smolvla_design.md`、`docs/fpo_smolvla_design.md`（FPO、中止）。
- **チェックポイント:** `tmp/hybrid_fpo/logs/<run>/<experiment_name>/checkpoints/global_step_30/actor/model_state_dict/full_weights.pt`（約 1.8GB、重みのみ。再開用の状態は保存しない）。**ディレクトリ名は `runner.logger.experiment_name`**（wandb 版は `EXP` の値、素の起動は `libero_10_smolvla_ppo_sde`）。

## 7. 運用ルール・落とし穴（重要）
- **ユーザーの標準ルール（CLAUDE.md）:** 一時作業は `tmp/` で。RLinf のファイルは直接編集せず、パッチにまとめる。区切りごとに commit（確認不要、メッセージは変更内容がわかるように、`Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>` で終える）。**`pkill` / `killall` は使わない**（`pgrep -af` で PID を確認し、`kill <PID>`）。
- データ・ログの削除は、ユーザーの明示的な指示があるまでしない。`models/` と、他の環境が使うファイルは触らない。
- 学習中に重い分析（メモリを使うもの）を並行しない（ホストメモリ 約 25GB）。長い実行は、デタッチ（`nohup`）で。
- **確認の仕方:** プロセスの有無でなく、**ログの最終更新時刻と更新数**で判断する（ハングや、落ちた後の残りを見誤った）。`tr '\r' '\n' < run.log | grep -a -E "Global Step:|exit=|Traceback"`。
- **他の環境（環境 1・2）への指示文:** 環境ごとに、**差分でなく全文を、別ブロックで**出す（元は Claude の記憶ファイル `feedback-per-env-instructions.md`。失われたので、この文の内容がすべて）。「同じ実験」と書かない。2 台とも、`~/.bashrc` に `WANDB_API_KEY` がある。共有ファイルシステムで、`tmp/hybrid_fpo/` は共通。各環境は、自分用の開発用コピー `rlinf_dev_<tag>` と `TAG` を使い、他の実行と干渉しない。
- wandb: プロジェクト `smolvla-libero-ppo`、エンティティ `junhand`。**キーは画面に出さない**（`~/.bashrc` から読み込むだけ）。学習の指標と、更新ごとの成功・失敗のロールアウト動画（学習データから作成、追加の推論なし）を送る。動画は 2Hz、1 コマ = 1 チャンク。送信前にユーザーへ確認すること（シミュレータの画像と設定の全体が送られる）。
- Hydra の指定: 設定にないキーは `+key=value`、既にあるキーは `key=value`（`+` を付けると、既にあるときエラー）。
- 評価の落とし穴: チェックポイントのパスが空だと、**ベースの重みで評価が走る**（今回、起動時に確認を怠り、誤って始めて、止めた）。`test -f <path>` で確認してから起動する。
- 保存時のメモリ: 再開用の状態（optimizer など）を保存すると、ホストメモリが上限に達する（落ちた）。`save_training_state: False`（既定の設定ファイルに入っている）。
- Ray の閾値: `RAY_memory_usage_threshold=0.995`（`run_cfg*.sh` に入っている。cgroup の使用量に、解放可能なキャッシュ約 3GB が含まれるため）。
- 評価の流儀: 評価は 50 エピソード（シード 950001）で十分。有望なものだけ、シードを足して 150 エピソードにする（ユーザー指示: 150 は時間の無駄）。評価の経路は、**学習と同じステップ数（10）、fp32 の同じラッパー**で、対照は、同じ条件のベースの重み。

## 8. 未解決・懸念
- B の評価は 0.42 で、対照を超えなかった。サンプル数（48 エピソード）でも足りない可能性がある。
- 1 エピソードの報酬が疎なため、成功が少ない更新では、アドバンテージが価値の誤差の雑音になる（崩壊の一因という仮説、未検証）。価値の説明分散は 0.0〜0.4 と低い。
- 評価時と学習時の成功率の差（学習時はノイズ付き SDE）。
- 共学習の効果は、学習率 5e-7 では見えなかった（0.41 vs PPO のみ 0.43）。共学習は 1 更新が約 2.7 倍遅い（原因の一部は未特定: プロファイルの見積もり 78 秒/エポックに対し、実測 178 秒/エポック）。
- 環境 1 の `l1e1`（PPO のみ 5e-7）の評価は未取得。環境 1・2 の `S1` 系の評価（A: 0.44、C: 0.38）は取得済み。
- 3 GPU の通信が未確認。

## 9. 新しい環境での立ち上げ（git clone から）
この文書を読む Claude は、`git clone` した新しいフォルダで作業を始める想定。`tmp/`・`data/`・`models/` と `RLinf/.venv-smolvla` は git に入っていないので、以下で作る。パスは clone した場所（以下 `$ROOT`）に読み替える。`scripts/fpo/*.sh` と、それをコピーした `tmp/hybrid_fpo/*.sh` は、スクリプトの 2 つ上のディレクトリを `$ROOT` とする（`ROOT=...` で上書き可）。
1. clone と submodule: `git clone --recurse-submodules <repo> $ROOT`（RLinf は `d1bf37cd` に固定）。
2. パッチ: `cd $ROOT && bash patches/apply_rlinf_patches.sh`。PPO（Flow-SDE）の実装は `patches/rlinf/smolvla-zzz-fpo.patch` に入っている（2026-10-09 に、クリーンな RLinf に全パッチを当てたものと、元環境の `tmp/rlinf_dev` で、PPO 関連のファイルが一致することを確認済み）。
3. Python 環境: README の「smolVLA 用の環境」（`bash requirements/install.sh embodied --model pi0_fast --env libero --venv .venv-smolvla --python 3.12.11` を `RLinf/` で実行、ほか注意点も README 参照）。スクリプトは `$ROOT/RLinf/.venv-smolvla` を使う。
4. モデル: `hf download lerobot/smolvla_libero --local-dir $ROOT/models/smolvla_libero` と `hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir $ROOT/models/SmolVLM2-500M-Video-Instruct`（スクリプトは `HF_HUB_OFFLINE=1` で動くので、事前に取得が必要）。
5. 開発用コピーと実行用ディレクトリ: スクリプトは、パッチ済みの RLinf のコピー `$ROOT/tmp/rlinf_dev` で動く。
   `mkdir -p $ROOT/tmp && rsync -a --exclude .git --exclude .venv-smolvla $ROOT/RLinf/ $ROOT/tmp/rlinf_dev/`、
   `mkdir -p $ROOT/tmp/hybrid_fpo && cp $ROOT/scripts/fpo/*.sh $ROOT/tmp/hybrid_fpo/`。
6. 動作確認: まず 1〜2 更新の短い学習（`cd $ROOT/tmp/hybrid_fpo && CONFIG=libero_10_smolvla_ppo_sde EPOCHS=2 ./run_cfg.sh env.train.total_num_envs=2 env.train.rollout_epoch=2`）で、`run.log` に `Global Step:` が出て `exit=0` で終わることを確かめてから、本番（§5）に進む。wandb を使うなら `~/.bashrc` に `WANDB_API_KEY` を置き、送信前にユーザーへ確認する。
7. ホストメモリ・GPU が元の環境（約 25GB、A5000 24GB）と違えば、環境数・エポック数（§4）を測り直す。メモリが多ければ、1 更新のエピソード数を増やせる（方針どおり）。
- 元の環境の `tmp/`（学習ログ、チェックポイント、評価ログ）は引き継がれない。結果の数値は §3 がすべて。B のチェックポイントが必要なら、元の環境の `tmp/hybrid_fpo/logs/` にある（ユーザーに依頼）。
- `.claude/skills/recap-smolvla-status` は RECAP（〜2026-10-02）の情報。PPO の現状はこの文書が一次情報。

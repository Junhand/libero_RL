# smolVLA 向け Flow-SDE PPO（πRL 方式）の設計（2026-10-07）

状態: 設計のみ（未実装）。根拠: πRL 論文（arXiv:2510.25889）、RLinf の `rlinf/models/embodiment/openpi/{tasks/rl.py, sampling/rl_sampler.py}`、`examples/embodiment/config/libero_10_ppo_openpi_pi05.yaml`、lerobot の `policies/common/flow_matching.py`、`smolvla/modeling_smolvla.py`。

## 1. 方式
学習時だけ、デノイズ（10 ステップ）のうち**無作為な 1 ステップを SDE（確率的）にし、残りは ODE（決定的）**にする（hybrid）。決定的なステップは確率 1 なので、確率的な 1 ステップのガウス対数確率が、この方策の厳密な尤度。PPO の比率は、その 1 ステップの `exp(logp_new − logp_old)`。評価時は全ステップ ODE（現在の smolVLA と同じ）。

## 2. smolVLA に当てはめられる点（読んで確認した）
- 時間の向きが同じ: lerobot の `euler_integrate` は t=1（ノイズ）→0（行動）、`dt = −1/num_steps`、`x ← x + dt·v`。RLinf の `get_timesteps(10)` は `[1, 0.9, …, 0.1, 0]` で一致。smolVLA の `num_steps` は 10。→ `sample_mean_var`（flow_sde）の式を、そのまま再利用できる見込み（式の導出の再確認は実装時に行う）。
- デノイズ 1 ステップ: `model.denoise_step(prefix_pad_masks, past_key_values, x_t, t)`（FPO で使用済み）。プレフィックスの KV キャッシュは、ステップ間で共有できる。
- FPO で作った部品を再利用: 価値ヘッド（プレフィックスのプール + 状態の MLP）、fp32 化、学習範囲の凍結（lm_expert と action 系）、`build_lerobot_batch_from_env_obs`、`default_forward` の入力の復元。

## 3. 実装案（`SmolVLAForFlowSDE`、`SmolVLAForFPO` と並べて追加）
- `predict_action_batch`（train）: プレフィックス 1 回 → 10 ステップのループ。無作為に選んだ 1 つの step で `flow_sde`、他は ODE。保存するのは、**選んだステップの (x_t, x_{t−1}, step インデックス)** と、行動のチャンク（全ステップの連鎖は、ホストメモリのため保存しない）。`prev_logprobs` は、`gaussian_logprob(x_next, mean, std)` を `[B, 10(実行ステップ), 7(実行次元)]` に切り出したもの（RLinf の π0.5 と同じ切り出し）。
- `default_forward`: 保存した x_t から 1 回の `denoise_step` で v を計算 → `sample_mean_var` → 保存した x_next の対数確率を再計算。計算量は、FPO（プレフィックス + デノイズ 4 回）より小さい。
- 価値: FPO と同じヘッド（π0.5 の recipe は VLM の出力の直後に 4 層 MLP）。
- 設定: `noise_method: flow_sde`、`noise_level 0.5`（πRL の最良。0.2 は不安定、0.8 は劣化）。

## 4. RLinf の π0.5 LIBERO-10 の recipe（`libero_10_ppo_openpi_pi05.yaml`）との比較
| 項目 | RLinf の π0.5 PPO | こちら（FPO の実行） |
|---|---|---|
| 環境数 × rollout_epoch | **64 × 8 = 512 エピソード/更新** | 3 × 5 = **15 エピソード/更新** |
| max_episode_steps | 480 | 520 |
| update_epoch / global batch | 4 / 2048 | 2 / 130 |
| clip（比率） | 0.2（dual clip c=3、value_clip 0.2） | 0.01（FPO） |
| lr / clip_grad | 5e-6 / 1.0 | 1e-5 / 25 |
| γ / λ | 0.99 / 0.95 | 0.995（チャンクあたり 0.951）/ 0.99 |
| logprob_type | chunk_level | token_level（FPO は chunk 単位の 1 つ） |
- **最大の差は、1 更新のエピソード数（512 と 15）**。3 環境（ホストメモリの上限）では、同じ recipe を再現できない。更新ごとの勾配の分散は、こちらのほうがはるかに大きい。この差は、FPO と PPO のどちらを使うかとは別に、安定性に影響する可能性が高い（未検証の推測）。

## 5. 未確認・リスク
- SDE の式（`σ_τ`、補正項）の符号と、smolVLA の時間の向きの一致は、実装時に数値で確認する（ODE との周辺分布の一致、a→0 で ODE に戻ること）。
- ノイズ入りの方策で、smolVLA の成功率が下がらないか（a=0.5）。ロールアウトの成功率で確認。
- 評価時（ODE）との不一致。πRL では問題にならなかった。
- 実績は、π0 / π0.5（few-shot の SFT から）。smolVLA（450M）で同じ効果が出るかは未検証。

## 6. 段階
P0: SDE のステップの数値テスト（a=0 で ODE と一致、ロールアウトと再計算の対数確率が一致）。P1: RLinf への組み込みと、2 更新のスモークテスト。P2: Task 0 で数十更新。P3: 150 エピソードの評価（ベース 0.42、FPO、RL Token と比較）。

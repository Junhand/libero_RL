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

## 7. 実装と最初の結果（2026-10-08）

### 実装（`patches/rlinf/smolvla-zzz-fpo.patch` に FPO と同居）
`smolvla/sde.py`（SDE の 1 ステップ、ガウス対数確率、プレフィックスのキャッシュ）、`smolvla/smolvla_sde_model.py`（`SmolVLAForFlowSDE`）、`examples/embodiment/config/libero_10_smolvla_ppo_sde.yaml`。`fsdp_config.save_training_state: False` で、再開用の状態の保存を省略できる（保存時のホストメモリ不足の回避）。単体テスト（`scripts/fpo/test_sde_model.py`）: ODE のループが lerobot の `sample_actions` と一致（誤差 5e-7）、ロールアウトと再計算の対数確率の差は 0、勾配は学習対象だけに流れる。

### 実行結果
1. 1 回目（`use_rel_reward: False`、デノイズ 10 ステップ、30 更新）: 更新 9 の success_once 0.66 をピークに悪化し、終盤は平均 約 0.11。最後の保存でホストメモリ不足。
2. **報酬の誤り**: π0.5 の recipe は `use_rel_reward: True`（環境の既定値）で、成功の報酬は 1 回だけ。上の実行は `False` で、成功状態が続く間、毎ステップ +1 が足され、`returns_max=10` になっていた。FPO の実行も同じ誤りがあった。
3. 2 回目（recipe に合わせて修正: `use_rel_reward: True`、デノイズ 5 ステップ、固定の初期状態、betas (0.9, 0.95)、weight decay 0.01、cosine）: 報酬は修正され（`returns_max=1.0`）、1 更新 約 3.8 分に高速化したが、**更新 10 以降の 4 更新で success_once が 0 に崩壊**（0.40, 0.13, 0.06, 0.20, 0.06, 0.26, 0.40, 0.06, 0.06, 0, 0, 0, 0）。13 更新で停止。
- 更新 1（学習前の方策）は 0.40 で、ベースと同じ。学習で悪化している。
- 推測（未検証）: 1 更新 15 エピソード（recipe は 512）では、成功が少ない更新で、アドバンテージが価値の誤差の雑音になり、正規化で引き伸ばされて、方策が悪化する。
- 次の候補: 学習率を下げる（1e-6）、価値のみの更新を 3 回に延長（`critic_warmup_steps 72`）。環境 1・2 に依頼する指示文を作成済み（未送付）。

## 8. 共学習・学習率・高速化の結果（2026-10-08）

### 学習率が決定的（`results` は学習時の success_once、各更新 15 エピソードの平均）
| 実行 | 設定 | 全体の平均 |
|---|---|---|
| 共学習 β=5、lr 5e-6 | 前回 | 0.29（評価 0.34、対照 0.44） |
| 共学習 + 把持報酬、lr 5e-6 | 環境 1（c1e1） | 0.48（評価 0.48） |
| 共学習 β=20、lr 5e-6 | 環境 2（c1e2） | 0.30 |
| 共学習 β=5、lr 5e-7 | 20/30 更新で停止 | 0.41 |
| PPO のみ、lr 5e-7 | 環境 1（l1e1） | 0.43（崩壊せず、改善もしない） |
- ベースは約 0.42（対照の評価 0.44）。**lr 5e-6 は大きすぎた。** 測定: 各重みを勾配の向きに 5e-6 動かすと、1 チャンクの対数確率が約 13 動く（clip 半径は 0.2）。lr 5e-7 では崩壊しないが、30 更新では、ベースを超えない。
- 共学習（SFT の項）の効果は、lr 5e-7 では見えない（0.41 と 0.43）。1 更新は PPO のみの約 2.7 倍の時間。
- 把持報酬の追加は、学習時の成功率で最良（c1e1）だが、評価では対照との差が小さい（+0.04）。

### 方針（2026-10-08）
素の PPO で学習する（FPO は中止、共学習は保留）。

### 高速化: 凍結した VLM のプレフィックスだけ bf16（`sde.prefix_bf16`）
- 全体の bf16/TF32 は不可: 小さな重みの変化が丸め誤差に埋もれ、PPO の比率が雑音になる（TF32 の誤差は信号の約 3 倍、bf16 は約 50 倍）。
- 凍結部分だけ bf16（KV キャッシュは fp32 に戻し、action expert と損失は fp32）なら、比率の誤差は信号の約 2%。PPO の micro-batch の順伝播 + 逆伝播が 514 → 193 ms（2.7 倍）。ロールアウトの 1 チャンクは 240 → 159 ms。
- バッチの形が違うと、bf16 の丸めが変わり、同じ重みでも 1 チャンクの対数比率が ±0.03 ずれる（clip 0.2 の約 15%）。→ `algorithm.recompute_logprobs: True`（学習前に、古い対数確率を学習側で再計算）を併用。RLinf の検証は OpenVLA 系のみ許可していたので、smolVLA も許可した（`rlinf/config.py`）。スモークテストで、`rollout_train_logprob_gap` は 0.008 程度。
- 評価の経路（`mode="eval"`）も、同じ精度の前処理を使う。
- 1 更新の見込み: 約 3.5 分 → 約 2.6 分（ロールアウトの大半はシミュレータの動作）。

### 把持報酬（`env.train.grasp_reward`、既定オフ）
把持（接触 + 2cm の持ち上げ）で 1 つ目 +0.2、2 つ目 +0.3。相対報酬の外側で足す。`env/grasp_count` を記録。

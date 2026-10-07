# FPO（Flow Policy Optimization）を smolVLA に実装する方法の検討

作成: 2026-10-07。調査元: verl-vla のドキュメントとコード（`tmp/verl_vla` に、浅い clone を置いた。`src/verl_vla/workers/engine/fpo/training_worker.py`、`src/verl_vla/models/base.py`、`src/verl_vla/models/pi0_torch/trainable_model.py`、`examples/rl/fpo/pi05/...`）、RLinf（`tmp/rlinf_dev`）、lerobot の SmolVLA（`modeling_smolvla.py`）。
結論: **実装可能（見込み）。推奨は、RLinf への実装。** verl-vla は、π0.5 / ACT / GR00T / Gaussian Actor のみ対応で、smolVLA は未対応（ドキュメントに記載なし）。

## 1. FPO の仕組み（verl-vla のコードで確認）
- 行動チャンクごとに、フロー時刻 τ ~ U(0,1)、ノイズ ε ~ N(0,I) を **N = 4 組**サンプルし、固定する（更新の間、同じ値を使う）。
- 条件付きフローマッチング（CFM）損失: x_t = τ ε + (1 − τ) a、目標の速度 v = ε − a、L = ‖v_θ(x_t, τ, obs) − v‖²（行動ステップごとに、次元で平均）。
- **対数比 = L_old − L_new**（更新前の方策で計算した損失と、現在の損失の差）。PPO の clip（半径 0.01）に使う。比は、(行動ステップ, サンプル) ごと。損失 = max(−A·r, −A·clip(r))。行動ステップは合計、サンプルは平均。
- L_old は、更新の前に、現在の重みで計算する（同期のオンポリシー）。ロールアウトの時点では、不要。
- 価値: MLP のヘッド（凍結した観測特徴のプール + 状態 → 1）。semi-Markov GAE（割引は、チャンクの環境ステップ数で指数）。最初の更新は、価値だけ学習。
- 主な設定（π0.5、LIBERO-Spatial タスク 2）: clip 0.01、target KL 0.1、更新エポック 2、lr 1e-5（方策）/ 1e-4（価値）、γ 0.995、λ 0.99、チャンク 10、ミニバッチ 128（マイクロ 4）、勾配クリップ 25、エピソード最大 200 ステップ、20 チャンク/軌跡。8 台の H20 と、4 つの CPU ワーカー（各 8 環境）。1 回の更新に約 80〜82 秒。成功率 58% → 100%（30 更新、10 更新で 92%）。

## 2. モデルに必要なもの（`SupportFPOTraining`）
- `fpo_cfm_loss(obs, tokenizer, actions, timesteps, noise) -> [B, H, N]`（損失は縮約しない）
- `fpo_forward_value(obs, tokenizer) -> [B]`
- `fpo_get_value_parameters()`
- π0 のアダプタは、プレフィックス（画像 + 言語）の KV キャッシュを **1 回だけ**作り、N 個のサンプルで共有して、`denoise_step` を呼ぶ。

## 3. smolVLA との対応（lerobot の SmolVLA で確認）
| FPO の要素 | smolVLA の対応 |
|---|---|
| プレフィックスの計算 | `model.embed_prefix(...)`、`vlm_with_expert.forward(..., fill_kv_cache=True, use_cache=True)`（`sample_actions` が使っている） |
| 速度の予測 | `model.denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)` |
| 初期ノイズ | `predict_action_chunk(batch, noise=...)`、`sample_noise` |
| 学習時の損失 | `model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)`（ノイズと時刻を引数で渡せる） |
| 行動の形 | チャンク 50、パディング込みの次元 32（環境の次元は 7）。私たちは、先頭 10 ステップを実行 |

## 4. 実装方式
**A. RLinf に実装する（推奨）**
- 既存の仕組み（smolVLA の環境・評価・RLT の経験、PPO の runner、`gae` の advantage、`actor_critic` の損失）を使える。
- 変更点:
  1. smolVLA のラッパに、`fpo_cfm_loss`（プレフィックス 1 回 + `denoise_step`、損失は、先頭 10 ステップ × 先頭 7 次元）、価値ヘッド（プレフィックスのプール 960 + 状態 8）、ロールアウトで、実行した行動のチャンク（正規化済み・0 でパディング）を保存する処理。
  2. `default_forward` で、`logprobs = −CFM 損失`（形 [B, ステップ, N]）と、`values` を返す。ロールアウトで、τ、ε を生成して、`prev_logprobs` を保存すれば、既存の PPO の clip 損失（clip 0.01）が、そのまま、FPO の損失になる。
  3. 設定（LIBERO-10 Task 0、全タスク、チャンク 10、γ はチャンク単位で 0.995^10 ≈ 0.951、GAE λ 0.99、価値だけの最初の更新）。
- RLinf のファイルは、パッチで変更する（これまでと同じ）。

**B. verl-vla に smolVLA のアダプタを書く**
- 利点: FPO のトレーナー・ワーカーが、そのまま使える。
- 欠点: smolVLA が未対応で、`DataProto`、verl の依存、8 GPU 前提の例など、新しい環境の構築が必要。1 GPU で動くかは未確認。

## 5. 設計の判断点
- **学習するパラメータ:** アクションエキスパート（と、行動の入出力・時刻の層）+ 価値ヘッド。VLM は凍結して、プレフィックスを no_grad で 1 回計算する（メモリと時間を節約）。全体の学習は、24 GB では重い見込み。
- **損失の対象:** 実行する先頭 10 ステップ × 先頭 7 次元（パディングの 25 次元は、除く。π0 のアダプタも、環境の次元だけ使っている）。
- **報酬とエピソード長:** タスク完了（+1）。Task 0 の成功は、230〜300 ステップで、FPO のドキュメントの 200 ステップでは足りない。学習のエピソードは、400〜520 ステップ（40〜52 チャンク）にする必要がある。報酬は疎で、ステップ数のペナルティ（−1/0 形式）は、選択肢。
- **サンプル数:** 3 環境（メモリの制約）で、1 ロールアウトが、約 120〜156 チャンク。1 回の更新に、10 ロールアウト程度を集める想定（FPO は、1 更新に、約 640 チャンク）。
- **評価:** 50 エピソードの標準誤差は約 0.07。重要な比較は、シードを変えて 150 エピソード。基準は、ベース（RLT 経路 0.46）、同じ条件の対照。

## 6. 時間の見積もり（推定）
- ロールアウト: 約 1 分/回（3 環境、400〜520 ステップ）。1 更新あたり約 10 分 + 更新の計算（アクションエキスパートだけ、約 1〜2 分）。30 更新で、約 6 時間。FPO の元の実験は、10 更新で大半の改善が出ている。

## 7. リスクと未確認
- RLinf の PPO の runner が、smolVLA のロールアウト・フォワードを受け付けるか（現在の `default_forward` は未実装）。
- `denoise_step` を、勾配つきで、KV キャッシュを繰り返して使えるか（smolVLA のキャッシュの実装）。
- clip 0.01 が、小さい環境数で、安定するか。
- FPO の学習で、方策が劣化する可能性（オフセットの報酬を使った実験では、ベースより悪化した例がある）。
- 報酬が必要という課題は、RL Token と同じ（シミュレータでは、成功の信号が使える）。

## 8. 段階的な計画
1. **S0（半日）:** `fpo_cfm_loss` の単体テスト。固定の τ、ε で、smolVLA の SFT の損失（`model.forward`）と一致すること。勾配がアクションエキスパートだけに流れること。
2. **S1:** ロールアウトでの保存、GAE、1 回の更新の動作確認。
3. **S2:** Task 0 で、5〜10 更新の学習曲線（学習中の成功率、比の統計、KL）。
4. **S3:** 評価（50 → 150 エピソード）。ベース 0.46、対照 0.56、RL Token（D 0.62、H 0.74）と比べる。

## 9. 変更点の一覧（RLinf への実装。パッチ `patches/rlinf/smolvla-aa-fpo.patch` にまとめる予定）
RLinf の PPO（`embodied_fsdp_actor_worker.py`）は、`model(forward_inputs=..., compute_logprobs=True, compute_values=True, use_cache=False)` を呼び、出力の `logprobs`（形: [B, ステップ, 次元]）と `values` から、`actor_critic` の損失（clip）を計算する。ロールアウトは、`prev_logprobs`、`prev_values`、`forward_inputs` を保存する。この契約に合わせる。

| ファイル | 変更 |
|---|---|
| `rlinf/models/embodiment/smolvla/smolvla_action_model.py` | (1) 価値ヘッド `value_head`（VLM のプレフィックスのプール 960 + 状態 8 → MLP → 1）。名前に `value_head` を含めれば、オプティマイザが、価値用の学習率を使う。(2) `fpo_cfm_loss`: プレフィックスを 1 回（no_grad）、KV を N 個に複製、`denoise_step`、損失は先頭 10 ステップ × 先頭 7 次元 → [B, N, ステップ]。(3) `default_forward`（現在は未実装）: `logprobs = −CFM 損失`、`values`。(4) `predict_action_batch`: 乱数ノイズで行動を生成し、τ、ε、行動のチャンク（正規化済み、0 でパディング）、観測のテンソルを `forward_inputs` に保存。`prev_logprobs = −CFM 損失`（no_grad、現在の重み）、`prev_values` を返す。(5) VLM の凍結と、アクションエキスパートの学習設定 |
| `rlinf/models/embodiment/smolvla/__init__.py` | FPO の設定（`n_action_samples`、価値ヘッドの層、凍結の指定）を受け取って、モデルを作る |
| `examples/embodiment/config/libero_10_smolvla_fpo.yaml`（新規） | `adv_type: gae`、`loss_type: actor_critic`、`clip_ratio_high` = `clip_ratio_low` = 0.01、チャンク単位の γ ≈ 0.951、λ 0.99、更新エポック 2、バッチ 128 / マイクロ 4、価値だけの最初の更新（`critic_warmup_steps`）、学習率 1e-5（方策）/ 1e-4（価値）、Task 0 のみ、3 環境、エピソード最大 400〜520 ステップ、1 更新あたりのロールアウト数 |
| 単体テスト（新規、`tmp/` で実行） | 固定の τ、ε で、`fpo_cfm_loss` が smolVLA の学習時の損失（`policy.model.forward`）と一致すること。勾配が、アクションエキスパートと価値ヘッドだけに流れること |

変更しないもの: 環境・評価の仕組み、GAE と PPO の損失のコード、runner、`embodied_fsdp_actor_worker.py`（実装時に、`logprob_type` と `logprobs` の形の扱いを確認する）。

## 10. `verl-vla/examples/rl/fpo` の確認で分かったこと（2026-10-07 追記。第 1〜9 節への訂正と補足）
確認したファイル: `examples/rl/fpo/pi05/libero_spatial_task2_online_from_sft_step100/{run_train.sh, fpo.yaml}`、`src/verl_vla/workflows/config/train/fpo.yaml`、`.../actor/fpo_actor.yaml`。（第 1〜9 節を書いた時点では、`fpo.yaml` の先頭 60 行と、ドキュメントだけを読んでいた。）

| 項目 | 内容 | 設計への影響 |
|---|---|---|
| 規模 | GPU 8 枚（`gpus_per_node=8`）、CPU の環境ワーカー 4 つ × 各 8 環境 = **32 環境**、`MUJOCO_GL=osmesa`（CPU 描画）、`pipeline_stage_num: 2`（環境と方策の重ね合わせ）、`max_interactions: 20`（1 軌跡 20 チャンク） | 1 更新 = 32 × 20 = **640 チャンク**。私たちの 3 環境では、1 ロールアウトが約 120〜156 チャンクなので、**約 5 ロールアウト**で 640 チャンクになる（第 5 節の「10 ロールアウト」を訂正） |
| 学習の長さ | `total_training_steps: 50`、評価は 5 更新ごと（50 エピソード）、`val_before_train: true`、5 更新ごとに保存、`resume_mode: auto` | 30〜50 更新が目安。更新ごとの評価は、私たちの環境では重い（約 20 分）ので、間引く |
| 出発点 | `Miical/pi05-libero-spatial-sft-step-100`（SFT を 100 step だけ行った弱い方策。成功率 58%） | 私たちのベース（成功率 0.46）と近い条件 |
| 割引・GAE | `gamma: 0.995`（環境ステップ単位。semi-Markov）、`gae_lambda: 0.99` | チャンク単位に換算（0.995^10 ≈ 0.951）。環境ステップ数が、チャンクで一定（10）なので、等価 |
| **アドバンテージの正規化** | `normalize_advantages: true`（分散した rollout 全体で） | **第 9 節で書き漏らした。** RLinf 側で、同等の正規化を有効にする |
| 価値の損失 | `vf_coef: 1.0`、価値 lr 1e-4、`value_only_updates: 1`（最初の更新は価値だけ） | 設定に追加 |
| 早期停止 | `target_kl: 0.1` | RLinf の PPO に同等の仕組みがなければ、省略するか、追加する |
| 勾配クリップ | 方策・価値とも 25.0 | 設定に追加 |
| 方策の lr | 1e-5、betas (0.9, 0.99)、定数スケジュール、weight decay 0 | そのまま |
| **ロールアウトでの価値・旧損失** | `rollout.output_critic_value: false`。旧損失（L_old）、旧価値は、**学習ワーカーが、更新の前に、現在の重みで計算**する（`_forward_old_statistics`） | RLinf には、`recompute_logprobs`（学習側の forward で、`prev_logprobs` を再計算）の仕組みがある。**ロールアウトでは、τ、ε、行動、観測だけを保存し、`recompute_logprobs: True` で旧損失を再計算する**と、元の実装と一致する（第 9 節の「ロールアウトで `prev_logprobs` と `prev_values` を保存」を簡略化できる） |
| ロールアウトの形 | `mode: async_envloop`、`trainer.async_rollout: false` | RLinf の同期のロールアウトで良い（同期のオンポリシー） |

### 時間の見積もりの訂正
- 1 更新 ≈ 5 ロールアウト（各 約 1 分、フルタスク 520 ステップ、3 環境）+ 更新の計算（アクションエキスパートのみ、2 エポック、128 のミニバッチ × 約 5）≈ **5〜8 分**。30 更新で **約 3〜4 時間**（第 6 節の「約 6 時間」より短い見込み）。ロールアウトの 1 分は、推定で、環境2 の測定（`docs/handoff_fpo_support.md` の F2）で、確かめる。

## 11. S0 の結果（2026-10-07）: `smolvla_fpo_cfm_loss` の単体テスト
実装: `patches/rlinf/smolvla-aa-fpo.patch`（新規ファイル `rlinf/models/embodiment/smolvla/fpo.py`）。テスト: `scripts/fpo/test_s0.py`、`bench_s0.py`、`debug_s0.py`。実データ（評価データの 1 エピソードの 4 フレーム）、固定の τ・ε（4 サンプル）、モデルは `smolvla_libero`。

| テスト | 結果 |
|---|---|
| T1: lerobot の学習時の損失（`policy.model.forward`、時刻とノイズを指定）との一致 | **policy 全体を fp32 にしたとき、一致**（最大絶対差 2.4e-6〜3.7e-6、相対 1e-5）。bf16 のままでは、不一致（最大絶対差 1.9e-2、相対 0.2） |
| T2: 決定性 | 一致（同じ入力で、完全に同じ値） |
| T3: 勾配 | アクションエキスパート（145 テンソル）と、行動の入出力・時刻の層（各 2）にだけ流れる。VLM（334 テンソル）と `state_proj`（プレフィックスの一部）には、流れない（凍結、no_grad） |
| T4: 重みの微小な変化に対する対数比（L_old − L_new） | **fp32: 変化の大きさに比例**（相対 1e-5 の摂動で標準偏差 2e-5、1e-3 で 2e-3）。**bf16: 摂動によらず、標準偏差 4〜5e-3（最大 2e-2〜9e-2）のノイズ** |

### 重要な発見: FPO は、policy 全体を fp32 で動かす必要がある
- `smolvla_libero` を読み込むと、**アクションエキスパート（`lm_expert`）と、VLM の文章部分（`vlm.model.text_model`）が bf16** で、視覚塔だけが fp32。
- 不一致の原因は、ロジックではなく、bf16 の数値誤差。同じ経路でも、バッチの構成（1 サンプルだけ / 4 サンプルのバッチ）で、予測した速度が、最大 8e-2 変わった。損失の差（対数比）に、**標準偏差 4〜5e-3 のノイズ**が入り、FPO の clip 半径（0.01）と同程度になる。
- **対処: `policy.model.float()`（全体を fp32）**。メモリの増加は、約 1.5 GB（パラメータ 450M × 4 バイト）。ノイズは、約 2e-6 に下がる。RECAP の Step 4（全パラメータ fp32）と、同じ。
- ロールアウトの推論は、bf16 のままでも、行動の分布の差は小さいと見込むが（未検証）、学習側の旧損失・新損失は、必ず、同じ fp32 の経路で計算する。`recompute_logprobs` で旧損失を再計算する設計と一致。

### 時間とメモリの測定（A5000、fp32、N = 4、順伝播 + 逆伝播）
| マイクロバッチ（チャンク数） | 時間 | 1 チャンクあたり | GPU メモリ（ピーク） |
|---|---|---|---|
| 4 | 0.264 秒 | 66 ms | 3.7 GB |
| 8 | 0.506 秒 | 63 ms | 5.2 GB |
| 16 | 0.976 秒 | 61 ms | 8.1 GB |
| 32 | 1.908 秒 | 60 ms | 14.0 GB |

- 640 チャンク × 更新エポック 2 の学習の計算: 約 80 秒。旧損失の再計算（順伝播のみ）: 約 13 秒（推定）。**1 更新の計算は、約 1.5〜2 分**（ロールアウトを除く）。第 10 節の見積もり（5〜8 分/更新、ロールアウト込み）と整合。
- マイクロバッチ 8〜16 が、時間とメモリの釣り合いが良い。元の設定（4）は、メモリが小さく、遅くない。

### S0 の判定
**合格。** 損失の計算は正しく（fp32）、勾配は意図どおり。次は S1（ロールアウトでの保存、価値ヘッド、`default_forward`、GAE、1 回の更新の動作確認）。

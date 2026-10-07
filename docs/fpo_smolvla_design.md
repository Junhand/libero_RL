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

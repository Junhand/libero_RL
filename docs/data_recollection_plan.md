# 学習データ（ロールアウト）再収集の方針

3 つの別環境（`/workspace` を共有、各 1 GPU）で分担して、LIBERO-10 Task 0 のロールアウトを集め直すための方針。
状態: **方針の草案**（未実行）。作成: 2026-10-01。

## 1. なぜ集め直すか

前回の収集（`train` 3,200 本、`train_extra` 1,081 本、`eval` 64 本）は、次の点が RECAP の前提と合っていなかった。

| 項目 | 前回 | 問題 |
|---|---|---|
| 成功時の扱い | `ignore_terminations: True`。成功しても**最後（520 ステップ）まで続行** | 全エピソードが 520 行。return が「フレーム番号と成否」だけで決まり、**早く成功したほど 0 に近い**という進捗の信号が消える（価値モデルの V は、フレーム番号と成否で約 92% 説明できた） |
| fps メタデータ | 10 | 公式データ（train/eval=10、sft=20）と同じ。smolVLA の学習データのラベルも 10。**行の刻みは同じ制御ステップ**なので問題は小さい |

公式の RECAP データは、成功した時点でエピソードが終わっており（長さ 210〜480、平均 381）、これと同じ形にする。

## 2. 変更する設定（前回との差）

| 設定 | 値 | 備考 |
|---|---|---|
| `env.eval.ignore_terminations` | **`False`** | 成功でエピソードを終了する。行動は 10 ステップ単位で実行するため、**打ち切りはチャンク境界**（成功後に最大 9 ステップ余分）。return の誤差は約 0.011 以内 |
| `env.eval.data_collection.fps` | `10` | 前回と同じ（smolVLA の学習データのラベルに合わせる） |
| `env.eval.total_num_envs` | **`2`**（当初 5 → 10 → 5 → **2**） | コンテナのメモリ上限（25GB）のため。**10 並列は即座に、5 並列も 5 エポックを完走できなかった**（スモークテストの 2 エポックは通ったが、実メモリが約 24GB まで増え続け、全 3 環境でエポック 3〜4 で Ray に止められた）。2 並列は、50 を割り切り、メモリに余裕がある見込み（**検証中**） |
| 環境変数 `RAY_memory_usage_threshold` | **`0.98`**（既定 0.95） | 停止閾値を 24.5GB に上げて、余裕を 1.3GB にする。**効果は未検証**。落ちた場合は、保存済みのエピソードは有効なので、後始末（§ 4.5）をして `_t2` で再実行する |
| `env.eval.max_episode_steps` | `520` | 前回と同じ（公式は 480） |
| `env.eval.max_steps_per_rollout_epoch` | `520` | `num_action_chunks`（10）で割り切れること |
| 方策 | `smolvla_libero` | ベースライン成功率 0.40 |

小規模テスト（5 並列 × 2 エポック）の実測: **10 エピソード、成功 3 本（270、310、490 フレームで終了）、失敗 7 本（520）**、1 エポック約 59 秒。

## 3. 最重要: 3 環境で「同じデータ」ができないようにする

### 3.1 乱数の出どころ（コードで確認した事実）

| 乱数 | 場所 | 現状 |
|---|---|---|
| 環境の seed | `LiberoEnv.__init__`: `self.seed = cfg.seed + seed_offset`、`seed_offset = rank * stage_num + stage_id` | `libero_10.yaml` の `seed: 0`。**3 環境とも rank 0 なら、同じ seed になる**。`env.seed(self.seed * len(env_idx))` でシミュレータにも使われる |
| 初期状態の並び | `_generator_ordered = np.random.default_rng(seed=0)`（**固定の 0**）で 50 個の初期状態をシャッフル、`start_idx = 0` から順に使う | `env.eval.seed` を変えても**変わらない**。3 環境とも同じ順に 50 個の初期状態を回る |
| 方策のノイズ（flow matching の x₁） | lerobot の `sample_noise` が `torch.normal(..., device=cuda)`。smolVLA のアダプタ（`smolvla_action_model.py`）は `predict_action_chunk(batch)` を、**noise を渡さずに**呼ぶ | RLinf には `seed_everything` があるが、**収集・評価の経路では呼ばれていない**。したがって、グローバルな乱数状態が使われる |

### 3.2 リスクの評価

- **多様性の源は、ほぼ方策のノイズだけ**。初期状態は 50 個の固定集合で、環境の物理は決定的なため、同じ初期状態から出発する軌跡の違いは、ノイズで決まる。
- 今の環境（torch 2.11）では、プロセスごとに CUDA の既定 seed が異なる（実測: 2 つのプロセスで `1098542714683744` と `2169250603517372`）。したがって、何もしなくても、ノイズが偶然そろう可能性は低い。
- ただし、**これに頼るのは危険**。(a) 別の環境で torch のバージョンが違う、(b) 誰かが `seed_everything` や環境変数で seed を固定する、(c) 将来の改修で seed が固定される、のいずれでも、**3 環境が同じ初期状態の並びと同じノイズで、完全に同じデータを作る**。環境の seed と初期状態の並びは、すでに 3 環境で同じである。
- 対策は、**明示的に、環境ごとに異なる seed を設定する**こと。かつ、**重複を検出する検証**を必ず入れる。

### 3.3 seed の設計

seed は **(分割, ラウンド, 環境番号 m)** ごとに一意にする。2 つの seed 族（環境用とノイズ用）が重ならないようにする。

| 用途 | 式（`R` = ラウンド番号、`m` = 環境番号 0〜2） |
|---|---|
| train の `env.eval.seed` | `100000 + 100*R + m` |
| eval の `env.eval.seed` | `200000 + m` |
| train のノイズ seed | `600000 + 100*R + m` |
| eval のノイズ seed | `700000 + m` |

- **ノイズの seed を設定できるようにする小さなパッチ（P1）が必要**（§ 6）。
- **実測（2026-10-01、5 並列 × 1 エポック × 4 回、`tmp/seed_test/`）**:
  - 環境 seed 1・ノイズ seed 11 を 2 回 → **完全に一致**（再現性あり。seed が効いている）。
  - **環境 seed だけを 2 に変更** → **完全に一致**。つまり、**環境 seed は、この設定ではデータに影響しない**（初期状態の並びが seed 0 固定のため）。
  - **ノイズ seed だけを 12 に変更** → **全て異なる**。
  - したがって、**データを分けるのはノイズ seed だけ**である。環境 seed は、念のため環境ごとに変えるが、**それだけでは重複を防げない**。3 環境で同じノイズ seed を使うと、完全に同じデータができる。
- 初期状態の並びが同じため、3 環境が、同じ時期に同じ初期状態を訪れる。軌跡の違いはノイズだけなので、§ 5 の検証で、重複がないことを必ず確認する。

## 4. 分担と収集量

| 項目 | 値 |
|---|---|
| train の目標 | 約 4,096 本（前回の train 3,200 + extra 1,081 を置き換える） |
| eval の目標 | 64 本（train とは別の seed 族） |
| 1 環境あたりの収集 | train 約 1,366 本、eval 約 22 本 |
| 1 エポック（5 並列）の収量 | 約 5 エピソード（実測: 2 エポックで 10 本） |
| 1 環境あたりのエポック数 | train 約 275、eval 5 |
| 所要時間（1 環境） | 約 4.5 時間（1 エポック 59 秒 × 275）。3 環境の並列で、全体も約 4.5 時間 |

- 成功で打ち切るため、エピソードの平均長が短くなり、エポックあたりの収量が変わる。**実収量は、スモークテストで測り直す**。
- 3 環境の GPU とメモリは同じ（ユーザー確認済み。A5000 24GB、メモリ上限 25GB）。

## 5. ディスクと保存先

### 5.1 保存先は環境ごとに分ける
- 各環境の `save_dir` は別にする: `$ROOT/collected_v2/<split>_r<R>/m<m>/`。
- 同じ `save_dir` に複数環境が書くと、`rank_0/id_0` が衝突する（どの環境も rank 0 のため）。

### 5.2 ディスクの使用量（クォータは 600GB。2026-10-01 に、ユーザーが 300GB から引き上げを確認）
- 収集は PNG で保存される（約 60MB/エピソード）。**4,096 エピソードで約 240GB**。
- 2026-10-01 時点の `/workspace` の使用量は約 310GB（`RLinf/logs` 82GB、`collected_v2` 96GB、`RLinf` の venv 36GB、`.cache` 29GB、`data` 28GB、`.pixi` 20GB、`models` 18GB など）。収集が終わると、PNG が約 250GB（eval と train）になり、**合計は約 460GB で、600GB に収まる**。
- したがって、**PNG は削除せず、全て残す**（動画への変換は、AV1 の損失圧縮で、元に戻せない。元データを残せるなら残すほうが安全）。変換は、各ラウンドの後に行うが、**変換後の PNG の削除は、必須ではない**。
- ラウンドに分ける理由は、ディスクではなく、**検査と変換を、収集と並行して進める**ためである。
- 使用量が 600GB の 85%（510GB）を超えそうになったら、再度、削除の候補（`RLinf/logs/steam_sft` 62GB など）を、ユーザーに相談する。
- 変換は、**ラウンドごと、環境ごとに、別のデータセット**にする（`libero10_task0_train_v2_r<R>_m<M>`。書き込みが 1 エピソードずつの処理で、環境ごとに分けて 3 並列にすると約 3 倍速い。実測: 1 本のとき 5.5 本/分、3 並列で約 16 本/分）。Step1〜3 は、複数データセットを扱える。最後に、lerobot の `aggregate_datasets` で 1 つにまとめる案（v3.0 変換後）がある。

### 5.2.5 落ちた収集の後始末
- 収集が途中で落ちると、書きかけの parquet が 1 つ残る。`scripts/remove_corrupt_parquets.py` で、**読めないファイルだけを削除**する（全環境で実施。手順書の § 4.5）。収集中の書きかけを消さないよう、更新から 10 分未満は飛ばし、収集のプロセスが終わってから実行する。
- 落ちたラウンドは、保存先を残したまま、`_t2`（seed + 5000）で再実行する。重複検査（`check_collected_duplicates.py`）は、壊れたファイルを警告して飛ばす。

### 5.3 変換
- **変換スクリプトの不具合を修正済み**: 成功で打ち切ると、短いエピソードが、1 つの parquet ファイルに**複数**詰め込まれる（実測: 1 ファイルに 260 と 270 の 2 エピソード）。修正前の `convert_rlinf_collected_to_v21.py` は「1 ファイル = 1 エピソード」と仮定して、**複数のエピソードを 1 つに結合していた**（8 エピソードが 6 になり、return も誤る）。`episode_index` で分けるように直し、`m0` で、8 エピソード・長さ一致を確認した。
- この環境（コンテナ）には `pixi` コマンドがない場合がある。その場合は、`.pixi/envs/lerobot-v21/bin/python scripts/convert_rlinf_collected_to_v21.py ...` を直接呼ぶ。
- `convert_rlinf_collected_to_v21.py` は、`<src>/rank_*/id_*` の形式を読む。3 環境の出力を、**ランク番号をずらしたシンボリックリンクで 1 つの `--src` にまとめる**（`rank_0`＝環境 0、`rank_1`＝環境 1、`rank_2`＝環境 2）。
- 変換は、`pixi run -e lerobot-v21 python scripts/convert_rlinf_collected_to_v21.py --src <まとめた dir> --dst <データセット> --max-episodes N`。

## 6. 実施前に必要な準備

### P1: ノイズ seed のパッチ（**実装済み**: `patches/rlinf/smolvla-t-noise-seed.patch`）
- `smolvla_action_model.py` の `predict_action_chunk(batch)` の呼び出しで、`noise=` を渡す。
- ノイズは、**`torch.Generator`（device は CUDA、seed は設定値 + rank）から生成**し、呼び出しをまたいで状態を引き継ぐ（同じノイズが繰り返されない）。
- 設定キー: `rollout.model.smolvla.noise_seed`（`null` なら現行の挙動）。
- ノイズの形は `(B, chunk_size, max_action_dim)` = `(B, 50, 32)`。`predict_action_chunk(batch, noise=...)` がこの引数を受け取ることは、lerobot のコードで確認済み。

### P2: 重複検出スクリプト（**実装済み**: `scripts/check_collected_duplicates.py`）
- 各エピソードの行動列全体（float32 のバイト列）と、最初の 3 チャンク（30 ステップ）のハッシュを計算する。
- 環境間、ラウンド間、環境内で、**同一ハッシュが 1 つもない**ことを確認する。
- 前回のデータ（`train`、`train_extra`）とも、重複しないことを確認する。

## 6.5 同じ Python 環境（`.venv-smolvla`）を 3 環境で使う場合の確認

`/workspace` は共有だが、`/root`、`/tmp`、システムのライブラリ（apt）、GPU は**環境ごと**に別。venv 本体（パッケージ）は `/workspace` にあるので共有できるが、**venv の外にある依存**が、各環境で揃っている必要がある。

### 共有されるもの（全環境で同一）
`RLinf/.venv-smolvla` のパッケージ、`RLinf` のソースとパッチ、`models/`、`data/`、`HF_HOME`（この環境では `/workspace/.cache/huggingface/` に設定されている）。venv の `activate` と、スクリプトの shebang のパスも `/workspace` 内で、環境間で同じ。

### 環境ごとに用意が必要なもの（今日、この環境で、それぞれ別のエラーになった）
| 依存 | 場所 | 欠けたときの症状 | 対処 |
|---|---|---|---|
| venv の Python 本体 | `.venv-smolvla/bin/python` が `/root/.local/share/uv/python/...` へのリンク | `python` が起動しない（`No module named hydra` など、別の Python が使われる） | `uv python install 3.12.11` |
| LIBERO の設定 | `~/.libero/config.yaml` | `import libero` が対話入力を待ち、`EOFError` で収集が落ちる | `echo n \| .venv-smolvla/bin/python -c 'import libero.libero'` |
| EGL ライブラリ | システムの `libEGL.so.1`、`libOpenGL.so.0` | シミュレータの描画の初期化で `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | `libegl1`、`libopengl0` を apt で導入（システムの変更なので、導入の可否は事前に確認する） |
| メモリ上限 | cgroup（25GB） | 5 並列でピーク約 22GB。上限に近づくと Ray が worker を落とす | 環境が同じなので問題ない想定。`total_num_envs=5` |

これらは、**各環境で事前チェックスクリプトを実行して確認**する: `bash scripts/preflight_collect_env.sh`。読み取りのみで、何もインストールしない。確認する項目は、Python 本体、パッケージの import と CUDA、LIBERO の設定、EGL、HF の参照先、メモリ上限、残った Ray、RLinf のツリー、`/workspace` の書き込みである。この環境で、通常系と異常系（`~/.libero` がない場合）の両方で動作を確認した。

### 共有ファイルシステム上で、同時に動かすときの注意（コードで確認した事実）
| 項目 | 内容 | 対策 |
|---|---|---|
| ログのディレクトリ | `run_eval.sh` が `RLinf/logs/$(date +'%Y%m%d-%H:%M:%S')-<config>` を作る。**秒単位**なので、3 環境を同じ秒に起動すると、**同じログディレクトリに混ざる** | 環境ごとに起動を 10 秒以上ずらす。標準出力は、環境ごとの別ファイルにリダイレクトする（`tmp/collect_m${M}.log`） |
| 出力先 | 全環境が rank 0 のため、同じ `save_dir` だと `rank_0/id_0` が衝突する | 環境ごとに別の `save_dir`（§ 5.1） |
| Hub へのアクセス | SmolVLM2 の設定とトークナイザを、`vlm_model_name: null` だと Hub から取得する（収集のログに Hub への要求が出ていた） | `rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct` を指定し、`HF_HUB_OFFLINE=1` にする。`models/` に実体がある |
| Ray | 各環境が、自分のローカルな Ray を起動する。起動時に `address="auto"` で既存のクラスタを探すが、`/tmp` が環境ごとなので、**他環境の Ray には接続しない**。ただし、**同じ環境に残った Ray**には接続しうる | 起動前に `ray stop --force`（自分のものだけ）。`RAY_ADDRESS`、`RLINF_NODE_RANK` は未設定にする |
| Hydra | `hydra.run.dir: .`、`output_subdir: null` で、出力ファイルを作らない | 問題なし |
| RLinf のツリー | 共有なので、パッチは 1 回適用すれば全環境に反映される | 事前チェックが、ツリーの指紋（`git diff` のハッシュ）を表示するので、**3 環境で一致することを確認** |
| `__pycache__` | 複数環境が同じ `.venv` の `__pycache__` に書く可能性がある | 内容が同じなので、実害は小さいと考える（未検証） |

**未検証**: 3 つの環境が、実際に同時に収集を動かしたときの挙動は、ここ（1 環境）からは確認できない。§ 7 のスモークテスト（3 環境で同時に実行）が、その検証になる。

## 7. 実施手順

0. **各環境で事前チェック**: `bash scripts/preflight_collect_env.sh`。`PASSED` になるまで直す（§ 6.5）。ツリーの指紋が 3 環境で一致することも確認する。
1. ~~P1、P2 を実装~~（済み。`patches/apply_rlinf_patches.sh` も、積み重ねたパッチの再実行で止まる不具合を直した）（`patches/apply_rlinf_patches.sh` で、3 環境の RLinf に同じパッチを適用する。`/workspace` が共有なので、1 回の適用で全環境に反映される想定。ただし、`.venv-smolvla` の整合は、各環境で確認する）。
2. **スモークテスト**（3 環境で同時に、各 2 エポック、eval の seed 族で）。確認すること:
   - 各環境が、メモリ上限内で完走する。
   - エピソードの長さ（成功で短くなる）、成功率が、前回の小規模テストと同程度。
   - **3 環境の出力で、行動列のハッシュが全て異なる**（P2）。同じ設定で seed だけが違うことを、ここで検証する。
   - seed を**同一にした**場合に、ハッシュが一致することも 1 回確かめる（seed が効いていることの確認）。
3. **eval を収集**（seed 族 eval、各環境約 22 本）。変換して 64 本に切り詰める。
4. **train をラウンドごとに収集**（seed 族 train、各環境 60 エポック × 約 5 ラウンド）。各ラウンドの終わりに、重複検査 → 変換 → PNG の削除。
5. 全ラウンド後に、データセット全体で、成功率、エピソード長の分布、重複の有無を最終確認する。
6. **旧データ（`train`、`train_extra`、`train_clean`、`eval`）は、新データの検証が終わるまで削除しない。**

### 起動コマンドの雛形（環境 `M` = 0, 1, 2、ラウンド `R`）

```bash
cd $ROOT/RLinf && source .venv-smolvla/bin/activate
export HF_HUB_OFFLINE=1
M=0; R=1                                  # 環境ごとに M を変える
sleep $((M * 10))                         # ログのディレクトリ名（秒単位）の衝突を避ける
bash evaluations/run_eval.sh libero libero_10_smolvla_collect \
  rollout.model.model_path=$ROOT/models/smolvla_libero \
  rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
  env.eval.data_collection.save_dir=$ROOT/collected_v2/train_r${R}/m${M} \
  env.eval.rollout_epoch=60 env.eval.total_num_envs=5 \
  env.eval.ignore_terminations=False env.eval.data_collection.fps=10 \
  env.eval.seed=$((100000 + 100*R + M)) \
  rollout.model.smolvla.noise_seed=$((600000 + 100*R + M)) \   # P1 の実装後
  > $ROOT/tmp/collect_r${R}_m${M}.log 2>&1
```

## 8. 収集後の工程（別途）

- Step1（return 計算）は約 16 秒で終わる。
- Step2（価値モデル）は約 25 時間。複数環境での分担は、ネットワークで通信できないため不可（FedAvg 型は未検証）。1 環境で実行する。
- Step3（advantage）は約 16 時間。Phase 1 の結果を保存する仕組み（再開機能）を土台に、データセット単位で 3 環境に分担し、最後に、全データ共通の閾値を計算する結合処理を作る案がある（未実装）。
- sft（人のデモ）の fps（20）を、ロールアウト（10）に合わせるかは未決定。

## 9. 未決定・要確認

- train の本数（4,096 本でよいか）、ラウンド数。
- `max_episode_steps` を 480（公式）にするか、520（前回）のままにするか。
- P1 の seed の扱い（`rank` の足し方を含む）の最終確認。
- Task 0 以外のデモ（sft 27 本）を、Step2、Step4 の学習に含めるか。

## 10. 参考（確認した事実の出典）

- 環境の seed: `RLinf/rlinf/envs/sim/libero/libero_env.py`（`self.seed = self.cfg.seed + seed_offset`、`_generator_ordered = np.random.default_rng(seed=0)`、`start_idx = 0`）
- ノイズ: `lerobot/policies/common/flow_matching.py` の `sample_noise`、`SmolVLAPolicy.predict_action_chunk(batch, noise=None)`
- アダプタ: `RLinf/rlinf/models/embodiment/smolvla/smolvla_action_model.py`（`predict_action_chunk(batch)`）
- 収集設定: `RLinf/evaluations/libero/libero_10_smolvla_collect.yaml`
- 小規模テスト（成功で打ち切り）: `tmp/collect_new_test/`（git 管理外）

## 収集の既知の不具合: 初期状態の並びの境界で出る、10 フレームの偽エピソード

- **症状**: ラウンド 2 の 2 つのシャード（`train_r2/m1/.../id_0` と `train_r2/m2/.../id_3`）に、**10 フレームで、失敗扱いのエピソード**が各 1 本あった（ラウンド 1 の 1,091 本、eval の 89 本、smoke には、なし）。1 本は、手先の位置が 10 ステップ間まったく動いていなかった。どちらも、シャード内の 49 番目で、**50 個の初期状態を 1 周して先頭に戻る境界**に当たる。
- **原因（推定）**: 並びの末尾をまたぐリセットの要求で、足りない環境にリセット ID `-1` が返り、**その環境はリセットされない**。終了済みのエピソードの上で、1 回の行動チャンクだけが実行され、別の「失敗エピソード」として記録される。コード（`libero_env.py` の `_get_ordered_reset_state_ids`、`valid_mask`）の読みと、データの状況（境界の位置、短さ）からの推定で、**実験では再現していない**。
- **頻度**: ラウンド 2 の 740 本のうち 2 本（0.27%）。ラウンド 1 の約 22 周の境界では 0 本、ラウンド 2 の約 15 周では 2 本。
- **対処**: データは削除せず、**変換のときに除外する**（`convert_rlinf_collected_to_v21.py --min-frames 100`、既定 100。実際のエピソードは約 250 フレーム以上）。除外したエピソードは、変換のログ（`[drop]`）に出る。検証スクリプト（`verify_converted_dataset.py`）も、同じ `--min-frames` に合わせてある。
- **注意**: 同じ現象が、**成功のエピソードの直後**で起きた場合に、そのエピソードが誤って「失敗」の 10 フレームとして残る形もありうる。完結した成功エピソード自体は、無傷（検査で、`done` と成否が整合）。

### 方針: RLinf 側の修正は行わない（2026-10-02、ユーザー判断）

- 偽エピソードは、頻度が低い（ラウンド 2 の 740 本のうち 2 本、0.27%）うえ、**変換時に `--min-frames 100` で除外できる**ため、RLinf のコードは修正しない。
- 一度、`LiberoEnv._get_ordered_reset_state_ids` を循環して配る形に直すパッチ（`smolvla-eval-reset-wrap.patch`）を作ったが、**共有ツリーには適用せず、取り消した**（commit `63bbfb9` に残っている。必要なら、そこから復元できる）。
- 原因の再現（関数単体、`scripts/test_libero_reset_ids.py`）: 1〜2 個の環境が同時にリセットされる状況の 27,962 回の要求で、現行コードは 0.583%（163 件）が `-1` を返す。これが、境界で偽エピソードが出る推定原因である。環境全体を動かした再現実験は、していない。
- 以後のラウンドでも、検査（`verify_collected_integrity.py`）で、100 フレーム未満のエピソードを確認し、変換時に除外する。

### RLinf の変更は、全てパッチに収まっている（2026-10-02 に検証）

- 検証方法: RLinf を、パッチの基準 commit（`d1bf37c`）にチェックアウトし直して、`patches/rlinf/*.patch` を順に適用したものと、実際の共有ツリーを、パッチが触る 18 ファイルで比較した。
- 結果: **1 件の漏れ**があった。`rlinf/data/datasets/recap/smolvla_cfg.py`（`smolvla-recap.patch` で新規作成される、Git の未追跡ファイル）への変更（advantage ラベルが欠けたフレームを、エラーにせず、学習から除外する処理）が、パッチに入っていなかった。未追跡ファイルの変更は、`git diff` では拾われないため。**`smolvla-s-recap-dataset-skip-unlabeled.patch` として切り出し**、再検証で、**18 ファイル全てが一致**（差分 0）。
- **パッチを作るときの注意**: パッチで新規作成されたファイル（`rlinf/models/embodiment/smolvla/`、`smolvla_cfg.py`、`smolvla.yaml` など）を、その後に変更するときは、`git diff` ではなく、**変更前のコピーと変更後のコピーを `diff -u` で比較**して、パッチにする（新規ファイルの変更は、パッチの順序で、作成するパッチより後に適用されるように、名前を付ける）。
- RLinf のうち、パッチ以外で残っているのは、インストールのスクリプトが作った `pyproject.toml.rlinf-torch-bak.*`（未追跡の退避ファイル 3 つ。私たちの変更ではない）のみ。

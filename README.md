# libero_RL

## RECAP（LIBERO-10 Task 0）の再現実験

RLinf のチュートリアル [RECAP](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/recap.html) を再現する手順。
few-shot の π₀.₅ が集めたデータ（成功率 48.8%）から、オフラインで方策を改善する。ドキュメントでは成功率が 66.5% になると報告されている。

```
Step 1 return 計算 → Step 2 価値モデル学習 → Step 3 アドバンテージ計算 → Step 4 CFG 学習 → 評価
```

[RLinf](https://github.com/RLinf/RLinf) は submodule として `RLinf/` に置いてある。

### 0. 前提

- Linux + NVIDIA GPU（CUDA）。評価では MuJoCo を EGL で描画する。
- ディスク: データセットに約 87GB。

```bash
git submodule update --init --recursive
```

以降は、このリポジトリのルートで次の変数を設定してから作業する。

```bash
export ROOT=$(pwd)
export DATA=$ROOT/data/RECAP-Libero10-Task0-48succ-Data
export MODELS=$ROOT/models
export SFT=$DATA/libero10_task0_sft
export ROLLOUT=$DATA/libero10_task0_train
export EVAL=$DATA/libero10_task0_eval
```

### 1. 環境構築

`RLinf/` で、次のどちらかを使う。

**Docker（推奨）**

```bash
cd $ROOT
docker run -it --rm --gpus all --shm-size 20g --network host --name rlinf \
  -v $ROOT:$ROOT -w $ROOT/RLinf \
  rlinf/rlinf:agentic-rlinf0.4-maniskill_libero
source switch_env openpi
```

コンテナ内でも、「0. 前提」の環境変数を設定し直す。

**UV**

```bash
cd $ROOT/RLinf
bash requirements/install.sh embodied --model openpi --env maniskill_libero
source .venv/bin/activate
```

### 2. データセット

[RLinf/RECAP-Libero10-Task0-48succ-Data](https://huggingface.co/datasets/RLinf/RECAP-Libero10-Task0-48succ-Data)（LeRobot v2 形式）を取得する。
Step 1 と Step 3 はデータセットの `meta/` に結果を書き込むので、HF のキャッシュではなく `--local-dir` で実体を置く。

```bash
hf download RLinf/RECAP-Libero10-Task0-48succ-Data --repo-type dataset --local-dir $DATA
```

| サブセット | 種類 | 中身 | 使う Step |
|---|---|---|---|
| `libero10_task0_sft` | `sft` | 人間デモ（LIBERO-10 の 10 タスク × 3 本、すべて成功）、20fps | 1〜4 |
| `libero10_task0_train` | `rollout` | few-shot π₀.₅ が Task 0 で集めた 4,096 本（成功率 48.8%）、10fps | 1〜4 |
| `libero10_task0_eval` | `rollout` | 同じ方策で別に集めた 64 本（成功率 42.2%） | 1, 2（価値モデルの過学習を監視） |

Task 0 は "put both the alphabet soup and the tomato sauce in the basket"（LIBERO 公式のスイート順で 0 番）。

### 3. モデル

| モデル | 用途 |
|---|---|
| SigLIP2-so400m | Step 2・3 の価値モデル（画像エンコーダ） |
| Gemma3-270M | Step 2・3 の価値モデル（言語モデル、トークナイザ） |
| π₀.₅ base（PyTorch 形式） | Step 4 の方策 |

```bash
hf download google/siglip2-so400m-patch14-224 --local-dir $MODELS/siglip2-so400m-patch14-224
hf download google/gemma-3-270m --local-dir $MODELS/gemma-3-270m   # ライセンスへの同意と hf auth login が必要
```

**π₀.₅ base を PyTorch 形式に変換する**

[openpi](https://github.com/Physical-Intelligence/openpi) の変換スクリプトを使う。JAX 版の重みは `gs://openpi-assets/checkpoints/pi05_base` から自動で取得される。

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git $MODELS/openpi
cd $MODELS/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv run examples/convert_jax_model_to_pytorch.py \
  --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
  --config_name pi05_libero \
  --output_path $MODELS/pi05_base_pytorch
```

RLinf は norm stats を `<model_path>/physical-intelligence/libero/norm_stats.json` から読むので、openpi 公開の LIBERO 用 norm stats を置く。

```bash
mkdir -p $MODELS/pi05_base_pytorch/physical-intelligence/libero
curl -fL https://storage.googleapis.com/openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json \
  -o $MODELS/pi05_base_pytorch/physical-intelligence/libero/norm_stats.json
```

### 4. パイプライン

以下はすべて `$ROOT/RLinf` で実行する。
設定ファイル（`examples/offline_rl/config/*.yaml`）のパスは、submodule を編集せずに済むよう Hydra のコマンドライン引数で上書きする。

ステップ間はタグで結果を受け渡す。

- Step 1 の `data.tag`、Step 2 の `data.tag`、Step 3 の `advantage.returns_tag` → `fail300`
- Step 3 の `advantage.tag`、Step 4 の `data.advantage_tag` → `fail300_N10_q30`

```bash
cd $ROOT/RLinf
export RETURNS_TAG=fail300
export ADV_TAG=fail300_N10_q30
```

#### Step 1: return の計算

割引なし（γ=1）で、1 ステップごとに報酬 −1、失敗した終端に −300 を与えて return を計算する。
Step 2 の評価でも return を使うので、eval データにも計算しておく。
結果は各データセットの `meta/returns_fail300.parquet` に書かれる。

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh recap_compute_returns \
  "data.train_data_paths=[{dataset_path:$SFT,type:sft},{dataset_path:$ROLLOUT,type:rollout},{dataset_path:$EVAL,type:rollout}]" \
  data.gamma=1.0 data.failure_reward=-300.0 data.tag=$RETURNS_TAG
```

確認:

```bash
python -c "import json; print(json.load(open('$ROLLOUT/meta/stats.json'))['return'])"
```

#### Step 2: 価値モデルの学習

SigLIP2 + Gemma3 + Critic Expert からなる価値モデルに、正規化した return（[-1, 0]、201 ビン）を予測させる。

```bash
bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh recap_value_model_sft \
  "data.train_data_paths=[{dataset_path:$SFT,type:sft,weight:1.0,robot_type:libero,model_type:pi05},{dataset_path:$ROLLOUT,type:rollout,weight:1.0,robot_type:libero,model_type:pi05}]" \
  "data.eval_data_paths=[{dataset_path:$EVAL,max_samples:10000,robot_type:libero,model_type:pi05}]" \
  data.tag=$RETURNS_TAG \
  actor.model.siglip_path=$MODELS/siglip2-so400m-patch14-224 \
  actor.model.gemma3_path=$MODELS/gemma-3-270m \
  actor.model.tokenizer_path=$MODELS/gemma-3-270m
```

- 見るべき指標は TensorBoard の `eval/spearman_correlation`（予測と実際の return の順位相関）。
- チェックポイントは `logs/value_sft/recap_value_model_sft-<timestamp>/value_sft/checkpoints/global_step_<N>/actor/model_state_dict` に保存される。既定では 8,000 ステップ学習し、3,000 ステップごとに保存する。

```bash
export VALUE_CKPT=$ROOT/RLinf/logs/value_sft/recap_value_model_sft-<timestamp>/value_sft/checkpoints/global_step_<N>/actor/model_state_dict
```

#### Step 3: アドバンテージの計算

10 ステップ先読みのアドバンテージ `A_t = normalize(r_{t:t+10}) + V(o_{t+10}) − V(o_t)` を計算し、上位 30% を正（`advantage=True`）とラベル付けする。
結果は各データセットの `meta/advantages_fail300_N10_q30.parquet` に書かれる。GPU が複数あれば torchrun で並列に推論する。

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh recap_compute_advantages \
  "data.train_data_paths=[{dataset_path:$SFT,robot_type:libero,type:sft,weight:1.0},{dataset_path:$ROLLOUT,robot_type:libero,type:rollout,weight:1.0}]" \
  data.advantage_lookahead_step=10 data.gamma=1.0 \
  advantage.value_checkpoint=$VALUE_CKPT \
  advantage.positive_quantile=0.3 \
  advantage.returns_tag=$RETURNS_TAG advantage.tag=$ADV_TAG \
  advantage.model.siglip_path=$MODELS/siglip2-so400m-patch14-224 \
  advantage.model.gemma3_path=$MODELS/gemma-3-270m \
  advantage.model.tokenizer_path=$MODELS/gemma-3-270m
```

GPU 数は `--nproc N` で指定する（既定は `nvidia-smi` で見えている数）。

#### Step 4: CFG 学習

π₀.₅ base を、アドバンテージのラベルを条件にした classifier-free guidance で学習する。

```bash
bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh cfg_rl_openpi \
  "data.train_data_paths=[{dataset_path:$SFT,type:sft,weight:1.0},{dataset_path:$ROLLOUT,type:rollout,weight:1.0}]" \
  data.advantage_tag=$ADV_TAG \
  actor.model.model_path=$MODELS/pi05_base_pytorch
```

主な設定は `positive_only_conditional: true`、`unconditional_prob: 0.1`、`cfgrl_guidance_scale: 1.0`、lr 1e-5（cosine）、30,000 ステップ、global batch 512。
チェックポイントは `logs/cfg_rl/cfg_rl_openpi-<timestamp>/cfg_sft/checkpoints/global_step_<N>/` に 3,000 ステップごとに保存される。

### 5. 評価（LIBERO シミュレータ）

学習した方策を LIBERO-10 の Task 0 で、固定の初期状態 50 個から 1 回ずつ動かし、成功率 `eval/success_once` を測る。

> RLinf には RECAP 専用の評価手順がない。以下は、汎用の評価設定（`evaluations/libero/libero_10_openpi_pi05_eval.yaml`）で CFG モデル（`model_type: cfg_model`）を読み込むように組み立てたもので、まだ動かして確認していない。

CFG モデルの読み込み処理（`rlinf/models/embodiment/openpi_cfg`）は、`model_path` 配下の `actor/model_state_dict/full_weights.pt` と `physical-intelligence/libero/norm_stats.json` を読む。そのため、先に norm stats をチェックポイントにコピーしておく。

```bash
export POLICY_CKPT=$ROOT/RLinf/logs/cfg_rl/cfg_rl_openpi-<timestamp>/cfg_sft/checkpoints/global_step_<N>
mkdir -p $POLICY_CKPT/physical-intelligence/libero
cp $MODELS/pi05_base_pytorch/physical-intelligence/libero/norm_stats.json $POLICY_CKPT/physical-intelligence/libero/

bash evaluations/run_eval.sh libero libero_10_openpi_pi05_eval \
  rollout.model.model_type=cfg_model \
  rollout.model.model_path=$POLICY_CKPT \
  +rollout.model.openpi.cfgrl_guidance_scale=1.0 \
  +env.eval.task_id_filter=[0] \
  env.eval.total_num_envs=50
```

- `task_id_filter=[0]` を指定すると、Task 0 の初期状態 50 個を 1 周したところで評価が終わる。`total_num_envs=50` で、その 50 個を並列に回す。
- 結果はターミナルに出る `eval/success_once`（目標は 66.5%）で確認する。ログは `logs/<timestamp>-libero_10_openpi_pi05_eval/eval_embodiment.log`、動画はその配下の `video/eval/` に保存される。
- GPU メモリが足りない場合は `env.eval.total_num_envs` を減らし、`max_steps_per_rollout_epoch` を `max_episode_steps`（520）の N 倍にして、50 個すべてを回るようにする。`max_steps_per_rollout_epoch` は 5 の倍数にする。
- EGL が使えないホストでは `export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa` を設定する。

ベースラインの 48.8% は `libero10_task0_train` の成功率（1,999/4,096）と一致しており、few-shot π₀.₅ のチェックポイントは公開されていない。

## RECAP を smolVLA（lerobot/smolvla_libero_plus）で行う

π₀.₅ 版と同じ 4 ステップの構成を、方策・価値モデル・学習データのすべてを smolVLA [lerobot/smolvla_libero_plus](https://huggingface.co/lerobot/smolvla_libero_plus) で行う。RECAP のアドバンテージは「データを集めた方策自身の、ふだんの行動からの改善度」なので、Step 1〜3 で使う rollout（train・eval とも）は smolVLA 自身で集め直す。π₀.₅ 版の `libero10_task0_train` / `libero10_task0_eval`（few-shot π₀.₅ が集めたデータ）はそのままでは使えない。学習と評価は RLinf で実行する。

```
ベースライン評価 → rollout 収集（train・eval）→ Step 1 return 計算 → Step 2 価値モデル学習
  → Step 3 アドバンテージ計算 → Step 4 アドバンテージ条件付き SFT → 評価
```

RLinf は SmolVLA に対応していないので、`patches/rlinf/smolvla-recap.patch` で RLinf に SmolVLA のサポートを追加する。パッチは RLinf のコミット `d1bf37cd` に対して作成した。

| パッチで追加・変更するもの | 内容 |
|---|---|
| `rlinf/models/embodiment/smolvla/` | LeRobot の `SmolVLAPolicy` を包むアダプタ（評価の推論と SFT の損失） |
| `rlinf/data/datasets/recap/smolvla_cfg.py` | アドバンテージ条件付き SFT のデータローダ |
| `rlinf/config.py` ほか 3 ファイル | モデル種別 `smolvla` の登録、SFT ワーカーと rollout ワーカーへの分岐追加 |
| `examples/embodiment/config/model/smolvla.yaml` | モデル設定 |
| `examples/sft/config/libero10_task0_recap_smolvla.yaml` | Step 4 の学習設定 |
| `evaluations/libero/libero_10_smolvla_eval.yaml` | LIBERO-10 の評価設定 |
| `evaluations/libero/libero_10_smolvla_collect.yaml` | Task 0 の rollout 収集設定 |
| `rlinf/data/storage/lerobot/writer.py` | 収集データの書き出しが LeRobot 0.4 以降（v3.0）でも完結するよう修正（3 行差分） |

**CFG 学習を SFT で置き換える理由**: RLinf の CFG モデルは、正のアドバンテージのサンプルを `"{task}\nAdvantage: positive"` というプロンプトで条件付けし（確率 0.1 で条件を外す）、負のサンプルは常に `"{task}"` で学習する。推論時の `cfgrl_guidance_scale=1.0` では条件付きの予測だけを使う。このため、同じプロンプト規則でタスク文を書き換えた SFT と、`Advantage: positive` を付けたプロンプトでの推論で同じことができる。条件を外す確率 0.1 は、RLinf と同じくサンプルを読むたびに引き直す。

### 0. 前提

- π₀.₅ 版の「0. 前提」を済ませ、環境変数 `ROOT` を設定しておく（データセットは自分で収集するので、π₀.₅ 版の「2. データセット」は不要）。
- [pixi](https://pixi.sh)（データ変換に使う）。
- ディスク: sft（人間デモ）だけ π₀.₅ 版のデータセットから取得する。rollout（train 約 4,096 本、eval 約 64 本）は収集したぶんだけ増える。目安として、libero10_task0_train は 1.4GB/64 episodes なので、4,096 episodes で約 90GB。

```bash
export DATA=$ROOT/data/smolvla_recap
export MODELS=$ROOT/models
export SFT=$DATA/libero10_task0_sft
export ROLLOUT=$DATA/libero10_task0_train
export EVAL=$DATA/libero10_task0_eval
hf download RLinf/RECAP-Libero10-Task0-48succ-Data --repo-type dataset \
  --include "libero10_task0_sft/*" --local-dir $DATA
```

### 1. パッチの適用

```bash
cd $ROOT
bash patches/apply_rlinf_patches.sh            # 適用（適用済みならスキップ）
bash patches/apply_rlinf_patches.sh --revert   # 元に戻す
```

### 2. 環境構築

用途ごとに 3 つの環境を使う。

| 環境 | 用途 | 作り方 |
|---|---|---|
| RLinf openpi 環境 | Step 1〜3（価値モデル） | π₀.₅ 版の「1. 環境構築」と同じ |
| RLinf SmolVLA 環境 | Step 4 の学習と評価 | 下記 |
| pixi `lerobot-v21` / `lerobot` | データの v3.0 変換 | `pixi install -e lerobot-v21 && pixi install -e lerobot` |

RLinf SmolVLA 環境は、LeRobot 0.6 系と LIBERO が入る pi0_fast 用の環境に、SmolVLA の追加依存を入れて作る。

```bash
cd $ROOT/RLinf
bash requirements/install.sh embodied --model pi0_fast --env libero --venv .venv-smolvla
source .venv-smolvla/bin/activate
uv pip install 'num2words>=0.5.14,<0.6.0'
```

### 3. モデル

```bash
hf download lerobot/smolvla_libero_plus --local-dir $MODELS/smolvla_libero_plus
```

SmolVLA は初回に `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` の設定とトークナイザを Hub から取得する。オフラインで使う場合は取得しておき、`rollout.model.smolvla.vlm_model_name`（学習では `actor.model.smolvla.vlm_model_name`）にそのパスを指定する。

### 4. ベースライン評価

学習前の smolvla_libero_plus を、LIBERO-10 の Task 0 で評価する（固定の初期状態 50 個から 1 回ずつ）。この成功率が、Step 4 の学習後と比べるベースラインになる（π₀.₅ 版の 48.8% にあたる）。

```bash
cd $ROOT/RLinf
source .venv-smolvla/bin/activate
bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
  rollout.model.model_path=$MODELS/smolvla_libero_plus \
  +env.eval.task_id_filter=[0] \
  env.eval.total_num_envs=50
```

- 1 回の推論で得た 50 ステップの行動のうち、先頭の `num_action_chunks`（既定 10）ステップを実行する。`env.eval.max_steps_per_rollout_epoch`（520）はこの値で割り切れる必要がある。
- 結果はターミナルの `eval/success_once` で確認する。ログは `logs/<timestamp>-libero_10_smolvla_eval/`、動画はその下の `video/eval/` に保存される。
- 成功率が極端に高い・低いと、アドバンテージによる成功/失敗の差がほとんど出ない。その場合は smolvla_libero_plus の代わりに、少数のデモで学習させた弱い方策を使うことを検討する。

### 5. rollout の収集

smolvla_libero_plus 自身に Task 0 を動かさせ、成功と失敗が混ざった rollout を集める。RLinf の `CollectEpisode` ラッパー（`env.eval.data_collection`）を、`libero_10_smolvla_eval.yaml` と同じ評価の仕組みの上で使う設定 `libero_10_smolvla_collect.yaml` をパッチに含めている。

固定の初期状態 50 個を、`rollout_epoch` の回数だけ繰り返し回る。SmolVLA の行動生成は毎回ノイズから始まるので、同じ初期状態でも試行のたびに違う軌跡になり、成功と失敗の両方が集まる。

```bash
cd $ROOT/RLinf
source .venv-smolvla/bin/activate

# train 用（目安 4,096 episodes）: 82 周（82 * 50 = 4,100）
bash evaluations/run_eval.sh libero libero_10_smolvla_collect \
  rollout.model.model_path=$MODELS/smolvla_libero_plus \
  env.eval.data_collection.save_dir=$ROOT/collected/libero10_task0_train \
  env.eval.rollout_epoch=82

# eval 用（目安 64 episodes）: 2 周（2 * 50 = 100）
bash evaluations/run_eval.sh libero libero_10_smolvla_collect \
  rollout.model.model_path=$MODELS/smolvla_libero_plus \
  env.eval.data_collection.save_dir=$ROOT/collected/libero10_task0_eval \
  env.eval.rollout_epoch=2
```

集めたエピソードは、env worker のランクごとに `<save_dir>/rank_<r>/id_0/` へ LeRobot v3.0 形式（画像は PNG）で書かれる。これを LeRobot v2.1 形式（動画）の 1 つのデータセットに統合し、目安の件数に切り詰める。

```bash
cd $ROOT
pixi run -e lerobot-v21 python scripts/convert_rlinf_collected_to_v21.py \
  --src $ROOT/collected/libero10_task0_train --dst $ROLLOUT --max-episodes 4096
pixi run -e lerobot-v21 python scripts/convert_rlinf_collected_to_v21.py \
  --src $ROOT/collected/libero10_task0_eval --dst $EVAL --max-episodes 64
```

`$ROLLOUT`（`$DATA/libero10_task0_train`）、`$EVAL`（`$DATA/libero10_task0_eval`）は、π₀.₅ 版の `libero10_task0_train` / `libero10_task0_eval` と同じ列（`image`、`wrist_image`、`state`、`actions`、`is_success`、`done`）を持つ。

### 6. Step 1〜3（価値モデルとアドバンテージ）

π₀.₅ 版の「4. パイプライン」の Step 1〜3 を、上で集めた `$SFT`・`$ROLLOUT`・`$EVAL` に対して実行する（RLinf openpi 環境）。

```bash
export ADV_TAG=fail300_N10_q30
```

Step 3 で各データセットの `meta/advantages_fail300_N10_q30.parquet` ができる。

### 7. データを LeRobot v3.0 に変換

Step 4 のデータローダは LeRobot 0.6 系（v3.0 形式）で読むので、sft と rollout（train）の v3.0 コピーを作る。元のデータ（Step 1〜3 で使う v2 形式）は変更しない。

```bash
cd $ROOT
bash scripts/convert_rlinf_to_lerobot_v30.sh $SFT $DATA/v30/libero10_task0_sft
bash scripts/convert_rlinf_to_lerobot_v30.sh $ROLLOUT $DATA/v30/libero10_task0_train
```

`scripts/convert_rlinf_to_lerobot_v30.sh` は次の順に処理する。

1. 元データをコピーする（`data/` と `videos/` はハードリンク、`meta/` は実体をコピー）。
2. v2.0 の場合（π₀.₅ 版のデータセットや、上で収集・統合した rollout が該当）は、LeRobot 0.3.3 の関数で v2.1 にする（`scripts/convert_rlinf_v20_to_v21.py`）。v2.0 から v2.1 への変換ツールは現在の LeRobot に含まれておらず、LeRobot 0.3.3 のものは Hub へ push する前提なので、同じ処理をローカル向けにしたもの。RLinf のデータにある文字列の `prompt` 列は統計の対象から外す。
3. LeRobot の変換ツール `lerobot.scripts.convert_dataset_v21_to_v30` で v3.0 にする。

エピソード番号とフレーム番号は変わらないので、元データの `meta/advantages_*.parquet` をそのまま使える。

### 8. Step 4: アドバンテージ条件付き SFT

`RLinf/examples/sft/run_vla_sft.sh` は Hydra の上書き引数を受け付けないので、同じ処理で引数を渡せる `scripts/run_rlinf_vla_sft.sh` を使う。

```bash
cd $ROOT
source RLinf/.venv-smolvla/bin/activate
bash scripts/run_rlinf_vla_sft.sh libero10_task0_recap_smolvla \
  "data.train_data_paths=[{dataset_path:$DATA/v30/libero10_task0_sft,advantages_path:$SFT/meta/advantages_$ADV_TAG.parquet,weight:1.0},{dataset_path:$DATA/v30/libero10_task0_train,advantages_path:$ROLLOUT/meta/advantages_$ADV_TAG.parquet,weight:1.0}]" \
  actor.model.model_path=$MODELS/smolvla_libero_plus
```

- 主な設定（`libero10_task0_recap_smolvla.yaml`）: 30,000 ステップ、global batch 256（micro batch 32）、lr 1e-5（cosine、warmup 1,000）、float32、`unconditional_prob: 0.1`、`balance_dataset_weights: true`（sft と rollout を件数によらず 1:1 で混ぜる。RLinf の CFG 学習と同じ）。
- sft（20fps）と rollout（10fps）は fps が違うが、どちらも 1 行が 1 制御ステップなので、行動チャンクはどちらも続く 50 行になる。
- 正規化の統計は smolvla_libero_plus のものをそのまま使う。
- チェックポイントは 3,000 ステップごとに `RLinf/logs/<timestamp>-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt` に保存される。

### 9. 評価

学習した重みを読み込み、プロンプトに `"\nAdvantage: positive"` を付けて評価する（`advantage_prompt=true`）。

```bash
cd $ROOT/RLinf
source .venv-smolvla/bin/activate
bash evaluations/run_eval.sh libero libero_10_smolvla_eval \
  rollout.model.model_path=$MODELS/smolvla_libero_plus \
  runner.ckpt_path=$ROOT/RLinf/logs/<timestamp>-libero10_task0_recap_smolvla/smolvla_recap_sft/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt \
  rollout.model.smolvla.advantage_prompt=true \
  +env.eval.task_id_filter=[0] \
  env.eval.total_num_envs=50
```

`eval/success_once` を 4 のベースラインと比べる。

### GPU 環境で最初に確認すること

以下はこのリポジトリの作成時に GPU・LIBERO のない環境で作ったため、まだ確認していない。

- RLinf SmolVLA 環境（pi0_fast 用の環境は transformers 5.5.4 に固定）で SmolVLA が読み込めること。
- ベースライン評価の成功率が妥当なこと。smolvla_libero_plus は LIBERO-plus のデータ（`pepijn223/libero_plus_lerobot`）で学習されている。RLinf の LIBERO 環境の観測（画像の向き、8 次元の state）は lerobot/libero と同じ形式で、RLinf の pi0_fast がこの前提で lerobot/pi0fast-libero を評価しているが、smolvla_libero_plus での成功率は未確認。
- `libero_10_smolvla_collect.yaml` での rollout 収集が実際に動くこと。特に `data_collection` が train/eval 双方の env に共通で使えるかどうかと、`rollout_epoch` から実際に集まるエピソード数の関係。
- Step 4 の学習が FSDP（`sharding_strategy: no_shard`）で動くこと。

確認済みのこと:
- パッチの適用・解除。
- π₀.₅ 版データセット（eval・sft サブセット）の v2 → v3.0 変換。元データが変更されないことを含む。
- `rlinf/data/storage/lerobot/writer.py` の修正込みで、`LeRobotDatasetWriter` が LeRobot 0.6 系（v3.0）でエピソードを書き出し、`finalize()` 後に読めること。
- `scripts/convert_rlinf_collected_to_v21.py` が、上記の書き出しを模した複数ランク・複数エピソードのデータを 1 つの v2.1 データセットに統合し、v3.0 へ変換できること。
- 変換後のデータに対する Step 4 のデータローダ（行動チャンクと state が元データと一致し、正のフレームの 90% に `Advantage: positive` が付くこと）。

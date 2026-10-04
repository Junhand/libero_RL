# 依頼書: 2 局面ゲートの学習の追加実験(fixed_std = 0.05)

別の GPU(別のコンテナ)で、**メインの実験と 1 つだけ条件を変えた実験**を行う。作業ディレクトリは `/workspace/libero_RL/tmp/rlt_stage2_b/`(専用。メインの実験は `tmp/rlt_stage2/` で動いており、**混ぜない**)。

## 背景
- RLT(RL token を使った、凍結 SmolVLA 上の軽い actor-critic の RL)で、LIBERO-10 の Task 0(2 つの物体をカゴに入れる)の成功率を上げる実験。
- 1 つ目の把持を RL の対象にすると、成功率が 0.46 から 0.62 に上がった(50 エピソード)。残る失敗の最大の原因は 2 つ目の把持。そこで、**2 つ目の把持も RL の対象にする**(2 局面ゲート)学習を、メインの GPU で実行中(`fixed_std=0.02`)。
- 今回の追加実験は、**同じ学習を `fixed_std=0.05`(探索のノイズを大きく)で行う比較**。他の条件(初期の重み、シード、ハイパーパラメータ、エポック数)は、メインと同じ。

## 実行するもの(すべて用意済み。コードは変更しない)
- `run_phase2_long.sh <名前> <最終のステップ>`: 学習(500 → 750 の 250 エポック、約 4 時間)と、最後の評価(50 エピソード、約 20 分)。落ちたら、最新のチェックポイントから、最大 3 回まで自動で再開する。
- `progress.py`: 進捗の確認(エポック数、1 エポックの時間、成功率の推移、メモリ不足の有無、保存済みのチェックポイント)。
- 初期の重み(読み取り専用): `/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_045825/rlt-s1cont-utd5-s1-grasp/checkpoints/global_step_500`

## 手順
1. 事前チェック(カレントは `/workspace/libero_RL`): `bash scripts/preflight_collect_env.sh`。EGL と robosuite の項目が FAIL なら、`libegl1`、`libopengl0` が足りない(apt は、**依頼者の許可を得てから**)。`venv interpreter` が FAIL なら `uv python install 3.12.11`。`LIBERO config` が FAIL なら `echo n | /workspace/libero_RL/RLinf/.venv-smolvla/bin/python -c 'import libero.libero'`。ユーザー領域のものは実行してよい。
2. GPU が空いていることを確認する(`nvidia-smi`。使用メモリが小さいこと)。
3. 起動:
```bash
cd /workspace/libero_RL/tmp/rlt_stage2_b
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
nohup bash -c 'while true; do echo "$(date +%T) rss=$(awk "\$1==\"rss\"{print int(\$2/1000000)}" /sys/fs/cgroup/memory/memory.stat)MB"; sleep 30; done' > mem.log 2>&1 &
echo $! > mem.pid
D=/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_045825/rlt-s1cont-utd5-s1-grasp/checkpoints/global_step_500
INIT_RESUME=$D FIXED_STD=0.05 ENV_SEED=960000 NOISE_SEED=960000 EVAL_SEED=930000 \
  nohup ./run_phase2_long.sh rlt-phase2-std005 750 > run.log 2>&1 &
disown
```
4. 起動の約 2 分後に、設定が読み込まれた値を確認する(`logs/smoke_*/tensorboard/config.yaml` の最新): `actor.model.fixed_std=0.05`、`env.train.rlt_policy_switch.num_phases=2`、`env.train.max_episode_steps=400`、`algorithm.rlt_segment_terminal=True`、`algorithm.rlt_phase_return=False`、`algorithm.update_epoch=25`、`algorithm.rlt_schedule.max_updates_per_train_step=400`、`runner.resume_dir=<上の D>`、`runner.max_epochs=750`。1 つでも違えば、止めて報告する。
5. 進捗は、30〜60 分おきに、`/workspace/libero_RL/RLinf/.venv-smolvla/bin/python progress.py`(最終行を見る程度)。

## 守ること
- **リポジトリのファイル、`RLinf/`、`tmp/rlinf_dev/`、`tmp/rlt_stage2/`、`tmp/rlt_stage1/`、`tmp/rlt_stage2/diag/` を編集・削除しない。** commit しない。書き込んでよいのは `tmp/rlt_stage2_b/` だけ。
- `pkill -f` を使わない(自分のシェルも落ちる)。止めるときは PID を指定する。
- システムの変更(apt など)は、依頼者の許可を得てから。
- wandb は使わない(この環境には認証がない。キーを求めない)。ログは tensorboard だけ。結果の wandb へのアップロードは、メインの環境で行う。
- 落ちた場合: スクリプトが自動で再開する。**同じエラーで 2 回続けて落ちたら、止めて、ログのエラーの前後を添えて報告する**(コードは直さない)。
- メモリ上限 25GB(学習のピークは約 20.5GB)。他の重い処理を同時に動かさない。

## 報告
- 起動後: 設定の確認の結果(4 の項目)と、最初のエポックが動いていること(1〜3 行)。
- 約 1 時間ごと: `progress.py` の出力の要点(エポック数、1 エポックの時間、成功率の推移、メモリ不足の有無)。
- 終了時: 学習の終了コード、最終の評価の結果(`eval_F_seed930000.log` の `eval/success_once` の行)、メモリの最大値(`mem.log` の rss の最大)、問題の有無。
- 終了後は、次の指示を待つ(新しい実験を始めない)。

---
# 追加の依頼 1: 学習済みモデル(step 750)の失敗の診断(約 20 分)

目的: 評価(0.66)の中身を、エピソードごとに調べる。評価と同じ条件、同じシード(930000)で、**各ステップの状態、行動、成否を記録**して、集計する。

1. メモリ記録のループを止める: `kill $(cat /workspace/libero_RL/tmp/rlt_stage2_b/mem.pid)`(PID 指定。自分が起動したもの)。
2. 実行(約 20 分。GPU は 1 つの評価だけ):
```bash
cd /workspace/libero_RL/tmp/rlt_stage2_b
source /workspace/libero_RL/RLinf/.venv-smolvla/bin/activate
ray stop --force
nohup ./diag_collect_G.sh G > diag/run_G.log 2>&1 &
disown
```
   終了は `diag/eval_G.log` の最終行が `exit=0`。保存先 `diag/G` は、既にあると中断する(再利用しない)。
3. 集計(画像は読まない。数秒):
```bash
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python diag_analyze.py diag/G
```
4. 報告: 集計の出力を、そのまま(1 エピソード数、タスク成功率、保持できなかった/1 回だけ/2 回以上の件数、失敗の内訳)。`diag/eval_G.log` の `eval/success_once`(0.66 になるはず)と、終了コード。問題があれば、エラーの前後。
5. 守ること: 書き込みは `tmp/rlt_stage2_b/` だけ(`diag/G` は画像を含み、約 3.5GB)。他のファイルは変更しない。commit しない。pkill -f を使わない。終わったら、次の指示を待つ。

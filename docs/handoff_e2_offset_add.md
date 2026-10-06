# 他の環境向け: E2（把持の報酬に、時間オフセットの報酬を足す）の指示文

作成: 2026-10-06。背景は `.claude/skills/steam-rlt-hybrid/SKILL.md`（E1-A、E1-B の結果）。

- 環境ごとに、全文の指示文（下の 2 つのブロック）を、そのまま渡す。違いは、タグ・ディレクトリ・`reward_scale` だけ。
- 各環境専用のスクリプト（`tmp/rlt_stage2_e1/`、`tmp/rlt_stage2_e2/`。`tmp/` は git の管理対象外）は、元のスクリプトから、次のように作った。
  共有の `last_log_dir` と評価ログの取り違えを避けるため。
  ```bash
  cd /workspace/libero_RL/tmp
  for n in e1 e2; do d=rlt_stage2_$n; mkdir -p $d
    for f in run_smoke.sh eval_one.sh run_e1.sh; do sed "s#tmp/rlt_stage2#tmp/rlt_stage2_$n#g" rlt_stage2/$f > $d/$f; chmod +x $d/$f; done
    sed -i 's#LOG=$(./eval_one.sh C $SEED#LOG=$(TAG=$TAG ./eval_one.sh C $SEED#' $d/run_e1.sh
  done
  ```
  `run_e1.sh` は、`run_stage1_long.sh` の複製（tensorboard のみ、wandb へは送らない、`OVR` で追加の設定を渡せる）で、`tmp/rlt_stage2/run_e1.sh` にある。
- `add` モードの動作確認は済み（2026-10-06、`reward = 環境の報酬 + オフセット`、3 エポック、exit=0）。
- 前提のファイル: `tmp/hybrid/offset_mlp.pt`（E0 の MLP。変更しないこと）、`tmp/rlinf_dev/`（RLinf の開発用コピー。パッチ `patches/rlinf/smolvla-x-offset-sft.patch` の内容を含む）。

---

## 訂正（2026-10-06）: 「D と同じ設定」は、不正確だった
- D（success_once 0.62）は、「250 エポックを UTD 1 で学習し、その重みとリプレイから、UTD 5 で 500 エポックまで延長」した二段階の学習だった（`results/rlt_libero10_task0_experiments.md` 6.3 節）。
- E1（A、B）と E2（e2a、e2b）は、**500 エポックを UTD 5 で、最初から学習**している。D とは、学習のスケジュールが違う。
- したがって、E2 の比較の基準は、D（0.62）ではなく、**同じ設定で、オフセットを加えない対照**（タグ `d5ctl`、2026-10-06 09:09 に私の環境で開始）。D との比較は、参考にとどめる。
- 指示文の本文は、変更していない（各環境の実行には影響しない）。

---

## 環境1への指示（`reward_scale` = 0.5、タグ e2a）

```
# 依頼 E2a: 把持の報酬 +1(D の設定)に、時間オフセットの報酬を足して、RL Token を学習・評価する(scale = 0.5)

## 背景(3 行)
STEAM と RL Token を組み合わせる検討をしている。時間オフセット(2 つの RL token から、その間に進んだ時間を予測する MLP)を報酬にした実験は、
置換(把持の報酬をやめる)で success_once 0.26、critic なしの SFT で 0.32 と、ベースの VLA(0.46)より悪かった。
今回は、把持の +1(D の設定、success_once 0.62)を残したまま、オフセットを足したら、良くなるか悪くなるかを確かめる。詳細: .claude/skills/steam-rlt-hybrid/SKILL.md の末尾(E1-A、E1-B)。

## 守ること
1. リポジトリのファイル、RLinf/、tmp/rlinf_dev/(共有のコード)、tmp/hybrid/(MLP など)、tmp/rlt_stage2/(別環境が使用)を、編集しない。commit しない。
   書き込むのは、tmp/rlt_stage2_e1/ だけ(この環境専用のスクリプトと出力が入っている)。既存のファイルを削除・変更しない。
2. pkill / pkill -f / killall を使わない。止めるときは pgrep -af で PID を確認して kill <PID>。
   (run_e1.sh は、各試行の後に `ray stop --force` を実行する。これは、この環境の Ray だけに作用する。)
3. メモリ: ノードの上限は約 23GB(確認: cat /sys/fs/cgroup/memory.max)。学習中は、他の重い処理を並行して動かさない。
   メモリ不足で落ちると、run_e1.sh は、自動で環境数を 3 → 2 に減らして再試行する。**これが起きたら、比較の条件が変わるので、報告に必ず書く**(ログの "out of memory -> retry with 2 envs")。
4. システムの変更(apt など)は、依頼者の許可を得てから。GPU は、割り当てられた 1 枚だけ。

## 準備(確認だけ)
cd /workspace/libero_RL
ls RLinf/.venv-smolvla/bin/python tmp/hybrid/offset_mlp.pt tmp/rlinf_dev/rlinf/algorithms/rlt/offset_score.py   # どれかが無ければ、ここで止めて報告
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # GPU が空いていること(A5000 24GB で実績あり)
cat /sys/fs/cgroup/memory.max

## 手順 1: 動作確認(約 5 分)
cd /workspace/libero_RL/tmp/rlt_stage2_e1
EPOCHS=3 ./run_smoke.sh env.train.total_num_envs=3 env.eval.total_num_envs=2 env.train.rlt_policy_switch.mode=grasp_subtask env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 algorithm.rlt_phase_return=False algorithm.rlt_offset.enable=True algorithm.rlt_offset.mode=reward algorithm.rlt_offset.reward_mode=add algorithm.rlt_offset.reward_scale=0.5 algorithm.rlt_offset.ckpt=/workspace/libero_RL/tmp/hybrid/offset_mlp.pt algorithm.update_epoch=25 algorithm.rlt_schedule.warmup_min_size=40
grep '^exit=' $(cat last_log_dir)/run.log     # exit=0 なら次へ。そうでなければ、ログの末尾 30 行を添えて報告して止める。

## 手順 2: 本番(学習 約 4〜5 時間 + 評価 約 20 分)
cd /workspace/libero_RL/tmp/rlt_stage2_e1
OVR="algorithm.rlt_offset.enable=True algorithm.rlt_offset.mode=reward algorithm.rlt_offset.reward_mode=add algorithm.rlt_offset.reward_scale=0.5 algorithm.rlt_offset.ckpt=/workspace/libero_RL/tmp/hybrid/offset_mlp.pt" nohup ./run_e1.sh e2a 500 > e2a_chain.log 2>&1 &
echo $! > e2a.pid
- 進捗: tail -5 e2a_chain.log。学習が終わると、自動で評価(50 エピソード、シード 930000)が走り、"all done" が出る。
- 設定は、D と同じ(1 つ目の把持のサブタスク、500 エポック、3 環境、UTD 5、fixed_std 0.02)。違いは、オフセットの報酬(scale 0.5)を足している点だけ。tensorboard のみに記録し、wandb へは送らない。
- 途中で落ちた場合: run_e1.sh が、最新のチェックポイントから、自動で再開する(最大 3 回)。ファイルを削除しない。繰り返し落ちるときは、ログの末尾を添えて報告する。

## 結果の取り出し(終了後)
cd /workspace/libero_RL/tmp/rlt_stage2_e1
grep -E "success_once|success_at_end" eval_e2a_seed930000.log | tail -2
L=$(ls -dt logs/smoke_* | head -1)   # 本番のログ(動作確認のものより新しい)。ls -dt logs/smoke_* で確認すること
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python - <<EOF
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, numpy as np
ea=EventAccumulator(glob.glob("$L/tensorboard")[0], size_guidance={"scalars":0}); ea.Reload()
for t in ("env/success_once","train/replay/reward_mean","train/critic/q_data","train/actor/action_ref_abs_mean"):
    v=[x.value for x in ea.Scalars(t)]; print(t,"n=%d first50 %.4f last50 %.4f"%(len(v),np.mean(v[:50]),np.mean(v[-50:])))
EOF

## 終了後の報告(5〜10 行)
- 評価の success_once と success_at_end(50 エピソード)。比較: ベース(VLA のみ)0.46、D 0.62、E1-A 0.32、E1-B 0.26。
- 学習中の rollout の success_once の、最初の 50 エポックと最後の 50 エポックの平均(悪化・改善の傾向)。
- 学習の所要時間(開始・終了時刻)。メモリ不足による環境数の削減の有無、再開の回数。
- 気づいた点、エラーの有無。
```

---

## 環境2への指示（`reward_scale` = 2.5、タグ e2b）

```
# 依頼 E2b: 把持の報酬 +1(D の設定)に、時間オフセットの報酬を足して、RL Token を学習・評価する(scale = 2.5)

## 背景(3 行)
STEAM と RL Token を組み合わせる検討をしている。時間オフセット(2 つの RL token から、その間に進んだ時間を予測する MLP)を報酬にした実験は、
置換(把持の報酬をやめる)で success_once 0.26、critic なしの SFT で 0.32 と、ベースの VLA(0.46)より悪かった。
今回は、把持の +1(D の設定、success_once 0.62)を残したまま、オフセットを足したら、良くなるか悪くなるかを確かめる。詳細: .claude/skills/steam-rlt-hybrid/SKILL.md の末尾(E1-A、E1-B)。

## 守ること
1. リポジトリのファイル、RLinf/、tmp/rlinf_dev/(共有のコード)、tmp/hybrid/(MLP など)、tmp/rlt_stage2/(別環境が使用)、tmp/rlt_stage2_e1/(別環境が使用)を、編集しない。commit しない。
   書き込むのは、tmp/rlt_stage2_e2/ だけ(この環境専用のスクリプトと出力が入っている)。既存のファイルを削除・変更しない。
2. pkill / pkill -f / killall を使わない。止めるときは pgrep -af で PID を確認して kill <PID>。
   (run_e1.sh は、各試行の後に `ray stop --force` を実行する。これは、この環境の Ray だけに作用する。)
3. メモリ: ノードの上限は約 23GB(確認: cat /sys/fs/cgroup/memory.max)。学習中は、他の重い処理を並行して動かさない。
   メモリ不足で落ちると、run_e1.sh は、自動で環境数を 3 → 2 に減らして再試行する。**これが起きたら、比較の条件が変わるので、報告に必ず書く**(ログの "out of memory -> retry with 2 envs")。
4. システムの変更(apt など)は、依頼者の許可を得てから。GPU は、割り当てられた 1 枚だけ。

## 準備(確認だけ)
cd /workspace/libero_RL
ls RLinf/.venv-smolvla/bin/python tmp/hybrid/offset_mlp.pt tmp/rlinf_dev/rlinf/algorithms/rlt/offset_score.py   # どれかが無ければ、ここで止めて報告
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # GPU が空いていること(A5000 24GB で実績あり)
cat /sys/fs/cgroup/memory.max

## 手順 1: 動作確認(約 5 分)
cd /workspace/libero_RL/tmp/rlt_stage2_e2
EPOCHS=3 ./run_smoke.sh env.train.total_num_envs=3 env.eval.total_num_envs=2 env.train.rlt_policy_switch.mode=grasp_subtask env.train.max_episode_steps=200 env.train.max_steps_per_rollout_epoch=200 algorithm.rlt_phase_return=False algorithm.rlt_offset.enable=True algorithm.rlt_offset.mode=reward algorithm.rlt_offset.reward_mode=add algorithm.rlt_offset.reward_scale=2.5 algorithm.rlt_offset.ckpt=/workspace/libero_RL/tmp/hybrid/offset_mlp.pt algorithm.update_epoch=25 algorithm.rlt_schedule.warmup_min_size=40
grep '^exit=' $(cat last_log_dir)/run.log     # exit=0 なら次へ。そうでなければ、ログの末尾 30 行を添えて報告して止める。

## 手順 2: 本番(学習 約 4〜5 時間 + 評価 約 20 分)
cd /workspace/libero_RL/tmp/rlt_stage2_e2
OVR="algorithm.rlt_offset.enable=True algorithm.rlt_offset.mode=reward algorithm.rlt_offset.reward_mode=add algorithm.rlt_offset.reward_scale=2.5 algorithm.rlt_offset.ckpt=/workspace/libero_RL/tmp/hybrid/offset_mlp.pt" nohup ./run_e1.sh e2b 500 > e2b_chain.log 2>&1 &
echo $! > e2b.pid
- 進捗: tail -5 e2b_chain.log。学習が終わると、自動で評価(50 エピソード、シード 930000)が走り、"all done" が出る。
- 設定は、D と同じ(1 つ目の把持のサブタスク、500 エポック、3 環境、UTD 5、fixed_std 0.02)。違いは、オフセットの報酬(scale 2.5)を足している点だけ。tensorboard のみに記録し、wandb へは送らない。
- 途中で落ちた場合: run_e1.sh が、最新のチェックポイントから、自動で再開する(最大 3 回)。ファイルを削除しない。繰り返し落ちるときは、ログの末尾を添えて報告する。

## 結果の取り出し(終了後)
cd /workspace/libero_RL/tmp/rlt_stage2_e2
grep -E "success_once|success_at_end" eval_e2b_seed930000.log | tail -2
L=$(ls -dt logs/smoke_* | head -1)   # 本番のログ(動作確認のものより新しい)。ls -dt logs/smoke_* で確認すること
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python - <<EOF
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, numpy as np
ea=EventAccumulator(glob.glob("$L/tensorboard")[0], size_guidance={"scalars":0}); ea.Reload()
for t in ("env/success_once","train/replay/reward_mean","train/critic/q_data","train/actor/action_ref_abs_mean"):
    v=[x.value for x in ea.Scalars(t)]; print(t,"n=%d first50 %.4f last50 %.4f"%(len(v),np.mean(v[:50]),np.mean(v[-50:])))
EOF

## 終了後の報告(5〜10 行)
- 評価の success_once と success_at_end(50 エピソード)。比較: ベース(VLA のみ)0.46、D 0.62、E1-A 0.32、E1-B 0.26。
- 学習中の rollout の success_once の、最初の 50 エポックと最後の 50 エポックの平均(悪化・改善の傾向)。
- 学習の所要時間(開始・終了時刻)。メモリ不足による環境数の削減の有無、再開の回数。
- 気づいた点、エラーの有無。
```

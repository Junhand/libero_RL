# 他の環境向け: FPO の smolVLA への実装を支える調査（2026-10-07）

環境ごとに、全文の指示文（下の 2 つのブロック）を、そのまま渡す。

- 背景: `docs/fpo_smolvla_design.md`（FPO の設計）。FPO の実装（損失の単体テスト、保存・GAE・更新）は、私の環境で進める。ここでは、実装に依存しない調査を依頼する。
- 環境1: ① ベース方策の、初期状態ごとの成功確率（ノイズのシード 8 回）、② 価値ヘッドのオフライン調査。約 4 時間。
- 環境2: ① ベース方策の、初期状態ごとの成功確率（ノイズのシード、環境1 と別の 8 回）、② ロールアウトの時間とメモリの測定。約 3.5 時間。
- 前提（作成済み）: `tmp/rlt_stage2_e1/`、`tmp/rlt_stage2_e2/`（`eval_one.sh`、`run_smoke.sh`。評価ログは、それぞれの中に出る）。
- 注意: STEAM と RL Token は、完了とした（ユーザーの判断）。これらの追加の実験は、依頼しない。

---

## 環境1への指示

```
# 依頼 F1: (1) ベース方策の、初期状態ごとの成功確率(ノイズ 8 回)と、(2) 価値ヘッドのオフライン調査

## 背景(4 行)
smolVLA に FPO(Flow Policy Optimization、オンポリシーの PPO。flow matching 方策の損失の差を、PPO の対数比として使う)を実装する。詳細: docs/fpo_smolvla_design.md。
FPO の探索は、行動の生成の初期ノイズだけ。ベース方策(VLA のみ)は、初期状態(Task 0 の 50 通り)で成功率が大きく違い(0% が 11 状態、90% 超が 8 状態)、
ノイズで成否が変わる状態(混合)が、どれだけあるかが、FPO の学習信号の量を決める。そこで、(1) 初期状態ごとの成功確率を、ノイズを変えて測る。
また、FPO の GAE は、価値ヘッドの質に左右されるので、(2) 手持ちのデータで、価値ヘッドをオフラインに学習して、どこまで予測できるかを調べる。

## 守ること
1. リポジトリのファイル、RLinf/、tmp/rlinf_dev/(共有のコード)、tmp/hybrid/、tmp/rlt_stage2/(他が使用)を、編集しない。commit しない。
   書き込むのは、tmp/rlt_stage2_e1/(1 の評価)と、tmp/hybrid_fpo_v1/(2 の出力。新規に作る)だけ。既存のファイルを削除・変更しない。入力のデータは、読み取りのみ。
2. pkill / pkill -f / killall を使わない。止めるときは pgrep -af で PID を確認して kill <PID>。
3. メモリ: ノードの上限は約 23GB(cat /sys/fs/cgroup/memory.max)。1 の評価は、1 つずつ、順番に(並列にしない)。評価の実行中に、重い分析(2)を並行しない。
4. ディスクの quota: 開始前に、書き込みを確認する(`head -c 20000000 /dev/zero > tmp/rlt_stage2_e1/quota_test.bin; ls -la tmp/rlt_stage2_e1/quota_test.bin` で 20000000 バイトになること)。
   `Disk quota exceeded` が出たら、止めて報告する(自分で削除しない)。確認用のファイルは、残してよい。
5. 長い処理は、切り離して実行する(nohup + &)。ツールのバックグラウンド実行には、約 2 時間の上限がある。
6. システムの変更(apt など)は、依頼者の許可を得てから。GPU は、割り当てられた 1 枚だけ。

## 準備(確認だけ)
cd /workspace/libero_RL/tmp/rlt_stage2_e1
ls eval_one.sh
ls /workspace/libero_RL/tmp/hybrid/zrl | head -3   # *.npy と *_meta.parquet があること
ls /workspace/libero_RL/results/steam_rlt_hybrid/initstate_ids.parquet
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
cat /sys/fs/cgroup/memory.max

## 手順 1: ベース方策の評価(シード 940001〜940008、1 回 約 20 分、合計 約 2.7 時間)
cat > run_f1.sh <<'EOS'
#!/bin/bash
cd /workspace/libero_RL/tmp/rlt_stage2_e1
for SEED in 940001 940002 940003 940004 940005 940006 940007 940008; do
  echo "[$(date +%T)] base seed $SEED"; TAG=f1_s$SEED ./eval_one.sh B $SEED > /dev/null
done
echo f1-done
EOS
chmod +x run_f1.sh
nohup ./run_f1.sh > f1.log 2>&1 &
echo $! > f1.pid
- B は、VLA のみ(RLT 経路)の評価。シードは、行動のノイズだけを変える(初期状態の 50 通りは同じ)。評価ログ: eval_f1_s<シード>_seed<シード>.log。
- 終了後、各試行の成否を集計する(ログに `trial_id=<番号>, success=<True|False>` の行がある。試行の番号は、どのシードでも、同じ初期状態):
  /workspace/libero_RL/RLinf/.venv-smolvla/bin/python - <<'PY'
  import re, glob, collections
  cnt = collections.Counter(); runs = 0
  for f in sorted(glob.glob('/workspace/libero_RL/tmp/rlt_stage2_e1/eval_f1_s*_seed*.log')):
      d = {}
      for l in open(f):
          m = re.search(r'trial_id=(\d+), success=(True|False)', l)
          if m: d[int(m.group(1))] = m.group(2) == 'True'
      if len(d) == 50: runs += 1; [cnt.update([k]) for k, v in d.items() if v]
  rows = [(k, cnt.get(k, 0)) for k in range(50)]
  open('/workspace/libero_RL/tmp/rlt_stage2_e1/per_trial_success_f1.csv', 'w').write('trial_id,n_success,n_runs\n' + ''.join(f'{k},{c},{runs}\n' for k, c in rows))
  print('runs', runs, 'per-trial success counts:', [c for _, c in rows])
  PY

## 手順 2: 価値ヘッドのオフライン調査(手順 1 が終わってから。GPU は、軽く使う。約 1.5 時間)
mkdir -p /workspace/libero_RL/tmp/hybrid_fpo_v1 && cd /workspace/libero_RL/tmp/hybrid_fpo_v1
スクリプト v1.py を書いて実行する。
入力(読み取りのみ):
- /workspace/libero_RL/tmp/hybrid/zrl/<元>.npy(RL token、フレーム × 960、fp16)と <元>_meta.parquet(列: ep, t, succ, s0..s7(状態 8 次元), gap, cmd)。元 = train_v2 の 11 個(r1_m0 … r4_m2)と eval_m0〜m2。sft_target3 は使わない。フレームは、t が 6 の倍数。
- 初期状態の ID: /workspace/libero_RL/results/steam_rlt_hybrid/initstate_ids.parquet(列: src, ep, succ, cid。cid は 0〜49)。(src, ep) で結合する。
- エピソードの長さ T: /workspace/libero_RL/data/smolvla_recap/libero10_task0_train_v2_<元>/meta/episodes.jsonl(eval は libero10_task0_eval_v2_m<番号>)の "length"。
- 成功の判定は、メタの succ(エピソードのどこかで成功)。episodes.jsonl の is_success は、結果と一致しないので、使わない。
目標: 報酬は、成功したエピソードの最後の環境ステップだけ +1。価値 V*(t) = 0.995^(T−1−t)(成功のエピソード)、0(失敗のエピソード)。t は環境のステップ。
モデル: [z_rl(960 次元) + 状態(8 次元)](標準化)→ 512 → 256 → 1 の MLP(ReLU)。損失は MSE、AdamW(学習率 1e-4)、30 エポック、バッチ 1024。
分割: (a) 初期状態ごとの 5 分割(cid を 5 つに分けて、4 つで学習、1 つで評価。未見の初期状態)、(b) 対照: エピソードをランダムに 20% 除外(初期状態は既知)。
比較する設定: 状態あり / なし。
指標(評価側):
- 説明分散 = 1 − Var(V* − V) / Var(V*)、Spearman(V*, V)。
- t の区間(0〜60、60〜120、120〜180、180 以上)ごとの、成功 vs 失敗のエピソードの AUROC(V の値で)。
- キャリブレーション: 予測 V の 10 分位ごとの、平均 V と、平均 V*。
- TD 残差のノイズ: エピソード内で 2 つ先(約 12 ステップ)の格子点を使い、δ = 0.995^12 · V(t+12) − V(t)(最後の区間は報酬を加える)の標準偏差を、V* の差の標準偏差と比べる。
出力: v1.py、v1_result.txt(表)、v1_calibration.png(日本語は /root/.fonts/NotoSansCJKjp-Regular.otf を matplotlib に登録。無ければ英語ラベルで可)。
メモリは 8GB 以内(特徴を GPU に載せ、バッチをインデックスで作る)。

## 終了後の報告(12〜18 行)
1. 手順 1: 8 シードの success_once と success_at_end の表(各シード)、平均。per_trial_success_f1.csv のパス。試行ごとの成功回数(8 回中)の分布: 0 回、1〜7 回(混合)、8 回の試行の数。
2. 手順 2: 説明分散と AUROC の表(未見の初期状態 / 対照 × 状態あり / なし)。TD 残差のノイズの結果。結論: 価値ヘッドは、FPO の GAE に使える質か。
3. 実行の開始・終了時刻、エラーの有無、quota の確認結果。
```

---

## 環境2への指示

```
# 依頼 F2: (1) ベース方策の、初期状態ごとの成功確率(ノイズ 8 回、環境1 とは別のシード)と、(2) ロールアウトの時間とメモリの測定

## 背景(4 行)
smolVLA に FPO(Flow Policy Optimization、オンポリシーの PPO)を実装する。詳細: docs/fpo_smolvla_design.md。
FPO の探索は、行動の生成の初期ノイズだけ。ベース方策は、初期状態(Task 0 の 50 通り)で成功率が大きく違い、ノイズで成否が変わる状態が、どれだけあるかが、学習信号の量を決める。
そこで、(1) 初期状態ごとの成功確率を、ノイズを変えて測る(環境1 が、別の 8 シードを担当。あとで合算する)。
また、(2) FPO の 1 回の更新の時間を見積もるために、学習の設定(3 環境、フルタスク 520 ステップ)での、ロールアウトの時間とメモリを測る。

## 守ること
1. リポジトリのファイル、RLinf/、tmp/rlinf_dev/(共有のコード)、tmp/hybrid/、tmp/rlt_stage2/、tmp/rlt_stage2_e1/(他が使用)を、編集しない。commit しない。
   書き込むのは、tmp/rlt_stage2_e2/ だけ(スクリプトが入っている)。既存のファイルを削除・変更しない。
2. pkill / pkill -f / killall を使わない。止めるときは pgrep -af で PID を確認して kill <PID>。
3. メモリ: ノードの上限は約 23GB(cat /sys/fs/cgroup/memory.max)。評価と測定は、1 つずつ、順番に(並列にしない)。
4. ディスクの quota: 開始前に、書き込みを確認する(`head -c 20000000 /dev/zero > tmp/rlt_stage2_e2/quota_test.bin; ls -la tmp/rlt_stage2_e2/quota_test.bin` で 20000000 バイトになること)。
   `Disk quota exceeded` が出たら、止めて報告する(自分で削除しない)。確認用のファイルは、残してよい。
5. 長い処理は、切り離して実行する(nohup + &)。ツールのバックグラウンド実行には、約 2 時間の上限がある。
6. システムの変更(apt など)は、依頼者の許可を得てから。GPU は、割り当てられた 1 枚だけ。

## 準備(確認だけ)
cd /workspace/libero_RL/tmp/rlt_stage2_e2
ls eval_one.sh run_smoke.sh
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
cat /sys/fs/cgroup/memory.max

## 手順 1: ベース方策の評価(シード 940009〜940016、1 回 約 20 分、合計 約 2.7 時間)
cat > run_f2.sh <<'EOS'
#!/bin/bash
cd /workspace/libero_RL/tmp/rlt_stage2_e2
for SEED in 940009 940010 940011 940012 940013 940014 940015 940016; do
  echo "[$(date +%T)] base seed $SEED"; TAG=f2_s$SEED ./eval_one.sh B $SEED > /dev/null
done
echo f2-done
EOS
chmod +x run_f2.sh
nohup ./run_f2.sh > f2.log 2>&1 &
echo $! > f2.pid
- B は、VLA のみ(RLT 経路)の評価。シードは、行動のノイズだけを変える(初期状態の 50 通りは同じ)。評価ログ: eval_f2_s<シード>_seed<シード>.log。
- 終了後、各試行の成否を集計する(ログの `trial_id=<番号>, success=<True|False>` の行。試行の番号は、どのシードでも、同じ初期状態):
  /workspace/libero_RL/RLinf/.venv-smolvla/bin/python - <<'PY'
  import re, glob, collections
  cnt = collections.Counter(); runs = 0
  for f in sorted(glob.glob('/workspace/libero_RL/tmp/rlt_stage2_e2/eval_f2_s*_seed*.log')):
      d = {}
      for l in open(f):
          m = re.search(r'trial_id=(\d+), success=(True|False)', l)
          if m: d[int(m.group(1))] = m.group(2) == 'True'
      if len(d) == 50: runs += 1; [cnt.update([k]) for k, v in d.items() if v]
  rows = [(k, cnt.get(k, 0)) for k in range(50)]
  open('/workspace/libero_RL/tmp/rlt_stage2_e2/per_trial_success_f2.csv', 'w').write('trial_id,n_success,n_runs\n' + ''.join(f'{k},{c},{runs}\n' for k, c in rows))
  print('runs', runs, 'per-trial success counts:', [c for _, c in rows])
  PY

## 手順 2: ロールアウトの時間とメモリの測定(手順 1 が終わってから。約 15 分)
cd /workspace/libero_RL/tmp/rlt_stage2_e2
( while true; do echo "$(date +%T) $(cat /sys/fs/cgroup/memory.current)"; sleep 10; done ) > t2_mem.log 2>&1 &
echo $! > t2_mem.pid
EPOCHS=5 ./run_smoke.sh env.train.total_num_envs=3 env.eval.total_num_envs=2 env.train.max_episode_steps=520 env.train.max_steps_per_rollout_epoch=520
kill $(cat t2_mem.pid)
grep '^exit=' $(cat last_log_dir)/run.log   # exit=0 を確認(そうでなければ、ログの末尾 30 行を添えて報告)
- これは、RL Token の学習の動作確認用の設定(フルタスク、3 環境、1 エポックは各環境で 520 ステップまで)。FPO のロールアウトは、特徴の抽出が不要なので、これより軽い(上限の目安)。
- 1 エポックの時間: TensorBoard の time/ で始まるタグ、または run.log の進捗から、エポックごとの時間(秒)を取り出す:
  /workspace/libero_RL/RLinf/.venv-smolvla/bin/python - <<'PY'
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  import glob
  d = open('/workspace/libero_RL/tmp/rlt_stage2_e2/last_log_dir').read().strip()
  ea = EventAccumulator(glob.glob(d + '/tensorboard')[0], size_guidance={'scalars': 0}); ea.Reload()
  tags = [t for t in ea.Tags()['scalars'] if 'time' in t.lower() or 'episode_len' in t or 'success_once' in t]
  for t in tags: print(t, [round(x.value, 2) for x in ea.Scalars(t)])
  PY
- メモリ: t2_mem.log の最大値(バイトを GB に換算)。

## 終了後の報告(10〜15 行)
1. 手順 1: 8 シードの success_once と success_at_end の表(各シード)、平均。per_trial_success_f2.csv のパス。試行ごとの成功回数(8 回中)の分布: 0 回、1〜7 回(混合)、8 回の試行の数。
2. 手順 2: 1 エポックの時間(5 エポックの各値と平均)、エピソードの長さ(episode_len)、ホストメモリの最大値(GB)、GPU メモリの最大値(nvidia-smi の目視でよい)。
3. 実行の開始・終了時刻、エラーの有無、quota の確認結果。
```

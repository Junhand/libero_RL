# 他の環境向け: 評価の再現性（複数シード）と、STEAM の凍結 critic の再現（2026-10-07）

**【2026-10-07 更新】STEAM は完了とする（ユーザーの判断）。環境2への指示（STEAM の凍結 critic の再現）は、実行しない。すでに送った場合は、実行中の学習を PID を指定して止める。環境1への指示（RL Token の評価の再現性）は、有効。**

環境ごとに、全文の指示文（下の 2 つのブロック）を、そのまま渡す。

- 環境1: RL Token のチェックポイントを、評価のシードを変えて評価する（約 3 時間）。
- 環境2: STEAM の critic（視覚と言語の両方を凍結）を、シードと学習率を変えて 2 回学習し、評価データで比べる（約 6 時間）。
- 前提のファイル・スクリプト（作成済み）:
  - 環境1: `tmp/rlt_stage2_e1/`（`eval_one.sh` など。評価ログは、この中に出る）。
  - 環境2: `tmp/steam_lora_e2/`（`run_variant.sh`、`train_frozen.sh`、`label_eval.sh`、`eval_critics.py`、評価データのコピー `shadow_eval/`）。`tmp/rlinf_steam_dev/`（RLinf の開発用コピー。STEAM の LoRA・凍結のコードを含む）は共有で、読み取りのみ（学習のログが、この中の `logs/` に出るのは、想定どおり）。
- 背景: `results/steam_smolvla_libero10_task0.md` の 11・12 節、`.claude/skills/steam-rlt-hybrid/SKILL.md`。
- 注意: e2a・e2b のチェックポイントは、すでに存在しない（再評価できない）。

---

## 環境1への指示（RL Token の評価の再現性）

```
# 依頼 R1: RL Token のチェックポイントを、評価のシードを変えて、3 回ずつ追加で評価する

## 背景(3 行)
50 エピソードの success_once は、標準誤差が約 0.07 で、条件の差(0.46 / 0.56 / 0.62 など)が、誤差の範囲内に見える。
評価は、同じシードなら決定的(同じ評価を繰り返すと、50 試行すべてが一致)で、シードを変えると、行動のノイズだけが変わる(初期状態の 50 通りは同じ)。
そこで、最終のチェックポイントを、シード 930001〜930003 で、追加で評価して、各条件の平均と、ばらつきを出したい。

## 守ること
1. リポジトリのファイル、RLinf/、tmp/rlinf_dev/(共有のコード)、tmp/hybrid/、tmp/rlt_stage2/(他が使用)を、編集しない。commit しない。
   書き込むのは、tmp/rlt_stage2_e1/ だけ。既存のファイルを削除・変更しない。チェックポイントは、読み取りのみ。
2. pkill / pkill -f / killall を使わない。止めるときは pgrep -af で PID を確認して kill <PID>。
3. メモリ: ノードの上限は約 23GB(cat /sys/fs/cgroup/memory.max)。評価は、1 つずつ、順番に実行する(並列にしない)。他の重い処理を並行しない。
4. ディスクの quota: 開始前に、書き込みを確認する(`head -c 20000000 /dev/zero > tmp/rlt_stage2_e1/quota_test.bin; ls -la tmp/rlt_stage2_e1/quota_test.bin` で 20000000 バイトになること)。
   `Disk quota exceeded` が出たら、止めて報告する(自分で削除しない)。確認用のファイルは、残してよい(後で報告する)。
5. システムの変更(apt など)は、依頼者の許可を得てから。GPU は、割り当てられた 1 枚だけ。

## 準備(確認だけ)
cd /workspace/libero_RL/tmp/rlt_stage2_e1
ls eval_one.sh
for f in /workspace/libero_RL/tmp/rlt_stage2/logs/smoke_045825/rlt-s1cont-utd5-s1-grasp /workspace/libero_RL/tmp/rlt_stage2/logs/smoke_090930/d5ctl-s1-grasp; do ls $f/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt; done   # 2 つとも存在すること
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # GPU が空いていること
cat /sys/fs/cgroup/memory.max

## 手順: 評価(9 回。1 回 約 20 分、合計 約 3 時間)
cat > run_multiseed.sh <<'EOS'
#!/bin/bash
cd /workspace/libero_RL/tmp/rlt_stage2_e1
D=/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_045825/rlt-s1cont-utd5-s1-grasp/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt
C=/workspace/libero_RL/tmp/rlt_stage2/logs/smoke_090930/d5ctl-s1-grasp/checkpoints/global_step_500/actor/model_state_dict/full_weights.pt
for SEED in 930001 930002 930003; do
  echo "[$(date +%T)] base seed $SEED"; TAG=base_s$SEED ./eval_one.sh B $SEED > /dev/null
  echo "[$(date +%T)] D seed $SEED";    TAG=D_s$SEED ./eval_one.sh C $SEED $D > /dev/null
  echo "[$(date +%T)] d5ctl seed $SEED"; TAG=d5ctl_s$SEED ./eval_one.sh C $SEED $C > /dev/null
done
echo multiseed-done
EOS
chmod +x run_multiseed.sh
nohup ./run_multiseed.sh > multiseed.log 2>&1 &
echo $! > multiseed.pid
- 進捗: tail -3 multiseed.log。評価ログは、eval_<タグ>_seed<シード>.log(例: eval_D_s930001_seed930001.log)。
- base は、VLA のみ(RLT 経路)。D は、1 つ目の把持だけ RL、500 エポック(段階 1 の 250 エポックを UTD 1 → UTD 5 で延長)。d5ctl は、同じ 500 エポックを UTD 5 で、最初から学習したもの。
- 途中で落ちた場合: 該当の評価だけ、ログの末尾を添えて報告し、残りを続ける(ファイルは削除しない)。

## 結果の取り出し(終了後)
cd /workspace/libero_RL/tmp/rlt_stage2_e1
for f in eval_base_s*_seed*.log eval_D_s*_seed*.log eval_d5ctl_s*_seed*.log; do echo "$f $(grep -o 'success_once=[0-9.]*\|success_at_end=[0-9.]*' $f | tail -2 | tr '\n' ' ')"; done
# 50 試行ごとの成否(各ログ): grep -o "trial_id=[0-9]*, success=[A-Za-z]*" <ログ> | sort -t= -k2 -n

## 終了後の報告(8〜12 行)
- 表: 条件(base、D、d5ctl)× シード(930001〜930003)の success_once と success_at_end。
- すでにあるシード 930000 の値(base 0.46、D 0.62、d5ctl 0.56)を含めた 4 シードの平均と、最小・最大。
- 条件間の順位が、シードで入れ替わるか。
- 実行の開始・終了時刻、エラーの有無、quota の確認結果。
```

---

## 環境2への指示（STEAM の凍結 critic の再現）

```
# 依頼 S1: STEAM の critic(視覚と言語を凍結)を、シードと学習率を変えて 2 回学習し、評価データで比べる

## 背景(3 行)
STEAM の critic(時間オフセットの予測器)で、視覚エンコーダ(SigLIP)と言語モデル(Gemma)の両方を凍結し、ヘッドだけを学習すると、2,000 step が約 2.4 時間(全体の学習は約 8.3 時間)。
評価データ(64 エピソード)で、全体の学習より、「空振り vs 保持」の AUROC が高く(0.893 対 0.798)、「成功 vs 失敗」のフレーム AUROC がやや低かった(0.871 対 0.913)。
各 1 回の学習の結果なので、(a) シードを変えると同じか、(b) 学習率の違い(2e-4 と 5e-5)の影響かを確かめる。詳細: results/steam_smolvla_libero10_task0.md の 11 節。

## 守ること
1. リポジトリのファイル、RLinf/、tmp/rlinf_steam_dev/(共有のコード)、tmp/steam_lora/(他が使用)を、編集しない。commit しない。
   書き込むのは、tmp/steam_lora_e2/ だけ(スクリプトと評価データのコピーが入っている)。学習のログは、tmp/rlinf_steam_dev/logs/steam_sft/ にも出るが、これは想定どおり(ほかのファイルを変更しない)。既存のファイルを削除・変更しない。
2. pkill / pkill -f / killall を使わない。止めるときは pgrep -af で PID を確認して kill <PID>。
3. メモリ: ノードの上限は約 23GB。学習中は、他の重い処理を並行しない。GPU は、割り当てられた 1 枚だけ。
4. ディスクの quota: 開始前に、書き込みを確認する(`head -c 20000000 /dev/zero > tmp/steam_lora_e2/quota_test.bin; ls -la tmp/steam_lora_e2/quota_test.bin` で 20000000 バイトになること)。
   `Disk quota exceeded` が出たら、止めて報告する(自分で削除しない)。確認用のファイルは、残してよい。
5. 長い処理は、必ず、切り離して実行する(nohup + &)。ツールのバックグラウンド実行には、約 2 時間の上限がある。
6. システムの変更(apt など)は、依頼者の許可を得てから。

## 準備(確認だけ)
cd /workspace/libero_RL/tmp/steam_lora_e2
ls run_variant.sh train_frozen.sh label_eval.sh eval_critics.py shadow_eval/eval_v2_m0/meta/advantages_n32e2000_eval.parquet shadow_eval/eval_v2_m0/meta/advantages_frozen2000_eval.parquet
ls /workspace/libero_RL/RLinf/.venv/bin/python   # 無ければ、止めて報告
nvidia-smi --query-gpu=memory.used,memory.total --format=csv   # GPU が空いていること
cat /sys/fs/cgroup/memory.max

## 手順 1: critic の学習(2 回、順番に。1 回 約 2.5 時間、合計 約 5 時間)
cat > run_replicates.sh <<'EOS'
#!/bin/bash
cd /workspace/libero_RL/tmp/steam_lora_e2
echo "[$(date +%T)] frz_s1 (seed 1, lr 2e-4)"
SEED=1 SAVE=1000 ./train_frozen.sh frz_s1 2000 > frz_s1.log 2>&1
echo "[$(date +%T)] frz_lr5e5 (seed 0, lr 5e-5)"
LR=5e-5 SEED=0 SAVE=1000 ./train_frozen.sh frz_lr5e5 2000 > frz_lr5e5.log 2>&1
echo replicates-done
EOS
chmod +x run_replicates.sh
nohup ./run_replicates.sh > replicates.log 2>&1 &
echo $! > replicates.pid
- 設定: 視覚と言語を凍結(学習するのは、射影・LayerNorm・ヘッドの 1.7M)、バッチ 64、2,000 step、N=32 ビン、k=32、warmup 50。frz_s1 は、シードだけが違う(学習率 2e-4)。frz_lr5e5 は、学習率だけが違う(5e-5、シード 0)。
- 1 step は、約 4.4 秒。学習のログ(ファイル frz_s1.log など)は、ログの出力がまとめて書かれるため、進行中は、空に見えることがある。進捗は、TensorBoard で見る:
  /workspace/libero_RL/RLinf/.venv/bin/python -c "
  from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
  import glob, os
  d=sorted(glob.glob('/workspace/libero_RL/tmp/rlinf_steam_dev/logs/steam_sft/*/tensorboard'), key=os.path.getmtime)[-1]
  ea=EventAccumulator(d, size_guidance={'scalars':0}); ea.Reload(); print(d, len(ea.Scalars('time/step')), 'steps')"
- 1,000 step と 2,000 step で、学習した部分だけ(約 7MB。ほかにトークナイザなど 約 40MB)のチェックポイントが保存される。

## 手順 2: 評価データのラベル付けと比較(各 critic、約 20 分 + 数分)
CK1=$(ls -d /workspace/libero_RL/tmp/rlinf_steam_dev/logs/steam_sft/steam_value_model_sft-*/frz_s1/checkpoints/global_step_2000/actor)
CK2=$(ls -d /workspace/libero_RL/tmp/rlinf_steam_dev/logs/steam_sft/steam_value_model_sft-*/frz_lr5e5/checkpoints/global_step_2000/actor)
nohup ./label_eval.sh $CK1 frz_s1_eval > label_frz_s1.log 2>&1 &     # 終わるまで待つ(label_frz_s1_eval.exit が "exit=0")
nohup ./label_eval.sh $CK2 frz_lr5e5_eval > label_frz_lr5e5.log 2>&1 &   # 1 つ目が終わってから(GPU を 1 つずつ)
# 比較(CPU のみ、数分)
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python eval_critics.py n32e2000_eval frz_s1_eval
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python eval_critics.py frozen2000_eval frz_s1_eval
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python eval_critics.py n32e2000_eval frz_lr5e5_eval
/workspace/libero_RL/RLinf/.venv-smolvla/bin/python eval_critics.py frozen2000_eval frz_lr5e5_eval
- n32e2000_eval は、全体を学習した基準の critic(2,000 step)。frozen2000_eval は、凍結の 1 回目(私の結果。シード 0、学習率 2e-4、1,000 + 1,000 step の続き)。

## 終了後の報告(10〜15 行)
- 学習の所要時間(各 2,000 step。TensorBoard の time/step の合計)、最終 50 step の損失と正解率(train/loss、train/accuracy、train/accuracy_neighbor)。
- 表: critic(基準、凍結の私の結果、frz_s1、frz_lr5e5)× 指標(エピソード AUROC の平均 Advantage と下位 10%、フレーム AUROC の成功 vs 失敗、空振り vs 保持の AUROC)。ブートストラップの差と 95% 区間(eval_critics.py の出力)。
- 結論: 「空振り vs 保持」が高く、「成功 vs 失敗のフレーム」がやや低い傾向は、シードと学習率を変えても再現したか。
- エラーの有無、quota の確認結果、実行の開始・終了時刻。
```

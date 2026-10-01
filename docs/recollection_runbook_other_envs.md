# 再収集の実行手順書（環境 1・環境 2 向け。環境 0 でも同じ手順）

このファイルは、**ロールアウト再収集を分担する各環境の担当（Claude Code）**が、そのまま実行するための手順書。
背景と設計の理由は `docs/data_recollection_plan.md` にある。**迷ったら、このファイルの手順だけに従い、勝手に変更しない。**

- 対象: LIBERO-10 Task 0 のロールアウト収集（成功で打ち切り、fps 10、**2 並列**。5 並列はメモリ不足で完走できなかった）
- 3 つの環境が `/workspace`（共有）を使い、**GPU とメモリはそれぞれ別・同一仕様**（A5000 24GB、メモリ上限 25GB）
- 状態: 手順書の初版（2026-10-01）。パッチ `smolvla-t-noise-seed.patch`（ノイズ seed）は適用済みで、seed の効き方も実測済み

## 0. 最初に確認すること（依頼者から伝えられる 3 つの値）

| 変数 | 意味 | 値の例 |
|---|---|---|
| `M` | **この環境の番号**（0、1、2）。環境ごとに必ず違う | 依頼者が指定する。**推測しない** |
| `R` | ラウンド番号（1 から） | 依頼者が指定する |
| フェーズ | `smoke`（動作確認）/ `eval` / `train` のどれか | 依頼者が指定する |

`M` が与えられていない場合は、**何も実行せず、依頼者に聞く**。同じ `M` を 2 つの環境で使うと、**同じデータが 2 つできる**。

## 1. 守ること（重要）

1. **リポジトリのファイルを編集しない・commit しない**。3 つの環境が同じ git リポジトリ（`/workspace/libero_RL`）を共有しているため、同時に commit すると競合する。**プロジェクトの `CLAUDE.md` にある「作業の区切りごとに commit する」は、この作業では適用しない**（依頼者が了承している）。commit は環境 0 だけが行う。
2. **`RLinf/` を編集しない**。`patches/apply_rlinf_patches.sh` も**実行しない**（適用済みで、共有ツリーなので全環境に反映されている）。
3. **seed を変えない**（§ 7 の表のとおり）。環境間で seed がそろうと、完全に同じデータができる。
4. **保存先を共有しない**。出力は必ず `collected_v2/<フェーズ>_r<R>/m<M>/`（環境ごとに別）に書く。
5. **データを削除しない**（旧データ `data/`、他の環境の出力を含む）。
6. 一時的なファイルは `tmp/`、**コマンドも `tmp/` をカレントにして実行する**ことを基本にする（ただし、収集コマンドは `RLinf/` で実行する必要があるため、例外とする）。
7. **システムを変更する操作（apt での導入など）は、依頼者の許可を得てから行う**。
8. 他の重い処理を、同時に動かさない。**並列数は 2 に固定**（5 並列は、全環境で、エポック 3〜4 でメモリ上限により落ちた。実メモリが約 24GB まで増え続けたため）。
9. 進捗の確認は、必要最小限にする（数十分おき。`tail` で最終行を見る程度）。

## 2. 手順 A: 事前チェック（フェーズに関係なく最初に必ず）

```bash
cd /workspace/libero_RL
bash scripts/preflight_collect_env.sh
```

- 最後が `PRE-FLIGHT PASSED` になることを確認する。`FAILED` なら、表示された `fix:` に従って直す。
- **パッチ状態の指紋（`patch-state fingerprint`）が、環境 0 の報告値と一致すること**。参考値（2026-10-01 時点）: `40a395e62d14`。一致しない場合は、**収集を始めずに**、依頼者に報告する。
- よくある不足（この環境ごとに必要なもの。共有されない）:

| 症状 | 原因 | 対処 |
|---|---|---|
| `venv interpreter` が FAIL | venv の Python 本体（`/root/.local/share/uv/python/...`）がない | `uv python install 3.12.11` |
| `LIBERO config` が FAIL | `~/.libero/config.yaml` がない | `echo n \| /workspace/libero_RL/RLinf/.venv-smolvla/bin/python -c 'import libero.libero'` |
| `EGL import` が FAIL | `libegl1`、`libopengl0` がない | **システムの変更なので、依頼者の許可を得てから** `apt-get install -y libegl1 libopengl0` |
| `Ray processes are running` | 前回の Ray が残っている（自分のものか確認する） | `ray stop --force` |

## 3. 手順 B: スモークテスト（`smoke`、依頼者の「GO」の合図で、3 環境が同時期に行う）

目的: メモリ・動作の確認と、**3 環境のデータが重複しないこと**の検証。短く（2 エポック、約 10 エピソード）実行する。

```bash
cd /workspace/libero_RL/RLinf && source .venv-smolvla/bin/activate
export HF_HUB_OFFLINE=1
export RAY_memory_usage_threshold=0.98     # 既定 0.95。5 並列の実メモリのピークは約 23.2GB で、0.95（23.75GB）の余裕が小さいため
ROOT=/workspace/libero_RL
M=<この環境の番号>
sleep $((M * 10))        # ログのディレクトリ名は秒単位。3 環境の同時起動による衝突を避ける
bash evaluations/run_eval.sh libero libero_10_smolvla_collect \
  rollout.model.model_path=$ROOT/models/smolvla_libero \
  rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
  rollout.model.smolvla.noise_seed=$((800000 + M)) \
  env.eval.seed=$((300000 + M)) \
  env.eval.data_collection.save_dir=$ROOT/collected_v2/smoke/m${M} \
  env.eval.rollout_epoch=5 env.eval.total_num_envs=2 \
  env.eval.ignore_terminations=False env.eval.data_collection.fps=10 \
  > $ROOT/tmp/collect_smoke_m${M}.log 2>&1
echo "exit=$?"
```

- バックグラウンドで実行する場合は、`nohup ... & disown` にして、ログを見る（セッションに紐づけない）。
- 約 5 分で終わる（2 並列、5 エポック）。終了後、**§ 5 の報告**を書く。
- スモークテストの結果（成功率、エピソード長、重複）は、**環境 0 が 3 環境分をまとめて検査**する。各環境は、自分の出力ができたことを報告すればよい。

## 4. 手順 C: 本番収集（`eval` または `train`）

依頼者から、フェーズ（`eval` か `train`）と、ラウンド `R` が指示された後に実行する。**スモークテストの検証が済むまで開始しない**（依頼者の「GO」を待つ）。

| フェーズ | `rollout_epoch` | `save_dir` | env の seed | ノイズの seed |
|---|---|---|---|---|
| `eval` | 13（約 26 エピソード） | `collected_v2/eval/m${M}` | `200000 + M` | `700000 + M` |
| `train` | 150（約 300 エピソード） | `collected_v2/train_r${R}/m${M}` | `100000 + 100*R + M` | `600000 + 100*R + M` |

```bash
cd /workspace/libero_RL/RLinf && source .venv-smolvla/bin/activate
export HF_HUB_OFFLINE=1
export RAY_memory_usage_threshold=0.98     # 既定 0.95。5 並列の実メモリのピークは約 23.2GB で、0.95（23.75GB）の余裕が小さいため
ROOT=/workspace/libero_RL
M=<番号>; R=<ラウンド>; PHASE=train          # eval の場合は PHASE=eval
if [ "$PHASE" = train ]; then
  EPOCHS=150; SAVE=$ROOT/collected_v2/train_r${R}/m${M}
  ENV_SEED=$((100000 + 100*R + M)); NOISE_SEED=$((600000 + 100*R + M)); TAG=train_r${R}_m${M}
else
  EPOCHS=13; SAVE=$ROOT/collected_v2/eval/m${M}
  ENV_SEED=$((200000 + M));         NOISE_SEED=$((700000 + M));         TAG=eval_m${M}
fi
[ -e "$SAVE" ] && { echo "save_dir already exists: $SAVE (do not reuse; see 'retry' below)"; exit 1; }
sleep $((M * 10))
nohup bash evaluations/run_eval.sh libero libero_10_smolvla_collect \
  rollout.model.model_path=$ROOT/models/smolvla_libero \
  rollout.model.smolvla.vlm_model_name=$ROOT/models/SmolVLM2-500M-Video-Instruct \
  rollout.model.smolvla.noise_seed=$NOISE_SEED \
  env.eval.seed=$ENV_SEED \
  env.eval.data_collection.save_dir=$SAVE \
  env.eval.rollout_epoch=$EPOCHS env.eval.total_num_envs=2 \
  env.eval.ignore_terminations=False env.eval.data_collection.fps=10 \
  > $ROOT/tmp/collect_${TAG}.log 2>&1 &
disown
```

- **1 エポック約 1 分**（1 エポック = 2 エピソード前後）。`train` の 1 ラウンド（150 エポック）で、約 2.5 時間。`eval` は約 13 分。
- 進捗: `tail -c 300 $ROOT/tmp/collect_${TAG}.log | tr '\r' '\n' | tail -2`（`Evaluating Rollout Epochs` の進捗バーが見える）。
- 終了の判定: ログの最後に評価結果の表（`success_once` など）が出て、プロセス（`eval_embodied_agent.py`）が消える。
- **再実行（retry）が必要になったとき**（落ちた、途中で止めた等）: **同じ seed で再実行しない**（途中までの分と同じデータができる）。`save_dir` の末尾を `_t2` にし、env の seed とノイズの seed の両方に `+5000` を足す（3 回目は `+10000`）。依頼者に報告する。
- **失敗した（落ちた）収集の出力は、再実行の前に、壊れたファイルを削除する**（§ 4.5）。

## 4.5 壊れたファイルの削除（落ちた収集のあと、全環境で行う）

収集が途中で落ちると（例: Ray のメモリ不足）、書きかけの parquet が 1 つ残る。変換スクリプトは、データファイルを直接読むので、**壊れたファイルを削除すれば、残りは使える**（メタデータの件数は古いままだが、使われない）。

```bash
cd /workspace/libero_RL
# 収集のプロセスが終わっていることを確認してから実行する（書きかけのファイルを消さないため）
ps -eo cmd | grep -c "[e]val_embodied_agent"        # 0 であること
RLinf/.venv-smolvla/bin/python scripts/remove_corrupt_parquets.py --dry-run $SAVE   # まず一覧だけ
RLinf/.venv-smolvla/bin/python scripts/remove_corrupt_parquets.py $SAVE             # 削除
```

- 読めないファイルだけを削除する。**完全なファイルは削除しない**。更新から 10 分未満のファイルは、収集中の可能性があるため、飛ばす（`--min-age-min` で変更できる）。
- **失敗した出力の保存先（`m0` など）は、削除せず残す**（壊れた 1 ファイルを除いた部分は、有効なデータとして使える）。

## 5. 完了報告（各フェーズの終了時に、環境ごとに書く）

`/workspace/libero_RL/collected_v2/status/<TAG>.txt` に、次の項目を書く（git 管理外）。

```bash
ROOT=/workspace/libero_RL; TAG=<smoke_m1 / eval_m1 / train_r1_m1 など>
# 先に、壊れたファイルがあれば削除する（§ 4.5）
$ROOT/RLinf/.venv-smolvla/bin/python $ROOT/scripts/remove_corrupt_parquets.py $SAVE
{
  echo "host=$(hostname)  M=$M  phase=$PHASE  R=${R:-}  finished=$(date '+%F %T')"
  echo "fingerprint=$(bash $ROOT/scripts/preflight_collect_env.sh 2>/dev/null | grep -o 'fingerprint: [0-9a-f]*')"
  echo "env_seed=$ENV_SEED noise_seed=$NOISE_SEED save_dir=$SAVE"
  echo "episodes=$($ROOT/RLinf/.venv-smolvla/bin/python $ROOT/scripts/check_collected_duplicates.py $SAVE 2>/dev/null | head -1)"
  echo "log_tail:"; tail -c 600 $ROOT/tmp/collect_${TAG}.log | tr '\r' '\n' | tail -8
} > $ROOT/collected_v2/status/${TAG}.txt
```

その後、**依頼者に、1〜3 行で報告**する（完了したこと、エピソード数、エラーの有無）。問題があれば、ログの該当箇所（エラー行の前後）を添える。

## 6. 既知のエラーと対処（今日、実際に起きたもの）

| 症状（ログ） | 原因 | 対処 |
|---|---|---|
| `EOFError: EOF when reading a line`（`libero/__init__.py` の `input(`） | `~/.libero/config.yaml` がない | § 2 の表 |
| `AttributeError: 'NoneType' object has no attribute 'eglQueryString'` | EGL ライブラリがない | § 2 の表（許可を得てから） |
| `ModuleNotFoundError: No module named 'hydra'`（`Using Python at /usr/local/bin/python`） | venv の Python 本体がなく、システムの Python が使われた | `uv python install 3.12.11` |
| `worker(s) were killed due to the node running low on memory` | 並列数が多すぎる（メモリ上限 25GB）。5 並列は、全環境でエポック 3〜4 で落ちた | **`total_num_envs=2` にする**（この手順書は 2。`RAY_memory_usage_threshold=0.98` も入れてある）。それでも落ちたら、§ 4.5 で後始末をして、`_t2`（seed + 5000）で再実行し、**私に報告する**。落ちるまでに保存されたエピソードは有効なので、削除しない |
| ログが他の環境と混ざる | 同じ秒に起動した | `sleep $((M * 10))` を入れる（手順に含めてある） |
| 同じ `save_dir` へ書いた | `M` を間違えた | **すぐに停止**し、依頼者に報告する。データは混ざっている可能性がある |
| `Connection closed by peer`（gloo） | 別のエラー（OOM など）の副次的な症状 | ログの先頭側のエラーを探す |

## 7. seed の一覧（変更しない）

| 用途 | env の seed | ノイズの seed |
|---|---|---|
| スモークテスト | `300000 + M` | `800000 + M` |
| `eval` | `200000 + M` | `700000 + M` |
| `train`（ラウンド `R`） | `100000 + 100*R + M` | `600000 + 100*R + M` |
| retry（`_t2`） | 上記 `+ 5000` | 上記 `+ 5000` |

- `env.eval.seed` は、環境側の乱数（`cfg.seed + seed_offset`）に使われる。初期状態の並びは seed に依存せず固定（seed 0）で、3 環境とも同じ順に 50 個の初期状態を回る。**軌跡の違いは、ノイズの seed で生まれる**。
- ノイズの seed は、`rollout.model.smolvla.noise_seed`（パッチ `smolvla-t-noise-seed.patch`）で、方策の flow matching のノイズを決める。**実測で確認済み（2026-10-01）**: ①同じ env seed・同じノイズ seed は、同じデータを再現する。②**env seed だけを変えても、データは変わらない**。③ノイズ seed を変えると、全て別のデータになる。したがって、**重複を防ぐのはノイズ seed**であり、`M` が違えば必ず違う値になるようにしてある。

## 8. 環境 0（取りまとめ）が行うこと（参考）

- 各フェーズの終了後、`scripts/check_collected_duplicates.py` で、**全環境の出力を横断して重複を検査**する（`m0=... m1=... m2=...`）。
- 検証が済んだラウンドを、変換する（環境ごとのデータセットに分けて、3 並列。ワーカーは 4）。**PNG は削除しない**（クォータが 600GB に引き上げられたため。ユーザーの指示があるまで、削除しない）。変換は環境 0 だけが行う。
- 方針（`docs/data_recollection_plan.md`）と、このファイルの更新、commit。

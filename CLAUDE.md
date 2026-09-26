# libero_RL

## 作業ルール

- 一時的な作業（動作確認、テスト用スクリプト、テストデータ、中間生成物など）は `libero_RL/tmp/` で行う。ファイルは `tmp/` に置き、コマンドも `tmp/` をカレントディレクトリにして実行する（`/tmp` やほかの場所は使わない）。`tmp/` は git の管理対象外。
- RLinf（submodule `RLinf/`）のファイルは直接編集しない。変更は `patches/rlinf/*.patch` にまとめ、`bash patches/apply_rlinf_patches.sh` で適用する（`--revert` で解除）。
- 作業が一区切りついたら、都度 commit まで実行する（ユーザーへの確認は不要）。commit メッセージは変更内容が分かるように書く。

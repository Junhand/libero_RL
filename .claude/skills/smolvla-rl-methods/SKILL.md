---
name: smolvla-rl-methods
description: RLinf と verl-vla に掲載されている、VLA (Vision-Language-Action) 向けの強化学習・オフライン方策最適化手法（RECAP, DSRL, STEAM, SAC-Flow, FPO, Sim-Real Co-Training, TD3+BC/SAC/CQL, IQL, OPD）を比較した調査結果を提供する。「smolVLA に使える RL 手法は？」「RECAP の次に何を試すべき？」「オンライン RL とオフラインRLのどちらが実機向き？」「学習時間が短い手法は？」「◯◯という手法は smolVLA に適用できる？」など、smolVLA（またはこのリポジトリが対象とする flow-matching 系 VLA 一般）への RL/オフライン最適化手法の適用可能性・学習パラダイム・実機対応・報告されている精度向上・原論文を尋ねられたら、必ずこのスキルを使うこと。手法名を言われていなくても「次に試すRL手法」「方策改善の選択肢」を尋ねられた場合も対象。
---

# smolVLA 向け RL/オフライン最適化手法 比較スキル

このスキルは、libero_RL リポジトリ（smolVLA で RECAP を再現する実験プロジェクト）のために行った、
RLinf と verl-vla の RL/オフライン最適化手法の横断調査結果を保持する。同じ調査をゼロからやり直す
必要がないよう、結論と根拠を再利用可能な形でまとめている。

## 使うタイミング

- 「smolVLA に使える RL 手法は？」「RECAP の次に試すなら？」のような質問
- 手法名（RECAP, DSRL, STEAM, SAC-Flow, FPO, Sim-Real Co-Training, TD3+BC/SAC/CQL, IQL, OPD）が
  会話に出てきて、その手法の性質・smolVLA との相性・実機対応・出典を答える必要があるとき
- 「オンラインRLとオフラインRLどちらが良いか」「学習時間が短い手法は」「実機で使える手法は」
  といった、手法選定の判断材料を求められたとき

## 前提: なぜ smolVLA が特別か

smolVLA は π0/π0.5 と同じ **flow matching（連続値のアクションを denoising で生成）** 方式を採る VLA。
これがどの既存手法と相性が良いかを決める最大の軸になる。「離散アクショントークン」を前提とする手法
（OPD, RL Token 系）や「状態ベースの単純な入力」しか扱わない手法（IQL for D4RL）は、smolVLA には
アーキテクチャ的に不向き。

## 結論（学習時間が短いと推定される順、smolVLA非対応を除外）

詳細な根拠・引用元・精度向上の数値は `references/comparison-table.md` を参照。要点だけ先に示す：

1. **SAC-Flow**（RLinf、2025-09） — オンライン。flow policy の velocity network を再帰的
   Transformer に置き換えて SAC で安定化。Franka 実機で30分学習の実証あり。
2. **DSRL**（RLinf/verl-vla、2025-06, CoRL 2025） — オンライン。ベース方策を凍結し、ノイズ空間に
   軽量 SAC アクターだけを追加学習。実機で 2/10→9/10 などの劇的な改善報告あり。
3. **RECAP**（RLinf/verl-vla、2025-11, Physical Intelligence "π*0.6"） — オフライン。
   **本リポジトリで既に smolVLA 対応済み**（`patches/rlinf/smolvla-recap.patch`）。rollout→
   リターン推定→価値モデル→アドバンテージ→条件付きSFT。実機での運用を前提に設計されている。
4. **STEAM**（RLinf、2026-06） — オフライン。RECAP の価値モデル部分の発展形（フレームペアの
   時間的順序学習＋最悪値アンサンブル）。実機4タスク平均 +38.1pt の報告。
5. **TD3+BC / SAC / CQL**（verl-vla） — オンライン（Critic warmup→Actor更新の反復）。
   Actor は TD3+BC、Critic は CQL、オプションで SAC エントロピー正則化。
   PI0.5 で 64%→80%、Gaussian Actor で 4%→96% の実測あり。
6. **FPO**（verl-vla、2025-07 vanilla FPO） — オンライン（PPO形式）。PPO風のクリップ目的関数を
   flow matching 方策に直接適用でき、正確な行動尤度が不要。flow-matching 前提の設計なので
   smolVLA との相性は理論上良さそう。
7. **Sim-Real Co-Training**（RLinf、2026-02 "Beyond Imitation"） — オンライン(sim)+オフライン(実機)。
   シムPPO＋実機データのSFTを同時最適化。実機成功率 OpenVLA +24%、π0.5 +20% の報告。

**除外**（smolVLA にアーキテクチャ的に不向き）：
- **IQL (D4RL)** — 画像・言語を扱わない状態ベース MLP 専用
- **OPD (OpenVLA-OFT)** — 離散アクショントークンの log-prob 比較が前提。smolVLA は連続値生成

## 使い方

1. ユーザーの質問が上記のどれかに当てはまるか確認する。
2. `references/comparison-table.md` を読み、該当する手法（または全体像が必要なら全部）の
   行を引用して回答する。表の列（公開年・フレームワーク・対応モデル・学習パラダイム・
   核心メカニズム・実機対応・smolVLAへの適用見込み・報告されている精度向上）はそのまま
   再利用してよい。
3. 出典URL（arXivリンクなど）を答えに含める。ユーザーが検証できるようにするため。
4. この調査は RLinf と verl-vla の**ドキュメント記載時点**の情報に基づく。新しいバージョンで
   記載が変わっている可能性があるので、確度が重要な場面（実装に着手する直前など）では
   `references/comparison-table.md` の出典URLを WebFetch で再確認してから回答することを勧める。

## 更新履歴

この調査は 2026-09-27 に libero_RL リポジトリのセッションで実施した。smolVLA での RECAP 再現実験
（`README.md`, `patches/rlinf/smolvla-recap.patch`）に続く「次の一手」を検討する目的で行った。
新しい手法がRLinf/verl-vlaに追加された場合や、smolVLAでの実装が進んだ場合は
`references/comparison-table.md` を更新すること。

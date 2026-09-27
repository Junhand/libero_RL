# smolVLA に適用可能な RL / オフライン最適化手法（RLinf・verl-vla 比較）

`smolvla-rl-methods` スキルの詳細参照ファイル。SKILL.md からポインタされる。

RLinf ([methods index](https://rlinf.readthedocs.io/en/latest/rst_source/examples/methods_index.html)) と
verl-vla ([reinforcement-learning index](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/index.html))
に掲載されている手法のうち、smolVLA（flow matching で連続値のアクションを生成する VLA）に
アーキテクチャ的に適用可能と考えられるものを、学習時間が短いと推定される順に並べた。

**smolVLA 非対応と判定して除外**: IQL（D4RL、状態ベース MLP 専用、画像・言語を扱わない）、
OPD（OpenVLA-OFT の離散アクショントークンの log-prob 比較が前提で、smolVLA には対応する
トークン概念がない）。

| 順位（学習時間） | 手法 | 公開年（原論文） | フレームワーク | 対応モデル | 学習パラダイム | オンポリシー/オフポリシー | 核心メカニズム | 実機対応 | smolVLAへの適用見込み | 報告されている精度向上 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **SAC-Flow** | 2025年9月（arXiv:2509.25756） | RLinf | JaxFlowTActor/FlowTActor | オンライン | オフポリシー（SACベース、リプレイバッファから学習） | velocity networkを再帰的Transformerに置き換えてSAC安定化 | ✅ Franka実機で実証（30分学習） | 🟡 velocity networkの構造適合が必要 | OGBenchで最大60%改善、HumanoidStandupでDIME比最大130%向上 |
| 2 | **DSRL** | 2025年6月（arXiv:2506.15799、CoRL 2025） | RLinf/verl-vla | π0（RLinf）／π0.5・GR00T N1.6（verl-vla） | オンライン | オフポリシー（操舵アクターをSACでオンラインリプレイから更新） | 事前学習済みflow policyを凍結し、ノイズ空間に軽量SACを被せる | ❌ RLinf文書には記載なし（論文には実機結果あり） | 🟡 smolVLAのノイズ空間APIが必要 | 実機Pick-and-Place：2/10→9/10。π0実機：5/20→18/20 等 |
| 3 | **RECAP** | 2025年11月（arXiv:2511.14759、Physical Intelligence「π*₀.₆」論文） | RLinf/verl-vla | π0.5（verl-vla版は+GR00T N1.6も対応） | オフライン | 方策勾配を使わないため区分外だが、混合データ（過去方策のrollout含む）から学ぶ点でオフポリシー的 | rollout→リターン推定→価値モデル→アドバンテージ→条件付きSFT | ✅ 実機想定設計・実証例あり | ✅ 本リポジトリで実施済み（`patches/rlinf/smolvla-recap.patch`） | LIBERO-10 Task 0：48.8%→66.5%（+17.7pt） |
| 4 | **STEAM** | 2026年6月（arXiv:2606.29834） | RLinf | π0.5（価値モデルはSigLIP+Gemma3） | オフライン | RECAPと同じ枠組み（オフポリシー的） | フレームペアの時間的順序学習＋最悪値アンサンブルでアドバンテージ推定 | 🟡 未明記（RECAPに準ずると推定） | 🟡 価値モデル部分は転用しやすいが未検証 | 実機4タスク平均+38.1pt（タオル折り+59pt、チップスチェックアウト+54.3pt、ピック&プレイス+16.2pt、コーラ補充+23pt） |
| 5 | **TD3+BC / SAC / CQL** | 各アルゴリズムは2018〜2021年発表（TD3+BC: 2021、SAC: 2018、CQL: 2020）。VLA向け組み合わせレシピ自体の単独論文は無し | verl-vla | π0.5、Gaussian Actor | オンライン（Critic warmup→Actor更新の反復） | オフポリシー（TD3/SAC/CQLいずれもリプレイバッファ＋off-policy補正が前提。正例90%/負例10%等の混在データを再利用） | TD3+BC（Actor）＋CQL（Critic）＋SACエントロピー正則化の組み合わせ | 不明 | 🟡 Gaussian Actor前提の記述もあり要確認 | PI0.5：64%→80%／Gaussian Actor：4%→96% |
| 6 | **FPO** | 2025年7月（arXiv:2507.21053、vanilla FPO） | verl-vla | π0.5 | オンライン（PPO形式） | オンポリシー（PPO由来、現在方策からの新規ロールアウトが前提） | PPO風のクリップ目的関数をflow matching方策に直接適用、正確な行動尤度不要 | 不明 | 🟢 flow-matching前提の設計で相性良好の可能性大 | MuJoCo Playground平均：667.8→759.3（+13.7%）。ヒューマノイド制御：29.8%→54.3%等（manipulationタスクの数値は後継のFPO++論文にのみ記載） |
| 7 | **Sim-Real Co-Training** | 2026年2月（arXiv:2602.12628「Beyond Imitation」） | RLinf | π0.5/OpenPi | オンライン(sim)+オフライン(実機) | 混合：sim側はPPOでオンポリシー、実機側はSFTでオフライン教師あり（区分外） | シムPPO＋実機データのSFTを同時最適化 | ✅ Franka実機で実証 | 🟡 PPO設計の追加実装が必要 | 実機成功率：OpenVLAで+24%、π0.5で+20%（4タスク平均） |

### TD3+BC / SAC / CQL の内訳（verl-vla 実装詳細）

- **TD3+BC**（Actor更新）：TD3にBehavior Cloning項を混ぜ、オフライン/少データでのQ値過大評価・分布外行動への逸脱を抑制
- **CQL**（Critic更新）：Q値に保守的な正則化を加え、データにない行動へのQ値の楽観的バイアスを防止
- **SAC/エントロピー正則化**（Gaussian Actor版のみ）：方策のエントロピーを自動調整（実験ではα=0.01→0.010585に自動収束）
- 損失重み：Actor目的はTD3+BC（BC重み0.5）、Critic目的はTD誤差+CQL（CQL重み0.5）
- リプレイサンプリング：Actor更新時は正例90%/負例10%、Critic更新時は正例50%/負例50%
- 学習の流れ：Warmup（最初400ステップ、Criticのみ更新）→本学習（2ステップごとにActor更新、TD3特有の遅延更新）→ターゲットネットワークはhard update（τ=1.0）

### 出典
- [RLinf methods index](https://rlinf.readthedocs.io/en/latest/rst_source/examples/methods_index.html)
- [verl-vla reinforcement-learning index](https://verl-vla.readthedocs.io/en/latest/reinforcement-learning/index.html)
- [Beyond Imitation: RL-Based Sim-Real Co-Training for VLA Models (arXiv:2602.12628)](https://arxiv.org/pdf/2602.12628)
- [π*₀.₆: a VLA That Learns From Experience (arXiv:2511.14759)](https://arxiv.org/html/2511.14759)
- [STEAM: Self-Supervised Temporal Ensemble Advantage Modeling (arXiv:2606.29834)](https://arxiv.org/html/2606.29834)
- [SAC Flow (arXiv:2509.25756)](https://arxiv.org/html/2509.25756)
- [DSRL: Steering Your Diffusion Policy with Latent Space RL (arXiv:2506.15799)](https://arxiv.org/html/2506.15799v2)
- [Flow matching policy gradients / vanilla FPO (arXiv:2507.21053)](https://arxiv.org/html/2507.21053)

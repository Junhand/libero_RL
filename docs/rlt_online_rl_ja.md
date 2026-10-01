# RL Token: VLA によるオンライン RL のブートストラップ（日本語訳）

> **翻訳について**
> - 原論文: *RL Token: Bootstrapping Online RL with Vision-Language-Action Models*, Charles Xu, Jost Tobias Springenberg, Michael Equi, Ali Amin, Adnan Esmail, Sergey Levine, Liyiming Ke（Physical Intelligence）, arXiv:2604.23073v2 [cs.LG], 2026-04-30
> - 原文: https://arxiv.org/abs/2604.23073 ／ プロジェクトページ: https://pi.website/research/rlt
> - **取得元について**: 依頼のあった `pi.website/download/rlt.pdf` と研究ページは、サイト側のボット確認（Vercel のチェックポイント）と 403 のため取得できなかった。そこで、同じ題名・著者の論文を arXiv（v2）から取得して訳した。pi.website の PDF と内容が完全に一致するかは確認していない。研究ページ（動画などの補足）の内容は、この翻訳に含まれていない。
> - この文書は原論文を Claude が日本語に訳したものである。**機械翻訳であり、正確な内容は原文を参照すること**。著作権は原著者に帰属する。
> - 数式は原文の記法を保った。専門用語（RL token、RLT、VLA、actor-critic など）は原語のまま用いている箇所がある。図そのものは掲載せず、キャプションのみ訳した。参考文献の番号（[1] など）は、原論文の参考文献リストに対応する。リストは末尾に原文のまま付した（PDF から機械的に抽出したため、抽出ミスが残る場合がある）。

---

**図 1: 本手法は、VLA の内部特徴から、コンパクトで意味のある表現を作るエンコーダとデコーダを学習することで、VLA に「RL token」を導入する。** 抽出した表現を使って、軽量な actor-critic ネットワークを、サンプル効率のよいオンライン RL で学習する。これにより、非常に精密なタスクを、数時間、あるいは数分のロボット経験でファインチューニングできる。（図中: データ → RL token を持つ VLA（VLA、アクションエキスパート、言語、エンコーダ、デコーダ、RL token）→ オンライン RL（actor、critic Q(s,a)、行動チャンク）→ 最終的な RL 方策。タスク: ネジの取り付け、結束バンドの固定、Ethernet と電源の挿入）

## 要旨

視覚言語行動（VLA）モデルは、多様な操作技能を「そのまま」学習できるが、実世界のタスクが要求する精度と速度を達成するには、たとえば強化学習（RL）による、さらなるファインチューニングが必要である。本論文では、事前学習済み VLA を、わずか数時間の実世界での練習で、サンプル効率よくオンライン RL でファインチューニングできる、軽量な手法を提案する。我々は、(1) VLA を適応させて「RL token」を出力できるようにする。これは、タスクに関連する事前学習済みの知識を保持しつつ、オンライン RL への効率的なインターフェースとして働く、コンパクトな読み出し表現である。(2) この RL token の上で小さな actor-critic ヘッドを学習し、学習した方策を VLA に紐づけたまま（anchoring）、行動を洗練する。RL token を用いたオンライン RL（RLT）により、大規模な VLA でも、RL で迅速かつ効率的にファインチューニングできる。4 つの実ロボットのタスク（ネジの取り付け、結束バンドの固定、充電器の挿入、Ethernet の挿入）で、RLT は、タスクの最も難しい部分の速度を最大 3 倍に改善し、数分から数時間の練習で、成功率を大きく引き上げる。一部のタスクでは、人間の遠隔操作の速度を上回ることもできる。

## I. はじめに

汎用の視覚言語行動（VLA）モデルは、データから幅広い多様な操作技能を学習できる。しかし、実行の「最後の 1 ミリ」で苦戦することが多い。動きが遅い、成功には停止とやり直しが必要になる、精密なタスクの重要な段階での小さな誤差が積み重なって失敗につながる、といったことが起きる。この課題に対処する自然な方法は、VLA を強化学習（RL）でファインチューニングすることである。対象のタスクで練習することで、RL は、成功に最も重要な段階を、まさに改善できる。そうした段階は、小さな誤差に最も敏感で、デモンストレーションだけでは確実にカバーするのが最も難しいことが多い。しかし、実世界のロボティクスには厳しい予算がある。どのエピソードにも時間がかかり、どの失敗にも労力と摩耗が伴い、意味のある適応は、数時間の練習のうちに行われなければならないことが多い。

しかし、VLA のサンプル効率のよいファインチューニングには、大きな課題がある。一方で、基盤モデルの RL 学習のための従来の手法 [1–3] は、大規模なデータに依存しており、迅速なオンライン適応には非効率なことがある。もう一方で、データ効率のよい実世界の RL 手法 [4, 5] は、通常、はるかに小さなモデルを学習するため、数時間で改善できるが、VLA の汎化能力を犠牲にしてしまう。したがって、中心的な問いは、VLA の汎化を活かしつつ、軽量なオンライン RL の速度とサンプル効率をどう実現するか、である。

我々は、事前学習済み VLA 方策から得た表現で、高速なオンライン強化学習をブートストラップする、実用的なレシピを提示する。鍵となる考えは、VLA を適応させて、サンプル効率のよいオンライン RL に使えるコンパクトなインターフェースを公開させることである。これは、VLA が **RL token** を出力するように学習することで達成する。RL token は、タスクに関連する事前学習済みの知識を、軽量なオンライン RL 方策が利用できるようにする、圧縮された表現である。この RL token を使って RL を実行する（RLT）ことで、単純な役割分担が生まれる。凍結した VLA が、広い知覚的理解と行動の推薦を提供し、軽量な actor と critic が、タスクの最も難しい部分で成功するように、方策をオンラインで適応させる。これをサンプル効率のよい実世界の条件で実用的にするため、本手法は、RL token の表現を使う小さな actor と critic のネットワークを学習するのに、サンプル効率のよいオンライン RL アルゴリズムを使う。さらに、actor を VLA の行動に紐づける追加の正則化項を加え、オンライン RL が、有望な挙動を、ゼロから学習するのではなく、洗練するようにする。

我々は、ミリメートルまたはサブミリメートルの精度を要しうる、4 つの難しいロボット操作タスクで RLT を評価する。ネジの取り付け、結束バンドの固定、Ethernet の挿入、充電器の挿入である。これらのタスクで、RLT は、数時間のオンライン学習で、成功率と実行速度の両方を改善する。最も大きな改善は、高い精度を要し、タスクの成否を決める、タスクの重要な局面に現れる。そこでは、RLT は実行を最大 3 倍に高速化し、成功率を大きく改善する。たとえば、難しいネジ挿入のタスクでは、20% から 65% になる。我々のタスクの中で最も器用さを要する部分の 1 つでは、本手法で学習した方策は、信頼性を保ったまま、専門家の遠隔操作の速度を上回ることができる。これらの結果は、VLA モデルと軽量なオンライン RL の組み合わせが、大規模なタスク固有のエンジニアリングなしに、高性能な操作を実現する、実用的な道筋を与えることを示唆している。

## II. 関連研究

**視覚言語行動モデル。** 大規模なデモンストレーションデータセットからの挙動クローニングは、最近、汎用ロボット操作方策を学習するための支配的なパラダイムになっている（たとえば [6–11] を参照）。この成功を可能にした 2 つの重要な要素は、複数の行動を、順次オープンループで実行するために予測する行動チャンキング [12] と、デモンストレーションデータに内在するマルチモダリティを捉えられる、拡散 [13] や自己回帰生成 [6] のような、表現力の高い出力分布の使用である。さらなる進歩は、言語条件付きの汎用方策のバックボーンとして、大規模な事前学習済み視覚言語モデルを使うことから生まれ、視覚言語行動（VLA）モデル [6, 7] が得られた。これらのモデルは、ウェブ規模の大きな事前知識を、閉ループのロボット方策に取り込む。最近の研究は、VLA のバックボーンを、拡散 [8] または自己回帰トークン化 [14, 15] によるチャンク化された行動生成と組み合わせ、最先端の汎用操作を達成している。これらの方策は印象的な汎化能力を示す [9, 16] が、あるタスクでの性能は、最終的には、学習に使った遠隔操作データの品質とカバレッジによって制約される。デモンストレーション自体にノイズや不整合がある場合、精密さが重要なタスクで確実な成功を達成するのは、難しいままである。

**実世界の強化学習。** 強化学習は、デモンストレーションデータの性能の上限を超える、自然な方法を提供する。タスクで練習することで、エージェントは、デモされたことのない、より速く、より精密で、より頑健な戦略を発見できる。実際には、ロボットの実世界 RL は厳しいサンプル予算のもとで動作する。ロボットのロールアウトはどれも、時間と摩耗を伴うためである。オフポリシーの actor-critic 手法（たとえば [17–20]）は、リプレイバッファに蓄えた遷移を再利用することでこれに対処する。更新対データの比 [21] を上げることで、サンプル効率をさらに高められるが、不安定さを避けるために正則化が必要になることがある [22]。重要なことに、オフポリシー手法は、人間のデモンストレーションデータを取り込んで学習をブートストラップすることもでき（たとえば [23]）、模倣と RL の長所を組み合わせられる。実ロボットで RL を使うための実用的なレシピが、数多く開発されている。自律的なデータ収集のパイプライン [24]、SERL [4, 25] や RL100 [5] のような効率的な学習フレームワーク、オペレータが介入して、自律実行中に修正を与えられる、ヒューマン・イン・ザ・ループの変種 [4] などである。これらのシステムは、オフポリシーの actor-critic 手法を、デモンストレーションと人間の修正と組み合わせることで、接触の多い操作タスクを、数時間のロボット時間で解けることを示してきた。しかし、これらは通常、標準的な事前学習済みの視覚エンコーダ（たとえば ResNet）の上で、小さな方策をゼロから学習するため、現代の VLA モデルが持つ豊かな挙動の事前知識を使っていない。RLT は、凍結した VLA を、軽量なオンライン RL 方策の知覚バックボーンと挙動の事前知識の両方として使うことで、このギャップを埋める。

**VLA モデルの RL ファインチューニング。** 事前学習済み VLA を RL でどう改善するかを研究する、急速に増えている一連の研究がある。これらのアプローチは、主に、何を更新するか、RL の信号をどう取り込むかで異なる。一方の端では、いくつかの手法が VLA モデル全体を更新する。RECAP [3] は、advantage-conditioned な方策抽出を使ったオフライン RL で、π\*₀.₆ モデル全体を end-to-end で学習する。分布型の価値関数が、各時刻の advantage を推定し、VLA は、収集した全てのデータ（デモンストレーション、自律的なロールアウト、人間の介入）で、高 advantage の行動の重みを上げる最適性指標とともに学習される。ロボット上でのデータ収集と、オフライン RL の更新を反復することで、RECAP は、エスプレッソ作り、洗濯物たたみ、箱の組み立てのような、複雑な長期ホライズンのタスクで、スループットを 2 倍以上にする。他の研究は、近接方策最適化（PPO）やその変種を VLA のファインチューニングに適用している（たとえば [1, 26, 27]）が、オンポリシー手法は、サンプル効率がよくスケーラブルな形で、実世界の RL に拡張するのが難しい。もう一方の端では、軽量な手法が、VLA 全体を更新するのを避け、凍結したモデルの上に、小さな補助モジュールを学習する。ConRFT [28] は、VLA のエンコーダを凍結し、一貫性（consistency）ベースの学習目的関数と、学習した二値の報酬分類器を使って、アクションヘッドをファインチューニングするが、短いホライズンのタスクで、チャンク化しない 1 ステップの行動で動作する。Policy Decorator [29] は、出力を手動調整のハイパーパラメータでスケーリングして、凍結した VLA の予測に加える、残差方策を学習するが、シミュレーションでのみ実証されており、必要なサンプル数も多い（数百万ステップのオーダー）。Probe-Learn-Distill（PLD）[30] は、ベース方策のロールアウトで、Cal-QL [31] を使って critic を事前学習し、その後、凍結した VLA の上に、1 ステップの残差方策を学習し、必要に応じて、教師ありファインチューニングで結果を VLA に蒸留し戻す。GR-RL [2] は、汎用 VLA を、長期ホライズンの靴紐結びのタスクに特化させるために、多段階のアプローチを取る。まず、オフラインのフィルタ付き BC を行い、次に、凍結した VLA の拡散過程を潜在空間で操舵するノイズ予測器 [32] を学習することで、オンライン RL を行う。DSRL [32] も同様に、拡散のノイズ空間で動作し、行動を高 return の領域へ向けて操舵するように、ノイズ除去の過程を調整する潜在方策を学習する。

RLT は、これらの手法と、VLA 全体の RL のコストをかけずに、事前学習済み VLA を改善するという目標を共有しているが、設計上のいくつかの重要な選択で異なる。第一に、RLT は RL token を導入する。これは、VLA の内部埋め込みを圧縮するように学習されたコンパクトな読み出し表現で、軽量な actor-critic の状態観測として働く。VLA の事前学習済みの知覚構造を保ちながら、効率的なオンライン学習を可能にする。第二に、RLT は、VLA 本来の行動インターフェースに合わせた、チャンク化された行動の上で動作する。これにより、高い制御周波数での疎な報酬のもとで、時間差分学習の実効的な意思決定ホライズンが短くなる。これは、はるかに長いクレジット割り当て問題に直面する、1 ステップの手法 [28–30] とは対照的である。第三に、残差やノイズの潜在変数を予測する代わりに、RLT の actor は、VLA がサンプリングした参照行動チャンクを直接条件とし、それに向けて正則化される。これにより、オンライン RL は、制約のない探索や、拡散過程の暗黙的な調整ではなく、VLA の良い事前の挙動方策の局所的な洗練になる。これらの選択が合わさって、実ロボットでの、サンプル効率のよいオンライン RL を可能にし、数時間の練習で、成功率と実行速度の両方を改善する。

## III. 準備

**視覚言語行動モデル。** 大規模な VLA モデルは、数万時間にわたる多様な人間のデモンストレーションデータセットから、操作の挙動を学習する。場合によっては、ロボット以外の視覚言語データで補強される [7, 9, 16]。典型的な VLA は、2 つの要素からなる。(i) VLM バックボーン、すなわち、マルチモーダルな入力（画像、言語指示、固有受容感覚の状態）を、共有のトークン列に符号化する視覚言語モデル。(ii) アクションエキスパート、すなわち、バックボーンのトークンを参照し、反復的なノイズ除去で連続行動を生成する、拡散ベースのモジュール。我々は π₀.₆ モデル [33] の上に構築する。最大 4 台のカメラ画像、言語指示 $\ell$、固有受容感覚の状態 $s^p_t$ が与えられると、π₀.₆ は、行動チャンクと呼ばれる行動列 $\tilde{a}_{t:t+H-1} = (\tilde{a}_t, \ldots, \tilde{a}_{t+H-1}) \in \mathbb{R}^{H \times d}$ を生成する。これは、制御の 1 秒に相当する、$H = 50$ 個の行動の列である。事前学習済み VLA が生成するチャンク化方策を $\pi_\text{vla}$ と書く。実際には、ロボットは、新しい観測から再計画する前に、このチャンクの先頭部分だけ（たとえば最初の 20 ステップ）を、オープンループで実行する。一部のタスク（たとえば高精度のタスク）の難しさのために、それらについて、大量の高品質な模倣学習データを大規模に収集するのは難しく、VLA のこれらのタスクでの性能が制限される。このことが、次節で開発するオンライン RL による洗練手法の動機となる。

**強化学習と actor-critic 手法。** ロボット制御を、マルコフ決定過程（MDP）$(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ として定式化する。ここで $\mathcal{S}$ は状態観測空間、$\mathcal{A}$ は連続行動空間、$p(s_{t+1}|s_t, a_t)$ は遷移ダイナミクス、$r(s_t, a_t)$ は報酬関数、$\gamma \in [0, 1)$ は割引率である。RL の目標は、期待割引 return $J(\pi) = \mathbb{E}_{\tau \sim \rho_\pi}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$ を最大化する方策 $\pi(a_t|s_t)$ を学習することである。ここで $\rho_\pi(\tau)$ は、方策 $\pi$ から導かれる軌跡の分布を表す。我々は、疎な二値の報酬のみにアクセスできると仮定する。人間の監督者が、各エピソードの終わりに、成功か失敗かをラベル付けし、成功なら $r_T = 1$、そうでなければ $r_T = 0$ とする。方策 $\pi$ の行動価値関数は、$Q^\pi(s_t, a_t) = \mathbb{E}_{\tau \sim \rho_\pi}\left[\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \,\middle|\, s_t, a_t\right]$ である。

本設定では、方策も critic も、行動チャンク $a_{t:t+C-1} = (a_t, \ldots, a_{t+C-1}) \in \mathbb{R}^{C \times d}$ の上で動作する。ここで $C$ は RL のチャンク長を表す（$H$ は VLA が予測するチャンクのホライズンを表す）。方策がより反応的になれるように、$C < H$ を選ぶ。チャンク化方策を $\pi(a_{t:t+C-1}|s_t)$ と定義し、対応するチャンク単位の $C$ ステップの価値推定を $Q^\pi(s_t, a_{t:t+C-1}) = \sum_{t'=t}^{t+C-1} \gamma^{t'-t} r_{t'} + \gamma^C \mathbb{E}_{a' \sim \pi|s_{t+C}}\left[Q^\pi(s_{t+C}, a')\right]$ と定義する。確率的な actor $\pi_\theta$ と critic $Q_\psi$ を共同で学習する、古典的なオフポリシーの actor-critic 手法 [17, 19, 34] の上に構築する。重要なのは、学習がオフポリシーであり、どの方策が生成したかにかかわらず、リプレイバッファ $\mathcal{B}$ に蓄えた遷移を使うことである。この性質は、$\mathcal{B}$ が、VLA 方策、RL の学習者、人間の遠隔操作による介入からのデータを集約する、我々の設定では不可欠である。

## IV. RL token からの強化学習

図 1 は、RLT によって、事前学習済み VLA モデルからの、高速で安定したオンライン RL を可能にする、我々のレシピをまとめている。中核となる考えは、事前学習済み VLA を最大限に活用して、RL 学習の過程の効率を高めることである。VLA 全体をオンライン RL で学習するのは、計算量とサンプル効率の面で、わずか数時間で改善された方策を得るには、不十分かもしれない。代わりに、凍結した VLA を使って、RL の状態表現を与え、参照行動を供給し、探索を VLA 自身の予測に近い行動へ導きつつ、小さな actor と critic のネットワークを使う。まず、少量のタスク固有のデモンストレーションデータで VLA を適応させる。これは、初期のタスク方策を改善するためであり、かつ、後続の RL のために RL token を公開するためでもある。次に、VLA を凍結し、RL token の表現と VLA の参照行動の両方を条件として、軽量なオフポリシーの actor と critic のネットワークを、オンラインで学習する。そして、学習した方策が VLA モデルに近い状態を保つよう、正則化する。本手法は、オンライン RL を、制約のない探索ではなく、有望な挙動の局所的な洗練にする。この設計により、オンライン RL 手法は、小さな actor-critic アルゴリズムの効率を持ちつつ、事前学習済み VLA モデルの表現と挙動を保持できる。

### A. VLA を適応させて RL インターフェースを公開する

サンプル効率のよいオンライン RL は、状態表現の選択に決定的に依存する。VLA モデル全体に直接 RL を適用するのは、迅速な実世界での適応には不向きである。表現が高次元であり、10 億パラメータ規模のモデルをオンラインで更新するのは、計算コストが高く、サンプル効率も悪いからである。同時に、事前学習後の VLA がすでに内部に持つ表現を活用したい。それは大規模なウェブデータとロボットデータで学習されており、多くのタスクの行動を生成するのに有用な情報をすでに含んでいるからである。しかし、transformer ベースの VLA のどの特徴が、オンライン RL にとって良い表現になるのかは、一般には明らかでなく、各 transformer 層の埋め込みは高次元である。そこで我々の目標は、VLA の表現を、RL のためのコンパクトな埋め込みに圧縮することである。この埋め込みは、タスクに関連する情報を保持しつつ、軽量なオンライン actor-critic 学習に十分なほど小さい。

これを、RL token（図 2）を加えることで実現する。RL token は、VLA の知識を、RL の状態として働く小さなベクトルに要約する、学習された読み出し埋め込みである。具体的には、事前学習済み VLA に追加する小さな transformer から、RL token を得る。この transformer は、エンコーダ・デコーダ [35] の形で学習し、エンコーダの最後の入力を RL token とする。RL token の表現は、デコーダが入力を再構成できるだけの情報を保持しなければならないので、ボトルネックとして働く。$z = f(s, \ell; \theta_\text{vla})$ を、状態 $s$ と言語指示 $\ell$ に対して、事前学習済み VLA が生成する最終層のトークン埋め込みとする。埋め込み $z$ は、$z_{1:M} = \{z_1, \ldots, z_M\}$ に分解され、各 $z_i$ は 1 つの入力トークンの埋め込みに対応する。学習された埋め込み $e_\text{rl} = e_\phi(\texttt{<rl>})$ を系列に追加し、拡張した系列を軽量なエンコーダ transformer $g_\phi$ で処理する。特殊トークンの位置でのエンコーダの出力を $z_\text{rl}$ と表し、これが我々の RL token である<sup>1</sup>:

$$z_\text{rl} = g_\phi\!\left([z_{1:M}, e_\text{rl}]\right)_{M+1}. \tag{1}$$

次に、線形の出力射影 $h_\phi$ を持つデコーダ transformer $d_\phi$ を、$z_\text{rl}$ から元の埋め込みを自己回帰的に再構成するように学習する。VLA の埋め込みに適用する stop-gradient 演算を $\bar{z}_i = \text{sg}(z_i)$ とすると、デモンストレーション $\mathcal{D}$ に対する自己回帰的な再構成の目的関数は次のとおりである:

$$\mathcal{L}_\text{ro} = \mathbb{E}_{\mathcal{D}}\left[\sum_{i=1}^{M} \left\| h_\phi\!\left(d_\phi([z_\text{rl}, \bar{z}_{1:i-1}])\right)_i - \bar{z}_i \right\|^2\right]. \tag{2}$$

パラメータ $\phi$ を、小さなタスク固有のデモンストレーションデータセットで学習する。このとき VLA は、$\mathcal{L}_\text{ro}$ に関して凍結されているものとみなす。また、（任意で）VLA（$\theta_\text{vla}$）の教師ありファインチューニングと組み合わせる。その後、$\theta_\text{vla}$ と $\phi$ の両方を凍結し、オンライン RL は RL token の表現 $z_\text{rl}$ の上で動作する。

**図 2: RL token の抽出の詳細。** RLT は、事前学習済み VLA に、エンコーダ・デコーダ型の transformer を追加する。これは、VLA の表現の圧縮された埋め込み（RL token）を生成する。この表現により、オンライン RL 中の、データとパラメータの効率のよいファインチューニングが可能になる。（図中: π₀.₆ VLA（事前学習済み VLM: SigLIP (400M) + Gemma (4B)、アクションエキスパート (860M)）、画像埋め込み N×2048、エンコーダ transformer、RL token 1×2048、デコーダ transformer、行動チャンク）

<sup>1</sup> 我々の実験では、各タスクに固定の言語指示があるため、この段階では言語埋め込みを省く。この構成は、一般には、全ての VLA の埋め込みに適用できる。

### B. オンライン RL による VLA の行動チャンクの洗練

最初の適応の段階の後、VLA と RL token の表現の両方を凍結する。その後、軽量な actor（$\pi_\theta$）と critic（$Q_\psi$）のネットワークを、オンラインで学習する。それらの入力 $x$ は、RL token と、閉ループ制御を可能にするのに有用な追加情報（たとえば、ロボットの固有受容感覚の状態）を組み合わせたものである。critic モデルは、状態と行動の価値を推定する: $Q_\psi(x, a_{1:C}) \in \mathbb{R}$。特に、行動をゼロから生成する代わりに、RL の actor $\pi_\theta(\cdot|x, \tilde{a}_{1:C})$ は、VLA が提案する行動列 $\tilde{a}_{1:C}$（行動チャンクと呼ぶ）を洗練するように学習される。

**critic の学習。** 我々の critic $Q_\psi(x, a_{1:C})$ は、状態と行動チャンク $a_{1:C}$ を入力とする。リプレイバッファ $\mathcal{B}$ からサンプリングした、行動チャンクの遷移に対して、標準的なオフポリシーの時間差分学習で critic を学習する:

$$\mathcal{L}_Q = \mathbb{E}_{(x, a_{1:C}, x') \sim \mathcal{B}}\left[\left(\hat{Q} - Q_\psi(x, a_{1:C})\right)^2\right], \qquad \hat{Q} = \sum_{t'=1}^{C} \gamma^{t'-1} r_{t'} + \gamma^C\, \mathbb{E}_{a' \sim \pi_\theta}\!\left[Q_{\psi'}(x', a')\right]. \tag{3}$$

ここで入力状態は $x = (z_\text{rl}, s^p)$ で、$s^p$ は固有受容感覚の状態情報、$z_\text{rl}(s)$ は状態 $s$ について抽出した RL token を表す。$x'$ は次の入力状態を表し、$a' \sim \pi_\theta$ は、RL 方策からのサンプルを取ることを表す。実際には TD3 [19] に従い、$\psi'$ はターゲットネットワークのパラメータである。

**RL 方策の学習。** actor ネットワーク $\pi_\theta(\cdot|x, \tilde{a}_{1:C})$ は、行動チャンク上のガウス型の行動分布を生成する。入力状態と参照行動チャンク $\tilde{a}_{1:C}$ を受け取り、次の行動分布を生成する:

$$\pi_\theta\!\left(a_{1:C}|x, \tilde{a}_{1:C}\right) = \mathcal{N}\!\left(\mu_\theta(x, \tilde{a}_{1:C}),\, \sigma^2 I\right), \tag{4}$$

ここで、前と同様に $x = (z_\text{rl}, s^p)$ である。$\tilde{a}$ を条件とすることで、actor は VLA の予測した行動に直接さらされ、オンライン RL は、ゼロから学習するのではなく、強い初期提案を洗練する。第二の利点は、サンプリングした参照チャンクが、VLA のマルチモーダルな行動分布のモード情報を保持することである。これは、単峰のガウス型 actor では、復元するのが難しいものである [36]。さらに、行動を参照行動に向けて正則化することで、学習を安定させる。具体的には、critic の価値を最大化しつつ、VLA の参照チャンク $\tilde{a}$ に近い状態を保つように、actor を最適化する。これは、KL 正則化 RL 手法（たとえば [20, 37–40] を参照）と、精神として似ている。これは、オンライン RL を、高次元の行動チャンク上の制約のない探索ではなく、VLA が生成する行動分布の周りでの、局所的な行動の編集にする。RL 方策を学習するための目的関数は、次のとおりである:

$$\mathcal{L}_\pi(\theta) = \mathbb{E}_{\substack{s \sim \mathcal{B} \\ a_{1:C} \sim \pi_\theta}}\left[-Q_\psi(x, a_{1:C}) + \beta \left\| a_{1:C} - \tilde{a}_{1:C} \right\|_2^2\right], \qquad \tilde{a}_{1:C} \sim \pi_\text{vla}(\cdot | s, \ell), \tag{5}$$

ここで係数 $\beta$ は、サンプリングした VLA の行動に向けて actor をどれだけ強く正則化するかを制御する。

**参照行動のドロップアウト。** 参照行動による条件付けの、実用上の失敗モードは、actor が $\tilde{a}$ を改善することを学習せず、単にコピーしてしまうことである。これは、critic がまだ有益な信号を出す前に特に起きやすい。$\tilde{a}$ による条件付けと、それに向けた正則化の両方が、actor を VLA の提案の近くに留めようとするためである。これを防ぐため、**参照行動のドロップアウト**を適用する。各学習バッチの、ランダムな一部の遷移について、参照チャンクを、actor に渡す前にゼロで置き換える。これにより、actor は独立した行動生成の経路を維持せざるを得なくなり、かつ、参照チャンクがあるときは、VLA の行動分布を活用できる。実際には、critic が有用な信号を与えるようになると、actor は、予測価値が上がるときはいつでも、自然に参照から逸脱することを学習する。

## V. システム全体

アルゴリズム 1 に、学習ループ全体をまとめる。ベース VLA 方策でエピソードを収集する最初のウォームアップの段階の後、学習は、ロボット上での経験の収集と、リプレイからのオフポリシー actor-critic の更新を交互に行う。リプレイバッファは、VLA のウォームアップのデータ、オンライン RL のロールアウト、任意の人間の介入を集約する。さらに、人間の監督者が、疎な成功/失敗のラベルを与える。各ステップを以下に詳述する。

**ウォームアップ。** RL token の表現を学習した後（IV-A 節）、VLA の参照方策を $N_\text{warm}$ 環境ステップ分ロールアウトして、リプレイバッファ $\mathcal{B}$ を事前に埋める。これにより、critic に初期の学習信号を与え、オンライン RL が、有能な VLA の挙動から始まることを保証する。

**ロールアウト。** オンライン収集中の、各行動チャンクの境界で、凍結した VLA が参照チャンク $\tilde{a}_{1:H}$ を生成し、RL token モジュールが $z_\text{rl}$ を抽出する。次に actor が、行動チャンク $a_{1:C} \sim \pi_\theta(\cdot | x, \tilde{a}_{1:C})$ を出力する。接触の多い、または安全上重要な挙動の学習を加速するため、人間のオペレータが、介入の間、actor の出力を上書きする遠隔操作コマンド $a^h_{1:C}$ を与えて、任意で介入できる。これが起きると、介入が、リプレイバッファ内の VLA の参照を置き換える。いずれの場合も、$\mathcal{B}$ に保存する各遷移は、実行した行動と、対応する参照を含み、actor が、自律的なロールアウトと人間の修正の両方から学習できるようにする。

**行動チャンクのサブサンプリング。** RL 方策は長さ $C$ の行動チャンクを使うが、観測は途中の全てのステップで得られる。したがって、途中のステップもリプレイバッファに保存することで、データを増やし、学習効率を高められる。具体的には、ストライド 2 を選び、$\langle x_0, a_{0:C} \rangle$、$\langle x_2, a_{2:C+2} \rangle$、$\langle x_4, a_{4:C+4} \rangle$、… に対応する遷移を、リプレイバッファに保存する。我々のRL アルゴリズムはオフポリシーなので、全ての行動チャンク（VLA が生成した行動と、人間の介入を含む）を使えることに注意されたい。

**アルゴリズム 1: RLT**

```
Require: 凍結した VLA バックボーン f_θvla と、VLA の行動分布 π_vla; デモデータ D、
         チャンク長 C、リプレイバッファ B、ウォームアップステップ N_warm、
         比 G、VLA のファインチューニングの重み α、方策の制約 β。
 1: RL token を学習し、（任意で）VLA をファインチューニングする
 2:   z_i = f_i(s, ℓ, θ_vla)、z_rl = g_φ([z_{1:M}, e_rl])_{M+1}、
      および θ_vla（α > 0 のときのみ）を用いて φ を学習する。
        L_ro(φ) = E_D[ Σ_{i=1}^{M} ‖ h_φ(d_φ([z_rl, z̄_{1:i-1}]))_i − z̄_i ‖² ]
 3:   φ, θ_vla = argmin_{φ,θ_vla} L_ro(φ) + α L_vla(θ_vla)
 4: RL の actor と critic を学習する
 5:   critic Q_ψ と RL 方策 π_θ を初期化する。
 6: for 環境ステップ t = 0, C, 2C, … do
 7:   VLA の参照チャンク ã_{t:t+C-1} ~ π_vla(s_t) をサンプリングする。
 8:   RL の状態 x_t = (z_rl(s_t), s^p_t) を作る。
 9:   a_{t:t+C-1} ←  a_human                    （介入があるとき）
                      ã_{t:t+C-1}                 （t < N_warm のとき）
                      ~ π_θ(· | x_t, ã)           （それ以外）
10:   a_{t:t+C-1} を実行し、r_t、s_{t+1}、s^p_{t+1} を観測する。
11:   ã_{t:t+C-1} ← a_human   （介入があるとき）
12:   遷移を B に保存する: ⟨x_t, a_{t:t+C-1}, ã, r_t, x_{t+1}⟩
13:   for g = 1, …, G do
14:     データのバッチ b ~ B をサンプリングする。
15:     ターゲットの Q 値を計算する:
          Q̂ = Σ_{t'=1}^{C} γ^{t'-1} r_{t'} + γ^C E_{a'~π_θ}[ Q_ψ'(x', a') ]
16:     TD バックアップで critic を学習する（式 (3)）:
          L_Q(ψ) = E_b[ (Q̂ − Q_ψ(x, a))² ]
17:     方策 a ~ π_θ(· | s, ã) を学習する（式 (5)）:
          L_π(θ) = E_b[ −Q_ψ(x, a) + β‖a − ã‖²₂ ]
18:   end for
19: end for
```

**更新。** 方策の更新は、アルゴリズム 1 に従い、リプレイバッファからオフポリシーで行う。学習中の計算と時間の効率を保つため、ロールアウトと学習を非同期に行う。実際には、actor の更新 1 回につき、critic を 2 回更新し、ウォームアップの段階の直後に学習を開始する。**更新対データの比（UTD）は 5** という高い値を使う。これは、データの少ないオンラインの領域では不可欠である。

**図 3: 実験のタスク。** 各タスクに、高い精度を要する重要な局面がある。（上）ドライバーでネジを取り付ける、（中）結束バンドを固定する、（下）Ethernet ケーブルを差し込む、および充電器を差し込む。

**タスクの重要な局面の、的を絞った改善。** 実用性と学習の効率のため、検討する各タスクの重要な局面、すなわち、高い精度を要する最も難しい区間に RLT を適用し、より易しい部分は、ベース VLA に実行させる。具体的には、各エピソードはベースモデルの実行から始まる。データ収集中に、人間のオペレータが、ベース VLA から RL 方策へ制御を渡す時点を選べる。これは、対話型の模倣学習 [41] における、人間の介入の判断に類似している。その後、本システムは、選ばれたタスクの区間に RL を適用し、この重要な局面の間の遷移を保存して学習する。これは、人間のオペレータから、RL タスクの成功または失敗を示す終了信号を受け取るまで続く。これにより、データ収集とクレジット割り当てが、オンラインの適応が最も重要な、挙動の部分に集中する。テスト時の自律実行を可能にするため、学習の最後に、VLA の短いファインチューニングの段階を設け、そこで VLA に、RL 方策へ実行を渡すべき時点も予測させる（人間の介入をラベルとして使う）。すると、テスト時に、方策の切り替えを自動でトリガーできる。

## VI. 実世界での実験

精密なサブミリメートルの精度と、器用な制御を要する、4 つの実世界の操作タスクで RLT を評価する。事前学習済みの VLA は、これらのタスクの大部分に強い初期化を与えるが、成功と速度は、最終的には、最も高い精度を要する、接触の多い重要な局面を洗練できるかにかかっている。我々の実験は、本手法の動機となる実用上の制約のもとで、すなわち、限られたロボットとの対話時間、疎な人間の監督、軽量なオンライン学習のもとで、本手法がそのような改善をもたらせるかを検証する。

評価は、次の問いに沿って構成する。

- **Q1.** RLT は、ベースの VLA モデルより、操作の性能を改善できるか？
- **Q2.** RLT は、これらのタスクで、他の RL アプローチと比べてどうか？
- **Q3.** 本手法の各構成要素（RL token、チャンク化した行動予測、方策の正則化、参照行動の通過）は、手法の性能にどれだけ寄与するか？
- **Q4.** RLT により、方策はより良い戦略を発見できるか？ その戦略は、元のデモンストレーションデータと比べてどうか？

### A. タスクとセットアップ

次のタスクで本手法を評価する（図 3）。

- **ネジの取り付け。** ロボットは、電動ドライバーを使って、M3 ネジをねじ穴付きの受け口に締め込む。ネジの頭とドライバーの先端の、サブミリメートルの位置合わせを要する。このタスクが特に難しいのは、(1) ネジが常に完全に直立しているとは限らない、(2) ドライバーを保持しているとき、エンドエフェクタの回転が、ドライバーの先端と把持点の間の 10 cm の距離で増幅される、(3) 重要な視覚的手がかりが、主に反対側のアームの広角の手首カメラからしか見えず、知覚が難しい問題になる、ためである。
- **結束バンドの固定。** ロボットは、結束バンドの端を、狭いロック用のスロットに通さなければならない。このタスクは、変形する物体の、厳しい公差での、双腕の協調制御を含む。挿入の成功には、先端とスロットの位置を、手首カメラだけから推定し、ミリメートルの精度で実行する必要がある。
- **Ethernet の挿入。** ロボットは、Ethernet コネクタを、奥まったポートに挿入しなければならない。正確な位置と角度の位置合わせと、それに続く、しっかりとした決定的な挿入動作が必要である。小さな向きの誤差や、ためらいがちな接触は、通常、コネクタがポートに入らず、筐体に引っかかる原因となるため、成功は、精度と接触のダイナミクスの両方に敏感である。
- **充電器の挿入。** ロボットは、充電器を、電源タップに位置合わせして挿入しなければならない。このタスクが難しいのは、方策が、センチメートルレベルの位置合わせを達成しなければならないのに、端子とソケットを常に明確に観測できるとは限らないためである。位置合わせの小さな誤差が、繰り返しの探り動作や、挿入の失敗につながることが多い。

各タスクは、把持、再配置、位置合わせを含み、30〜120 秒（50 Hz で約 1500〜6000 制御ステップ）に及ぶ。各タスクについて、**重要な局面**、すなわち、挿入、固定、回転の区間を特定する。そこは、精度の要件が最も高く、ベース VLA が最も頻繁に減速または失敗する部分である。これらの局面は、通常、5〜20 秒（250〜1000 制御ステップ）続く。

**重要な局面の評価。** 本手法は、まさにこれらの重要な局面を改善するよう設計されているので、まず、重要な局面だけで、手法とアブレーションを比較することに評価の焦点を当てる。この設定では、エピソードは、重要な局面の直前の、部分的に完了したタスクの状態にリセットされた後に始まり、わずかにランダム化された初期構成の集合を使う。たとえば、結束バンドの固定では、ロボットは、挿入を試みる前に、すでに結束バンドの 2 つの端を持っている状態から始まる。この設定は、RL が最も重要になると期待される、精度が重要な区間を切り分け、把持や運搬のような、ベース VLA がすでに適度にこなせる、タスクの前の局面からの、交絡する分散を減らす。この制御された設定では、各エージェントを、タスクあたり 50 エピソードで評価する。

**タスク全体の評価。** 制御された重要な局面の評価は、我々の手法が改善するよう設計されたボトルネックを切り分けるのに有用だが、長期ホライズンの実行の変動の全体は捉えない。そのため、より現実的な設定でのタスク全体の性能も評価する。そこでは、ロボットは「ホームポジション」から始め、タスクの前の段階をベース方策で実行し、その実行によって生じる状態の変動のもとで、重要な局面に入る。この設定は、RL で改善された挙動が、前の方策が生み出す、より広い状態分布のもとで有効であり続けなければならないため、はるかに難しい。タスク全体の学習では、まず、小さなランダム化のもとで、RL を重要な局面に集中させ、その後、タスク全体の設定に移る。

**実験の詳細。** RL 方策の入力は、RL token（2 つの手首カメラ画像と 1 つのベースカメラ画像から生成）と、追加の固有受容感覚の状態からなる。タスクによって、この補助的な状態には、関節位置（ネジ）や、エンドエフェクタの姿勢（結束バンド、Ethernet、充電器）が含まれる。ベース VLA 方策として π₀.₆ [33] を使う。ロボットは 50 Hz の制御周波数で動く。1 ステップあたり 14 次元の行動空間なので、RL の actor にとっては、140 次元のチャンク化された行動に相当する。実装の詳細は付録 B に示す。

### B. ベースラインとアブレーション

事前学習済みの VLA モデル π₀.₆ [33] から出発する。各タスクについて、1〜10 時間の遠隔操作のデモンストレーションを収集する。その後、RL token の表現を学習しながら、VLA モデルをファインチューニングする。これにより、全ての実験を通じて使う、ベースの VLA 方策が得られる。タスクの難易度に応じて、400〜1000 エピソードの RL 学習を行う。リセットや様々なオーバーヘッドを除くと、各実験は、約 15 分から 5 時間の、実際のロボットデータを生成する。性能は、人間のオペレータによる二値の報酬信号で判定した、各タスクの成功率で測る。また、頑健性と速度の両方の改善を評価するため、**スループット**（10 分間あたりの成功したタスクの完了数）も報告する。全てのタスクを重要な局面で評価し、より難しい 2 つのタスク（ネジと結束バンド）は、タスク全体の設定でも評価する。

RLT を、経験から方策を改善する 4 つのベースライン手法と比較する。公平な比較のため、各 RL 手法を、同じ量のデータで学習する（付録 C 参照）。

- **HIL-SERL [4]**: 我々の手法と同様に、経験と介入の組み合わせで、小さな actor と critic を学習するが、RLT と異なり、事前学習済み VLA の表現を使わず、標準的なコンピュータビジョンのタスクで事前学習された、単純な ResNet エンコーダを使う。
- **Probe-Learn-Distill [30]**: PLD は、各 1 ステップの行動に対する残差を出力する、残差方策を学習する。この残差をハイパーパラメータでスケーリングし、凍結した VLA の行動予測の 1 ステップと足して実行する。
- **DSRL [32]**: DSRL は、flow VLA モデルの潜在ノイズ空間で、オンライン RL 方策を学習する。凍結した VLA モデルの行動生成器に入力するノイズを選ぶことで、VLA の行動生成を「操舵」する。この手法は、探索を、VLA が生成できる行動に暗黙的に制約し、そのモードの間を探索する。
- **DAgger [41, 42]**: 我々の学習中に収集した人間の介入データで、ベース VLA モデルをファインチューニングする。

また、我々の手法の各構成要素を、個別に取り除くことで、その寄与を切り分ける。

- **RL token なし（w/o RL token）**: RL token を、[25] の、凍結した ImageNet 事前学習済み ResNet-10 エンコーダで置き換える。
- **チャンクなし（w/o Chunk）**: RL 方策が、行動チャンクではなく 1 ステップの行動（$C = 1$）を出力する。この方策は 50 Hz で実行する必要があり、ベース VLA モデルを 50 Hz で問い合わせるのは不可能なので、RL token を ResNet-10 エンコーダで置き換えざるを得ない。
- **BC 正則化なし（w/o BC Regularizer）**: 式 (5) で $\beta = 0$ とする。方策は Q 関数のみで学習される。
- **パススルーなし（w/o Pass-Through）**: 式 (4) の方策入力から $\tilde{a}$ を取り除く。RL の actor は、状態と RL token のみから行動を生成する。

**図 4: RLT は、ベース VLA 方策に対してスループットを大きく向上させ、各タスクの重要な局面の速度と一貫性の両方を改善する。** 改善は、VLA 方策がミスをしやすい、より難しいタスクで特に顕著である。（図中: ネジ、結束バンド、Ethernet、充電器、タスク全体（ネジ、結束バンド）。Base Policy と RLT (Ours)。縦軸: スループット（成功数/10 分））

**図 5: RLT は、複数のタスクで成功率を高められる。** VLA がすでに有能な場合（たとえば Ethernet のタスク）、成功率を維持してスループットを高める。ベース VLA 方策にとって難しいタスク（ドライバーと結束バンド）では、RLT は成功率を大きく改善する。

**図 6: 他の RL アルゴリズムとの比較。** RLT を、最近の RL 文献のいくつかのベースラインと比較する。行動チャンクではなく、1 ステップの行動のみを考える手法（HIL-SERL、PLD）は、性能が低い。DSRL は成功率は高いが、スループットでは大きく劣る。（Ethernet の挿入、ベース方策、DAgger、HIL-SERL、PLD、DSRL、RLT）

### C. 実験結果

**Q1: オンライン RL は、ベースの VLA 方策を改善するか？** 本手法を、2 つの条件で評価する。重要な局面を切り分ける制御された設定と、RL 方策により高い頑健性を要求する、タスク全体の設定である。オンライン RL は、どちらの設定でも、ベースモデルの成功率と実行速度を改善する。制御された設定では、RLT は、4 つ全てのタスクの重要な局面を一貫して改善する。ベース方策がすでに良い信頼性を達成している、比較的易しい充電器と Ethernet のタスクでも、RLT で学習した方策は、重要な局面で約 3 倍速い。成功率の向上は、より難しい結束バンドとドライバーのタスクで、より顕著である。タスク全体の評価では、タスクの前の部分（物体の把持や持ち上げなど）からの誤差の累積のため、全体の成功率は低くなるが、RLT は、それでも、ドライバーのタスクで成功率を 40%、結束バンドのタスクで 60% 改善する。

**Q2: RLT は、他の手法と比べてどうか？** 図 6 に示すとおり、RLT は、ベースラインと比べて、スループットを大きく向上させる。Ethernet のタスクで、4 つのベースラインと比較する。HIL-SERL と PLD（どちらも 1 ステップのオンライン RL 手法）は、数百ステップに及び、報酬が疎なこのタスクで、効果的に学習できない。行動チャンキングがないと、タスクのホライズンが非常に長くなり、価値関数の更新が、疎な報酬信号を伝播するのに効果的でない。この、より単純なタスクでは、DAgger と DSRL は、RLT に匹敵する成功率を達成する（図 6）が、速度の面での改善は、大幅に小さい。DAgger は模倣学習の手法で、人間のデモンストレーションと介入の速度に制限される。DSRL は、方策をベース VLA の近くに留めるよう強く制約する RL 手法で、学習は安定するが、改善の可能性は比較的小さい。対照的に、RLT は、ベース方策の高い成功率に匹敵しつつ、完了までの平均ステップ数を、ベース方策から 2 倍減らす。

**Q3: 各構成要素はどれだけ寄与するか？** 4 つの設計上の選択（RL token、行動チャンク、BC 正則化、参照行動のパススルー）は、全て有意に寄与する。我々のレシピの各構成要素が、正の寄与を与えることを確認した（図 7）。RL token を凍結した ImageNet 事前学習済みの ResNet-10 エンコーダで置き換えると、スループットが 50% 低下する。これは、我々のトークンが、標準的なコンピュータビジョンのタスクで学習された既製のエンコーダが与えない、操作に関連する構造を符号化していることを裏付ける。チャンク（$C = 10$）を 1 ステップの行動で置き換えると、価値関数が、はるかに長いホライズンにわたってクレジット割り当てを行う必要があるため、タスクの実効的なホライズンが劇的に増える。また、RL token を使って本手法を実行することも不可能になる。実際には、1 ステップの変種は、ベース方策の性能に確実には追いつけない。BC 正則化を取り除く（$\beta = 0$）と、単独の要素としては最大の性能低下が生じる。これは、actor に、Q 関数の勾配のみで、行動空間全体を探索することを強いるためである。参照行動のパススルーを取り除くと、学習が遅くなり、探索の初期のドリフトや、時折の退化した挙動につながる。このアブレーションは、この、より単純なタスクでは、最終的には RLT の性能に追いつくが、図 7 の学習曲線に見られるように、学習の過程で、より多くの失敗を経験する。

**図 7: Ethernet のタスクでの、学習の各時点でのスループット。** アブレーション研究は、本手法の各部分が、良い性能に重要であることを示している。完全なシステムが最も速く学習し、最終的に最も良い性能を示す。特に、RLT は、タスクの重要な部分のデータをわずか 5 分消費した時点で、代替の方策を上回る（実験の総時間は約 40 分）。actor の入力から参照行動を除く（「w/o Pass-Through」）場合も、最良の最終性能には到達できるが、学習が遅くなり、学習の過程で失敗が大幅に増える。（図中: 横軸 データの分数、縦軸 スループット（成功数/10 分）。RLT (Ours)、Base Policy、w/o Pass-Through、w/o RL Token、w/o Chunk、w/o BC Regularizer）

**図 8: Ethernet のタスクでの、学習中の成功率の評価。** RLT は、Ethernet の挿入タスクで、VLA 方策の成功率にすばやく追いつきつつ、スループットを高める。参照行動のパススルーを使わない場合や、RL token を使わない場合は、学習が遅くなる。

**Q4: RLT は、より効果的な創発的戦略につながるか？** 集約された指標を超えて、オンライン RL の効果は、ロボットがタスクを実行する方法の、質的な変化である。Ethernet のタスクの重要な局面で、遠隔操作のデモンストレーション、ベース方策、最終的な RL 方策の、速度の分布を可視化する（図 9）。ベース VLA は、接触の近くで、しばしば「探り」の挙動を示す。目標に近づき、少し引き、調整し直し、再び試みる。成功する前に、そのような試みを何度か繰り返すこともある。RLT は、代わりにポートに近づき、コネクタを滑らかな動きで挿入する。最初の試みで失敗した場合でも、RLT は圧力をかけ、コネクタを少し揺らして、コンプライアンスを利用し、より速い挿入につなげる。この挙動は、デモンストレーションデータには見られず、純粋にオンラインの探索から生まれたものであり、本手法が、人間の戦略の模倣を超えられることを示している。

**図 9: Ethernet のタスクでの速度。** RLT は、Ethernet のタスクの速度を大きく改善する。最終的な方策は、専門家のオペレータが作ったデモンストレーションより速く、ベース VLA モデルよりも大幅に速い。重要な挿入の局面での RL エピソードの半分（黄）は、遠隔操作のデモンストレーションの全て（緑）より速い。（横軸: エピソード長（タイムステップ）、縦軸: 割合（%）。Teleop: 中央値 146、Base Policy: 中央値 228、RLT (Ours): 中央値 66）

## VII. 結論

大規模な事前学習済み VLA から抽出した表現の上で、高速なオンライン RL を行う手法、RLT を提示した。VLA に、コンパクトな表現を出力するよう学習させることで、本手法は、軽量な actor と critic が、わずか数時間の実世界での練習で、非常に精密で繊細なタスクを改善できるようにする。精度と速度を要する 4 つの難しいタスクで、RLT は、成功率と実行速度の両方を一貫して改善し、各タスクの最も難しい局面で最大 3 倍の高速化を達成し、場合によっては、オンライン RL から生まれた戦略によって、専門家の人間の遠隔操作の速度を上回る。

RLT は、高速で効率的な学習を提供するが、学習中に、報酬信号、介入による修正、RL（重要な局面）とベース方策（それ以外の局面）の切り替えを与えるために、追加の人間の介入を必要とする。原理的には、これらの要素の一部は、たとえば、報酬モデルや進捗予測を使って、自動化できる。RLT に基づく、完全に自律的な RL による改善のパイプラインを開発することは、今後の研究の有望な方向である。より広く言えば、我々の手法は、デモンストレーションデータから学習するだけでなく、現場で直接改善できる、ロボットシステムに向けた重要な一歩を示していると考える。改善が速く、信頼できるとき、VLA の事前学習の段階は、下流の探索のための良い初期化を与えるだけでよく、最も成功し、最も高性能な戦略は、強化学習によって発見できる。RLT が、この未来に向けた一歩となることを願う。

## 謝辞

ロボティクスはチームの取り組みである。ハードウェア、データ収集、ロボットの運用、ロボットのインフラなど、この研究の多くの側面に貢献した、Physical Intelligence の全ての方々に感謝する。グリッパーの設計を手伝ってくれた Liam Murphy と Cameron Myers、ロボットオペレータと PI の運用・アノテーションチーム、ウェブサイトとブログ記事を手伝ってくれた Connor Jacobsen、図を手伝ってくれた Brian Ichter、校正を手伝ってくれた Kyle Vedder、ブログ記事の可視化を手伝ってくれた Claudio Guglieri、動画の撮影と編集を手伝ってくれた Donald Jewkes と Thomas Burton に感謝する。

## 付録

### A. 貢献

CX と LK がプロジェクトを開始した。CX がオンライン RL のインフラを構築した。JTS が RL token を設計し、学習した。ME が介入のインターフェースを構築した。AA と AE がグリッパーとロボットのハードウェアを設計・製作した。CX と LK が、システムの実装、タスク群、実験を設計した。SL と LK がプロジェクトを通じて助言した。LK、CX、JTS、SL、ME が、執筆、図版、動画に取り組んだ。

### B. 実験の追加の詳細

まず、対象タスクのデモンストレーションデータセットを収集する。その後、ベース VLA モデルをファインチューニングし、RL token を、単一タスクのデータで、2000〜10000 回の勾配ステップ学習する。VLA は、その後、オンライン RL の学習中は凍結される。

オンライン RL の間、結束バンドの固定、Ethernet、充電器の挿入のタスクでは、RL の actor と critic を、2 層の MLP（隠れ次元 256）でゼロから初期化する。より難しいネジの取り付けのタスクでは、3 層で隠れ次元 512 の MLP からなる、より大きなネットワークを使う。どちらのネットワークも、凍結したベース VLA モデルが生成した RL token と、固有受容感覚の位置および速度を入力として受け取る。critic は、Fujimoto ら [19] に従って、2 つの Q 関数のアンサンブルで学習し、ターゲット値の計算には、2 つの Q 関数の最小値を使う。actor は、さらに、VLA モデルが生成する参照行動チャンクを入力とするが、これは、学習中は **50% の確率でマスクされ**、推論時は常に与えられる。actor は、小さな固定の標準偏差を持つガウス型方策としてパラメータ化され、現在の観測から、$C = 10$ の行動チャンク $a_{t:t+C-1} \in \mathbb{R}^{C \times d}$ を出力する。サンプル効率を高めるため、学習中は、制御ステップで 2 つ離した行動チャンクをサブサンプリングし、データ 1 秒あたり、RL ネットワークに対して約 25 個のサンプルが得られる。RL タスクが完了したときに、学習中にオペレータから、疎な +1 の報酬が与えられる。

ネジの取り付けと結束バンドの固定のタスクでは、まず、重要な局面の設定のみで RL 学習を開始する。その後、タスク全体の局面に進み、まずベースモデルを動かしてタスクの重要でない局面を完了させ、重要な局面に達したときに RL 方策に切り替える。この 2 段階の学習戦略は、学習効率を高めつつ、RL 方策が、タスクの前の部分でベース方策が生む初期分布に対して、頑健であることを保証する。約 5 時間のデータを集めた後の方策の性能を報告する。

### C. ベースラインの追加の実験詳細

全てのベースライン手法で、我々の手法と同じ環境と行動空間の設定を使う。方策は、50 Hz のデルタ行動空間で実行する。

**PLD:** 原論文に従い、まず、より良いサンプル効率のため、50 回のベース方策のロールアウトで、Cal-QL [31] を使って critic ネットワークを事前学習する。その後、オンライン RL の段階に進む。

**DSRL:** 原実装に従い、我々の実装は、$(1, 32)$ 次元の潜在行動を予測する。これを、最初の次元で 50 回繰り返し、我々の行動チャンク VLA のノイズ入力空間に合わせる。

**HIL-SERL:** 原実装に従い、20 エピソードのデモンストレーションで RLPD の学習を初期化し、学習を通じて介入を与える。しかし、元のシステム（10 Hz）と比べて、制御周波数が高い（50 Hz）こと、および探索空間を減らす行動空間のバウンディングボックスがないことのため、我々の設定では成功できなかった。

**DAgger:** デモンストレーションデータと、オンライン RL の学習中に収集した同じ介入データの混合で、VLA をファインチューニングする。

---

## 参考文献（原文のまま）

[1] Haozhan Li, Yuxin Zuo, Jiale Yu, Yuhao Zhang, Zhao-hui Yang, Kaiyan Zhang, Xuekai Zhu, Yuchen Zhang, Tianxing Chen, Ganqu Cui, Dehui Wang, Dingxiang Luo, Yuchen Fan, Youbang Sun, Jia Zeng, Jiangmiao Pang, Shanghang Zhang, Yu Wang, Yao Mu, Bowen Zhou, and Ning Ding. Simplevla-rl: Scaling vla training via rein-forcement learning. arXiv preprint, arXiv:2509.09674, 2025. 1, 2

[2] Yunfei Li, Xiao Ma, Jiafeng Xu, Yu Cui, Zhongren Cui, Zhigang Han, Liqun Huang, Tao Kong, Yuxiao Liu, Hao Niu, Wanli Peng, Jingchao Qiao, Zeyu Ren, Haixin Shi, Zhi Su, Jiawen Tian, Yuyang Xiao, Shenyu Zhang, Liwei Zheng, Hang Li, and Yonghui Wu. Gr-rl: Going dexter-ous and precise for long-horizon robotic manipulation, 2025. URL https://arxiv.org/abs/2512.01801. 2

[3] Physical Intelligence. π∗ 0.6: a VLA That Learns From Experience, 2025. URL https://arxiv.org/abs/2511.14759. 1, 2

[4] Jianlan Luo, Charles Xu, Jeffrey Wu, and Sergey Levine. Precise and dexterous robotic manipulation via human-in-the-loop reinforcement learning. arXiv preprint arXiv:2410.21845, 2024. 1, 2, 7

[5] Kun Lei, Huanyu Li, Dongjie Yu, Zhenyu Wei, Lingxiao Guo, Zhennan Jiang, Ziyu Wang, Shiyu Liang, and Huazhe Xu. Rl-100: Performant robotic manipulation with real-world reinforcement learning, 2026. URL https://arxiv.org/abs/2510.14830. 1, 2

[6] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Flo-rence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alex Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Ste-fan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control. In arXiv preprint arXiv:2307.15818, 2023. 2

[7] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan Foster, Grace Lam, Pannag Sanketi, et al. Openvla: An open-source vision-language-action model. arXiv preprint arXiv:2406.09246, 2024. 2, 3

[8] Physical Intelligence. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024. 2

[9] Gemini Robotics Team, Saminda Abeyruwan, Joshua Ainslie, Jean-Baptiste Alayrac, Montserrat Gonzalez Arenas, Travis Armstrong, Ashwin Balakrishna, Robert Baruch, Maria Bauza, Michiel Blokzijl, Steven Bo-hez, Konstantinos Bousmalis, Anthony Brohan, Thomas Buschmann, Arunkumar Byravan, Serkan Cabi, Ken Caluwaerts, Federico Casarini, Oscar Chang, Jose En-rique Chen, Xi Chen, Hao-Tien Lewis Chiang, Krzysztof Choromanski, David D’Ambrosio, Sudeep Dasari, Todor Davchev, Coline Devin, Norman Di Palo, Tianli Ding, Adil Dostmohamed, Danny Driess, Yilun Du, Debidatta Dwibedi, Michael Elabd, Claudio Fantacci, Cody Fong, Erik Frey, Chuyuan Fu, Marissa Giustina, Keerthana Gopalakrishnan, Laura Graesser, Leonard Hasenclever, Nicolas Heess, Brandon Hernaez, Alexander Herzog, R. Alex Hofer, Jan Humplik, Atil Iscen, Mithun George Jacob, Deepali Jain, Ryan Julian, Dmitry Kalashnikov, M. Emre Karagozler, Stefani Karp, Chase Kew, Jerad Kirkland, Sean Kirmani, Yuheng Kuang, Thomas Lampe, Antoine Laurens, Isabel Leal, Alex X. Lee, Tsang- Wei Edward Lee, Jacky Liang, Yixin Lin, Sharath Maddi-neni, Anirudha Majumdar, Assaf Hurwitz Michaely, Robert Moreno, Michael Neunert, Francesco Nori, Car-olina Parada, Emilio Parisotto, Peter Pastor, Acorn Poo-ley, Kanishka Rao, Krista Reymann, Dorsa Sadigh, Ste-fano Saliceti, Pannag Sanketi, Pierre Sermanet, Dhruv Shah, Mohit Sharma, Kathryn Shea, Charles Shu, Vikas Sindhwani, Sumeet Singh, Radu Soricut, Jost Tobias Springenberg, Rachel Sterneck, Razvan Surdulescu, Jie Tan, Jonathan Tompson, Vincent Vanhoucke, Jake Varley, Grace Vesom, Giulia Vezzani, Oriol Vinyals, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Fei Xia, Ted Xiao, Annie Xie, Jinyu Xie, Peng Xu, Sichun Xu, Ying Xu, Zhuo Xu, Yuxiang Yang, Rui Yao, Sergey Yaroshenko, Wenhao Yu, Wentao Yuan, Jingwei Zhang, Tingnan Zhang, Allan Zhou, and Yuxiang Zhou. Gemini robotics: Bringing ai into the physical world, 2025. URL https://arxiv.org/abs/2503.20020. 2, 3

[10] Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, and Tao Kong. Unleashing large-scale video generative pre-training for visual robot manipulation, 2023.

[11] NVIDIA, :, Johan Bjorck, Fernando Casta˜neda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi ”Jim” Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025. URL https://arxiv.org/abs/2503.14734. 2

[12] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manip-ulation with low-cost hardware, 2023. URL https://arxiv. org/abs/2304.13705. 2

[13] Cheng Chi, Zhenjia Xu, Siyuan Feng, Eric Cousineau, Yilun Du, Benjamin Burchfiel, Russ Tedrake, and Shuran Song. Diffusion policy: Visuomotor policy learning via action diffusion. The International Journal of Robotics Research, page 02783649241273668, 2023. 2

[14] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Efficient action tokeniza-tion for vision-language-action models. arXiv preprint arXiv:2501.09747, 2025. 2

[15] Suneel Belkhale and Dorsa Sadigh. Minivla: A better vla with a smaller footprint, 2024. URL https://github.com/ Stanford-ILIAD/openvla-mini. 2

[16] Physical Intelligence. π0.5: a vision-language-action model with open-world generalization. In 9th Annual Conference on Robot Learning, 2025. 2, 3

[17] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. Pmlr, 2018. 2, 3

[18] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforce-ment learning. arXiv preprint arXiv:1509.02971, 2015.

[19] Scott Fujimoto, Herke van Hoof, and David Meger. Addressing function approximation error in actor-critic methods. arXiv preprint arXiv:1802.09477, 2018. 3, 4, 12

[20] Abbas Abdolmaleki, Jost Tobias Springenberg, Yuval Tassa, Remi Munos, Nicolas Heess, and Martin Ried-miller. Maximum a Posteriori Policy Optimisation. In International Conference on Learning Representations (ICLR), 2018. URL https://openreview.net/forum?id= S1ANxQW0b. 2, 4

[21] Marcel Hussing, Claas Voelcker, Igor Gilitschenski, Amir massoud Farahmand, and Eric Eaton. Dissecting deep rl with high update ratios: Combatting value divergence, 2024. URL https://arxiv.org/abs/2403.05996. 2

[22] Xinyue Chen, Che Wang, Zijian Zhou, and Keith Ross. Randomized ensembled double q-learning: Learning fast without a model. arXiv preprint arXiv:2101.05982, 2021. 2

[23] Philip J Ball, Laura Smith, Ilya Kostrikov, and Sergey Levine. Efficient online reinforcement learning with offline data. In International Conference on Machine Learning, pages 1577–1594. PMLR, 2023. 2

[24] Henry Zhu, Justin Yu, Abhishek Gupta, Dhruv Shah, Kristian Hartikainen, Avi Singh, Vikash Kumar, and Sergey Levine. The ingredients of real-world robotic re-inforcement learning. arXiv preprint arXiv:2004.12570, 2020. 2

[25] Jianlan Luo, Zheyuan Hu, Charles Xu, You Liang Tan, Jacob Berg, Archit Sharma, Stefan Schaal, Chelsea Finn, Abhishek Gupta, and Sergey Levine. Serl: A software suite for sample-efficient robotic reinforcement learning. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 16961–16969. IEEE, 2024. 2, 7

[26] Allen Z. Ren, Justin Lidard, Lars Lien Ankile, Anthony Simeonov, Pulkit Agrawal, Anirudha Majumdar, Ben-jamin Burchfiel, Hongkai Dai, and Max Simchowitz. Diffusion Policy Policy Optimization. In Proceedings of the 2025 International Conference on Learning Rep-resentations (ICLR), 2025. 2

[27] Kang Chen, Zhihao Liu, Tonghe Zhang, Zhen Guo, Si Xu, Hao Lin, Hongzhi Zang, Quanlu Zhang, Zhaofei Yu, Guoliang Fan, Tiejun Huang, Yu Wang, and Chao Yu. πRL: Online rl fine-tuning for flow-based vision-language-action models. arXiv preprint, arXiv:2510.25889, 2025. 2

[28] Yuhui Chen, Shuai Tian, Shugao Liu, Yingting Zhou, Haoran Li, and Dongbin Zhao. Conrft: A reinforced fine-tuning method for vla models via consistency policy. arXiv preprint arXiv:2502.05450, 2025. 2, 3

[29] Xiu Yuan, Tongzhou Mu, Stone Tao, Yunhao Fang, Mengke Zhang, and Hao Su. Policy decorator: Model-agnostic online refinement for large policy model. In The Thirteenth International Conference on Learning Representations, 2025. 2

[30] Wenli Xiao, Haotian Lin, Andy Peng, Haoru Xue, Tairan He, Yuqi Xie, Fengyuan Hu, Jimmy Wu, Zhengyi Luo, Linxi ”Jim” Fan, Guanya Shi, and Yuke Zhu. Self-improving vision-language-action models with data gen-eration via residual rl, 2025. 2, 3, 7

[31] Mitsuhiko Nakamoto, Simon Zhai, Anikait Singh, Max Sobol Mark, Yi Ma, Chelsea Finn, Aviral Kumar, and Sergey Levine. Cal-ql: Calibrated offline rl pre-training for efficient online fine-tuning. Advances in Neural Information Processing Systems, 36:62244–62269, 2023. 2, 12

[32] Andrew Wagenmaker, Mitsuhiko Nakamoto, Yunchu Zhang, Seohong Park, Waleed Yagoub, Anusha Naga-bandi, Abhishek Gupta, and Sergey Levine. Steering your diffusion policy with latent space reinforcement learning. In Proceedings of the 9th Conference on Robot Learning (CoRL), 2025. 2, 7

[33] Physical Intelligence. π0.6 model card, 2025. URL https: //website.pi-asset.com/pi06star/PI06 model card.pdf. 3, 7

[34] Nicolas Heess, Gregory Wayne, David Silver, Timothy Lillicrap, Tom Erez, and Yuval Tassa. Learning continuous control policies by stochastic value gradients. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper files/paper/2015/ file/148510031349642de5ca0c544f31b2ef-Paper.pdf. 3

[35] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. In Advances in neural information processing systems, pages 3104– 3112, 2014. 4

[36] Seohong Park, Qiyang Li, and Sergey Levine. Flow q-learning. In International Conference on Machine Learning (ICML), 2025. 4

[37] Xue Bin Peng, Erwin Coumans, Tingnan Zhang, Tsang- Wei Lee, Jie Tan, and Sergey Levine. Learning agile robotic locomotion skills by imitating animals. RSS, 2020. 4

[38] Jan Peters, Katharina M¨ulling, and Yasemin Alt¨un. Rel-ative entropy policy search. In Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelli-gence, AAAI’10, page 1607–1612. AAAI Press, 2010.

[39] Peter Dayan and Geoffrey E. Hinton. Using expectation-maximization for reinforcement learning. Neural Com-putation, 9(2):271–278, 1997. doi: 10.1162/neco.1997.9. 2.271.

[40] Sergey Levine. Reinforcement learning and control as probabilistic inference: Tutorial and review, 2018. URL https://arxiv.org/abs/1805.00909. 4

[41] Michael Kelly, Chelsea Sidrane, Katherine Driggs- Campbell, and Mykel J. Kochenderfer. Hg-dagger: In-teractive imitation learning with human experts, 2019. URL https://arxiv.org/abs/1810.02890. 6, 7

[42] Stephane Ross, Geoffrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Geoffrey Gordon, David Dunson, and Miroslav Dud´ık, editors, Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, volume 15 of Proceedings of Machine Learning Research, pages 627–635, Fort Lauderdale, FL, USA, 11–13 Apr 2011. PMLR. URL https://proceedings.mlr.press/v15/ross11a.html. 7


# keras バッチサイズの反映

### classification
(?,x,y,channel), (?,labels)


---

### 強化学習について

$s$ : 状態
$a$ : アクション
$\theta$:パラメータ
$\pi_\theta(s,a)$:パラメータθで決まる方策

---

### 方策反復法

$\theta_{s_ia_j}=\theta_{s_ia_j}+\eta\cdot\Delta\theta_{s,a_j}$

$\Delta\theta_{s,a_j}=\{N(s_i,a_j)+P(s_i,a_j)N(s_i,a_j)\}/T$

$\theta_{s_ia_j}$は、位置が$s_i$のときに行動$a_j$をとるパラメータ

$N(s_i,a_j)$は位置が$s_i$のときに行動$a_j$をとった回数

$N(s_i,a)$は位置が$s_i$のときに何らかの行動$a$をとった回数

$P(s_i,a_j)$は現在の方策で$s_i$のときに行動$a_j$をとる確率

$T$はゴールまでの試行回数

---


#### 価値反復法

価値の代わりに報酬をもらう。
即時報酬を時間割引して報酬を計算する。
それがベルマン方程式。



#### Valueベース と Policyベースの違い

行動選択の基準。
Value ベース(Off-policy)は価値が最大となる状態に遷移するよう行動を選択
Policyベース(On-Policy)は戦略に基づいて行動を選択

__行動価値関数__

$Q^\pi(s,a)$

* sが行、aが列の行列

ゴール手前のマスから別方向に移動するケース

$Q^\pi(s=7,a=0) = \gamma^2 * 1$


__状態価値関数__

$V^\pi(s)$

ゴールの手前のマス :  $V(s=7) = 1$
ゴールの手前から２つめのマス :  $V(s=4) = \gamma * 1$


__即時報酬__

移動したときに得られる報酬（ゴールに移動=1）

$R_t$


#### ベルマン方程式

$V^\pi(s)=\underset{a}{max}\mathbb{E}[R_{s,a}+\gamma*V^\pi(s(s,a))]$



#### 価値反復法 ：epsilon-greedy法

$\epsilon$ : 現在の行動価値関数の価値の高いものを実施

$1-\epsilon$　：　ランダムで行動を試行


#### Q-learning

Q値=Q(s,a)を学習する。TD法が一般的。

Q-learning では、価値が最大となる状態に遷移する行動をとる.
SARSAでは 次の行動は（更新前の）戦略に従って決めらる。


gain = reward + gamma*max(self.Q[state])

#### Monte-Carlo法

1エピソードを終わるまで実施してから評価する方法。

#### Multi-step Learning

TD法とモンテカルロ法の中間。


#### SARSA (State-Action-Reward-State-Action)

もし、行動価値関数が定まっていると、ベルマン方程式から、

$Q(s_t,a_t)=R_{t+1}+\gamma Q(s_{t+1},a_{t+1})$

の関係が成り立つ、そこで、0となるはずの

$R_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)$

を __TD誤差__ として、誤差を少なくなるように学習する。

$Q(s_t,a_t)=Q(s_t,a_t)-\eta \{R_{t+1}+\gamma Q(s_{t+1},a_{t+1})\}$

$R_{t+1}+\gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t)$　は現在のQ(s,t)をベースに次の行動を決められる。


gain = reward + gamma*self.Q[次のステート][次のアクション]

---

Q-learningとSARSAの違いは、gainの箇所に現れています。 　 


```plantuml

(Agent)->(環境):action
(環境)->(Agent):state

```

Agent : update_Q_function / get_ next_action







$V(s) = \underset{a}{\textrm{max}} (R(s, a) + \gamma\ V(s'))$
$\pi_\theta(s,a)$

__mathjax__

```
白抜き

\mathbb{ } .


下付き文字

_{xx}


上付き文字

^{xx}

下に書く(argmin/x)


\underset{a}{max}



```


```
fig = plt.figure() # Figureオブジェクトを用意

for step in range(0,200):
    img = env.render(mode='rgb_array') # 画像を生成
    #    print(img.__class__)
    frames.append(img)
    action = np.random.choice(2)
    observation, reward, done, info = env.step(action)


def show_animation_image(frames):
    ims = []
    for img in frames:
        im = plt.imshow(img,animated=True) # 画像を描画
        ims.append([im]) # imsにリストにして追加

    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
                                repeat_delay=100)
    plt.show()

show_animation_image(frames)
```



## Keras-RL


### Agent
rl.core.Agent(processor=None)


実装されているすべてのエージェントの抽象基本クラス。

各エージェントは、最初に環境の状態を観察することによって（Envクラスで定義されているように）環境と対話します。この観察に基づいて、エージェントはアクションを実行することによって環境を変更します。

この抽象基本クラスを直接使用せずに、代わりに実装された具象エージェントの1つを使用してください。各エージェントは強化学習アルゴリズムを実現する。すべてのエージェントは同じインターフェースに準拠しているので、それらを互換的に使用できます。

あなた自身のエージェントを実装するには、以下を実装する必要があります。

methods:

forward
backward
compile
load_weights
save_weights
layers
Arguments

__DQNAgent__
rl.agents.dqn.DQNAgent(model, policy=None, test_policy=None, enable_double_dqn=True, enable_dueling_network=False, dueling_type='avg')

__NAFAgent__

正規化アドバンテージ関数（NAF）エージェントは、DQNを連続アクション空間に拡張する方法であり、DDPGエージェントよりも単純です。

NAFAgentではＱ関数は、利点項Ａと状態値項Ｖとに分解される。したがって、エージェントは、３つのモデルを利用する。すなわち、Ｖ＿モデルは状態値項を学習し、利点項ＡはＬ＿モデル、およびｍｕ＿モデルに基づいて構成される。 mu_modelは常にQ関数を最大化するアクションです。

mu_modelはアクションを決定論的に選択するので、探索と利用のバランスをとるためにrandom_processを追加できます。 DQNと同様に、安定性のためにターゲットネットワークを使用し、再生バッファを保持します。


__DDPGAgent__


深層決定方策デターミニスティックポリシーグラデーション（DDPG）エージェントは、ポリシー外のアルゴリズムであり、連続アクションスペースのDQNとして考えることができます。それは方針（俳優）とQ-機能（評論家）を学びます。方針は決定論的であり、そのパラメータは学習されたＱ関数に連鎖規則を適用することに基づいて更新される（予想される報酬）。 Ｑ関数は、Ｑ学習と同様にベルマン方程式に基づいて更新される。

アクターモデルの入力は状態観測で、出力はアクションそのものです。 （環境のステップ関数で入力として与えられるアクションはアクターモデルの出力であることに注意してください。一方、離散空間のあるDQNでは、ポリシーはQ関数を学習するモデルに基づいてnb_actionsから1つのアクションを選択します。 。）

評論家モデルの入力は、状態観測と、この状態に基づいてアクターモデルが選択する行動を連結したものであるべきです。その出力は、各行動と状態のQ値を示します。形状nb_actionsのKeras入力層は、引数critic_action_inputとして渡されます。

搾取と探査のバランスをとるために、アクターモデルによって決定されたアクションにノイズを追加し、探査を可能にするrandom_processを導入することができます。原著論文では、Ornstein-Uhlenbeck法が使用されており、これは慣性を伴う物理的制御問題に適応しています。

DQNと同様に、DDPGもリプレイバッファとターゲットネットワークを使用します。

詳細については、DDPG振り子の例をご覧ください。


#### Processor
rl.core.Processor()

プロセッサを実装するための抽象基本クラス。

プロセッサは、エージェントとそのEnv間のカップリングメカニズムとして機能します。これは、あなたのエージェントが観察の形態、行動、そして環境の見返りに関して異なる要件を持っている場合に必要になることがあります。カスタムプロセッサを実装することで、エージェントや環境の基本となる実装を変更せずに2つの間で効果的に変換できます。

この抽象基底クラスを直接使わずに、代わりに具象実装を使うか独自のものを書いてください。


### Env
rl.core.Env()


すべてのエージェントによって使用される抽象環境クラス。このクラスはOpenAI Gymが使用しているのとまったく同じAPIを持っているので、統合することは簡単です。 OpenAI Gymの実装とは対照的に、このクラスは実際の実装なしで抽象メソッドを定義するだけです。



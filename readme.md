# Multiple robots avoid dynamic obstacles using model made by MADDPG

本レポジトリは[tf2multiagentrl](https://github.com/JohannesAck/tf2multiagentrl)をフォークし改良したものである．
環境構築等は上記のリポジトリに従う．

## 実行例
ルートディレクトリにおいて
```
python3 train.py
```
を実行．
```
input val[0 -> train, 1 -> add_learning, 2 -> display, 3 -> evaluate]:
```
と出力されるので適切な数字を入力する．
ここで
```
train -> 0から学習を行う
add_learning -> 指定したモデルを読み込んで，追加で学習を行う
display -> 指定したモデルを読み込んで，実行を行う．この際動画が表示される
evaluate -> 指定したモデルを読み込んで，評価を行う．指定した回数分試行し何回成功したかを記録する．
```
である．

## ディレクトリ構造
```
MADDPG_avoid_obstacles/
├── .gitignore
├── Dockerfile
├── graph_plot.py
├── readme.md
├── requirements.txt
├── reward_plot.py
├── learned_results/
|   └── 学習結果が保存される．
├── tf2marl/
│   ├── __init__.py
│   ├── agents/: アルゴリズム, layerを変更する場合はここのクラスを変更する．
│   │   ├── AbstractAgent.py
│   │   ├── __init__.py
│   │   ├── mad3pg.py
│   │   ├── maddpg.py
│   │   ├── masac.py
│   │   └── matd3.py
│   ├── common/: replay_bufferとそれに関連するクラス．layerを変更するとき以外いじらない．
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── replay_buffer.py
│   │   ├── segment_tree.py
│   │   ├── test_envs/
│   │   │   └── identity_env.py
│   │   └── util.py
│   └── multiagent/: MPE(シミュレーション環境)，自分の問題設定に合わせて変更する．
│       ├── __init__.py
│       ├── core.py: 学習対象のagent, follower, obstacleなどのクラスが記述されている．問題設定に応じて変更．
│       ├── environment.py: MultiAgentEnvクラスが記述されている．
│       ├── multi_discrete.py
│       ├── policy.py
│       ├── rendering.py
│       ├── scenario.py
│       └── scenarios/: 自分の問題設定に応じてシナリオを作成する．私の場合stage1~3を使用．
│           ├── __init__.py
│           ├── base_funcs.py: シナリオのコードが長くなりすぎたので，このファイルに必要な関数等をまとめた．
│           ├── others/: サンプルのシナリオ
│           │   ├── inversion.py
│           │   ├── maximizeA2.py
│           │   ├── simple.py
│           │   ├── simple_adversary.py
│           │   ├── simple_crypto.py
│           │   ├── simple_push.py
│           │   ├── simple_reference.py
│           │   ├── simple_speaker_listener.py
│           │   ├── simple_spread.py
│           │   ├── simple_tag.py
│           │   └── simple_world_comm.py
│           ├── stage1.py
│           ├── stage2.py
│           └── stage3.py
└── train.py
```
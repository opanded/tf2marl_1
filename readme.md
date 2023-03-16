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
.
├── README.md
├── pip.txt
├── example_setting: settingの例
│   └── setting.json
└── scripts
    ├── __init__.py
    ├── detect_convey.py: 前処理用のスクリプト
    ├── detect_alpha.py: alphaを決定するスクリプト
    ├── main_integurated.py: 検出，デバッグの全てが統合されたスクリプト
    ├── setting.json
    └── libs: クラス等がまとまっているフォルダ
      ├── magicEyeAPI.py
      ├── dataDriveFunc.py
      ├── measureDimensionFunc.py
      └── pointcloudFunc.py
```

## TensorFlow 2 Implementation of Multi-Agent Reinforcement Learning Approaches 

This repository contains a modular TF2 implementations of multi-agent versions of the RL methods DDPG 
([MADDPG](https://arxiv.org/abs/1706.02275)),
 TD3 ([MATD3](https://arxiv.org/abs/1910.01465)),
 [SAC](https://arxiv.org/abs/1801.01290) (MASAC) and
 [D4PG](https://arxiv.org/abs/1804.08617) (MAD4PG).
 It also implements [prioritized experience replay](https://arxiv.org/abs/1511.05952).
 
 In our experiments we found MATD3 to work the best and did not see find a benefit by using Soft-Actor-Critic
 or the distributional D4PG. However, it is possible that these methods may be benefitial in more
 complex environments, while our evaluation here focussed on the 
 [multiagent-particle-envs by openai](https://github.com/openai/multiagent-particle-envs).

## Code Structure
We provide the code for the agents in tf2marl/agents and a finished training loop with logging
powered by sacred in train.py.

We denote lists of variables corresponding to each agent with the suffix `_n`, i.e.
`state_n` contains a list of n state batches, one for each agent. 

## Useage

Use `python >= 3.6` and install the requirement with
```
pip install -r requirements.txt
```
Start an experiment with 
```
python train.py
```
As we use [sacred](https://github.com/IDSIA/sacred) for configuration management and logging, 
the configuration can be updated with their CLI, i.e.
```
python train.py with scenario_name='simple_spread' num_units=128 num_episodes=10000
```
and experiments are automatically logged to `results/sacred/`, or optionally also to a MongoDB.
To observe this database we recommend to use [Omniboard](https://github.com/vivekratnavel/omniboard).

 
## Acknowledgement
The environments in `/tf2marl/multiagent` are from [multiagent-particle-envs by openai](https://github.com/openai/multiagent-particle-envs)
with the exception of `inversion.py` and `maximizeA2.py`, which I added for debugging purposes.

The implementation of the segment tree used for prioritized replay is based on 
[stable-baselines](https://github.com/hill-a/stable-baselines)


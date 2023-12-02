# 简介

本项目是改自[MADDPG_avoid_obstacles](https://github.com/takumi-web/MADDPG_avoid_obstacles)，即[tf2multiagentrl](https://github.com/JohannesAck/tf2multiagentrl)的分支项目。
环境构建遵循[tf2multiagentrl]，但是略作改进，我更新了pip依赖项的表单和构建conda环境的yaml文件，当然我的工作流建立在 **WIN11 + WSL2 with Docker + Image with Conda + Container with biulded envs**这样的结构上，你可能需要根据需求去做更改。

# 原始项目简介
该存储库包含RL方法DDPG ([MADDPG](https://arxiv.org/abs/1706.02275))、TD3 ([MATD3](https://arxiv.org/abs/1910.01465))、SAC ([MASAC](https://arxiv.org/abs/1801.01290))和D4PG ([MAD4PG](https://arxiv.org/abs/1804.08617))的多智能体版本的模块化TF2实现。它还实现了[优先体验回放](https://arxiv.org/abs/1511.05952)。

在[tf2multiagentrl](https://github.com/JohannesAck/tf2multiagentrl)的实验中，他们发现MATD3效果最好，并没有发现使用SAC或DD4PG有什么好处。然而，这些方法可能在更复杂的环境中有益，而我们在这里的评估主要集中在openai的[multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)上。

# 示例
在根目录中
```
python3 train.py
```
执行主程序，在终端会显示如下的操作选项，同时也是用户端程序执行的流程：
```
input val[0 -> train, 1 -> add_learning, 2 -> display, 3 -> evaluate]:
```
在这里输入需要进行执行的阶段的对应的数字。
在这里对每个阶段略作解释：
```
train           ->     从头开始训练模型。
add_learning    ->     读入指定的模型，追加进行学习。
display         ->     读入指定的模型进行执行，显示视频。
evaluate        ->     加载指定的模型并进行评估。记录在指定的分钟数内尝试和成功的次数。
```

# 目录结构
```
MADDPG_avoid_obstacles/
├── .gitignore
├── Dockerfile
├── graph_plot.py
├── readme.md
├── requirements.txt
├── reward_plot.py
├── learned_results/
|   └── 保存学习结果
├── tf2marl/: 包含各种智能体的代码。
│   ├── __init__.py
│   ├── agents/: 要改变算法、layer时，先改变这里的类。
│   │   ├── AbstractAgent.py
│   │   ├── __init__.py
│   │   ├── mad3pg.py
│   │   ├── maddpg.py
│   │   ├── masac.py
│   │   └── matd3.py
│   ├── common/: replay_buffer及其相关类。只有在更改layer的时候才会摆弄。除了变更 layer 的时候以外不要乱动。
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── replay_buffer.py
│   │   ├── segment_tree.py
│   │   ├── test_envs/
│   │   │   └── identity_env.py
│   │   └── util.py
│   └── multiagent/: MPE(仿真环境)，根据您的问题设置进行更改。
│       ├── __init__.py
│       ├── core.py: 描述了诸如agent, follower, obstacle等的类。根据问题设置进行更改。
│       ├── environment.py: 描述了 MultiAgentEnv 类。
│       ├── multi_discrete.py
│       ├── policy.py
│       ├── rendering.py
│       ├── scenario.py
│       └── scenarios/: 根据您的问题设置创建一个方案。原作者设计的stage1~3。
│           ├── __init__.py
│           ├── base_funcs.py: 由于这个场景的代码太长，我们为这个文件收集了必要的函数和其他东西。
│           ├── others/: 样本场景。
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
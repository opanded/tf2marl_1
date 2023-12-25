import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from typing import List
import sys

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver

from tf2marl.common.logger import RLLogger
from tf2marl.agents import MADDPGAgent, MATD3Agent, MASACAgent, MAD3PGAgent
from tf2marl.agents.AbstractAgent import AbstractAgent
from tf2marl.multiagent.environment import MultiAgentEnv
from tf2marl.common.util import softmax_to_argmax

# setup gpu
list_gpu = tf.config.experimental.list_physical_devices('GPU')
print("length of GPU: ", len(list_gpu))
if list_gpu:
    for i in range(len(list_gpu)):
        tf.config.experimental.set_memory_growth(list_gpu[i], True)

train_ex = Experiment('avoide_obstacle')

### set global variable ###
input = int(input("input val[0 -> train, 1 -> add_learning, 2 -> display, 3 -> evaluate]:"))
if input == 0: 
    is_display = False
    is_evaluate = False
    add_learning = False
elif input == 1: 
    is_display = False
    is_evaluate = False
    add_learning = True
elif input == 2: 
    is_display = True
    is_evaluate = False
    add_learning = False
elif input == 3: 
    is_display = False
    is_evaluate = True
    add_learning = False
else: print("无效值"); sys.exit()


scenario = f'stage2'
load_dir = f"learned_results/stage2/10/models"
save_dir = f"learned_results/stage2"

if scenario == "stage2":
    learning_time = 5
elif scenario == "stage3" and add_learning:
    learning_time = 10
elif scenario == "stage3" and not add_learning:
    learning_time = 16
### set global variable ###

# This file uses Sacred for logging purposes as well as for config management.
# I recommend logging to a Mongo Database which can nicely be visualized with
# Omniboard.
@train_ex.config
def train_config():
    # Logging
    exp_name = 'default'            # name for logging

    display = is_display
    evaluate = is_evaluate
    if add_learning or display or evaluate: restore_fp = load_dir
    else: restore_fp = None
    save_path = save_dir                                
    save_rate = 500              # frequency to save policy as number of episodes
    num_eval_episodes = 1000
    
    # Environment
    scenario_name = scenario # environment name
    if scenario == "stage1":
        max_episode_len = 600           # timesteps per episodes
        num_episodes = 30000            # total episodes
    elif scenario == "stage2":
        if not is_display:  # 学習時
            max_episode_len = 800
        else:
            max_episode_len = 1000  # 評価時
        num_episodes = 30000
    else: 
        if not is_display:  # 学習時
            max_episode_len = 800
        else:
            max_episode_len = 1000  # 評価時
        num_episodes = 70000
    learning_time_log = learning_time
    
    # Agent Parameters
    good_policy = "masac"          # policy of "good" agents in env
    adv_policy = 'masac'           # policy of adversary agents in env
    # available agent: maddpg, matd3, mad3pg, masac

    # 一般训练超参数
    lr = 1e-3                       # learning rate for critics and policies
    gamma = 0.975                   # decay used in environments
    batch_size = 1024               # batch size for training
    num_layers = 2                  # hidden layers per network
    num_units = 64                  # units per hidden layer
    num_lstm_units = 16             # units per lstm layer 

    update_rate = 100               # update policy after each x steps
    critic_zero_if_done = False     # set the value to zero in terminal steps
    buff_size = 5e6                # size of the replay buffer
    # 学习不顺利时提前停止
    if buff_size <= batch_size * max_episode_len:
        print("Too big batch size!")
        sys.exit()
    tau = 0.01                      # Update for target networks
    hard_max = False                # use Straight-Through (ST) Gumbel

    priori_replay = False            # enable prioritized replay
    alpha = 0.6                     # alpha value (weights prioritization vs random)
    beta = 0.5                      # beta value  (controls importance sampling)

    use_target_action = True        # use target action in environment, instead of normal action

    # MATD3
    if good_policy == 'tf2marl':
        policy_update_rate = 1      # 与评论家相比，策略更新的频率
    else:
        policy_update_rate = 2
    critic_action_noise_stddev = 0.02  # 在评论家更新中增加了噪音
    action_noise_clip = 0.5         # 对噪音的限制

    # MASAC
    entropy_coeff = 0.05            # 软演员-评论家中熵的权重

    # MAD3PG
    num_atoms = 51                  # 支持的原子数量
    min_val = -400                  # minimum atom value
    max_val = 0                     # largest atom value


@train_ex.capture
def make_env(scenario_name) -> MultiAgentEnv:
    """
    Create an environment
    :param scenario_name:
    :return:
    """
    import tf2marl.multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    if is_display:  # rendring
        scenario.is_display = True
    if is_evaluate:  # evaluate
        scenario.is_evaluate = True
    # create world
    world = scenario.make_world()
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, done_callback = scenario.check_done)
    return env


@train_ex.main
def train(_run, exp_name, save_rate, display, evaluate, restore_fp,
          hard_max, max_episode_len, num_episodes, num_eval_episodes,
          batch_size, update_rate, use_target_action):
    """
    This is the main training function, which includes the setup and training loop.
    It is meant to be called automatically by sacred, but can be used without it as well.

    :param _run:            Sacred _run object for logging
    :param exp_name:        (str) Name of the experiment
    :param save_rate:       (int) Frequency to save networks at
    :param display:         (bool) Render the environment
    :param restore_fp:      (str)  File-Patch to policy to restore_fp or None if not wanted.
    :param hard_max:        (bool) Only output one action
    :param max_episode_len: (int) number of transitions per episode
    :param num_episodes:    (int) number of episodes
    :param batch_size:      (int) batch size for updates
    :param update_rate:     (int) perform critic update every n environment steps
    :param use_target_action:   (bool) use action from target network
    :return:    List of episodic rewards
    """
    # Create environment
    print(_run.config)
    env = make_env()

    # Create agents
    agents = get_agents(_run, env, env.n_adversaries)

    logger = RLLogger(exp_name, _run, len(agents), env.n_adversaries, save_rate)


    # Load previous results, if necessary
    if restore_fp is not None:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            fp = os.path.join(restore_fp, 'agent_{}'.format(ag_idx))
            # fp = os.path.join(restore_fp, 'agent_0')
            agent.load(fp)

    obs_n = env.reset()
    pos_list = []

    print('开始迭代...')
    while True:
        # get action
        if use_target_action:
            action_n = [agent.target_action(obs.astype(np.float32)[None])[0] for agent, obs in
                        zip(agents, obs_n)]
        else:
            action_n = [agent.action(obs.astype(np.float32)) for agent, obs in zip(agents, obs_n)]
        # environment step
        if hard_max:
            hard_action_n = softmax_to_argmax(action_n, agents)
            new_obs_n, rew_n, done_n, info_n = env.step(hard_action_n)
        else:
            action_n = [action.numpy() for action in action_n]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        logger.episode_step += 1

        done = any(done_n)
        terminal = (logger.episode_step >= max_episode_len)
        done = done or terminal

        # collect experience
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew_n[i], new_obs_n, done)
        obs_n = new_obs_n

        for ag_idx, rew in enumerate(rew_n):
            logger.cur_episode_reward += rew
            logger.agent_rewards[ag_idx][-1] += rew

        if done:
            if display:
                obj_info = [int(len(env.world.agents)), int(len(env.world.followers)), int(len(env.world.obstacles))]
                dest_info = [env.dest, env.rho_g]
                Os_info = [O for O in env.world.obstacles]
                logger.save_demo_result(pos_list, obj_info, dest_info, Os_info, env.reward_list_all)
                pos_list.clear()
            if evaluate:
                logger.save_eval_result(info_n, num_eval_episodes)
                
            obs_n = env.reset()
            logger.record_episode_end(agents, display, evaluate)
        logger.train_step += 1

        # policy updates
        train_cond = (not display) and (not evaluate)  # rendering時でも評価時でもない時
        for agent in agents:
            if train_cond and len(agent.replay_buffer) > batch_size * max_episode_len:
                if logger.train_step % update_rate == 0:  # only update every 100 steps
                    q_loss, pol_loss = agent.update(agents, logger.train_step)

        if display:  # rendering
            img = env.render('rgb_array')[0]
            logger.all_frames.append(img)
            # 各step毎のオブジェクトの位置を保存
            pos_list_1_step = []
            for L in env.world.agents:
                pos_list_1_step.append(L.state.p_pos)
            F_sum = np.array([0., 0.])
            for F in env.world.followers:
                pos_list_1_step.append(F.state.p_pos)
                F_sum += F.state.p_pos
            F_COM = F_sum / len(env.world.followers)
            pos_list_1_step.append(F_COM)
            for O in env.world.obstacles:
                pos_list_1_step.append(O.state.p_pos)
            pos_list.append(np.concatenate(pos_list_1_step))
            time.sleep(0.025)
        if evaluate and len(logger.episode_rewards) > num_eval_episodes:  # evaluate
            return None
        # saves logger outputs to a file similar to the way in the original MADDPG implementation
        if len(logger.episode_rewards) > num_episodes or (time.time() - logger.t_start) >= 3600 * learning_time:
            logger.experiment_end()
            return logger.get_sacred_results()


@train_ex.capture
def get_agents(_run, env, num_adversaries, good_policy, adv_policy, lr, batch_size,
               buff_size, num_units, num_lstm_units, num_layers, gamma, tau, priori_replay, alpha, num_episodes,
               max_episode_len, beta, policy_update_rate, critic_action_noise_stddev,
               entropy_coeff, num_atoms, min_val, max_val) -> List[AbstractAgent]:
    """
    This function generates the agents for the environment. The parameters are meant to be filled
    by sacred, and are therefore documented in the configuration function train_config.

    :returns List[AbstractAgent] returns a list of instantiated agents
    """
    agents = []
    for agent_idx in range(num_adversaries):
        if adv_policy == 'maddpg':
            agent = MADDPGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, num_lstm_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                _run=_run)
        elif adv_policy == 'matd3':
            agent = MATD3Agent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers,
                               num_units, num_lstm_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               policy_update_freq=policy_update_rate,
                               target_policy_smoothing_eps=critic_action_noise_stddev, _run=_run
                               )
        elif adv_policy == 'mad3pg':
            agent = MAD3PGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                num_atoms=num_atoms, min_val=min_val, max_val=max_val,
                                _run=_run
                                )
        elif adv_policy == 'masac':
            agent = MASACAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, num_lstm_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               entropy_coeff=entropy_coeff, policy_update_freq=policy_update_rate,
                               _run=_run
                               )
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    for agent_idx in range(num_adversaries, env.n):
        if good_policy == 'maddpg':
            agent = MADDPGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, num_lstm_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                _run=_run)
        elif good_policy == 'matd3':
            agent = MATD3Agent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, num_lstm_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               policy_update_freq=policy_update_rate,
                               target_policy_smoothing_eps=critic_action_noise_stddev, _run=_run
                               )
        elif adv_policy == 'mad3pg':
            agent = MAD3PGAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                                buff_size,
                                lr, num_layers,
                                num_units, gamma, tau, priori_replay, alpha=alpha,
                                max_step=num_episodes * max_episode_len, initial_beta=beta,
                                num_atoms=num_atoms, min_val=min_val, max_val=max_val,
                                _run=_run
                                )
        elif good_policy == 'masac':
            agent = MASACAgent(env.observation_space, env.action_space, agent_idx, batch_size,
                               buff_size,
                               lr, num_layers, num_units, num_lstm_units, gamma, tau, priori_replay, alpha=alpha,
                               max_step=num_episodes * max_episode_len, initial_beta=beta,
                               entropy_coeff=entropy_coeff, policy_update_freq=policy_update_rate,
                               _run=_run
                               )
        else:
            raise RuntimeError('Invalid Class')
        agents.append(agent)
    print('Using good policy {} and adv policy {}'.format(good_policy, adv_policy))
    return agents


def main():
    # use this code to attach a mongo database for logging
    # mongo_observer = MongoObserver(url='localhost:27017', db_name='sacred')
    # train_ex.observers.append(mongo_observer)
    if (not is_display) and (not is_evaluate): 
        # file_observer = FileStorageObserver(os.path.join('learned_results', scenario))
        file_observer = FileStorageObserver(os.path.join(save_dir))
    else:
        if is_evaluate: 
            ex_path = os.path.join(load_dir.replace('/models', ''), "eval")
        else:
            ex_path = os.path.join(load_dir.replace('/models', ''), "demo")
        file_observer = FileStorageObserver(ex_path)
    train_ex.observers.append(file_observer)
    train_ex.run_commandline()


if __name__ == '__main__':
    main()
